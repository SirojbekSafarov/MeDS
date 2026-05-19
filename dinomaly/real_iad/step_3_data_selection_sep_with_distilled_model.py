import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gc
import math
import time

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse

from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, global_cosine_hm_cosine_thr, WarmCosineScheduler,cal_anomaly_maps,cal_anomaly_maps_wo_rsize, get_gaussian_kernel, visualize

from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import csv
from scipy.stats import lognorm, gamma, weibull_min
from dataset import MVTecDataset, RealIADDataset, RealIADDataset_fuad, RealIADDataset_fuad_mem_recom_distill

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_mad_sd_ratio(data):
    # Ensure data is a NumPy array
    data = np.asarray(data.cpu().numpy())

    # Calculate Median
    median = np.median(data)

    # Calculate Median Absolute Deviation (MAD)
    abs_deviation = np.abs(data - median)
    mad = np.median(abs_deviation)

    # Calculate Mean
    mean = np.mean(data)

    # Calculate Standard Deviation (SD)
    sd = np.std(data, ddof=1)  # ddof=1 provides an unbiased estimator

    # Calculate the ratio MAD/SD
    ratio = mad / sd if sd != 0 else np.nan  # Avoid division by zero

    return mad, sd, ratio

def distilled_model_inference(distilled_model, data_list):
    distilled_model_samples = {}
    distilled_model.eval()
    for data in data_list:
        batch_size = min(len(data), 32)
        train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=10,
                                                       drop_last=False)
        anomaly_maps_distill = None
        with torch.no_grad():
            for idx, (img, _, _) in enumerate(train_dataloader):
                img = img.to(device)
                output_dis = distilled_model(img)
                en_dis, de_dis = output_dis[0], output_dis[1]

                anomaly_map_dis, _ = cal_anomaly_maps_wo_rsize(en_dis, de_dis)
                
                if idx == 0:
                    anomaly_maps_distill = anomaly_map_dis
                else:
                    anomaly_maps_distill = torch.cat([anomaly_maps_distill, anomaly_map_dis], dim=0)
        print_fn(f'{data.classes} Finished')
        distilled_model_samples[data.classes] =  anomaly_maps_distill

    return distilled_model_samples

def cosine_data_sample(model, distilled_model_anomaly_maps, data_list, save_dir, iter, total_iters, angle=25):
    subsets = []

    model.eval()
    for data in data_list:
        save_dir_class = os.path.join(save_dir, str(iter), 'plots', data.classes)
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)

        train_dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=10,
                                                       drop_last=False)
        anomaly_maps = None
        anomaly_maps_distill = distilled_model_anomaly_maps[data.classes]

        c_groups = {}
        # Map each 'C' group to the indices where they occur
        for i, s in enumerate(data.img_paths):
            for c in ['C1', 'C2', 'C3', 'C4', 'C5']:
                if c in s:
                    c_groups.setdefault(c, []).append(i)

        with torch.no_grad():
            for idx, (img, fname, label) in enumerate(train_dataloader):
                if iter == 0 and idx == 0:
                    anomaly_maps = anomaly_maps_distill
                    break
                elif iter != 0:
            
                    img = img.to(device)
                    output = model(img)

                    en, de = output[0], output[1]
                    anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)

                    if idx == 0:
                        anomaly_maps = anomaly_map
                    else:
                        anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)


        weight_memory = max(1 - (iter / (total_iters / 2)), 0)
        we = min(iter / (total_iters), 1)

        left_data_idxs_list = []
        for view, indexes in c_groups.items():
            anomaly_maps_single_view = anomaly_maps[indexes]
            anomaly_maps_distill_single_view = anomaly_maps_distill[indexes]

            top_1p = int(anomaly_maps_single_view.view(anomaly_maps_single_view.size(0), -1).shape[-1] * 0.01)

            mean_anomaly_maps_top_1 = torch.sort(anomaly_maps_single_view.view(anomaly_maps_single_view.size(0), -1), dim=1, descending=True)[
                                          0][:, :top_1p].mean(dim=1)


            mean_anomaly_maps_top_1_dis = \
            torch.sort(anomaly_maps_distill_single_view.view(anomaly_maps_distill_single_view.size(0), -1), dim=1, descending=True)[
                0][:, :top_1p].mean(dim=1)

            mean_anomaly_maps_top_1_comb = (1 - weight_memory) * mean_anomaly_maps_top_1.unsqueeze(
                0) + weight_memory * mean_anomaly_maps_top_1_dis.unsqueeze(0)

            mean_anomaly_maps_top_1_comb = mean_anomaly_maps_top_1_comb.view(-1)

            # Extract data points less than the left edge of the peak bin
            median = torch.median(mean_anomaly_maps_top_1_comb)


            abs_dev = torch.abs(mean_anomaly_maps_top_1_comb - median)
            mad = torch.median(abs_dev)

            thr = median + we*mad*1

            left_data_idxs =  torch.where(mean_anomaly_maps_top_1_comb < thr)[0]
            left_data_idxs_list.append(torch.tensor(indexes)[left_data_idxs.cpu()])

        final_denoised_indexes = torch.unique(torch.cat(left_data_idxs_list))

        # Create a new ImageFolder object with the same root directory
        new_dataset = copy.deepcopy(data)

        defect_count = 0
        normal_count = 0
        defctive_file_names = []
        for image_path in new_dataset.img_paths[final_denoised_indexes.cpu().numpy()]:
            # Extract the filename from the absolute path
            filename = os.path.basename(image_path).split('.')[0]

            # Check if the filename starts with any known defect prefix
            # or contains 'defect' anywhere in the filename
            if  'NG' in filename:
                defect_count += 1
                defctive_file_names.append(filename)
            else:
                normal_count += 1

        class_name = data.classes
        print_fn('')
        print_fn(f'Classname: {class_name}')
        print_fn(
            f'Defective file count: {defect_count} \
            \nNormal file count: {normal_count} \
            ')
            # \nDefective samples: {defctive_file_names} \
        print_fn(f'Weight_memory: {weight_memory}')
        print_fn(f'we: {we}')
        print_fn(f'')

        new_dataset.img_paths = new_dataset.img_paths[final_denoised_indexes.cpu().numpy()]
        new_dataset.gt_paths = new_dataset.gt_paths[final_denoised_indexes.cpu().numpy()]
        new_dataset.labels = new_dataset.labels[final_denoised_indexes.cpu().numpy()]
        new_dataset.types = new_dataset.types[final_denoised_indexes.cpu().numpy()]

        subsets.append(new_dataset)
    return subsets

def train(item_list, save_name):
    setup_seed(1)

    total_iters = 2500
    # total_iters = 50000
    # total_iters = 700000
    # total_iters = 2500
    batch_size = 16
    image_size = 448
    crop_size = 392
    # image_size = 448
    # crop_size = 448

    # test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #                         for test_data in test_data_list]

    # encoder_name = 'dinov2reg_vit_small_14'
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_large_14'

    # encoder_name = 'dinov2_vit_base_14'
    # encoder_name = 'dino_vit_base_16'
    # encoder_name = 'ibot_vit_base_16'
    # encoder_name = 'mae_vit_base_16'
    # encoder_name = 'beitv2_vit_base_16'
    # encoder_name = 'beit_vit_base_16'
    # encoder_name = 'digpt_vit_base_16'
    # encoder_name = 'deit_vit_base_16'

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # target_layers = list(range(4, 19))

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."

    data_transform, gt_transform, _ = get_data_transforms(image_size, crop_size)
    root_path = args.data_path

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_data = RealIADDataset_fuad(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                            phase='train',type=f'realiad_jsons_fuiad_{fuiad}')
        train_data.classes = item
        train_data.class_to_idx = {item: i}

        test_data = RealIADDataset_fuad(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                            phase="test",type=f'realiad_jsons_fuiad_{fuiad}')
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    print_fn('train image number:{}'.format(len(train_data)))
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)



    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))

    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        # blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)

    distilled_bottleneck = copy.deepcopy(bottleneck)
    distilled_decoder = copy.deepcopy(decoder)

    distilled_model = ViTill(encoder=encoder, bottleneck=distilled_bottleneck, decoder=distilled_decoder,
                             target_layers=target_layers, mask_neighbor_size=0,
                             fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    # model_state_dict = torch.load('visa_noise_ratio_10/visa_memory_to_recon_distill_iter_10000_loss_lambda_10_ensemble_100_wo_gausian/mode
    # l.pth', map_location='cpu')
    distill_output_dir = args.distill_output_dir + f'model_{item_list[0]}.pth'
    model_state_dict = torch.load(distill_output_dir, map_location='cpu')
    distilled_model.load_state_dict(model_state_dict, strict=True)
    distilled_model = distilled_model.to(device)
    distilled_model.eval()

    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    trainable = nn.ModuleList([model.bottleneck, model.decoder])

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    distilled_model_anomaly_maps = distilled_model_inference(distilled_model=distilled_model, data_list=train_data_list)

    it = 0
    first_epoch = False
    epoch = 0
    while True:
        loss_list = []
        if not first_epoch:
            train_data_list_sub = cosine_data_sample(model, distilled_model_anomaly_maps, train_data_list, save_name, it, total_iters, angle=22.5)

            train_data_sub = ConcatDataset(train_data_list_sub)
            print_fn('train image number:{}'.format(len(train_data_sub)))
            train_dataloader_sub = torch.utils.data.DataLoader(train_data_sub, batch_size=batch_size, shuffle=True, num_workers=4,
                                                               drop_last=True)
        else:
            train_dataloader_sub = train_dataloader

        first_epoch = False

        model.train()
        for img, _, label in train_dataloader_sub:
            img = img.to(device)

            en, de = model(img)

            p_final = 0.5
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        epoch += 1
        if it == total_iters:
            break
    
    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'model_{item_list[0]}.pth'))


    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                        num_workers=4)
        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse
    fuiad = '0.4'
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--data_path', type=str, default='./MVTech')
    parser.add_argument('--data_path', type=str, default='/path/to/datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='/path/to/experiments/dinomaly/real_iad/single_class/stage_3_data_selection/')
    parser.add_argument('--distill_output_dir', type=str, default=f'/path/to/experiments/dinomaly/real_iad/single_class/stage_2_distillation/fuiad_{fuiad}_iter_2500/')

    parser.add_argument('--save_name', type=str,
                        default=f'fuiad_{fuiad}')
    args = parser.parse_args()
    #
    # item_list = ['audiojack']
    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    for item in item_list:
        train([item], os.path.join(args.save_dir, args.save_name))

    print_fn(
    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))