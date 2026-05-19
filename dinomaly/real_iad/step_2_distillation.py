import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset, RealIADDataset, RealIADDataset_fuad, RealIADDataset_fuad_mem_recom_distill

import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler, visualize, cal_anomaly_maps_wo_rsize, visualize_global_min_max, distillation_SmoothL1_loss, \
    get_gaussian_kernel
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW, AdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import matplotlib
matplotlib.use('Agg')
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

def plot_save(model, data_list, save_dir, iter):
    model.eval()

    for data in data_list:
        save_dir_class = os.path.join(save_dir, str(iter), 'plots', data.classes)
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)

        train_dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=10,drop_last=False)
        anomaly_maps = None
        anomaly_maps_memory = None
        binary_list = []
        with torch.no_grad():
            for idx, (img, label, filename) in enumerate(train_dataloader):
                img = img.to(device)
                output = model(img)
                en, de = output[0], output[1]
                anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)

                label = label.to(device)
                label = label.reshape((1,28,28))
                anomaly_map_memory = label

                if idx == 0:
                    anomaly_maps = anomaly_map
                    anomaly_maps_memory = anomaly_map_memory
                else:
                    anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)
                    anomaly_maps_memory = torch.cat([anomaly_maps_memory, anomaly_map_memory], dim=0)

                # filename = os.path.basename(data.imgs[idx][0]).split('.')[0]

                # Check if the filename starts with any known defect prefix
                # or contains 'defect' anywhere in the filename
                if 'NG' in filename[0]:
                    binary_list.append(1)
                else:
                    binary_list.append(0)

        top_05p_as_num = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.005)

        trimmed_max_anomaly_maps_top_2_recon = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                            0][:,top_05p_as_num:].max(dim=1)[0]

        trimmed_max_anomaly_maps_top_2_memory = torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
                                            0][:,top_05p_as_num:].max(dim=1)[0]


        top_1p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.01)
        top_2p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.02)
        top_5p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.05)

        mean_anomaly_maps_top_1_recon = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_1p].mean(dim=1)

        mean_anomaly_maps_top_1_memory = torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_1p].mean(dim=1)

        mean_anomaly_maps_top_2_recon = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_2p].mean(dim=1)

        mean_anomaly_maps_top_2_memory = torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_2p].mean(dim=1)

        mean_anomaly_maps_top_5_recon = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_5p].mean(dim=1)

        mean_anomaly_maps_top_5_memory = torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
                                            0][:,:top_5p].mean(dim=1)


        # Min-max normalization

        plot_list = {
            'trimmed_max_anomaly_maps_top_0.5_recon': trimmed_max_anomaly_maps_top_2_recon.cpu().numpy(),
            'top_2_mean_anomaly_maps_top_2_recon': mean_anomaly_maps_top_2_recon.cpu().numpy(),
            'trimmed_max_anomaly_maps_top_0.5_memory': trimmed_max_anomaly_maps_top_2_memory.cpu().numpy(),
            'top_2_mean_anomaly_maps_top_2_memory': mean_anomaly_maps_top_2_memory.cpu().numpy(),
            'mean_anomaly_maps_top_1_recon': mean_anomaly_maps_top_1_recon.cpu().numpy(),
            'mean_anomaly_maps_top_1_memory': mean_anomaly_maps_top_1_memory.cpu().numpy(),
            'mean_anomaly_maps_top_5_recon': mean_anomaly_maps_top_5_recon.cpu().numpy(),
            'mean_anomaly_maps_top_5_memory': mean_anomaly_maps_top_5_memory.cpu().numpy(),
        }

        binary_array = np.array(binary_list)

        for key, value in plot_list.items():
            plt.figure(figsize=(10, 6))

            defective = value * binary_array
            # Plot each tensor
            plt.plot(value, label=key)
            plt.plot(defective, label='defect samples')


            # Add labels and title
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'{data.classes} {key} Plot')

            # Add a legend
            plt.legend()

            plt.grid(True)
            plt.savefig(f'{save_dir_class}/{data.classes}_{key}.png')# Saves as PNG

def train(item_list, save_name):
    setup_seed(0)

    total_iters = 50000
    batch_size = 16
    image_size = 448
    crop_size = 392
    root_path = args.data_path


    data_transform, gt_transform , _ = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):

        train_data = RealIADDataset_fuad_mem_recom_distill(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                            phase='train',type='realiad_jsons_fuiad_0.4', memory_dir='memory_anomaly_maps_ensemble_100_fuad_0.4')
        train_data.classes = item
        train_data.class_to_idx = {item: i}

        test_data = RealIADDataset_fuad(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                            phase="test",type='realiad_jsons_fuiad_0.4')
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    encoder_name = 'dinov2reg_vit_base_14'


    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

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

    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = AdamW([{'params': trainable.parameters()}],
                            lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-4, final_value=2e-5, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label, _ in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)
            # loss = global_cosine(en, de)
            anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)
            anomaly_map = anomaly_map.view(anomaly_map.size(0),-1)

            loss = F.mse_loss(anomaly_map, label) * 10

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

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    
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
    
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))





    # for item, test_data in zip(item_list, test_data_list):
    #     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
    #                                                   num_workers=4)
    #     visualize(model,test_dataloader,device=device, _class_=item, save_name=save_name)
    
    # return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/research/workspaces/sirojbek/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='/research/experiments/siroj/academic/noisy_ad/dinomaly/real_iad/dino_v2_vit_base_backbone/multi_class/stage_2_distillation/')
    parser.add_argument('--save_name', type=str, default='fuiad_0.4')
    args = parser.parse_args()
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
    train(item_list,os.path.join(args.save_dir, args.save_name))

