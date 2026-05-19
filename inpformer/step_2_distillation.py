import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import glob

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models import vit_encoder
from dinov1.utils import trunc_normal_

from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, \
    WarmCosineScheduler, cal_anomaly_maps_wo_rsize
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
import threading
import matplotlib

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

def get_logger(save_path=None, level='INFO'):
    logger = logging.getLogger()
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

class CustomImageFolder(ImageFolder):
    def __init__(self, root, memory_output_dir, transform=None, target_transform=None, item=None,extra_data=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.extra_data = extra_data  # Add extra data like bounding boxes, metadata, etc.
        self.item = item
        self.memory_scores = torch.load(os.path.join(memory_output_dir, 'memory_scores.pth'))

    def __getitem__(self, index):
        # Get the original image and label
        image, _ = super().__getitem__(index)

        file_name = os.path.splitext(self.samples[index][0].split('/')[-1])[0]

        if len(file_name) > 5:
            anomaly_flag = 1
        else:
            anomaly_flag = 0
        
        label = self.memory_scores[self.item][file_name]
        return image, label, anomaly_flag

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
            for idx, (img, label, anomaly_flag) in enumerate(train_dataloader):
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
                binary_list.extend(anomaly_flag)
                # if len(filename) > 5:
                #     binary_list.append(1)
                # else:
                #     binary_list.append(0)

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


def train(item_list, output_dir, memory_output_dir, total_iters=10000, batch_size=16, encoder_name='dinov2reg_vit_base_14', loss_function='l2'):
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = CustomImageFolder(root=train_path, transform=data_transform, memory_output_dir=memory_output_dir, item=item)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    # encoder_name = 'dinov2reg_vit_small_14'
    # encoder_name = 'dinov2reg_vit_base_14'
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

   # Model Preparation
    Bottleneck = []
    INP_Guided_Decoder = []
    INP_Extractor = []

    # bottleneck
    Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
    Bottleneck = nn.ModuleList(Bottleneck)

    # INP
    INP = nn.ParameterList(
                    [nn.Parameter(torch.randn(args.INP_num, embed_dim))
                     for _ in range(1)])

    # INP Extractor
    for i in range(1):
        blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Extractor.append(blk)
    INP_Extractor = nn.ModuleList(INP_Extractor)

    # INP_Guided_Decoder
    for i in range(8):
        blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        INP_Guided_Decoder.append(blk)
    INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

    model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    model = model.to(device)

    # Model Initialization
    trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    # define optimizer
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=total_iters,
                                        warmup_iters=100)
    print_fn('train image number:{}'.format(len(train_data)))
    
    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label, _ in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de, g_loss = model(img)
            anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)
            anomaly_map = anomaly_map.view(anomaly_map.size(0),-1)

            if loss_function == 'l2':
                loss = F.mse_loss(anomaly_map, label) * 10
            elif loss_function == 'l1':
                loss = F.l1_loss(anomaly_map, label) * 10
            else:
                assert f'{loss_function} is not implemented' 

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % total_iters == 0:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{it}.pth'))
                # plot_save(model, train_data_list, output_dir, it+1)
                thread = threading.Thread(target=plot_save, args=(model, train_data_list, output_dir, it+1))
                thread.start()

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

                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
        if it == total_iters:
            break
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    random_seed = 1
    setup_seed(random_seed)

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA'
    parser.add_argument('--data_path', type=str, default='/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio10/MVTech_nr10_seed_0')
    parser.add_argument('--output_dir', type=str, default='/path/to/experiments/inpformer/mvtec/stage_2_distillation/noise_ratio_10/seed_0/ensemble_100_p10/l2')
    parser.add_argument('--memory_output_dir', type=str, default='/path/to/experiments/inpformer/mvtec/memory_scores/noise_ratio_10/seed_0/ensemble_100_p10')
    parser.add_argument('--loss_function', type=str, default=f'l2', help='loss can be l1 or l2')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    # category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        
    logger = get_logger(args.output_dir)
    print_fn = logger.info
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    print_fn(f'loss_function = {args.loss_function}')
    print_fn(f'encoder = {args.encoder}')
    print_fn(f'n_iters = {args.n_iters}')
    print_fn(f'batch_size = {args.batch_size}')

    train(item_list, args.output_dir, args.memory_output_dir, args.n_iters, args.batch_size, encoder_name=args.encoder, loss_function=args.loss_function) 

