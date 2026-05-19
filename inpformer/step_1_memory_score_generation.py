import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import gc
import math
import time

import torch
import torch.nn as nn
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

from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, \
    global_cosine_hm_cosine_thr, WarmCosineScheduler, cal_anomaly_maps, cal_anomaly_maps_wo_rsize, get_gaussian_kernel, \
    visualize, evaluation_memory

from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, extra_data=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.extra_data = extra_data  # Add extra data like bounding boxes, metadata, etc.

    def __getitem__(self, index):
        # Get the original image and label
        image, label = super().__getitem__(index)
        file_name = os.path.splitext(self.samples[index][0].split('/')[-1])[0]
        if len(file_name) > 5:
            anomaly_flag = 1
        else:
            anomaly_flag = 0
            
        return image, file_name, anomaly_flag
    
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

def save_tensor(tensor, folder_path, file_name):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define full file path
    file_path = os.path.join(folder_path, str(file_name) + '.pth')

    # Save tensor as .pth file
    torch.save(tensor, file_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train(item_list, save_name, ensemble, image_sampling_ratio, encoder_name='dinov2reg_vit_base_14'):
    setup_seed(1)

    image_size = 448
    crop_size = 392

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

    encoder = vit_encoder.load(encoder_name)

    data_transform, gt_transform, _ = get_data_transforms(image_size, crop_size)

    model = encoder.to(device)
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    memory_score_dict = {}

    for item in item_list:

        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = CustomImageFolder(root=train_path, transform=data_transform)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

        save_dir_class = os.path.join(save_name, 'plots', item)
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)

        batch_size = min(len(train_data), 32)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False)
        train_features = torch.Tensor()
        filename_list = []
        binary_list = []
        with torch.no_grad():
            for idx, (img, filenames, anomaly_flags) in enumerate(train_dataloader):
                img = img.to(device)

                feature = model.prepare_tokens(img)
                for i, blk in enumerate(model.blocks):
                    feature = blk(feature)

                feature = model.norm(feature)
                feature = feature[:, 1 + model.num_register_tokens:, :]

                if idx == 0:
                    train_features = feature
                else:
                    train_features = torch.cat((train_features, feature), dim=0)

                filename_list.extend(filenames)
                binary_list.extend(anomaly_flags)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        results = evaluation_memory(model, test_dataloader,train_features, device, max_ratio=image_sampling_ratio, resize_mask=256, ensemble=ensemble)
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

        train_features = F.normalize(train_features, p=2, dim=-1)
        num_image_samples = math.ceil(train_features.size(0) * image_sampling_ratio)
        ensamble_anomaly_maps = None
        for idx in range(ensemble):
            random_image_indices = torch.randint(0, train_features.size(0), (num_image_samples,))

            rand_image_features = train_features[random_image_indices].view(-1, train_features.size(-1))
            memory_bank = rand_image_features

            anomaly_maps = None
            for i in range(train_features.size(0)):
                # Compute dot product
                cosine_similarities = torch.mm(train_features[i], memory_bank.T)

                # Convert cosine similarity to cosine distance
                cosine_distances = 1 - torch.round(cosine_similarities, decimals=5)

                # Find the minimum cosine distance for each test patch
                min_cosine_distances, _ = torch.min(cosine_distances, dim=1)

                anomaly_map = min_cosine_distances.unsqueeze(0)
                if i == 0:
                    anomaly_maps = anomaly_map
                else:
                    anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)

            anomaly_maps = anomaly_maps.unsqueeze(0)
            if idx == 0:
                ensamble_anomaly_maps = anomaly_maps
            else:
                ensamble_anomaly_maps = torch.cat([ensamble_anomaly_maps, anomaly_maps], dim=0)

        anomaly_maps_memory = ensamble_anomaly_maps.permute(1, 0, 2)
        anomaly_maps_memory = anomaly_maps_memory.mean(1)

        top_1p = int(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1).shape[-1] * 0.01)
        top_2p = int(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1).shape[-1] * 0.02)
        top_5p = int(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1).shape[-1] * 0.05)

        mean_anomaly_maps_top_1_memory = \
        torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
            0][:, :top_1p].mean(dim=1)

        mean_anomaly_maps_top_2_memory = \
        torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
            0][:, :top_2p].mean(dim=1)

        mean_anomaly_maps_top_5_memory = \
        torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
            0][:, :top_5p].mean(dim=1)


        plot_list = {
            'mean_anomaly_maps_top_1_memory': mean_anomaly_maps_top_1_memory.cpu().numpy(),
            'mean_anomaly_maps_top_2_memory': mean_anomaly_maps_top_2_memory.cpu().numpy(),
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
            plt.title(f'{item} {key} Plot')

            # Add a legend
            plt.legend()

            plt.grid(True)
            plt.savefig(f'{save_dir_class}/{item}_{key}.png')  # Saves as PNG
       
        classe_memory_score_dict = {}
        for i, anomaly_map in enumerate(anomaly_maps_memory):
            classe_memory_score_dict[filename_list[i]] = anomaly_map.cpu()
        memory_score_dict[item] = classe_memory_score_dict
        print_fn(item)

    save_tensor(memory_score_dict, os.path.join(args.output_dir), 'memory_scores')

    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA'
    parser.add_argument('--data_path', type=str, default='/research/workspaces/sirojbek/mvtec_noisy/mvtec_noise_ratio10/MVTech_nr10_seed_0')
    parser.add_argument('--output_dir', type=str, default='/research/experiments/siroj/academic/noisy_ad/dinomaly/mvtec/dino_v2_vit_base_backbone/memory_scores_folder/noise_ratio_10/seed_0/ensemble_100_p10')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--ensemble_size', type=int, default=100)
    parser.add_argument('--subsampling_ratio', type=float, default=0.1)
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
    print_fn(f'encoder = {args.encoder}')
    print_fn(f'ensemble_size = {args.ensemble_size}')
    print_fn(f'subsampling_ratio = {args.subsampling_ratio}')

    train(item_list, args.output_dir, args.ensemble_size, args.subsampling_ratio, encoder_name=args.encoder)

