import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math

import torch
import torch.nn as nn
from sympy import legendre

from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
from dataset import MVTecDataset, RealIADDataset,RealIADDataset_fuad
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine_hm_percent, regional_cosine_focal, \
    regional_cosine_hm, WarmCosineScheduler, evaluation_memory_riad
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
import torch
import gc
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
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def save_tensor(tensor, folder_path, file_name):
    # Check if folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define full file path
    file_path = os.path.join(folder_path, str(file_name[0]) + '.pth')

    # Save tensor as .pth file
    torch.save(tensor, file_path)

import time
def train(item_list, save_name):
    setup_seed(1)

    image_size = 448
    crop_size = 392
    ensemble = 100
    data_transform, gt_transform, _ = get_data_transforms(image_size, crop_size)
    root_path = args.data_path

    # encoder_name = 'dinov2reg_vit_small_14'
    encoder_name = 'dinov2reg_vit_base_14'
    encoder = vit_encoder.load(encoder_name)

    model = encoder.to(device)
    memory_score_dict = {}
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
    model.eval()

    for item in item_list:
        save_dir_class = os.path.join(save_name, 'plots', item)
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)

        train_data = RealIADDataset_fuad(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train',type='realiad_jsons_fuiad_0.0')

        test_data = RealIADDataset_fuad(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                    phase='test',type='realiad_jsons_fuiad_0.0')
        
        c_groups = {}
        # Map each 'C' group to the indices where they occur
        for i, s in enumerate(train_data.img_paths):
            for c in ['C1', 'C2', 'C3', 'C4', 'C5']:
                if c in s:
                    c_groups.setdefault(c, []).append(i)


        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=False, num_workers=10,  drop_last=False)
        train_features = torch.Tensor()
        filename_list = []
        binary_list = []
        with torch.no_grad():
            # for idx, (img, filename) in enumerate(train_dataloader):
            for idx, (img, filename, label) in enumerate(train_dataloader):
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


                binary_list.extend(label.tolist())
                # if 'NG' in filename[0]:
                # else:
                #     binary_list.append(0)

                filename_list.append(filename)
            
            print_fn(f'{item} feature extraction finished')
            
            train_features = F.normalize(train_features, p=2, dim=-1)
            
            # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
            # results = evaluation_memory_riad(model, test_dataloader,train_features, device, max_ratio=0.1, resize_mask=256, ensemble=ensemble)
            # auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            # auroc_sp_list.append(auroc_sp)
            # ap_sp_list.append(ap_sp)
            # f1_sp_list.append(f1_sp)
            # auroc_px_list.append(auroc_px)
            # ap_px_list.append(ap_px)
            # f1_px_list.append(f1_px)
            # aupro_px_list.append(aupro_px)

            # print_fn(
            #     '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            #         item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

            num_image_samples = math.ceil(train_features.size(0) * 0.1)
            train_features_cpu = train_features.to('cpu')
            # 2. Delete the GPU reference
            del train_features 

            # 3. Force Garbage Collection and Clear Cache
            gc.collect()
            torch.cuda.empty_cache()
            
            ensamble_anomaly_maps = None
            for idx in range(ensemble):
                # random_image_indices = torch.randint(0, train_features.size(0), (num_image_samples,))
                selected_indexes = []
                # Ensure at least one index from each 'C' group is included
                for indices in c_groups.values():
                    if indices:
                        num_samples = math.ceil(len(indices) * 0.1)
                        random_selection = random.sample(indices, num_samples)
                        selected_indexes.extend(random_selection)

                random_image_indices = torch.tensor(selected_indexes)
                rand_image_features = train_features_cpu[random_image_indices].view(-1, train_features_cpu.size(-1))
                memory_bank = rand_image_features.to(device)
                # memory_bank = rand_image_features

                anomaly_maps = None
                num_images = train_features_cpu.size(0)
                num_patches = train_features_cpu.size(1)
                # BATCH_SIZE=16
                BATCH_SIZE=20
                # for i in range(train_features.size(0)):
                start = time.time()
                for i in range(0, num_images, BATCH_SIZE):
                    batch_features = train_features_cpu[i : i + BATCH_SIZE].to(device)
                    # batch_features = train_features_cpu[i : i + BATCH_SIZE]
                    curr_batch_size = batch_features.size(0)
                    
                    # Flatten batch for matrix multiplication: (Batch * Patches, Dim)
                    flat_batch = batch_features.reshape(-1, train_features_cpu.size(-1))

                    # Compute dot product for the whole batch at once
                    # (Batch * Patches, Dim) @ (Dim, M) -> (Batch * Patches, M)

                    # Convert to distance and find minimums
                    cosine_distances = 1 - torch.round(torch.mm(flat_batch, memory_bank.T), decimals=5)
                    min_distances, _ = torch.min(cosine_distances, dim=1)
                    anomaly_map = min_distances.view(curr_batch_size, num_patches).to('cpu')

                    if i == 0:
                        anomaly_maps = anomaly_map
                    else:
                        anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)
                    

                    # 2. Delete the GPU reference
                    del cosine_distances 
                    del min_distances 
                    del anomaly_map 

                    # 3. Force Garbage Collection and Clear Cache
                    gc.collect()
                    torch.cuda.empty_cache()

                print_fn(f'{time.time() - start} Time for one ensamble')
                
                del memory_bank  

                # 3. Force Garbage Collection and Clear Cache
                gc.collect()
                torch.cuda.empty_cache()
            
                anomaly_maps = anomaly_maps.unsqueeze(0)
                if idx == 0:
                    ensamble_anomaly_maps = anomaly_maps
                else:
                    ensamble_anomaly_maps = torch.cat([ensamble_anomaly_maps, anomaly_maps], dim=0)

            print_fn(f'{item} ensemble anomaly map creation finished')

            anomaly_maps_memory = ensamble_anomaly_maps.permute(1, 0, 2)
            anomaly_maps_memory = anomaly_maps_memory.mean(1)

            top_1p = int(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1).shape[-1] * 0.01)

            mean_anomaly_maps_top_1_memory = \
            torch.sort(anomaly_maps_memory.view(anomaly_maps_memory.size(0), -1), dim=1, descending=True)[
                0][:, :top_1p].mean(dim=1)


            plot_list = {
                'top_2_mean_anomaly_maps_top_1_memory': mean_anomaly_maps_top_1_memory.cpu().numpy(),
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
                plt.savefig(f'{save_dir_class}/{item}_{key}.png') 

            classe_memory_score_dict = {}
            for i, anomaly_map in enumerate(anomaly_maps_memory):
                classe_memory_score_dict[filename_list[i]] = anomaly_map.cpu()
            memory_score_dict[item] = classe_memory_score_dict
            print_fn(item)

    save_tensor(memory_score_dict, os.path.join(args.output_dir), 'memory_scores')

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/research/workspaces/sirojbek/Real-IAD')
    # parser.add_argument('--data_path', type=str, default='/home/aiv/D/_datasets/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='/research/experiments/siroj/academic/noisy_ad/dinomaly/real_iad/dino_v2_vit_base_backbone/memory_scores_folder/fuad_0')
    parser.add_argument('--save_name', type=str,
                        default='real_aid_memory_ensemble_save_fuad_0.0')
    args = parser.parse_args()
    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    # item_list = ['button_battery','phone_battery', 'porcelain_doll', 'rolled_strip_base']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train(item_list,os.path.join(args.save_dir, args.save_name))
