import _meds_paths  # noqa: F401
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import gc
import csv
import time
import copy
import torch
import random
import logging
import warnings
import numpy as np
import torch.nn as nn
from functools import partial
from optimizers import StableAdamW

from torch.utils.data import Subset
from matplotlib import pyplot as plt
from dataset import get_data_transforms
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from dinov1.utils import trunc_normal_
from models.uad import ViTill
from models import vit_encoder
from dataset import MVTecDataset
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler,cal_anomaly_maps_wo_rsize, visualize
import threading
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, item=None,extra_data=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.extra_data = extra_data  # Add extra data like bounding boxes, metadata, etc.
        self.item = item

    def __getitem__(self, index):
        # Get the original image and label
        image, label = super().__getitem__(index)
        # Fetch additional data from the extra_data
        # extra = self.extra_data[index] if self.extra_data else None
        file_name = os.path.splitext(self.samples[index][0].split('/')[-1])[0]

        if len(file_name) > 5:
            anomaly_flag = 1
        else:
            anomaly_flag = 0

        return image, anomaly_flag

class CustomMemoryDataset(ImageFolder):
    def __init__(self, root, memory_output_dir, transform=None, target_transform=None, item=None,extra_data=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.extra_data = extra_data  # Add extra data like bounding boxes, metadata, etc.
        self.item = item
        self.memory_scores = torch.load(os.path.join(memory_output_dir, 'memory_scores.pth'))


    def __getitem__(self, index):
        # Get the original image and label
        image, label = super().__getitem__(index)
        # Fetch additional data from the extra_data
        # extra = self.extra_data[index] if self.extra_data else None
        file_name = os.path.splitext(self.samples[index][0].split('/')[-1])[0]
        
        memory_score = self.memory_scores[self.item][file_name]
        return memory_score
    
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_memory_scores(data_list):

    anomaly_maps_memory = {}
    for data in data_list:
        batch_size = min(len(data), 32)
        train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=10,
                                                       drop_last=False)
        memory_anomaly_maps = None
        with torch.no_grad():
            for idx, (memory_anomaly_map) in enumerate(train_dataloader):
                memory_anomaly_map = memory_anomaly_map.to(device)
                
                if idx == 0:
                    memory_anomaly_maps = memory_anomaly_map
                else:
                    memory_anomaly_maps = torch.cat([memory_anomaly_maps, memory_anomaly_map], dim=0)
        print_fn(f'{data.item} Finished')
        anomaly_maps_memory[data.item] =  memory_anomaly_maps

    return anomaly_maps_memory

def visualize_selected_samples(anomaly_maps, anomaly_maps_distill, weight_memory, save_dir_class, class_name, binary_list, thr):
    top_1p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.01)
    top_2p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.02)
    top_5p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.05)

    mean_anomaly_maps_top_1 = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                    0][:, :top_1p].mean(dim=1)
    mean_anomaly_maps_top_2 = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                    0][:, :top_2p].mean(dim=1)
    mean_anomaly_maps_top_5 = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                    0][:, :top_5p].mean(dim=1)

    mean_anomaly_maps_top_1_dis = \
        torch.sort(anomaly_maps_distill.view(anomaly_maps_distill.size(0), -1), dim=1, descending=True)[
            0][:, :top_1p].mean(dim=1)
    mean_anomaly_maps_top_2_dis = \
        torch.sort(anomaly_maps_distill.view(anomaly_maps_distill.size(0), -1), dim=1, descending=True)[
            0][:, :top_2p].mean(dim=1)
    mean_anomaly_maps_top_5_dis = \
        torch.sort(anomaly_maps_distill.view(anomaly_maps_distill.size(0), -1), dim=1, descending=True)[
            0][:, :top_5p].mean(dim=1)


    mean_anomaly_maps_top_1_comb = (1 - weight_memory) * mean_anomaly_maps_top_1.unsqueeze(
        0) + weight_memory * mean_anomaly_maps_top_1_dis.unsqueeze(0)
    mean_anomaly_maps_top_2_comb = (1 - weight_memory) * mean_anomaly_maps_top_2.unsqueeze(
        0) + weight_memory * mean_anomaly_maps_top_2_dis.unsqueeze(0)
    mean_anomaly_maps_top_5_comb = (1 - weight_memory) * mean_anomaly_maps_top_5.unsqueeze(
        0) + weight_memory * mean_anomaly_maps_top_5_dis.unsqueeze(0)

    mean_anomaly_maps_top_1_comb = mean_anomaly_maps_top_1_comb.view(-1)
    mean_anomaly_maps_top_2_comb = mean_anomaly_maps_top_2_comb.view(-1)
    mean_anomaly_maps_top_5_comb = mean_anomaly_maps_top_5_comb.view(-1)

    # Extract data points less than the left edge of the peak bin
    mean_org = torch.mean(mean_anomaly_maps_top_1_comb)
    std_org = torch.std(mean_anomaly_maps_top_1_comb)
    median = torch.median(mean_anomaly_maps_top_1_comb)
    abs_dev = torch.abs(mean_anomaly_maps_top_1_comb - median)
    mad = torch.median(abs_dev)

    binary_tensor = torch.tensor(binary_list, dtype=torch.bool).to(anomaly_maps.device)
    anomaly_samle_scores_recon = mean_anomaly_maps_top_1[binary_tensor]
    normal_sample_scores_recon = mean_anomaly_maps_top_1[~binary_tensor]

    anomaly_samle_scores_memory = mean_anomaly_maps_top_1_dis[binary_tensor]
    normal_sample_scores_memory = mean_anomaly_maps_top_1_dis[~binary_tensor]

    anomaly_samle_scores_comb = mean_anomaly_maps_top_1_comb[binary_tensor]
    normal_sample_scores_comb = mean_anomaly_maps_top_1_comb[~binary_tensor]

    hist_range = (float(mean_anomaly_maps_top_1_comb.min().cpu().numpy()),
                    float(mean_anomaly_maps_top_1_comb.max().cpu().numpy()))
    # Create figure and axis
    fig, ax = plt.subplots()
    # Plot histograms
    ax.hist(normal_sample_scores_recon.cpu().numpy(), bins=30, alpha=0.5, label='Normal samples', range=hist_range)
    ax.hist(anomaly_samle_scores_recon.cpu().numpy(), bins=30, alpha=0.5, label='Anomaly samles', range=hist_range)

    # Customize the plot
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of recon samples')
    ax.legend()
    plt.savefig(f'{save_dir_class}/{class_name}_recon_sample_histogram.png')  # Saves as PNG

    fig, ax = plt.subplots()
    ax.hist(normal_sample_scores_memory.cpu().numpy(), bins=30, alpha=0.5, label='Normal samples', range=hist_range)
    ax.hist(anomaly_samle_scores_memory.cpu().numpy(), bins=30, alpha=0.5, label='Anomaly samles', range=hist_range)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of memory samples')
    ax.legend()
    plt.savefig(f'{save_dir_class}/{class_name}_distill_sample_histogram.png')  # Saves as PNG

    fig, ax = plt.subplots()
    ax.hist(normal_sample_scores_comb.cpu().numpy(), bins=30, alpha=0.5, label=f'Normal samples, std={std_org}',
            range=hist_range)
    ax.hist(anomaly_samle_scores_comb.cpu().numpy(), bins=30, alpha=0.5, label=f'Anomaly samles, mad = {mad}',
            range=hist_range)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of combined samples')
    ax.legend()
    ax.axvline(x=mean_org.cpu().numpy(), color='red', linestyle='-', linewidth=1,
                label=f'Mean = {mean_org.cpu().numpy()}')
    ax.axvline(x=thr.cpu().numpy(), color='green', linestyle='--', linewidth=1,
                label=f'Thr = {thr.cpu().numpy()}')
    ax.axvline(x=median.cpu().numpy(), color='blue', linestyle='-.', linewidth=1,
                label=f'median = {median.cpu().numpy()}')
    ax.legend()
    plt.savefig(f'{save_dir_class}/{class_name}_comb_sample_histogram.png')  # Saves as PNG

    min_val = mean_anomaly_maps_top_1_comb.min()
    max_val = mean_anomaly_maps_top_1_comb.max()
    normalized_tensor = (mean_anomaly_maps_top_1_comb - min_val) / (max_val - min_val)

    anomaly_normalised_scores_comb = normalized_tensor[binary_tensor]
    normal_normalised_scores_comb = normalized_tensor[~binary_tensor]

    fig, ax = plt.subplots()
    ax.hist(normal_normalised_scores_comb.cpu().numpy(), bins=30, alpha=0.5, label=f'Normal samples')
    ax.hist(anomaly_normalised_scores_comb.cpu().numpy(), bins=30, alpha=0.5, label='Anomaly samles')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of combined min max normalised')
    ax.legend()
    plt.savefig(f'{save_dir_class}/{class_name}_comb_normalised_histogram.png')  # Saves as PNG
    # Min-max normalization
    plot_list = {
        'combined_top2_mean_anomaly_score': mean_anomaly_maps_top_2_comb.cpu().numpy(),
        'model_top2_mean_anomaly_score': mean_anomaly_maps_top_2.cpu().numpy(),
        'memory_mean_anomaly_score': mean_anomaly_maps_top_2_dis.cpu().numpy(),
        'normalized_tensor_comb': normalized_tensor.cpu().numpy(),
        'mean_anomaly_maps_top_1': mean_anomaly_maps_top_1.cpu().numpy(),
        'mean_anomaly_maps_top_5': mean_anomaly_maps_top_5.cpu().numpy(),
        'mean_anomaly_maps_top_1_memory': mean_anomaly_maps_top_1_dis.cpu().numpy(),
        'mean_anomaly_maps_top_5_memory': mean_anomaly_maps_top_5_dis.cpu().numpy(),
        'mean_anomaly_maps_top_1_comb': mean_anomaly_maps_top_1_comb.cpu().numpy(),
        'mean_anomaly_maps_top_5_comb': mean_anomaly_maps_top_5_comb.cpu().numpy(),


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
        plt.title(f'{class_name} {key} Plot')

        # Add a legend
        plt.legend()

        plt.grid(True)
        plt.savefig(f'{save_dir_class}/{class_name}_{key}.png')  # Saves as PNG

def cosine_data_sample(model, memory_anomaly_maps, data_list, save_dir, iter, total_iters, mad_factor=3, beta_end=0.5):
    subsets = []

    model.eval()
    start_sampler = time.time()
    for data in data_list:
        start = time.time()
        save_dir_class = os.path.join(save_dir, str(iter), 'plots', data.classes)
        if not os.path.exists(save_dir_class):
            os.makedirs(save_dir_class)

        batch_size = min(len(data), 32)
        train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=10,
                                                       drop_last=False)
        anomaly_maps = None
        class_memory_anomaly_maps = memory_anomaly_maps[data.classes]

        binary_list = []
        with torch.no_grad():
            for idx, (img, label) in enumerate(train_dataloader):
                img = img.to(device)
                output = model(img)
                en, de = output[0], output[1]

                anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)

                if idx == 0:
                    anomaly_maps = anomaly_map
                else:
                    anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)

                binary_list.extend(list(label))


        weight_memory = max(1 - (iter / (total_iters * beta_end)), 0)
        top_1p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.01)
        mean_anomaly_maps_top_1 = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                    0][:, :top_1p].mean(dim=1)
        mean_anomaly_maps_top_1_dis = \
            torch.sort(class_memory_anomaly_maps.view(class_memory_anomaly_maps.size(0), -1), dim=1, descending=True)[
                0][:, :top_1p].mean(dim=1)
        mean_anomaly_maps_top_1_comb = (1 - weight_memory) * mean_anomaly_maps_top_1.unsqueeze(
            0) + weight_memory * mean_anomaly_maps_top_1_dis.unsqueeze(0)
        mean_anomaly_maps_top_1_comb = mean_anomaly_maps_top_1_comb.view(-1)

        median = torch.median(mean_anomaly_maps_top_1_comb)

        we =min(iter / (total_iters), 1)
        abs_dev = torch.abs(mean_anomaly_maps_top_1_comb - median)
        mad = torch.median(abs_dev)

        h = median

        thr = h + we*mad*mad_factor

                # Create and start the thread
        thread = threading.Thread(target=visualize_selected_samples, args=(anomaly_maps, class_memory_anomaly_maps,weight_memory, save_dir_class, data.classes, binary_list, thr))
        thread.start()

        left_data_idxs = torch.where(mean_anomaly_maps_top_1_comb < thr)[0]

        final_denoised_indexes = torch.unique(torch.cat([left_data_idxs]))
        new_imagefolder = ImageFolder(root=data.root, transform=data.transform)

        defect_count = 0
        normal_count = 0
        defctive_file_names = []
        for image_path, _ in np.array(data.samples)[final_denoised_indexes.cpu().numpy()]:
            # Extract the filename from the absolute path
            filename = os.path.basename(image_path).split('.')[0]

            # Check if the filename starts with any known defect prefix
            # or contains 'defect' anywhere in the filename
            if len(filename) > 5:
                defect_count += 1
                defctive_file_names.append(filename)
            else:
                normal_count += 1

        print_fn(f'Class {data.classes}')
        print_fn(f'Defective file count: {defect_count} \nNormal file count: {normal_count} \nDefective samples: {defctive_file_names}')
        print_fn(f'Weight_memory: {weight_memory}')
        print_fn(f'Weight_epoch: {we}')
        print_fn(f'')

        result_list = [iter, data.classes, normal_count, defect_count, mad.cpu().numpy(), median.cpu().numpy(), we, thr.cpu().numpy()]

        matrix = np.array(result_list)
        flat_matrix = matrix.flatten()
        csv_path = os.path.join(save_dir, 'results.csv')

        # Write header only if file does not exist or is empty
        write_header = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if write_header:
                header = ['iter', 'class', 'normal_count', 'defect_count', 'mad', 'median', 'we', 'thr']
                writer.writerow(header)

            writer.writerow(flat_matrix.tolist())

        new_imagefolder.samples = np.array(data.samples)[final_denoised_indexes.cpu().numpy()].tolist()
        new_imagefolder.imgs = np.array(data.imgs)[final_denoised_indexes.cpu().numpy()].tolist()
        new_imagefolder.targets = np.array(data.targets)[final_denoised_indexes.cpu().numpy()].tolist()

        subsets.append(new_imagefolder)
        print_fn(f'Time for class {data.classes}: {time.time()-start}')
    print_fn(f'Time for one full data sample: {time.time()-start_sampler}')
    return subsets

warnings.filterwarnings("ignore")
def train(item_list, output_dir, memory_output_dir, total_iters=10000, batch_size=16, encoder_name='dinov2reg_vit_base_14', mad_factor=1, beta_end=0.5):
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

    data_transform, gt_transform, _ = get_data_transforms(image_size, crop_size)

    train_data_list = []
    train_data_list_for_memory = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")

        train_data = CustomImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data_list.append(train_data)

        train_data_mem = CustomMemoryDataset(root=train_path, memory_output_dir=memory_output_dir, transform=data_transform)
        train_data_mem.item = item
        train_data_list_for_memory.append(train_data_mem)
        
        test_data_list.append(test_data)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    train_data = ConcatDataset(train_data_list)
    print_fn('train image number:{}'.format(len(train_data)))

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

    trainable = nn.ModuleList([model.bottleneck, model.decoder])

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    memory_anomaly_maps = load_memory_scores(data_list=train_data_list_for_memory)

    it = 0
    epoch = 0
    while True:
        loss_list = []

        train_data_list_sub = cosine_data_sample(model=model, memory_anomaly_maps=memory_anomaly_maps, data_list=train_data_list, save_dir=args.output_dir, iter=it, total_iters=total_iters, mad_factor=mad_factor, beta_end=beta_end)
        train_data_sub = ConcatDataset(train_data_list_sub)
        print_fn('train image number:{}'.format(len(train_data_sub)))
        train_dataloader_sub = torch.utils.data.DataLoader(train_data_sub, batch_size=batch_size, shuffle=True, num_workers=4,
                                                           drop_last=True)

        model.train()
        for img, _ in train_dataloader_sub:
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


            if (it + 1) % int(total_iters / 4) == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

            if (it + 1) % total_iters == 0:
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

        epoch += 1
        if it == total_iters:
            break

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))

    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                      num_workers=4)
        visualize(model,test_dataloader,device=device,_class_=item, output_dir=args.output_dir)

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA'
    parser.add_argument('--data_path', type=str, default='/research/workspaces/sirojbek/mvtec_noisy/mvtec_noise_ratio10/MVTech_nr10_seed_0')
    parser.add_argument('--output_dir', type=str, default=f'/research/experiments/siroj/academic/noisy_ad/dinomaly/mvtec/dino_v2_vit_base_backbone/multi_class/stage_3_pseudo_label_selection/memory_as_dataselector/noise_ratio_10/seed_0/z3/ensemble_100_p10/l2')
    parser.add_argument('--memory_output_dir', type=str, default=f'/research/experiments/siroj/academic/noisy_ad/dinomaly/mvtec/dino_v2_vit_base_backbone/memory_scores_folder/noise_ratio_10/seed_0/ensemble_100_p10')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--beta_end', type=float, default=0.5, help='Use memory scores for data selection until a fraction of the total iterations is reached.')
    parser.add_argument('--mad_factor', type=int, default=3, help='MAD factor to make thr for data selection. Values can be 0,1,2,3')
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
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
    print_fn(f'beta_end = {args.beta_end}')
    print_fn(f'mad_factor = {args.mad_factor}')
    print_fn(f'n_iters = {args.n_iters}')
    print_fn(f'batch_size = {args.batch_size}')

    train(item_list, args.output_dir, args.memory_output_dir, total_iters=args.n_iters, batch_size=args.batch_size, encoder_name=args.encoder, mad_factor=args.mad_factor, beta_end=args.beta_end)
