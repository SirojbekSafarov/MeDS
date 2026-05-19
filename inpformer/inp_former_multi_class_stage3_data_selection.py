import _meds_paths  # noqa: F401
import torch
import torch.nn as nn
import numpy as np
import os
import copy
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, cal_anomaly_maps_wo_rsize
import csv
import time
import threading

# Dataset-Related Modules
from dataset import MVTecDataset, RealIADDataset
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block
from matplotlib import pyplot as plt

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
            label = 1
        else:
            label = 0

        return image, label

def distilled_model_data_sampling(distilled_model, data_list):

    distilled_model_samples = {}
    distilled_model.eval()
    for data in data_list:
        batch_size = min(len(data), 32)
        train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=10,
                                                       drop_last=False)
        anomaly_maps_distill = None
        binary_list = []
        with torch.no_grad():
            for idx, (img, label) in enumerate(train_dataloader):
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

def visualize_selected_samples(anomaly_maps, anomaly_maps_distill,weight_memory, save_dir_class, class_name, binary_list, thr):
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
    ax.set_title('Histogram of distill samples')
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
        'distilled_model_top2_mean_anomaly_score': mean_anomaly_maps_top_2_dis.cpu().numpy(),
        'normalized_tensor_comb': normalized_tensor.cpu().numpy(),
        'mean_anomaly_maps_top_1': mean_anomaly_maps_top_1.cpu().numpy(),
        'mean_anomaly_maps_top_5': mean_anomaly_maps_top_5.cpu().numpy(),
        'mean_anomaly_maps_top_1_dis': mean_anomaly_maps_top_1_dis.cpu().numpy(),
        'mean_anomaly_maps_top_5_dis': mean_anomaly_maps_top_5_dis.cpu().numpy(),
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
            
def cosine_data_sample(model, distilled_model_samples, data_list, save_dir, iter, total_iters, z=3):
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
        anomaly_maps_distill = distilled_model_samples[data.classes]

        binary_list = []
        with torch.no_grad():
            for idx, (img, label) in enumerate(train_dataloader):
                if iter == 0 and idx == 0:
                    anomaly_maps = anomaly_maps_distill
                elif iter != 0:
                    img = img.to(device)
                    output = model(img)
                    en, de = output[0], output[1]

                    anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)

                    if idx == 0:
                        anomaly_maps = anomaly_map
                    else:
                        anomaly_maps = torch.cat([anomaly_maps, anomaly_map], dim=0)


                binary_list.extend(list(label))


        weight_memory = max(1 - (iter / (total_iters / 2)), 0)
        top_1p = int(anomaly_maps.view(anomaly_maps.size(0), -1).shape[-1] * 0.01)
        mean_anomaly_maps_top_1 = torch.sort(anomaly_maps.view(anomaly_maps.size(0), -1), dim=1, descending=True)[
                                    0][:, :top_1p].mean(dim=1)
        mean_anomaly_maps_top_1_dis = \
            torch.sort(anomaly_maps_distill.view(anomaly_maps_distill.size(0), -1), dim=1, descending=True)[
                0][:, :top_1p].mean(dim=1)
        mean_anomaly_maps_top_1_comb = (1 - weight_memory) * mean_anomaly_maps_top_1.unsqueeze(
            0) + weight_memory * mean_anomaly_maps_top_1_dis.unsqueeze(0)
        mean_anomaly_maps_top_1_comb = mean_anomaly_maps_top_1_comb.view(-1)

        median = torch.median(mean_anomaly_maps_top_1_comb)

        we =min(iter / (total_iters), 1)
        abs_dev = torch.abs(mean_anomaly_maps_top_1_comb - median)
        mad = torch.median(abs_dev)

        h = median

        thr = h + we*mad*z

                # Create and start the thread
        thread = threading.Thread(target=visualize_selected_samples, args=(anomaly_maps, anomaly_maps_distill,weight_memory, save_dir_class, data.classes, binary_list, thr))
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
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)

            train_data = CustomImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif args.dataset == 'Real-IAD' :
        train_data_list = []
        test_data_list = []
        for i, item in enumerate(args.item_list):
            train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                        gt_transform=gt_transform,
                                        phase='train')
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            test_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                       gt_transform=gt_transform,
                                       phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                       drop_last=True)
    # Adopting a grouping-based reconstruction strategy similar to Dinomaly
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Encoder info
    encoder = vit_encoder.load(args.encoder)
    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
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

    distilled_model = INP_Former(encoder=copy.deepcopy(encoder), bottleneck=copy.deepcopy(Bottleneck), aggregation=copy.deepcopy(INP_Extractor), decoder=copy.deepcopy(INP_Guided_Decoder),
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)

    model_state_dict = torch.load(args.distilled_model_path, map_location='cpu')
    distilled_model.load_state_dict(model_state_dict)
    distilled_model = distilled_model.to(device)
    distilled_model.eval()

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    if args.phase == 'train':
        # Model Initialization
        trainable = nn.ModuleList([Bottleneck, INP_Guided_Decoder, INP_Extractor, INP])
        # for m in trainable.modules():
        #     if isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)
        # define optimizer
        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        it = 0
        epoch = 0
        total_iters=args.total_epochs*len(train_dataloader)

        distilled_model_samples = distilled_model_data_sampling(distilled_model=distilled_model, data_list=train_data_list)
        # for epoch in range(args.total_epochs):
        while True:
            
            train_data_list_sub = cosine_data_sample(model=model, distilled_model_samples=distilled_model_samples, data_list=train_data_list, save_dir=os.path.join(args.save_dir, args.save_name), iter=it, total_iters=total_iters, z=3)
            train_data_sub = ConcatDataset(train_data_list_sub)
            print_fn('train image number:{}'.format(len(train_data_sub)))
            train_dataloader_sub = torch.utils.data.DataLoader(train_data_sub, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                            drop_last=True)
        
            model.train()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print_fn(name, param.grad.abs().mean())

            loss_list = []
            for img, _ in tqdm(train_dataloader_sub, ncols=80):
                img = img.to(device)
                en, de, g_loss = model(img)
                loss = global_cosine_hm_adaptive(en, de, y=3)
                loss = loss + 0.2 * g_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()

                # if (it + 1) % int(total_iters / 20) == 0:
                #     print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                #     loss_list = []

                if (it + 1) % (int(total_iters / 5)) == 0:
                    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                    for item, test_data in zip(args.item_list, test_data_list):
                        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
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

                    print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                model.train()
                
                
                if it + 1 >= total_iters:
                    break

                it += 1

            print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
            loss_list = []
            epoch += 1
            
            if it + 1 >= total_iters:
                break

    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        model.eval()
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
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


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio10/MVTech_nr10_seed_0') # Replace it with your path.
    parser.add_argument('--distilled_model_path', type=str, default=f'./saved_noisy_results/mvtec_noisy/Distillation/40_epoch/ratio_10/seed_0/INP-Former-Multi-Class_dataset=MVTec-AD_Encoder=dinov2reg_vit_base_14_Resize=448_Crop=392_INP_num=6/model.pth') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_noisy_results/mvtec_noisy/Dataselection_MeDS/with_40_epoch_distilled_model_and_0.2gloss_and_hm_loss_y3_train_only_40_epoch/ratio_10/seed_0')
    parser.add_argument('--save_name', type=str, default='INP-Former-Multi-Class')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--INP_num', type=int, default=6)

    # training info
    parser.add_argument('--total_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        # args.data_path = 'E:\IMSN-LW\dataset\mvtec_anomaly_detection' # '/path/to/dataset/MVTec-AD/'
        args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    main(args)
