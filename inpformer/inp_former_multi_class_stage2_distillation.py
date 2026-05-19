import _meds_paths  # noqa: F401
import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import argparse
from optimizers import StableAdamW
from utils import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, cal_anomaly_maps_wo_rsize

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
import matplotlib
matplotlib.use('Agg')
from torch.nn import functional as F

class CustomImageFolder(ImageFolder):
    def __init__(self, root, score_dir, transform=None, target_transform=None, item=None,extra_data=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.extra_data = extra_data  # Add extra data like bounding boxes, metadata, etc.
        self.item = item
        # self.glob_mean, self.glob_std = mean_and_stds(self.root)
        self.glob_mean, self.glob_std = 0, 0
        self.memory_anomaly_score_folder = score_dir

    def __getitem__(self, index):
        # Get the original image and label
        image, label = super().__getitem__(index)

        file_name = os.path.splitext(self.samples[index][0].split('/')[-1])[0]
        label_path = os.path.join(os.path.split(self.root)[0],self.memory_anomaly_score_folder, file_name + '.pth')
        label = torch.load(label_path) * 0.8
        return image, label

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
            for idx, (img, label) in enumerate(train_dataloader):
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

                filename = os.path.basename(data.imgs[idx][0]).split('.')[0]

                # Check if the filename starts with any known defect prefix
                # or contains 'defect' anywhere in the filename
                if len(filename) > 3:
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

            train_data = CustomImageFolder(root=train_path, transform=data_transform, score_dir=args.score_dir)
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
    model = model.to(device)

    if args.phase == 'train':
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
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=1e-4, total_iters=args.total_epochs*len(train_dataloader),
                                           warmup_iters=100)
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            loss_list = []
            for img, label in tqdm(train_dataloader, ncols=80):
                img = img.to(device)
                label = label.to(device)

                en, de, g_loss = model(img)

                anomaly_map, _ = cal_anomaly_maps_wo_rsize(en, de)
                anomaly_map = anomaly_map.view(anomaly_map.size(0),-1)
                loss = F.mse_loss(anomaly_map, label) * 10
                # loss = global_cosine_hm_adaptive(en, de, y=3)
                # loss = loss + 0.2 * g_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                loss_list.append(loss.item())
                lr_scheduler.step()
            print_fn('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args.total_epochs, np.mean(loss_list)))
            if (epoch + 1) % args.total_epochs == 0:

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
                plot_save(model, train_data_list, args.save_name, epoch+1)
                model.train()
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
    parser.add_argument('--data_path', type=str, default=r'/path/to/datasets/MVTec-AD-noisy/mvtec_noise_ratio40/MVTech_nr40_seed_0') # Replace it with your path.
    parser.add_argument('--score_dir', type=str,
                        default=f'memory_anomaly_maps_ensemble_{100}_p{10}')
    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_noisy_results/mvtec_noisy/Distillation/40_epoch/ratio_40/seed_0')
    # parser.add_argument('--save_dir', type=str, default='./saved_noisy_results/mvtec_noisy/Distillation/ratio_10/seed_0')
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
