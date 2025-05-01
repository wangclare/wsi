#!/usr/bin/env python3
import os
import argparse
import logging
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

class ColorNormH5Dataset(Dataset):
    """读取 HDF5 中已归一化的 patches（224×224）和坐标"""
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as f:
            self.imgs   = f['color_tensor'][:]  # shape [N,224,224,3]
            self.coords = f['coords'][:]        # shape [N,2]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # numpy uint8 [224,224,3]
        if self.transform:
            img = self.transform(img)  # Tensor [3,224,224]
        coord = self.coords[idx]
        return img, coord

class ResNet50FeatureExtractor(torch.nn.Module):
    """截取 ResNet50 layer3 → 全局池化 → [B,1024]"""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'resnet50', pretrained=True,
            features_only=True, out_indices=(3,)
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        fm = self.backbone(x)[0]               # [B,1024,H,W]
        v  = self.pool(fm).reshape(x.size(0), -1)  # [B,1024]
        return v

def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def main(args):
    # 日志
    os.makedirs(args.feat_dir, exist_ok=True)
    log_file = os.path.join(args.feat_dir, 'process.log')
    setup_logging(log_file)
    logging.info(f'Running on device: {args.device}')

    # 输出目录
    pt_dir = os.path.join(args.feat_dir, 'pt_files')
    os.makedirs(pt_dir, exist_ok=True)

    # 只做 ToTensor + Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),  # uint8 [0,255]→float [0,1]
        transforms.Normalize(   # ImageNet mean/std
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        ),
    ])

    # 模型准备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model  = ResNet50FeatureExtractor().to(device)
    model.eval()

    # 遍历 h5
    h5_files = sorted(f for f in os.listdir(args.data_h5_dir) if f.endswith('.h5'))
    for fname in h5_files:
        slide_id = os.path.splitext(fname)[0]
        out_pt   = os.path.join(pt_dir, slide_id + '.pt')
        if os.path.exists(out_pt):
            logging.info(f'SKIP {slide_id}')
            continue

        h5_path = os.path.join(args.data_h5_dir, fname)
        ds = ColorNormH5Dataset(h5_path, transform=transform)
        loader = DataLoader(ds,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True)

        all_feats, all_coords = [], []
        with torch.no_grad():
            for imgs, coords in loader:
                imgs = imgs.to(device)
                feats = model(imgs).cpu()
                all_feats.append(feats)
                all_coords.append(coords)

        feats_tensor  = torch.cat(all_feats, dim=0)
        coords_tensor = torch.cat(all_coords, dim=0)
        torch.save({'features': feats_tensor, 'coords': coords_tensor}, out_pt)
        logging.info(f'DONE {slide_id}: patches={feats_tensor.size(0)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ResNet50 features from normalized H5')
    parser.add_argument('--data_h5_dir', required=True,
                        help='目录，包含 *.h5（224×224 patches）')
    parser.add_argument('--feat_dir',    required=True,
                        help='输出目录，生成 pt_files/slide_id.pt')
    parser.add_argument('--batch_size',  type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device',      default='cuda',
                        help='cuda 或 cpu')
    args = parser.parse_args()
    main(args)
#python extract_feature_resnet.py --data_h5_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/Testslide" --feat_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/Testslide_ft"