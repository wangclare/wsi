import os
import argparse
import logging
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50


class ColorNormH5Dataset(Dataset):
    """读取 HDF5 中已归一化的 patches（224×224）和坐标"""
    def __init__(self, h5_path, transform=None):
        with h5py.File(h5_path, 'r') as f:
            self.imgs = f['color_tensor'][:]  # uint8 [N,224,224,3]
            self.coords = f['coords'][:]      # [N,2]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # numpy uint8 [224,224,3]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # float [3,224,224] in [0,1]
        if self.transform:
            img = self.transform(img)
        coord = self.coords[idx]
        return img, coord


class SimCLRResNet50Extractor(torch.nn.Module):
    """加载 SimCLR 训练的 ResNet50 encoder，去掉分类头"""
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)  # 初始化结构，不加载 ImageNet 权重
        self.model.fc = torch.nn.Identity()  # 去掉分类头，
        ckpt_path = "/data/leuven/373/vsc37341/wsi_code/encoder_resnet50_simclr.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt) #输出 [B,2048]
        
    def forward(self, x):
        return self.model(x)  # 输出: Tensor[B, 2048]


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'simclr_feature_extraction.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def main(args):
    setup_logging(args.feat_dir)
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # 标准 ImageNet 归一化
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225])

    model = SimCLRResNet50Extractor().to(device)
    model.eval()

    pt_dir = os.path.join(args.feat_dir, 'pt_files')
    os.makedirs(pt_dir, exist_ok=True)

    h5_files = sorted(f for f in os.listdir(args.data_h5_dir) if f.endswith('.h5'))
    for fname in h5_files:
        slide_id = os.path.splitext(fname)[0]
        out_pt = os.path.join(pt_dir, slide_id + '.pt')
        if os.path.exists(out_pt):
            logging.info(f'SKIP {slide_id}')
            continue

        h5_path = os.path.join(args.data_h5_dir, fname)
        ds = ColorNormH5Dataset(h5_path, transform=transform)
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

        all_feats, all_coords = [], []
        with torch.no_grad():
            for imgs, coords in loader:
                imgs = imgs.to(device, non_blocking=True)  # [B,3,224,224]
                feats = model(imgs).cpu()                  # [B,2048]
                all_feats.append(feats)
                all_coords.append(coords)

        feats_tensor = torch.cat(all_feats, dim=0)   # [N,2048]
        coords_tensor = torch.cat(all_coords, dim=0) # [N,2]
        torch.save({'features': feats_tensor, 'coords': coords_tensor}, out_pt)
        logging.info(f'DONE {slide_id}: patches={feats_tensor.size(0)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract SimCLR-ResNet50 features from normalized H5 patches')
    parser.add_argument('--data_h5_dir', required=True, help='输入 H5 patch 文件所在目录')
    parser.add_argument('--feat_dir', required=True, help='输出特征保存目录')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda', help='cuda 或 cpu')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)

    main(args)
#python extract_feature_simclr.py --data_h5_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/Testslide" --feat_dir    "/scratch/leuven/373/vsc37341/TCGA-BRCA/Testslide_ftsimclr" --batch_size 256 --num_workers 4 --device cuda