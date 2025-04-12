import sys
sys.path.append("/data/leuven/373/vsc37341/CLAM")
import os
import argparse
import time
import h5py
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchstain
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from utils.file_utils import save_hdf5



class TorchstainWithStandardRef:
    def __init__(self, backend='torch'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend=backend)
        self.normalizer.HERef = self.normalizer.HERef.to(self.device)
        self.normalizer.maxCRef = self.normalizer.maxCRef.to(self.device)
        


        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
    
    def apply_tensor(self, img_tensor):
        img_tensor = img_tensor.to(self.device)
        #Io = torch.tensor(240.0, device=self.device)  # 这行代码不用了，但是留着以作提醒，标量例如这里的float会被pytorch自动broadcast到GPU上
        #beta = torch.tensor(0.15, device=self.device)  
        #norm, _, _ = self.normalizer.normalize(I=img_tensor, Io=Io, beta=beta, stains=True)
        norm, _, _ = self.normalizer.normalize(I=img_tensor, stains=True)
        # Debug 打印
        #print("🎯 DEBUG: img_tensor.device:", img_tensor.device)
        #print("🎯 DEBUG: HERef.device:", self.normalizer.HERef.device)
        #print("🎯 DEBUG: Io.device:", Io.device)

        return norm.to("cpu")  # 确保后续 numpy 转换不会出错
    
    

def normalize_and_save_patch_features(slide_id, h5_file_path, slide_file_path, output_path, normalizer, batch_size):
    wsi = openslide.open_slide(slide_file_path)
    dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=normalizer.T)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    mode = 'w'
    for batch in tqdm(loader, desc=f"归一化中: {slide_id}"):
        batch_imgs = batch['img']
        coords = batch['coord'].numpy().astype(np.int32)

        for i in range(len(batch_imgs)):
            try:
                norm_tensor = normalizer.apply_tensor(batch_imgs[i])
                norm_np = norm_tensor.byte().numpy()
            except Exception as e:
                print(f"❌ 归一化失败 patch: {coords[i]} 错误: {e}")
                continue

            asset_dict = {
                'color_tensor': norm_np.reshape(1, *norm_np.shape),
                'coords': coords[i].reshape(1, -1)
            }
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    bags = [f for f in os.listdir(args.data_h5_dir) if f.endswith('.h5')]
    normalizer = TorchstainWithStandardRef()

    if torch.cuda.is_available():
        print("✅ 正在使用 GPU 加速归一化处理")
    else:
        print("⚠️ 当前未使用 GPU，处理可能较慢")

    for bag_name in bags:
        slide_id = bag_name.split('.h5')[0]
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        output_path = os.path.join(args.output_dir, bag_name)

        if not os.path.exists(slide_file_path):
            print(f"❌ slide 不存在: {slide_file_path}")
            continue

        print(f"\n处理 {slide_id}")
        start = time.time()
        normalize_and_save_patch_features(slide_id, h5_file_path, slide_file_path, output_path, normalizer, args.batch_size)
        print(f"✅ {slide_id} 完成，用时: {time.time() - start:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI patch 归一化保存为 HDF5')
    parser.add_argument('--data_h5_dir', type=str, required=True, help='CLAM 生成的 patch 坐标 .h5 文件目录')
    parser.add_argument('--data_slide_dir', type=str, required=True, help='原始 WSI 文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='归一化后 patch 保存目录')
    parser.add_argument('--slide_ext', type=str, default='.svs', help='WSI 文件扩展名')
    parser.add_argument('--batch_size', type=int, default=1, help='DataLoader 每批加载数量')
    args = parser.parse_args()

    main(args)
