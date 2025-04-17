'''
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
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    buffer_imgs = []
    buffer_coords = []
    buffer_size = 100  # 可调节：50 是一个安全起点

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

            buffer_imgs.append(norm_np.reshape(1, *norm_np.shape))
            buffer_coords.append(coords[i].reshape(1, -1))

            # 写入条件满足
            if len(buffer_imgs) >= buffer_size:
                asset_dict = {
                    'color_tensor': np.concatenate(buffer_imgs, axis=0),
                    'coords': np.concatenate(buffer_coords, axis=0)
                }
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
                buffer_imgs = []
                buffer_coords = []

    # 写入剩余没写的（最后一组 < buffer_size）
    if buffer_imgs:
        asset_dict = {
            'color_tensor': np.concatenate(buffer_imgs, axis=0),
            'coords': np.concatenate(buffer_coords, axis=0)
        }
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)



def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    bags = [f for f in os.listdir(args.data_h5_dir) if f.endswith('.h5')]
    normalizer = TorchstainWithStandardRef()

    if torch.cuda.is_available():
        print("✅ 正在使用 GPU 加速归一化处理")
    else:
        print("⚠️ 当前未使用 GPU，处理可能较慢")

    # 参数控制
    sleep_every = 10        # 每处理 N 张 slide
    sleep_seconds = 5       # 休息秒数
    log_path = os.path.join(args.output_dir, "color_norm_log.csv")
    slide_count = 0

    # 初始化日志
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("slide_id,patch_count,time_sec,status\n")

    for bag_name in bags:
        slide_id = bag_name.split('.h5')[0]
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        output_path = os.path.join(args.output_dir, bag_name)

        if not os.path.exists(slide_file_path):
            print(f"❌ slide 不存在: {slide_file_path}")
            continue
        if os.path.exists(output_path):
            print(f"✅ 已存在，跳过: {slide_id}")
            continue

        print(f"\n🎯 处理 {slide_id}")
        try:
            start = time.time()
            normalize_and_save_patch_features(slide_id, h5_file_path, slide_file_path, output_path, normalizer, args.batch_size)
            duration = time.time() - start
            status = "OK"
        except Exception as e:
            print(f"❌ 错误: {slide_id} - {e}")
            duration = -1
            status = f"FAIL:{str(e)}"

        # 写日志
        patch_count = "?"
        try:
            with h5py.File(output_path, 'r') as f:
                patch_count = f['color_tensor'].shape[0]
        except:
            pass

        with open(log_path, 'a') as f:
            f.write(f"{slide_id},{patch_count},{duration:.2f},{status}\n")

        # 显存清理 + sleep 控制
        torch.cuda.empty_cache()
        slide_count += 1
        if slide_count % sleep_every == 0:
            print(f"😴 已处理 {slide_count} 张，休息 {sleep_seconds} 秒...")
            time.sleep(sleep_seconds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI patch 归一化保存为 HDF5')
    parser.add_argument('--data_h5_dir', type=str, required=True, help='CLAM 生成的 patch 坐标 .h5 文件目录')
    parser.add_argument('--data_slide_dir', type=str, required=True, help='原始 WSI 文件目录')
    parser.add_argument('--output_dir', type=str, required=True, help='归一化后 patch 保存目录')
    parser.add_argument('--slide_ext', type=str, default='.svs', help='WSI 文件扩展名')
    parser.add_argument('--batch_size', type=int, default=1, help='DataLoader 每批加载数量')
    args = parser.parse_args()

    main(args)
'''
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
import pandas as pd

import torch
import torchstain
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from utils.file_utils import save_hdf5

from collections import Counter
import numpy as np
from PIL import Image

def is_patch_valid(img_pil, std_thresh=3, v_thresh=240, dom_thresh=0.98):
    # Convert to RGB numpy array
    img = np.array(img_pil.convert("RGB"))
    
    # 1. RGB Standard Deviation Check
    stds = img.std(axis=(0, 1))
    if not (stds > std_thresh).all():
        return False

    # 2. Brightness Check (HSV V-channel)
    hsv = Image.fromarray(img).convert("HSV")
    v = np.array(hsv)[:, :, 2]
    if v.mean() >= v_thresh:
        return False

    # 3. Dominant Color Check
    small = img_pil.resize((32, 32))
    pixels = np.array(small).reshape(-1, 3)
    pixels = [tuple(p) for p in pixels]
    most_common_ratio = Counter(pixels).most_common(1)[0][1] / len(pixels)
    if most_common_ratio > dom_thresh:
        return False

    return True


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
        norm, _, _ = self.normalizer.normalize(I=img_tensor, stains=True)
        return norm.to("cpu")

def normalize_and_save_patch_features(slide_id, h5_file_path, slide_file_path, output_path, normalizer, batch_size, filtered_log, failed_log):
    wsi = openslide.open_slide(slide_file_path)
    dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, wsi=wsi, img_transforms=normalizer.T)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    buffer_imgs, buffer_coords = [], []
    failed_any = False
    buffer_size = 100

    for batch in tqdm(loader, desc=f"归一化中: {slide_id}"):
        batch_imgs = batch['img']
        coords = batch['coord'].numpy().astype(np.int32)

        for i in range(len(batch_imgs)):
            pil_img = transforms.ToPILImage()(batch_imgs[i] / 255.0)
            if not is_patch_valid(pil_img):
                filtered_log.append([slide_id, coords[i][0], coords[i][1]])
                continue
            try:
                norm_tensor = normalizer.apply_tensor(batch_imgs[i])
                norm_np = norm_tensor.byte().numpy()
            except Exception as e:
                failed_any = True
                failed_log.append([slide_id, coords[i][0], coords[i][1], str(e)])
                continue

            buffer_imgs.append(norm_np.reshape(1, *norm_np.shape))
            buffer_coords.append(coords[i].reshape(1, -1))

    if not buffer_imgs or failed_any:
        return False

    mode = 'w'
    for i in range(0, len(buffer_imgs), buffer_size):
        asset_dict = {
            'color_tensor': np.concatenate(buffer_imgs[i:i+buffer_size], axis=0),
            'coords': np.concatenate(buffer_coords[i:i+buffer_size], axis=0)
        }
        save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
        mode = 'a'
    return True

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    bags = [f for f in os.listdir(args.data_h5_dir) if f.endswith('.h5')]
    normalizer = TorchstainWithStandardRef()

    filtered_log, failed_log = [], []

    if torch.cuda.is_available():
        print("✅ 正在使用 GPU 加速归一化处理")
    else:
        print("⚠️ 当前未使用 GPU，处理可能较慢")

    log_path = os.path.join(args.output_dir, "color_norm_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("slide_id,patch_count,time_sec,status\n")

    for bag_name in bags:
        slide_id = bag_name.split('.h5')[0]
        h5_file_path = os.path.join(args.data_h5_dir, bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        output_path = os.path.join(args.output_dir, bag_name)

        if not os.path.exists(slide_file_path):
            continue
        if os.path.exists(output_path):
            continue

        print(f"🎯 处理 {slide_id}")
        try:
            start = time.time()
            success = normalize_and_save_patch_features(
                slide_id, h5_file_path, slide_file_path, output_path,
                normalizer, args.batch_size,
                filtered_log, failed_log
            )
            duration = time.time() - start
            status = "OK" if success else "FAIL:no valid patch"
        except Exception as e:
            duration = -1
            status = f"FAIL:{str(e)}"

        patch_count = "?"
        try:
            if status == "OK":
                with h5py.File(output_path, 'r') as f:
                    patch_count = f['color_tensor'].shape[0]
        except:
            pass

        with open(log_path, 'a') as f:
            f.write(f"{slide_id},{patch_count},{duration:.2f},{status}\n")

        torch.cuda.empty_cache()

    pd.DataFrame(filtered_log, columns=["slide_id", "x", "y"]).to_csv(
        os.path.join(args.output_dir, "filtered_patches.csv"), index=False)
    pd.DataFrame(failed_log, columns=["slide_id", "x", "y", "error"]).to_csv(
        os.path.join(args.output_dir, "failed_patches.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI patch 归一化保存为 HDF5')
    parser.add_argument('--data_h5_dir', type=str, required=True)
    parser.add_argument('--data_slide_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()
    main(args)

#python color_norm.py --data_h5_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/BRCATest" --data_slide_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/BRCATestSlide" --output_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/BRCATestNorm"
#srun --account=intro_vsc37341 --partition=gpu_p100 --clusters=genius --nodes=1 --gpus-per-node=1 --time=01:00:00 --mem=24G --pty bash
