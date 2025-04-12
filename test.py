
'''
import torch
print(torch.version.cuda)  # 看 PyTorch 用的 CUDA 版本
print(torch.cuda.is_available())  # 看环境里能不能识别 GPU（目前应该是 False）
print(torch.backends.cudnn.version())  # 看 cudnn 版本

import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

print("包都装好了，没问题！🎉")

import histomicstk
print(histomicstk.__version__)  # 查看版本号，确认安装成功

import numpy as np
from histomicstk.preprocessing.color_normalization import macenko_stain_normalization

# 随机生成一个小的“图像”矩阵（100x100的RGB图像）
random_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
reference_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

try:
    # 尝试进行颜色归一化
    normalized_image = macenko_stain_normalization(random_image, reference_image)
    print("Macenko normalization works fine on supercomputer!")
except Exception as e:
    print(f"Something went wrong: {e}")'

import histomicstk
print(histomicstk.__version__)

from histomicstk.preprocessing import color_normalization
print(dir(color_normalization))

import openslide
print(openslide.__version__)


import time
import threading
import pyvips

# 假设你有一个较大的图像文件用于测试
test_image_path = '/scratch/leuven/373/vsc37341/TCGA_BRCA/gdc_download_20250321_194522.991199/b7742447-a8c9-4772-894d-30bed44d8b98/TCGA-OL-A5RV-01Z-00-DX1.920AC243-1DAC-4854-BEB6-1CBCC950F26B.svs'

def worker():
    # 这里进行一个简单的降采样操作
    img = pyvips.Image.new_from_file(test_image_path)
    img_downsampled = img.resize(0.5, kernel='lanczos3')
    # 不保存文件，仅进行计算
    _ = img_downsampled.write_to_memory()

def test_threads(num_threads):
    threads = []
    start = time.time()
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    end = time.time()
    print(f"{num_threads} threads took {end - start:.2f} seconds")

if __name__ == '__main__':
    # 串行测试：依次执行
    start = time.time()
    for _ in range(16):
        worker()
    end = time.time()
    print(f"Sequential execution took {end - start:.2f} seconds")
    
    # 并行测试：使用不同线程数
    for n in [2, 4, 8, 16]:
        test_threads(n)

#!/usr/bin/env python3
import openslide
import pyvips

# 获取 OpenSlide 版本
openslide_version = getattr(openslide, '__version__', 'Unknown')
print("OpenSlide version:", openslide_version)

# 获取 pyvips 版本
pyvips_version = getattr(pyvips, '__version__', 'Unknown')
print("pyvips version:", pyvips_version)

import h5py
from PIL import Image
import numpy as np
import os

# 设置路径
h5_path = "路径/TCGA-xxx.h5"     # <- 改成你的实际 h5 文件路径
save_dir = "sample_patches"       # <- 保存图片的文件夹
os.makedirs(save_dir, exist_ok=True)

# 打开 h5 文件并提取图像和坐标
with h5py.File(h5_path, "r") as f:
    patches = f['color_tensor'][:]   # shape: (N, 1, H, W, 3)
    coords = f['coords'][:]

# 去除多余维度（1 通道）
patches = patches.squeeze(1)  # -> shape: (N, H, W, 3)

# 保存前几张为 PNG 图片
for i in range(min(5, len(patches))):
    try:
        img = Image.fromarray(patches[i])
        img.save(os.path.join(save_dir, f"patch_{i}_at_{coords[i][0]}_{coords[i][1]}.png"))
    except Exception as e:
        print(f"⚠️ 第 {i} 张 patch 保存失败: {e}")

python create_patches_fp.py --source "/scratch/leuven/373/vsc37341/TCGA-BRCA/problem3" --save_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/problem3_patch" --patch_size 224 --step_size 224 --preset tcga.csv --seg --patch --stitch         


import os

# 路径按你自己的实际情况替换
wsi_dir = '/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_collection'
h5_dir = '/scratch/leuven/373/vsc37341/TCGA-BRCA/20x_norm/patches'

# 原始文件（.svs 或 .tif）
wsi_names = [os.path.splitext(f)[0] for f in os.listdir(wsi_dir) if f.endswith(('.svs', '.tif'))]

# 成功切图的 .h5 文件
h5_names = [os.path.splitext(f)[0] for f in os.listdir(h5_dir) if f.endswith('.h5')]

# 找出没生成 h5 的 WSI
missing = sorted(set(wsi_names) - set(h5_names))
print("未生成 .h5 文件的 WSI：")
for name in missing:
    print(name)
'''
import h5py
import os
import csv
from pathlib import Path

def get_patch_count(h5_path):
    """根据字段判断使用哪个 dataset 获取 patch 数量"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'color_tensor' in f:
                count = f['color_tensor'].shape[0]
                source = 'color_tensor'
            elif 'coords' in f:
                count = f['coords'].shape[0]
                source = 'coords'
            else:
                print(f"⚠️  文件中没有 color_tensor 或 coords：{h5_path}")
                return 0, 'none'
        return count, source
    except Exception as e:
        print(f"❌ 无法读取 {h5_path}，错误：{e}")
        return 0, 'error'

def count_all_patches(folder_path, save_csv_path=None):
    folder = Path(folder_path)
    results = []
    total = 0

    print(f"\n📁 扫描文件夹: {folder_path}")
    print("-" * 60)
    for h5_file in sorted(folder.glob("*.h5")):
        slide_id = h5_file.stem
        count, source = get_patch_count(str(h5_file))
        total += count
        print(f"{slide_id:<40} ➜ {count:>5} patches  (from: {source})")
        results.append([slide_id, count, source])

    print("-" * 60)
    print(f"✅ 总计 patch 数量: {total}")

    if save_csv_path:
        with open(save_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['slide_id', 'patch_count', 'source'])
            writer.writerows(results)
        print(f"📄 结果已保存为: {save_csv_path}")

    return results
if __name__ == "__main__":
    # 你的 patch .h5 文件夹路径
    folder = "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch/patches"

    # 是否保存为 CSV（可选）
    output_csv = "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch/patch_count_summary.csv"

    count_all_patches(folder, save_csv_path=output_csv)



