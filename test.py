
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
'''
#!/usr/bin/env python3
import openslide
import pyvips

# 获取 OpenSlide 版本
openslide_version = getattr(openslide, '__version__', 'Unknown')
print("OpenSlide version:", openslide_version)

# 获取 pyvips 版本
pyvips_version = getattr(pyvips, '__version__', 'Unknown')
print("pyvips version:", pyvips_version)
