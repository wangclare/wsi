
import torch
print(torch.version.cuda)  # 看 PyTorch 用的 CUDA 版本
print(torch.cuda.is_available())  # 看环境里能不能识别 GPU（目前应该是 False）
print(torch.backends.cudnn.version())  # 看 cudnn 版本
'''
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

'''