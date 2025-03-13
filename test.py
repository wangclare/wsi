'''
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
