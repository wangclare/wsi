import torch
ckpt = torch.load("/data/leuven/373/vsc37341/wsi_code/checkpoints/simclr_bracs/simclr/3/bracs-simclr-3-ep=190.ckpt")
print(f"Checkpoint contains epoch: {ckpt['epoch']}")  # 应该输出190
print(f"Model keys: {ckpt['state_dict'].keys()}")  # 确认模型参数存在