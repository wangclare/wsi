import torch

ckpt_path = '/data/leuven/373/vsc37341/wsi_code/checkpoints/simclr_bracs/simclr/4/bracs-simclr-4-ep=199.ckpt'  # 改成你的 ckpt 路径
ckpt = torch.load(ckpt_path, map_location='cpu')


# 打印所有 key
print("Top-level keys in ckpt:")
for k in ckpt.keys():
    print(f"- {k}")

print("\nKeys in state_dict:")
for k in ckpt['state_dict']:
    print(k)


