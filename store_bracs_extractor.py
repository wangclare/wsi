import torch

ckpt_path = "/data/leuven/373/vsc37341/wsi_code/checkpoints/simclr_bracs/simclr/4/bracs-simclr-4-ep=199.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

# 提取以 "backbone." 开头的参数
encoder_state_dict = {
    k.replace("backbone.", ""): v
    for k, v in ckpt["state_dict"].items()
    if k.startswith("backbone.")
}

# 保存为 encoder.pth
torch.save(encoder_state_dict, "encoder_resnet50_simclr.pth")
print("✅ encoder 提取完成，保存在 encoder_resnet50_simclr.pth")
