'''
from PIL import Image
import numpy as np
import torch
import torchstain
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

class TorchstainWithStandardRef:
    def __init__(self, backend='torch'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend=backend)
        self._set_standard_reference()

    def _set_standard_reference(self):
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        maxCRef = np.array([1.9705, 1.0308])

        self.normalizer.stain_matrix = torch.tensor(HERef, dtype=torch.float32).to(self.device)
        self.normalizer.maxC = torch.tensor(maxCRef, dtype=torch.float32).to(self.device)

    def apply_tensor(self, img_tensor):
        norm, _, _ = self.normalizer.normalize(I=img_tensor.to(self.device), stains=True)
        norm_img = norm.byte().cpu().numpy()
        return Image.fromarray(norm_img)


class PatchDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.T(img), os.path.basename(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像文件夹')
    parser.add_argument('--output_dir', type=str, required=True, help='保存归一化图像的文件夹')
    parser.add_argument('--batch_size', type=int, default=32, help='GPU批处理大小')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PatchDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    normalizer = TorchstainWithStandardRef()

    for batch_imgs, batch_names in tqdm(loader, desc="归一化处理中"):
        for i in range(len(batch_imgs)):
            try:
                norm_img = normalizer.apply_tensor(batch_imgs[i])
                norm_img.save(os.path.join(args.output_dir, batch_names[i]))
            except Exception as e:
                print(f"❌ 处理失败: {batch_names[i]}, 错误: {e}")

    print("🎉 全部完成！")
'''
#!/usr/bin/env python3
import os
import csv
import h5py
from collections import defaultdict

# —— 用户配置 —— 
filtered_csv = "/scratch/leuven/373/vsc37341/TCGA-BRCA/color_norm_58059717.csv"
norm_dir     = "/scratch/373/vsc37341/TCGA-BRCA/downsampled_20x_patch/patches"   # e.g. …/BRCATestNorm
slide_ext    = ".h5"
# ————————

# 1) 读 filtered CSV
failed = defaultdict(list)
with open(filtered_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["slide_id"]
        x, y = int(row["x"]), int(row["y"])
        failed[sid].append((x, y))

# 2) 对每个 slide 验证
all_ok = True
for sid, coords in failed.items():
    h5_path = os.path.join(norm_dir, sid + slide_ext)
    if not os.path.exists(h5_path):
        print(f"⚠️ 归一化 H5 不存在: {h5_path}")
        all_ok = False
        continue

    # 读归一化后的 coords
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            print(f"❌ 文件里没 coords 数据集: {h5_path}")
            all_ok = False
            continue
        norm_coords = set(map(tuple, f["coords"][:]))

    # 计算 CSV 坐标里哪些反而出现在了 norm_coords
    intersect = [c for c in coords if c in norm_coords]
    if intersect:
        print(f"❌ {sid} 有 {len(intersect)} 个本该被过滤的坐标竟然还在归一化文件里！示例: {intersect[:5]}")
        all_ok = False
    else:
        print(f"✅ {sid} 的所有 {len(coords)} 个失败坐标均未写入归一化文件。")

# 3) 总结
if all_ok:
    print("\n🎉 验证通过：所有失败坐标都未出现在归一化后的 HDF5 中！")
else:
    print("\n🚨 有部分 slide 验证未通过，请检查上述输出！")
