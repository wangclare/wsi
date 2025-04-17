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
            transforms.Resize((224, 224)),  # 👈 保证所有图像大小一样,仅测试时使用
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
import os
import argparse
import csv

def count_pngs(root_dir):
    """
    遍历 root_dir 下所有子目录（递归），
    收集每个目录中 .png 文件的数量（只记录 >0 的目录）。
    返回列表 [(dirpath, png_count), ...]。
    """
    results = []
    for dirpath, _, filenames in os.walk(root_dir):
        png_count = sum(1 for fname in filenames if fname.lower().endswith('.png'))
        if png_count > 0:
            results.append((dirpath, png_count))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计目录下各子目录的 PNG 数量，并输出 CSV')
    parser.add_argument('root_dir', help='要检查的根目录')
    parser.add_argument('--output_csv', default='png_counts.csv', help='输出的 CSV 文件路径')
    args = parser.parse_args()

    data = count_pngs(args.root_dir)
    # 写入 CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['directory', 'png_count'])
        writer.writerows(data)

    print(f"✅ 已将结果保存到 CSV: {args.output_csv}")