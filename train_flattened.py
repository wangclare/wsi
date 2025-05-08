import os
import shutil
import argparse
from tqdm import tqdm  # 导入进度条库

def flatten_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    # 先统计所有图片文件的总数（用于进度条）
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                image_files.append((root, file))
    
    # 使用tqdm显示进度条
    for root, file in tqdm(image_files, desc="Flattening images"):
        src_path = os.path.join(root, file)
        dst_path = os.path.join(target_dir, file)
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(file)
            dst_path = os.path.join(target_dir, f"{base}_{hash(root)}{ext}")
        shutil.copy(src_path, dst_path)
    
    print(f"\nAll images flattened to: {target_dir}")
    print(f"Total images processed: {len(image_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Path to source directory with subfolders")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to target flattened directory")
    args = parser.parse_args()
    flatten_images(args.source_dir, args.target_dir)