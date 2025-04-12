import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

def compute_patch_starts(image_size, patch_size):
    if image_size <= patch_size:
        return [0]
    n = int(np.ceil((image_size - patch_size) / patch_size)) + 1
    stride = (image_size - patch_size) / (n - 1) if n > 1 else 0
    starts = [round(i * stride) for i in range(n)]
    return starts

def process_image(image_path, root_dir, output_base, patch_size, log_file_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            with open(log_file_path, 'a') as f:
                f.write(f"[FAIL] {image_path} => Cannot read image\n")
            return

        h, w, _ = img.shape
        x_starts = compute_patch_starts(w, patch_size)
        y_starts = compute_patch_starts(h, patch_size)

        relative_path = image_path.relative_to(root_dir)
        output_dir = output_base / relative_path.parent / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for x0 in x_starts:
            for y0 in y_starts:
                patch = img[y0:y0 + patch_size, x0:x0 + patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue  # skip incomplete patches
                patch_filename = f"{image_path.stem}_x{x0}_y{y0}.png"
                cv2.imwrite(str(output_dir / patch_filename), patch)
                count += 1

        with open(log_file_path, 'a') as f:
            if count > 0:
                f.write(f"[OK] {image_path} => {count} patches\n")
            else:
                f.write(f"[SKIP] {image_path} => No valid patches (too small?)\n")

    except Exception as e:
        with open(log_file_path, 'a') as f:
            f.write(f"[ERROR] {image_path} => {str(e)}\n")

def extract_and_save_patches_from_folder(
    root_dir,
    output_base,
    patch_size=224,
    num_workers=8
):
    root_dir = Path(root_dir)
    output_base = Path(output_base)
    log_file_path = output_base / "patch_log.txt"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    if log_file_path.exists():
        log_file_path.unlink()  # 清空旧日志

    subfolders = [f for f in root_dir.glob("*") if f.is_dir()]
    all_images = []
    for subfolder in subfolders:
        images = list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.tif"))
        all_images.extend(images)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, img_path, root_dir, output_base, patch_size, log_file_path)
            for img_path in all_images
        ]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            pass

if __name__ == '__main__':
    extract_and_save_patches_from_folder(
        root_dir="/scratch/leuven/373/vsc37341/BRACS/train_20x_norm",
        output_base="/scratch/leuven/373/vsc37341/BRACS/train_20x_norm_patches",
        patch_size=224,
        num_workers=8  # 根据你的 CPU 核心数自行调整
    )
