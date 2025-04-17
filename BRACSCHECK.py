import os
from PIL import Image
import csv
import numpy as np

log_path = "/scratch/leuven/373/vsc37341/BRACS/train_20x_norm_patches/patch_log.txt"  # 你的日志路径
output_csv = "/scratch/leuven/373/vsc37341/BRACS/train_20x_norm_patches/skipped_patch_stats.csv"

results = []




with open(log_path, 'r') as f:
    for line in f:
        if line.startswith('[SKIP]'):
            # 提取 PNG 图像路径部分
            if ".png" in line:
                png_index = line.find(".png") + 4
                path_raw = line[:png_index].strip()
                path = path_raw.split(']', 1)[1].strip()  # 去掉 [SKIP]

                # 尝试读取图像
                if os.path.exists(path):
                    try:
                        img = Image.open(path).convert('RGB')
                        img_np = np.array(img)

                        height, width = img_np.shape[:2]
                        mean = img_np.mean()
                        std = img_np.std()

                        suspect_empty = mean > 220 and std < 15  # 自定义判断标准
                        results.append([path, width, height, round(mean, 2), round(std, 2), suspect_empty])

                    except Exception as e:
                        results.append([path, "ERROR", "ERROR", "ERROR", "ERROR", str(e)])
                else:
                    results.append([path, "NOT FOUND", "NOT FOUND", "", "", ""])

# 写入 CSV 文件
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'width', 'height', 'mean', 'std', 'suspect_empty'])  # 表头
    writer.writerows(results)

print(f"✅ 分析完成，结果保存在 {output_csv}")

