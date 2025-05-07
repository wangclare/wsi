import os
import re
import argparse
import csv
from shapely.geometry import box
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def compute_patch_overlap(folder, out_csv):
    patch_groups = defaultdict(list)
    pattern = re.compile(r'_x(\d+)_y(\d+)\.png$')

    for img_path in Path(folder).rglob("*.png"):
        match = pattern.search(img_path.name)
        if not match:
            continue
        x, y = map(int, match.groups())
        case_id = img_path.parent.name
        patch_groups[case_id].append((x, y, img_path.name))

    overlap_results = []

    for case_id, patches in tqdm(patch_groups.items(), desc="分析病例"):
        n = len(patches)
        patches.sort()
        for i in range(n):
            x1, y1, name1 = patches[i]
            box1 = box(x1, y1, x1+224, y1+224)
            for j in range(i+1, min(i+11, n)):
                x2, y2, name2 = patches[j]
                box2 = box(x2, y2, x2+224, y2+224)
                inter_area = box1.intersection(box2).area
                if inter_area > 0:
                    ratio = inter_area / (224 * 224)
                    overlap_results.append({
                        'case_id': case_id,
                        'img1': name1,
                        'img2': name2,
                        'overlap_ratio': round(ratio * 100, 2)  # 转为百分比格式
                    })

    if not overlap_results:
        print("❗ 没有检测到重叠 patch")
        return

    ratios = [r['overlap_ratio'] for r in overlap_results]
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    max_ratio = np.max(ratios)
    min_ratio = np.min(ratios)

    print(f"\n✅ 分析完成（共检测 {len(overlap_results)} 对重叠 patch）")
    print(f"📊 平均重叠率: {mean_ratio:.2f}%")
    print(f"🔺 中位数重叠率: {median_ratio:.2f}%")
    print(f"🔍 最大重叠率: {max_ratio:.2f}%")
    print(f"🔽 最小重叠率: {min_ratio:.2f}%")

    # 分段统计
    bins = list(range(0, 101, 10))
    bin_labels = [f"{bins[i]}–{bins[i+1]}%" for i in range(len(bins) - 1)]
    counts = [0] * (len(bins) - 1)

    for r in ratios:
        for i in range(len(bins) - 1):
            if bins[i] <= r < bins[i+1] or (r == 100 and i == len(bins) - 2):
                counts[i] += 1
                break

    print("\n📦 重叠率分布：")
    for label, count in zip(bin_labels, counts):
        percent = count / len(ratios) * 100
        print(f"  {label:<10}: {count:>7} ({percent:.2f}%)")

    # 写入 CSV 文件（附加统计信息）
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'img1', 'img2', 'overlap_ratio'])
        writer.writeheader()
        writer.writerows(overlap_results)

        # 空行后写统计
        f.write("\n# 统计信息\n")
        f.write(f"# 总共重叠对数: {len(overlap_results)}\n")
        f.write(f"# 平均重叠率: {mean_ratio:.2f}%\n")
        f.write(f"# 中位数重叠率: {median_ratio:.2f}%\n")
        f.write(f"# 最大重叠率: {max_ratio:.2f}%\n")
        f.write(f"# 最小重叠率: {min_ratio:.2f}%\n")
        for label, count in zip(bin_labels, counts):
            percent = count / len(ratios) * 100
            f.write(f"# 区间 {label:<10}: {count:>7} ({percent:.2f}%)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    compute_patch_overlap(args.folder, args.out_csv)
