
import os
import re
import openslide
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# === 参数 ===
OUT_FILE = "/scratch/leuven/373/vsc37341/TCGA-BRCA/color_norm_58059717.csv"
H5_DIR = "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch/patches"  # 含有 .h5 文件的目录
SLIDE_DIR = "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_collection"  # 含有 .svs 的目录
OUTPUT_DIR = "/scratch/leuven/373/vsc37341/TCGA-BRCA/badpatches_verified"  # 输出目录
PATCH_SIZE = 224
SLIDE_EXT = ".svs"

# 宽松匹配：处理 patch: [...] 里任意空格、可选逗号
coord_re = re.compile(r"patch:.*?\[\s*(\d+)\s+(\d+)\s*\]")

slide_re = re.compile(r"处理\s+([^\s]+)")

bad_coords = {}
current_slide = None
total_patch_count = 0

with open(OUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # 跳过空行
        if not line:
            continue

        # 匹配 slide 行
        m = slide_re.search(line)
        if m:
            current_slide = m.group(1)
            bad_coords.setdefault(current_slide, [])
            continue

        # 匹配 patch 行
        if "patch" in line and current_slide:
            cm = coord_re.search(line)
            if cm:
                x, y = map(int, cm.groups())
                bad_coords[current_slide].append((x, y))
                total_patch_count += 1
            else:
                # 这段现在应该不会触发
                print(f"❗ 未匹配到坐标: {line}")

print(f"✅ slide 总数: {len(bad_coords)}")
print(f"✅ patch 总数: {total_patch_count}")

# —— 只保留 norm_dir, slide_ext
norm_dir  = "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch_colornorm"
slide_ext = ".h5"

# 假设你前面已经得到了 bad_coords：
# bad_coords = {...}

all_ok = True
for slide_id, coords in bad_coords.items():
    h5_path = os.path.join(norm_dir, slide_id + slide_ext)
    if not os.path.exists(h5_path):
        print(f"⚠️ 归一化 H5 不存在: {h5_path}")
        all_ok = False
        continue

    with h5py.File(h5_path, 'r') as f:
        norm_coords = set(map(tuple, f['coords'][:]))

    bad_in_norm = [c for c in coords if c in norm_coords]
    if bad_in_norm:
        print(f"❌ {slide_id} 有 {len(bad_in_norm)} 个本该被过滤的坐标竟然出现在归一化文件中：{bad_in_norm[:5]}")
        all_ok = False
    else:
        print(f"✅ {slide_id} 的 {len(coords)} 个失败坐标均未写入归一化文件。")

if all_ok:
    print("🎉 验证通过：所有失败坐标都没写入！")

'''
# === 创建输出目录 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 开始提取 ===
missing_slide = []
missing_h5 = []
mismatch_coords = []

for slide_id, coords in tqdm(bad_coords.items(), desc="提取 patch"):
    slide_path = os.path.join(SLIDE_DIR, slide_id + SLIDE_EXT)
    h5_path = os.path.join(H5_DIR, slide_id + ".h5")
    if not os.path.exists(slide_path):
        missing_slide.append(slide_id)
        continue
    if not os.path.exists(h5_path):
        missing_h5.append(slide_id)
        continue

    try:
        wsi = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"❌ 无法打开 slide {slide_id}: {e}")
        continue

    # 校验 patch 是否真实存在于 h5 坐标中
    try:
        with h5py.File(h5_path, 'r') as h5f:
            h5_coords = h5f['coords'][:]
    except Exception as e:
        print(f"❌ 无法读取 h5: {h5_path} 错误: {e}")
        continue

    h5_coord_set = set(map(tuple, h5_coords))

    slide_output_dir = os.path.join(OUTPUT_DIR, slide_id)
    os.makedirs(slide_output_dir, exist_ok=True)

    for i, (x, y) in enumerate(coords):
        if (x, y) not in h5_coord_set:
            mismatch_coords.append((slide_id, x, y))
            continue

        try:
            patch = wsi.read_region((x, y), level=0, size=(PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            save_path = os.path.join(slide_output_dir, f"{slide_id}_{x}_{y}.png")
            patch.save(save_path)
        except Exception as e:
            print(f"⚠️ 读取 patch ({x},{y}) 失败于 {slide_id}: {e}")

# === 总结报告 ===
print(f"\n✅ patch 提取完毕，共成功处理 {len(bad_coords)} 个 slide")
print(f"📁 有 {len(missing_slide)} 个 slide 缺少 .svs 文件")
print(f"📁 有 {len(missing_h5)} 个 slide 缺少 .h5 文件")
print(f"🚫 有 {len(mismatch_coords)} 个 patch 坐标在 .h5 中未找到（可能是坐标不一致）")
'''

