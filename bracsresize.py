import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

def process_double(csv_path, out_dir, path_col='path'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Double'):
        src = row[path_col]    # 直接用绝对路径
        img = Image.open(src)

        w, h = img.size
        # 短边224，长边448
        if w >= h:
            img2 = img.resize((448, 224), Image.BILINEAR)
            patches = [ img2.crop((0,   0, 224, 224)),
                        img2.crop((224, 0, 448, 224)) ]
        else:
            img2 = img.resize((224, 448), Image.BILINEAR)
            patches = [ img2.crop((0,   0, 224, 224)),
                        img2.crop((0, 224, 224, 448)) ]

        base = os.path.splitext(os.path.basename(src))[0]
        for i, p in enumerate(patches):
            out_f = os.path.join(out_dir, f"{base}_{i}.png")
            p.save(out_f)

def process_resized(csv_path, out_dir, path_col='path'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Resized'):
        src = row[path_col]
        img = Image.open(src).resize((224,224), Image.BILINEAR)
        base = os.path.splitext(os.path.basename(src))[0]
        out_f = os.path.join(out_dir, f"{base}.png")
        img.save(out_f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['double','resized'], required=True)
    parser.add_argument('--csv',    required=True)
    parser.add_argument('--out_dir',required=True)
    args = parser.parse_args()

    if args.mode == 'double':
        process_double(args.csv, args.out_dir)
    else:
        process_resized(args.csv, args.out_dir)
