import torch
import os
import argparse
import random

def analyze_feature_file(pt_path):
    try:
        data = torch.load(pt_path)
        feats = data['features']  # [N, D]

        print(f"➡️ File: {os.path.basename(pt_path)}")
        print(f"Shape: {feats.shape}")
        print(f"Min:   {feats.min().item():.4f}")
        print(f"Max:   {feats.max().item():.4f}")
        print(f"Mean:  {feats.mean().item():.4f}")
        print(f"Std:   {feats.std().item():.4f}")
        print(f"NaN count: {torch.isnan(feats).sum().item()}")
        print(f"Inf count: {torch.isinf(feats).sum().item()}")
        print('-'*50)
    except Exception as e:
        print(f"⚠️  Error loading {pt_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str, required=True, help='包含 .pt 特征文件的目录')
    parser.add_argument('--num_samples', type=int, default=5, help='随机抽查几个文件')
    args = parser.parse_args()

    pt_files = [f for f in os.listdir(args.pt_dir) if f.endswith('.pt')]
    if not pt_files:
        print("❌ No .pt files found.")
        exit()

    samples = random.sample(pt_files, min(args.num_samples, len(pt_files)))
    for fname in samples:
        analyze_feature_file(os.path.join(args.pt_dir, fname))
