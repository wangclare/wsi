#!/usr/bin/env python3
import os
import cv2
import json
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import argparse
from datetime import datetime


def get_file_stats(filepath):
    """获取文件统计信息"""
    try:
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "无法读取图像"}
        return {
            "size_mb": round(size, 2),
            "resolution": f"{img.shape[1]}x{img.shape[0]}"
        }
    except Exception as e:
        return {"error": str(e)}


def process_image(args):
    """处理单个图像 - 修复版"""
    img_path, output_dir, input_dir = args  # 现在明确接收input_dir
    record = {
        "filename": Path(img_path).name,
        "original_path": str(Path(img_path).relative_to(input_dir)),
        "status": "failed",
        "timestamp": datetime.now().isoformat()
    }

    try:
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV读取失败")

        # 计算目标路径
        rel_path = Path(img_path).relative_to(input_dir)
        output_path = Path(output_dir) / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 执行降采样 (40x → 20x)
        h, w = img.shape[:2]
        downsampled = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LANCZOS4)

        # 保存图像
        cv2.imwrite(str(output_path), downsampled, [
            cv2.IMWRITE_PNG_COMPRESSION, 5
        ])

        # 更新记录
        record.update({
            "status": "success",
            "processed_path": str(rel_path),
            "original_resolution": f"{w}x{h}",
            "processed_resolution": f"{w // 2}x{h // 2}",
            "file_stats": {
                "original": get_file_stats(img_path),
                "processed": get_file_stats(output_path)
            }
        })

    except Exception as e:
        record["error"] = str(e)

    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BRACS降采样工具(40x→20x)')
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='并发工作数')
    args = parser.parse_args()

    # 准备输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 收集所有图像文件
    img_files = []
    for ext in ('*.png', '*.jpg', '*.tiff', '*.tif'):
        img_files.extend(Path(args.input).rglob(ext))

    print(f"找到 {len(img_files)} 个图像文件，使用 {args.workers} 个工作线程...")

    # 使用ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 注意现在传递三个参数：img_path, output_dir, input_dir
        futures = [
            executor.submit(process_image, (str(img), args.output, args.input))
            for img in img_files
        ]

        # 处理结果
        processing_log = []
        size_stats = []
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="处理中",
                unit="图像"
        ):
            record = future.result()
            processing_log.append(record)

            if record["status"] == "success":
                orig = record["file_stats"]["original"]["size_mb"]
                proc = record["file_stats"]["processed"]["size_mb"]
                size_stats.append({
                    "file": record["filename"],
                    "original_size": orig,
                    "processed_size": proc,
                    "ratio": round(proc / orig, 3)
                })

    # 保存日志
    log_path = Path(args.output) / "processing_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            "metadata": {
                "input_dir": args.input,
                "output_dir": args.output,
                "workers": args.workers,
                "timestamp": datetime.now().isoformat()
            },
            "records": processing_log,
            "summary": {
                "total": len(processing_log),
                "success": sum(1 for r in processing_log if r["status"] == "success"),
                "average_compression": round(
                    sum(s["ratio"] for s in size_stats) / len(size_stats), 3
                ) if size_stats else 0
            }
        }, f, indent=2)

    print(f"\n处理完成！日志已保存到 {log_path}")
    print(f"成功率: {len(size_stats)}/{len(processing_log)}")