#!/usr/bin/env python3
import os
import openslide
import json
import pyvips
from tqdm import tqdm
import shutil
import concurrent.futures
from functools import partial
import threading
import argparse


def get_magnification(properties):
    """从属性中提取倍率信息（增强版）"""
    mag = properties.get('aperio.AppMag') or \
          properties.get('openslide.objective-power') or \
          properties.get('hamamatsu.SourceLens')
    try:
        return float(mag) if mag else None
    except:
        return None


def get_file_stats(filepath):
    """获取文件基础统计信息"""
    try:
        size = os.path.getsize(filepath) / (1024 * 1024)  # 转换为MB
        with openslide.OpenSlide(filepath) as slide:
            width, height = slide.dimensions
        return {
            "size_mb": round(size, 1),
            "resolution": f"{width}x{height}"
        }
    except:
        return {"error": "unable_to_read"}


def process_single_file(svs_path, input_folder, output_folder, size_stats):
    """处理单个SVS文件"""
    try:
        slide = openslide.OpenSlide(svs_path)
        properties = slide.properties
        filename = os.path.basename(svs_path)
        rel_path = os.path.relpath(svs_path, input_folder)

        mag = get_magnification(properties)
        output_path = os.path.join(output_folder, filename)
        action = "skipped"
        orig_stats = get_file_stats(svs_path)

        # 处理逻辑
        if mag == 40:
            img = pyvips.Image.new_from_file(svs_path)
            img_downsampled = img.resize(0.5, kernel='lanczos3')
            img_downsampled.tiffsave(output_path, pyramid=True, compression='jpeg')
            action = "downsampled (40x→20x)"
        elif mag is not None:
            shutil.copy2(svs_path, output_path)
            action = f"copied (original {mag}x)"
        else:
            shutil.copy2(svs_path, output_path)
            action = "copied (unknown magnification)"

        processed_stats = get_file_stats(output_path)

        record = {
            'filename': filename,
            'original_path': rel_path,
            'processed_path': filename,
            'original_mag': mag,
            'processed_mag': 20 if mag == 40 else mag,
            'status': 'success',
            'file_stats': {
                'original': orig_stats,
                'processed': processed_stats
            }
        }

        if mag == 40 and 'size_mb' in orig_stats and 'size_mb' in processed_stats:
            with stats_lock:
                size_stats.append({
                    'file': filename,
                    'original_size': orig_stats['size_mb'],
                    'processed_size': processed_stats['size_mb'],
                    'compression_ratio': round(processed_stats['size_mb'] / orig_stats['size_mb'], 2)
                })

        slide.close()
        return record

    except Exception as e:
        return {
            'filename': os.path.basename(svs_path),
            'error': str(e),
            'status': 'failed'
        }


def process_svs_files(input_folder, output_folder, num_workers=8):
    """多线程处理所有SVS文件"""
    svs_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.svs'):
                svs_files.append(os.path.join(root, file))

    magnification_info = {}
    processing_log = []
    size_stats = []
    global stats_lock
    stats_lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        process_func = partial(process_single_file,
                               input_folder=input_folder,
                               output_folder=output_folder,
                               size_stats=size_stats)

        futures = [executor.submit(process_func, svs_path) for svs_path in svs_files]

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing WSI files"):
            result = future.result()
            processing_log.append(result)
            if 'filename' in result:
                magnification_info[result['filename']] = result

    if size_stats:
        avg_compression = sum(s['compression_ratio'] for s in size_stats) / len(size_stats)
        summary_report = {
            'total_files_processed': len(svs_files),
            'files_downsized': len([x for x in processing_log if x.get('processed_mag') == 20]),
            'average_compression_ratio': round(avg_compression, 2),
            'size_reduction': f"{round((1 - avg_compression) * 100)}%",
            'sample_stats': size_stats[:3]
        }
        with open(os.path.join(output_folder, 'summary_report.json'), 'w') as f:
            json.dump(summary_report, f, indent=2)

    return magnification_info, processing_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WSI Downsampling Tool')
    parser.add_argument('--input', required=True, help='Input folder containing SVS files')
    parser.add_argument('--output', required=True, help='Output folder for processed files')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Starting processing with {args.workers} workers...")
    magnification_info, processing_log = process_svs_files(args.input, args.output, args.workers)

    with open(os.path.join(args.output, 'processed_magnification_info.json'), 'w') as f:
        json.dump(magnification_info, f, indent=2)

    with open(os.path.join(args.output, 'processing_log.json'), 'w') as f:
        json.dump(processing_log, f, indent=2)

    print("Processing completed successfully!")
    