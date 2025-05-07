import os
from collections import defaultdict

def count_png_files(root_dir):
    """
    递归统计目录及其子目录中的所有PNG文件数量
    
    参数:
        root_dir (str): 要扫描的根目录路径
        
    返回:
        tuple: (总数量, 各子目录数量字典)
    """
    total_count = 0
    dir_counts = defaultdict(int)
    
    for root, dirs, files in os.walk(root_dir):
        png_count = sum(1 for f in files if f.lower().endswith('.png'))
        if png_count > 0:
            relative_path = os.path.relpath(root, root_dir)
            dir_counts[relative_path] = png_count
            total_count += png_count
            
    return total_count, dir_counts

def print_statistics(total, dir_counts):
    """打印统计结果"""
    print(f"\n{' 统计结果 ':=^40}")
    
    if dir_counts:
        # 计算最大列宽
        max_dir_len = max(len(d) for d in dir_counts)
        max_count_len = max(len(str(c)) for c in dir_counts.values())
        
        # 打印表头
        print(f"\n{' 子文件夹路径':<{max_dir_len}} | {'数量':>{max_count_len}}")
        print('-' * (max_dir_len + max_count_len + 3))
        
        # 打印各目录统计
        for dir_path, count in sorted(dir_counts.items()):
            print(f" {dir_path:<{max_dir_len}} | {count:>{max_count_len}}")
        
        # 打印分隔线和总数
        print('-' * (max_dir_len + max_count_len + 3))
    else:
        print("警告: 未发现任何PNG文件")
    
    # 始终显示总数（带颜色高亮）
    print(f"\n\033[1;32m总PNG文件数: {total}\033[0m")  # 绿色高亮

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='统计目录中的PNG文件数量')
    parser.add_argument('directory', help='要扫描的目录路径')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"错误: 目录不存在 - {args.directory}")
        exit(1)
        
    total, dir_counts = count_png_files(args.directory)
    print_statistics(total, dir_counts)