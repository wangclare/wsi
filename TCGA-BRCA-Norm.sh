#!/bin/bash
#SBATCH --job-name=TCGA-BRCA-Stain-Normlization              # 作业名称
#SBATCH --account=intro_vsc37341              # 使用的账户
#SBATCH --clusters=genius                       # 所在集群（如你用的是 a100 节点）
#SBATCH --partition=gpu_p100                  # 分区（使用 A100 显卡）
#SBATCH --nodes=1                             # 1 个节点
#SBATCH --ntasks=1                            # 单任务
#SBATCH --cpus-per-task=4                     # CPU 核数（根据需求设置）
#SBATCH --gpus-per-node=1                              # 使用 1 张 GPU
#SBATCH --mem=24G                             # 分配内存
#SBATCH --time=72:00:00                       # 最大运行时间
#SBATCH --output=logs/color_norm_%j.out       # 标准输出日志
#SBATCH --error=logs/color_norm_%j.err        # 错误输出日志
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zeyang.wang@student.kuleuven.be

set -e  # 脚本出错立即终止

echo " Job ID: $SLURM_JOB_ID"
echo " Node: $(hostname)"
echo " Started at: $(date)"

# 激活 conda 环境
source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

# 切换到代码目录
cd /data/leuven/373/vsc37341/wsi_code

# 执行颜色归一化主程序
python color_norm.py \
  --data_h5_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch/patches" \
  --data_slide_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_collection" \
  --output_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch_colornorm" \
  --batch_size 8


echo "✅ Finished at: $(date)"
