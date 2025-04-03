#!/bin/bash
#SBATCH --job-name=bracs_norm                 # 作业名称
#SBATCH --account=intro_vsc37341             # 信用账户
#SBATCH --clusters=genius                    # 指定集群
#SBATCH --partition=batch                    # 分区（例如batch分区对应CPU节点）
#SBATCH --nodes=1                            # 申请1个节点
#SBATCH --ntasks=1                           # 任务数
#SBATCH --cpus-per-task=8                    # 分配8个CPU核心
#SBATCH --mem=64G                            # 分配内存（根据实际需要调整）
#SBATCH --time=24:00:00                      # 最大运行时间（调整为你估计处理BRACS所需的时间）
#SBATCH --output=bracs_norm_%j.out           # 标准输出文件
#SBATCH --error=bracs_norm_%j.err            # 标准错误文件

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node: $(hostname)"
echo "Job started at: $(date)"

# 激活conda环境
source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate wsi_env

# 切换到代码所在目录
cd /data/leuven/373/vsc37341/wsi_code

# 运行BRACS归一化脚本
python bracs_norm.py --input_dir "/scratch/leuven/373/vsc37341/BRACS/train_20x" --output_dir "/scratch/leuven/373/vsc37341/BRACS/train_20x_norm" --threads 8

echo "Job finished at: $(date)"
