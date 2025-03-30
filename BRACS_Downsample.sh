#!/bin/bash
#SBATCH --job-name=bracs_downsample          # 作业名称
#SBATCH --account=intro_vsc37341             # 信用账户
#SBATCH --clusters=genius                   # 指定集群
#SBATCH --partition=batch                   # 分区（例如batch分区对应CPU节点）
#SBATCH --nodes=1                           # 申请1个节点
#SBATCH --ntasks=1                          # 任务数
#SBATCH --cpus-per-task=8                  # 分配16个CPU核心
#SBATCH --mem=64G                          # 分配内存（根据需求调整，如16核可能需要更多内存）
#SBATCH --time=24:00:00                     # 最大运行时间，根据数据量调整
#SBATCH --output=downsample_%j.out          # 标准输出文件
#SBATCH --error=downsample_%j.err           # 标准错误文件

set -e # 出错立即退出
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node: $(hostname)"
echo "Job started at: $(date)"

# 激活conda环境
source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate wsi_env

# 切换到代码所在目录
cd /data/leuven/373/vsc37341/wsi_code

# 运行WSI降采样Python脚本，--workers 参数设置为8
python BRACS_Downsample.py --input "/scratch/leuven/373/vsc37341/BRACS/train/" --output "/scratch/leuven/373/vsc37341/BRACS/train_20x" --workers 8

echo "Job finished at: $(date)"
