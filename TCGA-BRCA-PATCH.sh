#!/bin/bash
#SBATCH --job-name=TCGA_BRCA_Patches          # 作业名称
#SBATCH --account=intro_vsc37341             # 信用账户
#SBATCH --clusters=genius                   # 指定集群
#SBATCH --partition=batch                   # 分区（例如batch分区对应CPU节点）
#SBATCH --nodes=1                           # 申请1个节点
#SBATCH --ntasks=1                          # 任务数
#SBATCH --cpus-per-task=8                  # 分配16个CPU核心
#SBATCH --mem=64G                          # 分配内存（根据需求调整，如16核可能需要更多内存）
#SBATCH --time=01:00:00                     # 最大运行时间，根据数据量调整
#SBATCH --output=downsample_%j.out          # 标准输出文件
#SBATCH --error=downsample_%j.err           # 标准错误文件

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated node: $(hostname)"
echo "Job started at: $(date)"

# 激活conda环境
source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

# 切换到代码所在目录
cd /data/leuven/373/vsc37341/CLAM

# TCGA-BRCA-PATCHES
python create_patches_fp.py --source "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_collection" --save_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/20x_patch" --patch_size 224 --step_size 224 --seg --patch --stitch --preset tcga.csv

echo "Job finished at: $(date)"
