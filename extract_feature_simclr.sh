#!/bin/bash
#SBATCH --job-name=extract_feature_simclr
#SBATCH --account=intro_vsc37341
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/simclr_%j.out
#SBATCH --error=logs/simclr_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zeyang.wang@student.kuleuven.be

set -e
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "CMD: $0 $@"

source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate clam_latest

cd /data/leuven/373/vsc37341/wsi_code

python extract_feature_simclr.py \
  --data_h5_dir "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch_colornorm" \
  --feat_dir    "/scratch/leuven/373/vsc37341/TCGA-BRCA/downsampled_20x_patch_colornorm_ftsimclr" \
  --batch_size 256 \
  --num_workers 4 \
  --device cuda

echo "✅ Finished at: $(date)"
