#!/bin/bash
#SBATCH --job-name=bracs_extractor
#SBATCH --account=intro_vsc37341
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bracs_extractor_%j.out
#SBATCH --error=logs/bracs_extractor_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zeyang.wang@student.kuleuven.be

set -e
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "CMD: $0 $@"

source /data/leuven/373/vsc37341/miniconda3/etc/profile.d/conda.sh
conda activate wsi_env

cd /data/leuven/373/vsc37341/wsi_code

python /data/leuven/373/vsc37341/solo-learn/main_pretrain.py --config-path /data/leuven/373/vsc37341/wsi_code --config-name bracs_simclr

echo "✅ Finished at: $(date)"
