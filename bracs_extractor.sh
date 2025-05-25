#!/bin/bash
#SBATCH --job-name=bracs_extractor
#SBATCH --account=intro_vsc37341
#SBATCH --clusters=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=45G
#SBATCH --time=48:00:00
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
export WANDB_API_KEY=8777e3287e608a44536da9a9d9b96907418bfb80
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
cd /data/leuven/373/vsc37341/wsi_code

python -m torch.distributed.run --nproc_per_node=2 \
  /data/leuven/373/vsc37341/solo-learn/main_pretrain.py \
  --config-path /data/leuven/373/vsc37341/wsi_code/configs \
  --config-name bracs_simclr

echo "✅ Finished at: $(date)"
#8777e3287e608a44536da9a9d9b96907418bfb80