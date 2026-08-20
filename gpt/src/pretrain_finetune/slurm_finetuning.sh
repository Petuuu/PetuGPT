#!/bin/bash
#SBATCH --account=project_2020334
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50G

module load python-pytorch/2.10

python3 -m pip install tiktoken

srun torchrun --master_port=29501 -m src.pretrain_finetune.classification/instructions
