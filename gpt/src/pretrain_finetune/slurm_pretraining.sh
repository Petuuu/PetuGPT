#!/bin/bash
#SBATCH --account=project_2020070
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=200G

module load python-pytorch/2.10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun --nproc_per_node=4 -m src.build.pretrain
