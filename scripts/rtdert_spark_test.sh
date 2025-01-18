#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 14
#SBATCH -G 0
#SBATCH --time=00:20:00
#SBATCH --partition gpu
#SBATCH --qos normal

nvidia-smi

uv run main.py pipeline rtdetr test
