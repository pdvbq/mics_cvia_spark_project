#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 14
#SBATCH -G 1
#SBATCH --time=24:00:00
#SBATCH --partition gpu
#SBATCH --qos normal


print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

nvidia-smi

uv run main.py pipeline maskrcnn train
