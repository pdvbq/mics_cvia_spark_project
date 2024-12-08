#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 17
#SBATCH -G 1
#SBATCH --time=02:00:00
#SBATCH -p gpu

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
module load  lang/Anaconda3/2020.11 || print_error_and_exit "Anaconda module not found"

conda env create -f environment.yml
conda activate spark
uv run main.py pipeline yolo train
