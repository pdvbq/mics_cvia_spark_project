# mics_cvia_spark_project
Project repository for the course of Computer Vision &amp; Image Analysis

## HPC Train

```sh
#   nodes                    cpus   gpus est time        use gpu   script to run
run -N 1 --ntasks-per-node=1 -c 17 -G 1 --time=02:00:00 -p gpu ./hpc_yolo_train.sh
```
## Environment Setup
### OS
Linux.
### Conda
The dependencies can be resolved with conda as follows:
```
conda env create -f environment.yml
```
To activate the environment:
```
conda activate spark
```
### UV
Instructions for UV are coming soon
