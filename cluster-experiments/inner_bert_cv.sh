#!/bin/bash

#SBATCH --qos=medium

#SBATCH --job-name=inner-bert

#SBATCH --output=out/inner-bert.out
#SBATCH --error=out/inner-bert.err
#SBATCH --ntasks=9
#SBATCH --partition=ram_gpu
#SBATCH --gres=gpu:v100:1

source activate huggingface-tf

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

srun --mpi=pmi2 -n 9 inner_bert_cv.py 3 distilbert-base-uncased False INCLUDE False -mpi 1
