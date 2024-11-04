#!/bin/bash

#SBATCH --qos=short

#SBATCH --job-name=make-predictions

#SBATCH --output=out/make-predictions.out
#SBATCH --error=out/make-predictions.err
#SBATCH --ntasks=3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8

source activate huggingface-tf

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

srun --mpi=pmi2 -n 3 make_predictions.py 3 distilbert-base-uncased False INCLUDE True 
