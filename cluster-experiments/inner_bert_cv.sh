#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --job-name=inner-bert

#SBATCH --output=out/inner-bert.out
#SBATCH --error=out/inner-bert.err
#SBATCH --ntasks=3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8

source activate huggingface-tf

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

# n_splits model_name roundup y_prefix resume
srun --mpi=pmi2 -n 3 inner_bert_cv.py 3 distilbert-base-uncased False "9 - 0. Ex-post" True
#srun --mpi=pmi2 -n 3 inner_bert_cv.py 3 distilbert-base-uncased False "4 -" False 

#srun --mpi=pmi2 -n 3 inner_bert_cv.py 3 distilbert-base-uncased False INCLUDE False 

#srun --mpi=pmi2 -n 3 inner_bert_cv.py 3 allenai/scibert_scivocab_uncased False "4 -"  True 
