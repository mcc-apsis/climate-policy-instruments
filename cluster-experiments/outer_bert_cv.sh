#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=00:15:00
#SBATCH --job-name=outer-bert

#SBATCH --output=out/outer-bert.out
#SBATCH --error=out/outer-bert.err
#SBATCH --ntasks=3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8

source activate huggingface-tf

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

#srun --mpi=pmi2 -n 3 outer_bert_cv.py 3 distilbert-base-uncased False "4 -" True
#srun --mpi=pmi2 -n 3 outer_bert_cv.py 3 distilbert-base-uncased False INCLUDE True

srun --mpi=pmi2 -n 3 outer_bert_cv.py 3 allenai/scibert_scivocab_uncased False INCLUDE True 
#srun --mpi=pmi2 -n 3 outer_bert_cv.py 3 allenai/scibert_scivocab_uncased False "4 -" True 
