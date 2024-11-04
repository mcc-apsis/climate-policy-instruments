#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=06:00:00

#SBATCH --job-name=cv-svm

#SBATCH --output=out/cv-svm.out
#SBATCH --error=out/cv-svm.err
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8

source activate huggingface-tf

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

#srun --mpi=pmi2 -n 3 svm_cv.py 3 svm False INCLUDE True 
srun --mpi=pmi2 -n 3 svm_cv.py 3 svm False "4 -" False
