#!/bin/bash
#SBATCH --job-name=classify-policy-instruments

job1=$(sbatch --parsable inner_bert_cv.sh)

job2=$(sbatch --parsable --dependency=afterok:$job1 outer_bert_cv.sh)

job3=$(sbatch --parsable --dependency=afterok:$job2 make_predictions.sh)
