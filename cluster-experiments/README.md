# Machine learning experiments run on the PIK cluster

## Relevant documents

`sbatch inner_bert_cv.sh` args = 

This runs through the inner loop of the BERT classifier

for each of SVM, SciBERT, DistilBERT:
    - Get the data
    - Define parameter space
    - For each outer split:
        - for each inner split:
            - for each model candidate from parameter space:
                - train / validate
        - with best model:
            - train / validate
        - for each model candidate from parameter space:
            - train / validate
    - with best model:
        - train / predict
