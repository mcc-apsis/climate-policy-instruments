# Import libraries
import sys
import argparse
import time

parser = argparse.ArgumentParser(description="Run the inner loop of a nested cross validation process")
parser.add_argument("n_splits", type=int)
parser.add_argument("model_name", type=str)
parser.add_argument("roundup", type=str)
parser.add_argument("y_prefix", type=str)
parser.add_argument("resume", type=str)

args = parser.parse_args()

t0 = time.time()

# Establish what task number we have if running from slurm, otherwise just get a random number
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    test = False
except:
    import random
    rank = random.randint(0,args.n_splits**2)
    print(rank)
    test = True
    
rank_i = rank//args.n_splits
rank_j = rank%args.n_splits

# Import the rest of the libraries
import gc
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold
import pickle
import re

# Load data
df = pd.read_csv('../data/0_labelled_documents.csv')

df = (df
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

if test:
    print("TESTING WITH 100 documents")
    df = df.head(150)


if len(args.y_prefix) < 2:
    args.y_prefix+=" "
cols = [x for x in df.columns if re.match(f"^{args.y_prefix}",x)]
print(cols)
num_labels=len(cols)
if len(cols)==1:
    y_var = cols[0]
    df = df.loc[pd.notna(df[y_var]),:].reset_index(drop=True)
    print(df.shape)
    df['labels'] = df[y_var]
    df['random'] = df['representative_sample']
    cw = df[(df['random']==1) & (df[y_var]==0)].shape[0] / df[(df['random']==1) & (df[y_var]==1)].shape[0]
    class_weight={0:1, 1:cw}
else:
    df ['random'] = df['representative_relevant']

random_index = df[df['random']==1].index

# Start setting up the Deep learning things
from transformers import BertTokenizer, DistilBertTokenizer,  TFBertForSequenceClassification, TFDistilBertForSequenceClassification
import tensorflow as tf
import tensorflow_addons as tfa
import cv_setup as cvs

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

bert_params = cvs.bert_params
bert_params['class_weight'].append(class_weight)
param_space = list(cvs.product_dict(**bert_params))

if test:
    param_space = param_space[:2]

outer_cv = cvs.KFoldRandom(args.n_splits, df.index, random_index)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_i:
        continue
    inner_cv = cvs.KFoldRandom(args.n_splits, train, random_index, discard=False)
    inner_scores = []
    for l, (l_train, l_test) in enumerate(inner_cv):
        if l!=rank_j:
            continue
        fname = f'cv/cv_results_{args.y_prefix}_{args.model_name}_{rank_i}_{rank_j}.csv'
        if args.resume=="True":
            try:
                pr = param_space[0]
                cv_results=pd.read_csv(fname).to_dict('records')
                params_tested=pd.read_csv(fname)[list(pr.keys())].to_dict('records')
            except:
                cv_results = []
                params_tested = []
        else:
            cv_results = []
            params_tested = []
            
        for pr in param_space:
            if pr in params_tested:
                continue
            cv_results.append(cvs.train_eval_bert(args.model_name, pr, df=df, targets=cols, train=l_train, test=l_test, roundup=args.roundup))
            cv_df = pd.DataFrame.from_dict(cv_results)
            cv_df['dataset_size'] = df.shape[0]
            cv_df.to_csv(fname,index=False)
            gc.collect()


t1 = time.time() - t0

minutes = t1//60
hrs = t1//(60*60)


print(f"Completed in {hrs} hours, {minutes} minutes")
