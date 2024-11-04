#!/usr/bin/env python

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
import ast

# Load data
seen_df = pd.read_csv('../data/0_labelled_documents.csv')

seen_df = (seen_df
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)
seen_df['seen']=1
unseen_df = pd.read_csv('../data/0_unlabelled_documents.csv')

df = (pd.concat([seen_df,unseen_df])
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

df.content = df.content.astype(str)

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
    df = df.loc[(pd.notna(df[y_var])) | (df.seen==0),:].reset_index(drop=True)
    print(df.shape)
    df['labels'] = df[y_var]
    df['random'] = df['representative_sample']
    cw = df[(df['random']==1) & (df[y_var]==0)].shape[0] / df[(df['random']==1) & (df[y_var]==1)].shape[0]
    class_weight={0:1, 1:cw}
    scorer = "F1"
else:
    df ['random'] = df['representative_relevant']

random_index = df[df['random']==1].index
seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

# Start setting up the Deep learning things
from transformers import BertTokenizer, DistilBertTokenizer,  TFBertForSequenceClassification, TFDistilBertForSequenceClassification
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
tf.keras.backend.set_floatx('float16')
#tf.config.threading.set_intra_op_parallelism_threads(1)
#tf.config.threading.set_inter_op_parallelism_threads(1)

import tensorflow_addons as tfa
import cv_setup as cvs

bert_params = cvs.bert_params
bert_params['class_weight'].append(class_weight)
param_space = list(cvs.product_dict(**bert_params))

params = list(bert_params.keys())

if test:
    param_space = param_space[:2]

outer_cv = cvs.KFoldRandom(args.n_splits, seen_index, random_index)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    inner_scores = []
    fname = f'cv/df_{seen_df.shape[0]}_model_selection_{args.y_prefix}_{args.model_name}_{rank_j}.csv'
    if args.resume=="True":
        try:
            pr = param_space[0]
            cv_results=pd.read_csv(fname).to_dict('records')
            cv_df = pd.DataFrame.from_dict(cv_results)
            params_tested=pd.read_csv(fname)[list(pr.keys())].to_dict('records')
            for pr in params_tested:
                if pd.isna(pr["class_weight"]):
                    pr["class_weight"] = None
                elif isinstance(pr['class_weight'],str):
                    pr['class_weight'] = ast.literal_eval(pr['class_weight'])
        except:
            cv_results = []
            params_tested = []
    else:
        cv_results = []
        params_tested = []

    # Test each combination of parameters

    for pr in param_space:
        if pr in params_tested:
            continue
        cv_results.append(cvs.train_eval_bert(args.model_name, pr, df=df, targets=cols, train=seen_index[train], test=seen_index[test], roundup=args.roundup))
        cv_df = pd.DataFrame.from_dict(cv_results)
        cv_df['dataset_size'] = seen_df.shape[0]
        cv_df.to_csv(fname,index=False)
        gc.collect()
        tf.keras.backend.clear_session()
        gc.collect()

    cv_df.loc[pd.notna(cv_df['class_weight']),'class_weight'] = cv_df.loc[pd.notna(cv_df['class_weight']),'class_weight'].astype(str)
    print(cv_df)
    print(params)
    print(scorer)
    # Now select te best combination
    best_model = (cv_df[pd.notna(cv_df[scorer])]
                  .groupby(params)[scorer]
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index() 
                 ).to_dict('records')[0]

    del best_model[scorer]
    print("Best model from this round: ", best_model)
    if best_model['class_weight']==-1:
        best_model['class_weight']=None
    elif isinstance(best_model['class_weight'],str):
        best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])

    print(train)
    print(unseen_index)
    y_preds = cvs.train_eval_bert(args.model_name, best_model, df=df, targets=cols, train=seen_index[train], test=unseen_index, evaluate=False, roundup=args.roundup, return_predictions=True)

    fname = f'cv/df_{seen_df.shape[0]}_y_preds_{args.y_prefix}_{args.model_name}_{k}'
    np.save(fname,y_preds)
    fname = f'cv/df_{seen_df.shape[0]}_y_preds_{args.y_prefix}_{args.model_name}_pred_ids.csv'
    df.loc[unseen_index,"id"].to_csv(fname,index=False)


t1 = time.time() - t0

minutes = t1//60
hrs = t1//(60*60)


print(f"Completed in {hrs} hours, {minutes} minutes")
