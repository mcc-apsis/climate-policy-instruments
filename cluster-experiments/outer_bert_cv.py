#!/usr/bin/env python

# Import libraries
import sys
import argparse
import time
import ast

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
    scorer = "F1"
else:
    scorer = "F1 macro"
    num_labels = len(cols)
    df = df[df['INCLUDE']==1] 
    df = df.replace(2,1)
    df['labels'] = list(df[cols].values.astype(int))
    df = df.dropna(subset=cols)
    df['random'] = df['representative_relevant']
    df = df.reset_index(drop=True)
    class_weight = {}
    for i, t in enumerate(cols):
        cw = df[(df['random']==1) & (df[t]==0)].shape[0] / df[(df['random']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw


nonrandom_index = df[df['random']!=1].index

if "distilbert" in args.model_name.lower():
    # Start setting up the Deep learning things
    from transformers import BertTokenizer, DistilBertTokenizer,  TFBertForSequenceClassification, TFDistilBertForSequenceClassification
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    #tf.config.threading.set_intra_op_parallelism_threads(1)
    #tf.config.threading.set_inter_op_parallelism_threads(1)

    import tensorflow_addons as tfa
    use_tf = True
else:
    import torch
    use_tf = False

import cv_setup as cvs


bert_params = cvs.bert_params
bert_params['class_weight'].append(class_weight)
param_space = list(cvs.product_dict(**bert_params))

params = list(bert_params.keys())

if test:
    param_space = param_space[:2]


outer_cv = cvs.KFoldRandom(args.n_splits, df.index, nonrandom_index, discard=False)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    inner_scores = []
    for l in range(args.n_splits):
        try:
            fname = f'cv/df_{df.shape[0]}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
            inner_df = pd.read_csv(fname)
        except:
            fname = f'cv/df_master_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
            inner_df = pd.read_csv(fname)
        inner_df = inner_df.sort_values(scorer,ascending=False).reset_index(drop=True)
        inner_scores += inner_df.to_dict('records')

    inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
    best_model = (inner_scores
                  .groupby(params)[scorer]
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index() 
                 ).to_dict('records')[0]

    del best_model[scorer]
    print("Best model from this round: ", best_model)
    if best_model['class_weight']==-1:
        best_model['class_weight']=None
    else:
        best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])
    
    outer_scores, y_preds = cvs.train_eval_bert(args.model_name, best_model, df=df, targets=cols, train=train, test=test, roundup=args.roundup, return_predictions=True)

    fname = f'cv/df_{df.shape[0]}_cv_results_outer_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    pd.DataFrame.from_dict([outer_scores]).to_csv(fname, index=False)
    fname = f'cv/df_{df.shape[0]}_y_preds_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,y_preds)
    fname = f'cv/df_{df.shape[0]}_y_pred_ids_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,df.loc[test,"id"])

outer_cv = cvs.KFoldRandom(args.n_splits, df.index, [], discard=False)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    inner_scores = []
    for l in range(args.n_splits):
        try:
            fname = f'cv/df_{df.shape[0]}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
            inner_df = pd.read_csv(fname)
        except:
            fname = f'cv/df_master_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
            inner_df = pd.read_csv(fname)

        inner_df = inner_df.sort_values(scorer,ascending=False).reset_index(drop=True)
        inner_scores += inner_df.to_dict('records')

    inner_scores = pd.DataFrame.from_dict(inner_scores).fillna(-1)
    best_model = (inner_scores
                  .groupby(params)[scorer]
                  .mean()
                  .sort_values(ascending=False)
                  .reset_index() 
                 ).to_dict('records')[0]

    del best_model[scorer]
    print("Best model from this round: ", best_model)
    if best_model['class_weight']==-1:
        best_model['class_weight']=None
    else:
        best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])
    
    outer_scores, y_preds = cvs.train_eval_bert(args.model_name, best_model, df=df, targets=cols, train=train, test=test, roundup=args.roundup, return_predictions=True)

    fname = f'cv/df_{df.shape[0]}_cv_results_outer_nonrandom_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    pd.DataFrame.from_dict([outer_scores]).to_csv(fname, index=False)
    fname = f'cv/df_{df.shape[0]}_y_preds_nonrandom_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,y_preds)
    fname = f'cv/df_{df.shape[0]}_y_pred_nonrandom_ids_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,df.loc[test,"id"])

t1 = time.time() - t0

minutes = t1//60
hrs = t1//(60*60)


print(f"Completed in {hrs} hours, {minutes} minutes")
