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
# This means we are just running the script in test mode
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
    
rank_i = rank%args.n_splits
rank_j = rank//args.n_splits

print("Rank I ", rank_i, "Rank j", rank_j)
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


# Get the target labels from the y_prefix argument passed to this script
if len(args.y_prefix) < 2:
    args.y_prefix+=" "
cols = [x for x in df.columns if re.match(f"^{args.y_prefix}",x)]
print(cols)
num_labels=len(cols)

# If the target is inclusion, use only those documents for which we have a non-na value
# Otherwise, only use those documents which are included
# also define what subset is to be treated as a random representative sample
# For labels beyond inclusion, we treat all those that are representative of the included
# studies as representative
if "INCLUDE" in args.y_prefix:
    df = df.loc[pd.notna(df[y_var]),:].reset_index(drop=True)
    df['random'] = df['representative_sample']
else:
    df = df[df['INCLUDE']==1]
    df['random'] = df['representative_relevant']
    

# Turn the columns into target variables and get class-weights to counteract class imbalances
if len(cols)==1:
    y_var = cols[0]
    df = df.loc[pd.notna(df[y_var]),:].reset_index(drop=True)
    print(df.shape)
    df['labels'] = list(df[y_var].values.astype(int))
    cw = df[(df['random']==1) & (df[y_var]==0)].shape[0] / df[(df['random']==1) & (df[y_var]==1)].shape[0]
    class_weight={0:1, 1:cw}
    scorer = "F1"
else:
    num_labels = len(cols) 
    df = df.replace(2,1)
    df['labels'] = list(df[cols].values.astype(int))
    df = df.dropna(subset=cols)
    df = df.reset_index(drop=True)
    scorer = "F1 macro"
    class_weight = {}
    for i, t in enumerate(cols):
        cw = df[(df['random']==1) & (df[t]==0)].shape[0] / df[(df['random']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw

# This is the index of nonrandom/nonrepresentative documents, and these will be removed from validation sets
nonrandom_index = df[df['random']!=1].index

# Try this with tensorflow if this is True, otherwise we do it with pytorch
try_tf = False
if "distilbert" in args.model_name.lower() and try_tf:
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

# import the helper functions defined in cv_setup.py
import cv_setup as cvs

# Get the BERT parameters, and include class_weight as a parameter to be tested
bert_params = cvs.bert_params
bert_params['class_weight'].append(class_weight)
param_space = list(cvs.product_dict(**bert_params))
params = list(bert_params.keys())

if test:
    param_space = param_space[:2]

# Get the spilts for our outer fold, discard=False means to pass the nonrandom documents that
# would ordinarily be in the validation set back into the test set
outer_cv = cvs.KFoldRandom(args.n_splits, df.index, nonrandom_index, discard=False)

# Iterate through the folds
for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_i:
        continue
        # Skip if the fold does not match the job number of this process
    # Get the splits for the inner CV
    inner_cv = cvs.KFoldRandom(args.n_splits, train, nonrandom_index, discard=False)
    inner_scores = []
    for l, (l_train, l_test) in enumerate(inner_cv):
        if l!=rank_j:
            pass
            #continue
            # If we want to be very very parallel then continue here, and increase the number or tasks
        # File name for the results of this split across params
        fname = f'cv/df_{df.shape[0]}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
        # Try to load existing results if resume is true, otherwise initialize new one
        if args.resume=="True":
            try:
                pr = param_space[0]
                cv_results=pd.read_csv(fname).to_dict('records')
                params_tested=pd.read_csv(fname)[list(pr.keys())].to_dict('records')
                if len(params_tested) >= len(param_space):
                    continue
                for pr in params_tested:
                    if pd.isna(pr["class_weight"]):
                        pr["class_weight"] = None
            except:
                cv_results = []
                params_tested = []
        else:
            cv_results = []
            params_tested = []
            
        # train and validate a model for each combination of parameters
        for pr in param_space:
            if pr in params_tested:
                continue
            cv_results.append(cvs.train_eval_bert(args.model_name, pr, df=df, targets=cols, train=l_train, test=l_test, roundup=args.roundup))
            cv_df = pd.DataFrame.from_dict(cv_results)
            cv_df['dataset_size'] = df.shape[0]
            cv_df.to_csv(fname,index=False)
            gc.collect()
            if use_tf:
                tf.keras.backend.clear_session()
            else:
                torch.cuda.empty_cache()
            gc.collect()

    # Now do the outer cv loop

    inner_scores = []
    for l in range(args.n_splits):
        fname = f'cv/df_{df.shape[0]}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
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
