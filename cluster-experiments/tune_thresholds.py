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
parser.add_argument("make_predictions", type=str)

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
seen_df = pd.read_csv('../data/0_labelled_documents.csv')

seen_df = (seen_df
      .sort_values('id')
      .sample(frac=1, random_state=1)
      .reset_index(drop=True)
)

weights_df = pd.read_csv('../data/0_label_weights.csv')

# Get the target labels from the y_prefix argument passed to this script
if len(args.y_prefix) < 2:
    args.y_prefix+=" "
cols = [x for x in seen_df.columns if re.match(f"^{args.y_prefix}",x)]
print(cols)
num_labels=len(cols)

# If the target is inclusion, use only those documents for which we have a non-na value
# Otherwise, only use those documents which are included
# also define what subset is to be treated as a random representative sample
# For labels beyond inclusion, we treat all those that are representative of the included
# studies as representative
if "INCLUDE" in args.y_prefix:
    y_var = cols[0]
    seen_df = seen_df.loc[pd.notna(seen_df[y_var]),:].reset_index(drop=True)
    seen_df['random'] = seen_df['representative_sample']
else:
    seen_df = seen_df[seen_df['INCLUDE']==1]
    seen_df['random'] = seen_df['representative_relevant']
    

# Turn the columns into target variables and get class-weights to counteract class imbalances
if len(cols)==1:
    y_var = cols[0]
    seen_df = seen_df.loc[pd.notna(seen_df[y_var]),:].reset_index(drop=True)
    print(seen_df.shape)
    seen_df['labels'] = list(seen_df[y_var].values.astype(int))
    cw = seen_df[(seen_df['random']==1) & (seen_df[y_var]==0)].shape[0] / seen_df[(seen_df['random']==1) & (seen_df[y_var]==1)].shape[0]
    class_weight={1:cw}
    scorer = "F1"
    weights_df["sample_weight"] = list(weights_df[y_var+"_sample_weight"].fillna(1).values)
else:
    num_labels = len(cols) 
    weights_df['sample_weight'] = list(weights_df[[x+"_sample_weight" for x in cols]].fillna(1).values)
    seen_df = seen_df.replace(2,1)
    seen_df['labels'] = list(seen_df[cols].values.astype(int))
    seen_df = seen_df.dropna(subset=cols)
    seen_df = seen_df.reset_index(drop=True)
    scorer = "F1 macro"
    class_weight = {}
    for i, t in enumerate(cols):
        cw = seen_df[(seen_df['random']==1) & (seen_df[t]==0)].shape[0] / seen_df[(seen_df['random']==1) & (seen_df[t]==1)].shape[0]
        class_weight[i] = cw

# Remove unneccessary columns
seen_df = seen_df[["id","title","content","labels","random"]+cols].merge(
    weights_df[["doc__id","sample_weight"]].rename(columns={"doc__id":"id"})
)


# Merge with the unseen data if necessary
seen_df['seen']  = 1
if args.make_predictions=="True":
    unseen_df = pd.read_csv('../data/0_unlabelled_documents.csv')
    unseen_df['seen'] = 0
    df = (pd.concat([seen_df,unseen_df])
          .sort_values('id')
          .sample(frac=1, random_state=1)
          .reset_index(drop=True)
    )
    df.content = df.content.astype(str)
else:
    df = seen_df

# This is the index of nonrandom/nonrepresentative documents, and these will be removed from validation sets
nonrandom_index = df[(df['random']!=1) & (df['seen']==1)].index
random_index = df[df['random']==1].index
seen_index = df[df['seen']==1].index
unseen_index = df[df['seen']==0].index

print("seen_index", seen_index)
print("nonrandom_index", nonrandom_index)

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
bert_params['class_weight'] = [class_weight]
param_space = list(cvs.product_dict(**bert_params))
params = list(bert_params.keys())

if test:
    param_space = param_space[:2]

# Get the spilts for our outer fold, discard=False means to pass the nonrandom documents that
# would ordinarily be in the validation set back into the test set
outer_cv = cvs.KFoldRandom(args.n_splits, seen_index, nonrandom_index, discard=False)

# Iterate through the folds
for k, (train, test) in enumerate(outer_cv):    
    inner_scores = []
    for l in range(args.n_splits):
        fname = f'cv/df_{len(seen_index)}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}_{l}.csv'
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
    if best_model['class_weight']==-1:
        best_model['class_weight']=None
    elif isinstance(best_model['class_weight'],str):
        best_model['class_weight'] = ast.literal_eval(best_model['class_weight'])
    
    print("Best model from this round: ", best_model)

    inner_cv = cvs.KFoldRandom(args.n_splits, train, nonrandom_index, discard=False)
    inner_scores = []
    
    threshold_results = []
    
    for l, (l_train, l_test) in enumerate(inner_cv):
        outer_scores, y_preds = cvs.train_eval_bert(
            #args.model_name, 
            "distilbert-base-uncased",
            best_model, df=df, targets=cols, 
            train=l_train, test=l_test, roundup=args.roundup, return_predictions=True
        )
        
        if "INCLUDE" in args.y_prefix:
            for t in np.linspace(0.1, 0.9, 50):
                y_pred_bin = np.where(y_preds[:,1]>t,1,0)
                threshold_results.append({
                    "t": t, 
                    "f1": f1_score(df['labels'][l_test], y_pred_bin),
                    "k": l,
                    "label_i": 0
                })
        else:
            for label_i, col in enumerate(cols):
                for t in np.linspace(0.1, 0.9, 50):
                    y_pred_bin = np.where(y_preds[:,label_i]>t,1,0)
                    y_true = [x[label_i] for x in df.labels[l_test]]
                    threshold_results.append({
                        "t": t, 
                        "f1": f1_score(y_true, y_pred_bin),
                        "k": l,
                        "label_i": label_i
                    })
            
    thresh_df = pd.DataFrame.from_dict(threshold_results)
    optimal_t = (thresh_df.groupby(["label_i","t"])["f1"]
     .mean()
     .sort_values(ascending=False)
     .reset_index()
     .groupby(["label_i"])
     .first()
    )
    fname = f'cv/df_{len(seen_index)}_tune_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    thresh_df.to_csv(fname, index=False)
    
    outer_scores, y_preds = cvs.train_eval_bert(
        args.model_name, best_model, df=df, targets=cols, 
        train=train, test=test, roundup=args.roundup, 
        return_predictions=True, threshold=optimal_t["t"].values
    )
    print(y_preds.shape)
    fname = f'cv/df_{len(seen_index)}_cv_results_outer_thresh_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    pd.DataFrame.from_dict([outer_scores]).to_csv(fname, index=False)
    fname = f'cv/df_{len(seen_index)}_y_preds_thresh_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,y_preds)

    fname = f'cv/df_{len(seen_index)}_y_pred_ids_thresh_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,df.loc[test,"id"])



t1 = time.time() - t0

minutes = t1//60
hrs = t1//(60*60)


print(f"Completed in {hrs} hours, {minutes} minutes")
