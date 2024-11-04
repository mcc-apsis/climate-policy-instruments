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
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

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


# If the target is inclusion, use only those documents for which we have a non-na value
# Otherwise, only use those documents which are included
# also define what subset is to be treated as a random representative sample
# For labels beyond inclusion, we treat all those that are representative of the included
# studies as representative

if "INCLUDE" in args.y_prefix:
    df = df.loc[pd.notna(df["INCLUDE"]),:].reset_index(drop=True)
    df['random'] = df['representative_sample']
else:
    df = df[df['INCLUDE']==1]
    df['random'] = df['representative_relevant']
    

# Turn the columns into target variables and get class-weights to counteract class imbalances
if len(cols)==1:
    scorer = "f1"
    y_var = cols[0]
    df = df.loc[pd.notna(df[y_var]),:].reset_index(drop=True)
    print(df.shape)
    df['labels'] = list(df[y_var].values.astype(int))
    y = df['labels']
    cw = df[(df['random']==1) & (df[y_var]==0)].shape[0] / df[(df['random']==1) & (df[y_var]==1)].shape[0]
    class_weight={0:1, 1:cw}
    scorer = "F1"
else:
    scorer = "f1_macro"
    num_labels = len(cols) 
    df = df.replace(2,1)
    df['labels'] = list(df[cols].values.astype(int))
    df = df.dropna(subset=cols)
    df = df.reset_index(drop=True)
    y = np.matrix(df[cols])
    X = df['content'].values

    print(y.shape)
    print(X.shape)

    scorer = "f1_macro"
    class_weight = {}
    for i, t in enumerate(cols):
        cw = df[(df['random']==1) & (df[t]==0)].shape[0] / df[(df['random']==1) & (df[t]==1)].shape[0]
        class_weight[i] = cw


import cv_setup as cvs


if len(cols)>1:
    pipeline = cvs.mpipeline
    parameters = cvs.mparameters
    cwparam = "param_clf__estimator__class_weight"
    for p in parameters:
        #p['clf__estimator__class_weight'] = ['balanced',class_weight]
        p['clf__estimator__class_weight'] = ['balanced',None]

else:
    pipeline = cvs.pipeline
    parameters = cvs.parameters
    cwparam = "clf__class_weight"
    for p in parameters:
        p['param_clf__class_weight'] = ['balanced',class_weight]
        
nonrandom_index = df[df['random']!=1].index

outer_cv = cvs.KFoldRandom(args.n_splits, df.index, nonrandom_index, discard=False)

for k, (train, test) in enumerate(outer_cv):    
    if k!=rank_j:
        continue
    fname = f'cv/df_{df.shape[0]}_cv_results_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    if args.resume == "True":
        try:
            inner_scores = pd.read_csv(fname)
            best_model = inner_scores.sort_values('rank_test_score').to_dict('records')[0]
            if best_model[cwparam] != "balanced":
                best_model[cwparam] = ast.literal_eval(best_model[cwparam])
            if len(cols)>1:
                clf = Pipeline([
                    ('vect', TfidfVectorizer(
                        max_df=best_model['param_vect__max_df'],
                        min_df=best_model['param_vect__min_df'],
                        ngram_range=ast.literal_eval(best_model['param_vect__ngram_range'])
                    )),
                    ('clf', OneVsRestClassifier(SVC(
                        probability=True,
                        C=best_model['param_clf__estimator__C'],
                        class_weight=best_model['param_clf__estimator__class_weight'],
                        kernel=best_model['param_clf__estimator__kernel'],
                        gamma=best_model['param_clf__estimator__gamma']
                    ))),    
                ])
            else:
                clf = Pipeline([
                    ('vect', TfidfVectorizer(
                        max_df=best_model['param_vect__max_df'],
                        min_df=best_model['param_vect__min_df'],
                        ngram_range=ast.literal_eval(best_model['param_vect__ngram_range'])
                    )),
                    ('clf', SVC(
                        probability=True,
                        C=best_model['param_clf__C'],
                        class_weight=best_model['param_clf__class_weight'],
                        kernel=best_model['param_clf__kernel'],
                        gamma=best_model['param_clf__gamma']
                    )),    
                ])
            clf.fit(X[train],y[train])
        except:
            print("couldn't find old results")
    else:
        inner_cv = cvs.KFoldRandom(args.n_splits, train, nonrandom_index, discard=False)
        for a in inner_cv:
            print(a)
        inner_cv = cvs.KFoldRandom(args.n_splits, train, nonrandom_index, discard=False)
        #clf = GridSearchCV(pipeline, parameters, scoring="f1", n_jobs=8, verbose=1, cv=inner_cv)
        clf = GridSearchCV(pipeline, parameters, scoring=scorer, n_jobs=8, verbose=1, cv=inner_cv)
        #print(X[train])
        #print(y[train,:])
        clf.fit(X, y)

        inner_scores = pd.DataFrame(clf.cv_results_) 
        inner_scores.to_csv(fname, index=False)

    y_preds = clf.predict_proba(df.loc[test,'content'])
    if len(cols)==1:
        y_preds = y_preds[:,1]
    eps = cvs.evaluate_preds(y[test], y_preds, cols)  
    best_params = inner_scores.sort_values('mean_test_score',ascending=False).to_dict('records')[0]['params']
    if type(best_params) is str:
        best_params = ast.literal_eval(best_params)
    for key, value in best_params.items():
        eps[key] = value
    eps["rank_k"] = rank_j

    fname = f'cv/df_{df.shape[0]}_cv_results_outer_nonrandom_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}.csv'
    pd.DataFrame.from_dict([eps]).to_csv(fname, index=False)
    fname = f'cv/df_{df.shape[0]}_y_preds_nonrandom_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,y_preds)
    fname = f'cv/df_{df.shape[0]}_y_pred_nonrandom_ids_{args.y_prefix}_{args.model_name.replace("/","__")}_{k}'
    np.save(fname,df.loc[test,"id"])


t1 = time.time() - t0

minutes = t1//60
hrs = t1//(60*60)


print(f"Completed in {hrs} hours, {minutes} minutes")
