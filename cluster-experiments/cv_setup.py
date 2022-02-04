from transformers import BertTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, TFDistilBertForSequenceClassification, TFBertForSequenceClassification
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, BertTokenizer
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers import Trainer, TrainingArguments
import gc
import torch
from sklearn.multiclass import OneVsRestClassifier

pipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', SVC(probability=True)),
])

mpipeline = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(SVC(probability=True))),
])


parameters = [
    {
        'vect__max_df': (0.5,0.75,0.9),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),  
        'clf__kernel': ['rbf'], 
        'clf__gamma': [1e-3, 1e-4], 
        'clf__C': [1e1, 1e2]
    },
    {
        'vect__max_df': (0.5,75,0.9),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),  
        'clf__kernel': ['linear'], 
        'clf__C': [1e1,1e2, 1e3]
    }
]


mparameters = [
    {
        'vect__max_df': (0.5,0.75,0.9),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),  
        'clf__estimator__kernel': ['rbf'], 
        'clf__estimator__gamma': [1e-3, 1e-4], 
        'clf__estimator__C': [1e1, 1e2]
    },
    {
        'vect__max_df': (0.5,75,0.9),
        'vect__min_df': (5, 10, 15),
        'vect__ngram_range': ((1, 1), (1, 2)),  
        'clf__estimator__kernel': ['linear'], 
        'clf__estimator__C': [1e1,1e2, 1e3]
    }
]

def create_train_val(tokenizer, x,y,train,val, tensorflow):
    if tensorflow:
        import tensorflow as tf
        train_encodings = tokenizer(list(x[train].values),
                                    truncation=True,
                                    padding=True)
        val_encodings = tokenizer(list(x[val].values),
                                    truncation=True,
                                    padding=True) 

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            list(y[train].values)
        ))
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            list(y[val].values)
        ))


        MAX_LEN = train_dataset._structure[0]['input_ids'].shape[0]
    else:
        MAX_LEN = 512
        train_encodings= tokenizer(list(x[train]),truncation=True,padding=True,max_length=512)
        val_encodings = tokenizer(list(x[val]),truncation=True,padding=True,max_length=512)
        import torch
        class TDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx],dtype=torch.float32)
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = TDataset(train_encodings, list(y[train]))
        val_dataset = TDataset(val_encodings, list(y[val]))        
    return train_dataset, val_dataset, MAX_LEN


class BCWTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        cw = torch.tensor(self.class_weight[1])
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=cw)
        loss = loss_fct(logits.view(-1, 1),
                        labels.float().view(-1, 1))
        return (loss, outputs) if return_outputs else loss

class CWTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        cw = torch.tensor(self.class_weight[1])
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=cw)
        loss = loss_fct(logits.view(-1, model.num_labels),
                        labels.float().view(-1, model.num_labels))
        return (loss, outputs) if return_outputs else loss

def init_model(model, params,tensorflow=True):
    if tensorflow:
        import tensorflow as tf
        import tensorflow_addons as tfa
        optimizer = tfa.optimizers.AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])
        # Should this be adjusted for multilabel?
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        trainer = None
    else:
        if "distilbert" in model.name_or_path:
            gradient_checkpointing=False
        else:
            gradient_checkpointing=True
        trainer_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=params['num_epochs'],
            per_device_train_batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            fp16=True,
            gradient_checkpointing=gradient_checkpointing
        )
        if params['class_weight'] is not None:
            if model.num_labels<3:
                trainer = BCWTrainer(
                    model=model,
                    args=trainer_args
                )
                trainer.class_weight = params['class_weight']
            else:
                trainer = CWTrainer(
                    model=model,
                    args=trainer_args,
                )
                trainer.class_weight = params['class_weight']
        else:
            trainer = Trainer(
                model=model,
                args=trainer_args
            )
    return model, trainer

def evaluate_preds(y_true, y_pred, targets):
    if len(targets)==1:
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except:
            roc_auc = np.NaN
        f1 = f1_score(y_true, y_pred.round())
        p, r = precision_score(y_true, y_pred.round()), recall_score(y_true, y_pred.round())
        acc = accuracy_score(y_true, y_pred.round())
        print(f"ROC AUC: {roc_auc:.0%}, F1: {f1:.1%}, precision: {p:.1%}, recall {r:.1%}, acc {acc:.0%}")
        return {"ROC AUC": roc_auc, "F1": f1, "precision": p, "recall": r, "accuracy": acc}
    else:
        res = {}
        for average in ["micro","macro","weighted", "samples"]:
            try:
                res[f'ROC AUC {average}'] = roc_auc_score(y_true, y_pred, average=average)
            except:
                res[f'ROC AUC {average}'] = np.NaN
            res[f'F1 {average}'] = f1_score(y_true, y_pred.round(), average=average)
            res[f'precision {average}'] = precision_score(y_true, y_pred.round(), average=average)
            res[f'recall {average}'] = recall_score(y_true, y_pred.round(), average=average)

        for i, target in enumerate(targets):
            try:
                res[f'ROC AUC - {target}'] = roc_auc_score(y_true[:,i], y_pred[:,i])
            except:
                res[f'ROC AUC - {target}'] = np.NaN
            res[f'precision - {target}'] = precision_score(y_true[:,i], y_pred[:,i].round())
            res[f'recall - {target}'] = recall_score(y_true[:,i], y_pred[:,i].round())
            res[f'F1 - {target}'] = f1_score(y_true[:,i], y_pred[:,i].round())
            res[f'accuracy - {target}'] = accuracy_score(y_true[:,i], y_pred[:,i].round())
            res[f'n_target - {target}'] = y_true[:,i].sum()

        return res        

def train_eval_bert(model_name, params, df, targets, train, test, roundup, return_predictions=False, evaluate=True):

    num_labels = min([2,len(targets)])
    num_labels = len(targets)
    
    if "distilbert" in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained('./transformers/dbtokenizer')
        #model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir='transformers')
        #tensorflow=True
        #import tensorflow as tf
        #import tensorflow_addons as tfa
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir='transformers')
        import torch
        tensorflow=False
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir='transformers')
        #tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir='transformers')
        tokenizer= BertTokenizer.from_pretrained('./transformers/transformers/tokenizer')
        tensorflow=False
        import torch
    
    
    train_dataset, val_dataset, MAX_LEN = create_train_val(tokenizer, df['content'].astype(str), df['labels'], train, test, tensorflow)
    
    print("training bert with these params")
    print(params)
    model, trainer = init_model(model,  params, tensorflow)
    if trainer:
        trainer.train_dataset=train_dataset
        trainer.train()
        preds = trainer.predict(val_dataset)
        activation = torch.nn.Sigmoid()
        y_pred = activation(torch.tensor(preds.predictions)).numpy()

    else:
        model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
                  epochs=params['num_epochs'],
                  batch_size=params['batch_size'],
                  class_weight=params['class_weight']
        )

        preds = model.predict(val_dataset.batch(1)).logits
        y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()

    del model
    gc.collect()
    if trainer:
        torch.cuda.empty_cache()
    if roundup=="True":
        ai = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
        maximums = np.maximum(y_pred.max(1),0.51)
        np.put_along_axis(y_pred, ai, maximums.reshape(ai.shape), axis=1)
    if not evaluate:
        return y_pred
    eps = evaluate_preds(np.array(df.loc[test,targets]), y_pred, targets)
    print(eps)
    for key, value in params.items():
        eps[key] = value
    if return_predictions:
        return eps, y_pred
    return eps

bert_params = {
  "class_weight": [None],
  "batch_size": [16, 32],
  "weight_decay": (0, 0.3),
  "learning_rate": (1e-5, 5e-5),
  "num_epochs": [2, 3, 4]
}

import itertools

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

param_space = list(product_dict(**bert_params))

def KFoldRandom(n_splits, X, no_test, shuffle=False, discard=True):
    kf = KFold(n_splits=n_splits, shuffle=shuffle)
    for train, test in kf.split(X):
        X = np.array(X)
        train = X[train]
        test = X[test]
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]
    yield (train, test)

