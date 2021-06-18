from transformers import BertTokenizer, DistilBertTokenizer, TFDistilBertForSequenceClassification, TFBertForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import precision_score, recall_score

def create_train_val(tokenizer, x,y,train,val):
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
    
    return train_dataset, val_dataset, MAX_LEN

def init_model(model, params): 
    optimizer = tfa.optimizers.AdamW(learning_rate=params['learning_rate'], weight_decay=params['weight_decay'])
    # Should this be adjusted for multilabel?
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model

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

def train_eval_bert(model_name, params, df, targets, train, test, roundup):

    num_labels = min([2,len(targets)])
    
    if "distilbert" in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = TFBertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    
    
    train_dataset, val_dataset, MAX_LEN = create_train_val(tokenizer, df['content'].astype(str), df['labels'], train, test)
    
    print("training bert with these params")
    print(params)
    model = init_model(model,  params)
    model.fit(train_dataset.shuffle(100).batch(params['batch_size']),
              epochs=params['num_epochs'],
              batch_size=params['batch_size'],
              class_weight=params['class_weight']
    )

    preds = model.predict(val_dataset.batch(1)).logits
    y_pred = tf.keras.activations.sigmoid(tf.convert_to_tensor(preds)).numpy()
    if roundup=="True":
        ai = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
        maximums = np.maximum(y_pred.max(1),0.51)
        np.put_along_axis(y_pred, ai, maximums.reshape(ai.shape), axis=1)
    eps = evaluate_preds(np.array(df.loc[test,targets]), y_pred, targets)
    print(eps)
    for key, value in params.items():
        eps[key] = value
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
        if not discard:
            train = list(train) +  [x for x in test if x in no_test]
        test = [x for x in test if x not in no_test]
        yield (train, test)
