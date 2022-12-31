from datasets import Dataset

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from tqdm.auto import tqdm

from sklearn.model_selection import LeavePGroupsOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification


def tokenize_seq2seq_input(examples, tokenizer, source_max_length: int=1024, target_max_length: int=128):
    model_inputs = tokenizer(examples["inputs"], max_length=source_max_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], max_length=target_max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def tokenize_classification_input(examples, tokenizer, labels, source_max_length: int=512):
    model_inputs = tokenizer(examples["inputs"], max_length=source_max_length,
                                padding=True, truncation=True, return_tensors='pt')

    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(examples["inputs"]), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    model_inputs["labels"] = labels_matrix.tolist()

    return model_inputs


def compute_multilabel_metrics(eval_preds):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = F.sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, probs, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def convert_to_vector(decoded, label2id):
    matrix = np.zeros((len(decoded), 31))

    for i, discourses in enumerate(decoded):
        discourses = discourses.split(',')
        encoded = [label2id[disc] for disc in discourses if disc in label2id.keys()]
        matrix[i, encoded] = 1

    return matrix


def compute_seq2seq_metrics(eval_preds, tokenizer, label2id):

    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    labels_matrix = convert_to_vector(decoded_labels, label2id)
    preds_matrix = convert_to_vector(decoded_preds, label2id)

    jaccard = 0.0
    for pred, label in zip(decoded_preds, decoded_labels):
        pred, label = set(pred.split(',')), set(label.split(','))
        jaccard += len(pred & label) / len(pred | label)

    jaccard /= len(decoded_preds)

    # not sure if this is the correct way to calculate accuracy
    # match_accuracy = accuracy_score(labels_matrix, preds_matrix)

    match_accuracy = (labels_matrix == preds_matrix).mean()

    f1_micro_average = f1_score(y_true=labels_matrix, y_pred=preds_matrix, average='micro')

    metrics = {
        'jaccard': jaccard,
        'accuracy': match_accuracy,
        'f1': f1_micro_average,
    }
    return metrics


def prepare_metrics(model):
    if model.startswith('t5'):
        metrics = compute_seq2seq_metrics

    else:
        metrics = compute_multilabel_metrics

    return metrics


def prepare_model(args, labels=None):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.model.startswith('t5'):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    else:
        num_labels = len(labels)
        id2label = {i: c for i, c in enumerate(labels)}
        label2id = {c: i for i, c in id2label.items()}
        problem_type = 'multi_label_classification'
        model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                                   problem_type=problem_type,
                                                                   num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id)

    return model, tokenizer


def prepare_training_data(df, tokenizer, labels, args):

    # this is temporary group split
    # ideally we would do the entire cross-validation
    if args.split_type == 'group':
        lpgo = LeavePGroupsOut(10)
        for (train_index, test_index) in lpgo.split(df, groups=df['tree_id']):
            break

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    elif args.split_type == 'time':
        trees = df['tree_id'].unique()

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for tree in tqdm(trees, total=len(trees)):
            current_tree = df[df['tree_id'] == tree]
            current_tree_train = current_tree[current_tree['time'] < current_tree['time'].quantile(.8)]
            current_tree_test = current_tree[current_tree['time'] >= current_tree['time'].quantile(.8)]
            train_df = pd.concat([train_df, current_tree_train])
            test_df = pd.concat([test_df, current_tree_test])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    if args.model.startswith('t5'):
        train_dataset = train_dataset.map(tokenize_seq2seq_input,
                                          batched=True,
                                          fn_kwargs={'tokenizer': tokenizer,
                                                     'source_max_length': args.source_max_length,
                                                     'target_max_length': args.target_max_length,
                                                     },
                                          remove_columns=train_dataset.column_names)
        test_dataset = test_dataset.map(tokenize_seq2seq_input,
                                          batched=True,
                                          fn_kwargs={'tokenizer': tokenizer,
                                                     'source_max_length': args.source_max_length,
                                                     'target_max_length': args.target_max_length,
                                                     },
                                          remove_columns=test_dataset.column_names)

    else:
        train_dataset = train_dataset.map(tokenize_classification_input,
                                          batched=True,
                                          fn_kwargs={'tokenizer': tokenizer,
                                                     'labels': labels,
                                                     'source_max_length': args.source_max_length,
                                                     },
                                          remove_columns=train_dataset.column_names)
        test_dataset = test_dataset.map(tokenize_classification_input,
                                          batched=True,
                                          fn_kwargs={'tokenizer': tokenizer,
                                                     'labels': labels,
                                                     'source_max_length': args.source_max_length,
                                                     },
                                          remove_columns=test_dataset.column_names)

    return train_dataset, test_dataset
