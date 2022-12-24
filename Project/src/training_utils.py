from datasets import Dataset

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import LeavePGroupsOut

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification


def create_seq2seq_tokenize_fn(tokenizer, source_max_legnth: int=512, target_max_legnth: int=128):
    def tokenize_input(examples):
        model_inputs = tokenizer(examples["inputs"], max_length=source_max_legnth, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["labels"], max_length=target_max_legnth, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return tokenize_input


def create_clf_tokenize_fn(tokenizer, labels, source_max_legnth: int=512):
    def tokenize_input(examples):
        model_inputs = tokenizer(examples["inputs"], max_length=source_max_legnth,
                                 padding=True, truncation=True, return_tensors='pt')

        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(examples["inputs"]), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        model_inputs["labels"] = labels_matrix.tolist()

        return model_inputs

    return tokenize_input


def prepare_metrics(tokenizer, accuracy):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = np.argmax(preds[0], -1)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = accuracy.compute(predictions=preds.ravel(), references=labels.ravel())

        return results

    return compute_metrics


def prepare_model(args, labels=None):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.model.startswith('t5'):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer_fn = create_seq2seq_tokenize_fn(tokenizer, args.source_max_length, args.target_max_length)

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

        tokenizer_fn = create_clf_tokenize_fn(tokenizer, label2id.keys(), args.source_max_length)

    return model, tokenizer, tokenizer_fn


def prepare_training_data(df, tokenizer_fn, split_type):

    # this is temporary group split
    # ideally we would do the entire cross-validation
    if split_type == 'group':
        lpgo = LeavePGroupsOut(5)
        for (train_index, test_index) in lpgo.split(df, groups=df['tree_id']):
            break

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

    elif split_type == 'time':
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
    train_dataset = train_dataset.map(tokenizer_fn, batched=True)

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(tokenizer_fn, batched=True)

    return train_dataset, test_dataset
