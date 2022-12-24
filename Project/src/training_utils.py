from datasets import Dataset

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import LeavePGroupsOut

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
        lpgo = LeavePGroupsOut(5)
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
