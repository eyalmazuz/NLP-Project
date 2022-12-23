import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()

def create_post_comment_pairs(df: pd.DataFrame) -> pd.DataFrame:
    tuples = []
    for row in tqdm(df.itertuples(), total=len(df)):
        if row.parent == -1:
            continue
      
        tree_id = row.tree_id
        comment = row.text
        root = df[(df['tree_id'] == tree_id) & (df['parent'] == -1)]['text'].values[0]
      
        tuples.append((root, comment, tree_id, row.timestamp, row.labels))

    tuples_df = pd.DataFrame(tuples, columns=['post', 'comment', 'tree_id', 'time', 'labels'])
    tuples_df['inputs'] = 'comment: ' + tuples_df.comment.str.cat(' post: ' + tuples_df.post)

    return tuples_df

def remove_bad_comments(df: pd.DataFrame) -> pd.DataFrame:
    removed_tokens = ['[removed]', '[deleted]']

    df = df[~(df.post.isin(removed_tokens)) & ~(df.comment.isin(removed_tokens))]

    return df

def rename_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ['node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent',
       'Aggressive', 'Agree But', 'Agree To Disagree', 'Alternative', 'Answer',
       'Attack Validity', 'BAD', 'Clarification', 'Complaint', 'Convergence',
       'Counter Argument', 'Critical Question', 'Direct No', 'Double Voicing',
       'Extension', 'Irrelevance', 'Moderation', 'Neg Transformation',
       'Nitpicking', 'No Reason Disagreement', 'Personal', 'Positive',
       'Repetition', 'Rephrase Attack', 'Request Clarification', 'Ridicule',
       'Sarcasm', 'Softening', 'Sources', 'Viable Transformation',
       'W Qualifiers']

    return df

def get_labels(row: pd.Series) -> str:
    row = row[6:] # filter all columns that aren't labels
    labels = list(row[row != 0].index.values)
    labels = ','.join(labels) if labels else 'No Label'
        
    return labels + '</s>'

def convert_labels_to_text(df: pd.DataFrame) -> pd.DataFrame:
    df['labels'] = df.progress_apply(get_labels, axis=1)

    return df

def create_tokenize_fn(tokenizer, source_max_legnth: int, target_max_legnth: int):
    def tokenize_input(examples):
        model_inputs = tokenizer(examples["inputs"], max_length=source_max_legnth, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["labels"], max_length=target_max_legnth, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    return tokenize_input

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = np.argmax(preds[0], -1)
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    results = exact_match_metric.compute(predictions=preds.ravel(), references=labels.ravel())

    return results
