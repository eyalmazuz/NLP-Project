import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def create_post_comment_pairs(df: pd.DataFrame, args) -> pd.DataFrame:
    tuples = []
    for row in tqdm(df.itertuples(), total=len(df)):
        if row.parent == -1:
            continue

        tree_id = row.tree_id
        comment = row.text
        root = df[(df['tree_id'] == tree_id) & (df['parent'] == -1)]['text'].values[0]

        if 't5' in args.model:
            tuples.append((root, comment, tree_id, row.timestamp, row.labels))

        else:
            # row starts from 7 because itertuples also returns the index in the tuple.
            tuples.append((root, comment, tree_id, row.timestamp, *row[7:]))

    if 't5' in args.model:
        tuples_df = pd.DataFrame(tuples, columns=['post', 'comment', 'tree_id', 'time', 'labels'])

    else:
        tuples_df = pd.DataFrame(tuples, columns=['post', 'comment', 'tree_id', 'time'] + df.columns[6:].tolist())

    tuples_df['inputs'] = 'comment: ' + tuples_df.comment

    if args.add_post_context:
        tuples_df['inputs'] = tuples_df['inputs'].str.cat(' post: ' + tuples_df.post)

    # This makes sure that the labels are the last columns in the dataframe
    new_columns_order = tuples_df.columns[:4].tolist() + [tuples_df.columns[-1]] + tuples_df.columns[4:-1].tolist()
    tuples_df = tuples_df[new_columns_order]

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
    row = row[6:]  # filter all columns that aren't labels
    labels = list(row[row != 0].index.values)
    labels = ','.join(labels) if labels else 'No Label'

    return labels + '</s>'


def convert_labels_to_text(df: pd.DataFrame) -> pd.DataFrame:
    df['labels'] = df.progress_apply(get_labels, axis=1)

    return df


def prepare_data(data_path, args):
    df = pd.read_csv(data_path, index_col=0)

    df = rename_df_columns(df)

    labels = df.columns[6:].tolist()

    if 't5' in args.model:
        df = convert_labels_to_text(df)

    pairs_df = create_post_comment_pairs(df, args)
    pairs_df = remove_bad_comments(pairs_df).drop(['post', 'comment'], axis=1)

    return pairs_df, labels
