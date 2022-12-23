import argparse
import os
os.environ["WANDB_DISABLED"] = "true"
from datasets import Dataset
import evaluate
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeavePGroupsOut

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer, IntervalStrategy
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

from src.utils import create_post_comment_pairs, remove_bad_comments, rename_df_columns, convert_labels_to_text, create_tokenize_fn, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', type=str,
                        required=True, help='Path to the discourse parsing CSV file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['t5-small', 't5-base', 't5-large', 'bert-base-uncased'],
                        help='Which classification model to use')
    parser.add_argument('--source-max-length', type=int, default=1024,
                        help='The maximum sequence size that the model can receive')
    parser.add_argument('--target-max-length', type=int, default=128,
                        help='The maximum sequence size that the model can create')
    parser.add_argument('--split-type', type=str, default='group', choices=['time', 'group'],
                        help='Which split to perform on the dataset. Time based (take the last 20% from each tree according to timestemp or Group (remove entire trees)')
    parser.add_argument('--output-dir', type=str, default='./results/', help='Where to save the model')
    parser.add_argument('-lr', '--learning-rate', type=float, default=2e-5,
                        help='Learning rate to use in the experiments')
    parser.add_argument('-b', '--batch-size', type=int, default=4, help='Batch size to use during training')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='How many steps of gradient accumulation to perform before updating weights')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs to perform')
    parser.add_argument('--fp16', action='store_true', help='if to use fp16 backend when training on compatible devices')
    parser.add_argument('--save-steps', type=int, default=100, help='After how many steps to save the model')
    parser.add_argument('--logging-steps', type=int, default=50, help='After how many steps to perform evaluation to the model')

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data, index_col=0)
    df = rename_df_columns(df)

    if args.model.startswith('t5'):
        df = convert_labels_to_text(df)

    pairs_df = create_post_comment_pairs(df)
    pairs_df = remove_bad_comments(pairs_df) 

    pairs_df.to_csv('./foo.csv', index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.model.startswith('t5'):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)

    lpgo = LeavePGroupsOut(5)
    for (train_index, test_index) in lpgo.split(pairs_df, groups=pairs_df['tree_id']):
        break
    
    train_df = pairs_df.iloc[train_index]
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(create_tokenize_fn(tokenizer, args.source_max_length, args.target_max_length), batched=True)
    
    test_df = pairs_df.iloc[test_index]
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(create_tokenize_fn(tokenizer, args.source_max_length, args.target_max_length), batched=True)

    exact_match_metric = evaluate.load("accuracy")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        eval_accumulation_steps=args.accumulation_steps,
        num_train_epochs=args.epochs,
        # warmup_steps=4000,
        fp16=args.fp16,
        #lr_scheduler_type='cosine_with_restarts',
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,
        # report_to='wandb',
        # run_name=f'T5 Reddit discourse',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        sortish_sampler=True,
        save_total_limit=5
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == "__main__": 
     main()
