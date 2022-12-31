import argparse
import os

import evaluate

from src.utils import prepare_data
from src.training_utils import prepare_metrics, prepare_model, prepare_training_data
from src.training import train

os.environ["WANDB_DISABLED"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data-path', type=str,
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
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='How many steps of gradient accumulation to perform before updating weights')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs to perform')
    parser.add_argument('--fp16', action='store_true',
                        help='if to use fp16 backend when training on compatible devices')
    parser.add_argument('--save-steps', type=int, default=100, help='After how many steps to save the model')
    parser.add_argument('--logging-steps', type=int, default=50,
                        help='After how many steps to perform evaluation to the model')

    return parser.parse_args()


def main():
    args = parse_args()

    df, labels = prepare_data(args.data_path, args.model)

    model, tokenizer = prepare_model(args, labels)

    train_dataset, test_dataset = prepare_training_data(df, tokenizer, labels, args)

    metrics = prepare_metrics(args.model)

    train(model, tokenizer, train_dataset, test_dataset, metrics, args, labels=labels)


if __name__ == "__main__":
    main()
