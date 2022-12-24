from transformers import TrainingArguments, Trainer, IntervalStrategy
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments, DataCollatorForSeq2Seq


def train(model, tokenizer, train_dataset, test_dataset, metrics, args):
    if args.model.startswith('t5'):
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
            compute_metrics=metrics,
            data_collator=data_collator
        )

    else:
        training_args = TrainingArguments(
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
            save_total_limit=5
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=metrics,
        )


    trainer.train()
