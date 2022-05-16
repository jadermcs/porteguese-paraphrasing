import argparse
from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk

def actor_train(raw_args=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Size of the batch.")
    parser.add_argument("--token_length", type=int, default=256,
        help="Size of token sequence.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
        help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20,
        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=20,
        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args(raw_args)

    model = MT5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = MT5Tokenizer.from_pretrained("t5-small")

    data = load_from_disk("data/actor_data")

    def tokenize(example):
        # add prefix for T5 and tokenize
        a = tokenizer("paraphrase: " + example['setA'], max_length=args.token_length,
                padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            b = tokenizer.encode(example['setB'], max_length=args.token_length,
                padding="max_length", truncation=True)
        return {**a, 'labels': b}

    col_names = data['train'].features
    data = data.map(
        tokenize,
        remove_columns=col_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    args = Seq2SeqTrainingArguments(
        "models/actor",
        run_name="actor",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="no",
        evaluation_strategy="steps",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=data['train'],
        eval_dataset=data['valid'],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    actor_train()
