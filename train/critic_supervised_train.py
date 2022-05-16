import argparse
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    TrainingArguments, Trainer
)
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, precision_recall_fscore_support, r2_score, max_error,
    mean_absolute_percentage_error)

def critic_train(raw_args=None):
    parser = argparse.ArgumentParser(description="Finetune a transformers "
                                    "model on a causal language modeling task")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Size of the batch.")
    parser.add_argument("--token_length", type=int, default=256,
        help="Size of token sequence.")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
        help="Initial learning rate to use.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=20,
        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=20,
        help="Number of steps for the warmup in the lr scheduler.")
    args = parser.parse_args(raw_args)
    
    # config for regression:
    model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)
    tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

    data = load_from_disk("data/critic_data")

    def tokenize(example):
        result = tokenizer(example['setA'], example['setB'], max_length=args.token_length,
                            padding="max_length", truncation=True)
        return result
    
    data = data.map(
        tokenize,
        remove_columns=["setA", "setB"],
        batched=True,
        num_proc=8,
    )

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     return {
    #         "rmse": mean_squared_error(labels, predictions, squared=False),
    #         "mae": mean_absolute_error(labels, predictions),
    #         "r2_score": r2_score(labels, predictions),
    #         "max_error": max_error(labels, predictions),
    #         "mape": mean_absolute_percentage_error(labels, predictions),
    #         }

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    args = TrainingArguments(
        "models/critic",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="no",
        evaluation_strategy="steps",
        warmup_steps=args.num_warmup_steps,
        weight_decay=args.weight_decay,
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=data['train'],
        eval_dataset=data['valid'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    critic_train()