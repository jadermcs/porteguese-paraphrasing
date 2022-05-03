from transformers import (
    BertForSequenceClassification, BertTokenizer,
    TrainingArguments, Trainer
)
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

data = load_from_disk("data/critic_data")

def tokenize(example):
  result = tokenizer(example['setA'], example['setB'], max_length=256,
                  padding="max_length", truncation=True)
  return result

col_names = data['train'].features
data = data.map(
    tokenize,
    remove_columns=["setA", "setB"],
    batched=True,
    num_proc=8,
)
data.save_to_disk("data/critic_data")
data = load_from_disk("data/critic_data")

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
    "models/bert_fake_paraphrase_detector",
    num_train_epochs=20,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=64,
    save_strategy="no",
    evaluation_strategy="steps",
    eval_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
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