import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from itertools import permutations
from eda import eda
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    TrainingArguments, Trainer
)
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, load_metric
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

np.random.seed(42)
model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

data = load_dataset("tapaco", "en")
data['train'].to_csv("data/tapaco_en.csv", index=False)

sample_data = False

df = pd.read_csv("data/tapaco_en.csv")
df.drop(columns=["lists", "tags", "language"], inplace=True)
df["paraphrase"] = df["paraphrase"].str.lower()
if sample_data:
  indexes = random.choice(df["paraphrase_set_id"].unique(), k=1000)
  df = df[df["paraphrase_set_id"].isin(indexes)]

train_indexes = df[df.paraphrase_set_id % 10 != 0].index
valid_indexes = df[df.paraphrase_set_id % 10 == 0].index

def match_pairs(df, index):
    df = df.loc[index]
    df.set_index(['paraphrase_set_id', 'sentence_id'], inplace=True)
    new_df = []
    for id, group in tqdm(df.groupby(level=0)):
        for seta, setb in permutations(group['paraphrase'], 2):
            new_df.append({'id': id, 'setA':seta, 'setB':setb})
    return pd.DataFrame.from_records(new_df)

train_df = match_pairs(df, train_indexes)
valid_df = match_pairs(df, valid_indexes)

def get_other(df):
    df['other'] = np.roll(df['setB'], df.groupby("id").count().max()["setA"])
    return df

train_df = get_other(train_df)
valid_df = get_other(valid_df)

train = Dataset.from_pandas(train_df, split="train")
valid = Dataset.from_pandas(valid_df, split="valid")
data = DatasetDict({"train": train, "valid": valid})
data.save_to_disk("data/critic_data")

data = load_from_disk("data/critic_data")

def batched_eda(examples):
    return [eda(example, num_aug=1)[0] for example in examples]

def gen_examples(examples):
  len_examples = len(examples["setA"])
  result = {
      "labels": [1]*len_examples + [0]*len_examples,
      "setA": examples["setA"] + examples["setA"],
      "setB": examples["setB"] + (
                random.choices(batched_eda(examples["setA"])+examples["other"],
                k=len_examples)
            ) # create fake paraphrasing
    }
  return result

data = data.map(
    gen_examples,
    remove_columns=["id", "other"],
    batched=True,
).shuffle()

def tokenize(example):
  result = tokenizer(example['setA'], example['setB'], max_length=256,
                  padding="max_length", truncation=True, return_tensors="pt")
  return result

col_names = data['train'].features
data = data.map(
    tokenize,
    remove_columns=["setA", "setB"],
)

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
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
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