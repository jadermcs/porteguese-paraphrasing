from transformers import (
    MT5ForConditionalGeneration, MT5Tokenizer,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_from_disk

model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

data = load_from_disk("data/mt5data")

def tokenize(example):
    a = tokenizer("paraphrase: "+example['setA'], max_length=256,
            padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        b = tokenizer.encode(example['setB'], max_length=256,
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
    "models/paraphrasing_pt",
    run_name="paraphrasing_pt",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    weight_decay=.1,
    save_strategy="epoch",
    num_train_epochs=50,
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
