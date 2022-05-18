import argparse
from datasets import load_from_disk
import numpy as np
import torch
from tqdm import tqdm
import wandb
import time

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)

from utils.ppo import PPOTrainer

def ppo_trainer(raw_args=None):
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate for a backward/update pass.")
    parser.add_argument("--num_warmup_steps", type=int, default=20,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--actor", type=str, default="models/actor",
        help="")
    parser.add_argument("--critic", type=str, default="models/critic",
        help="")
    args = parser.parse_args(raw_args)
    
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = T5ForConditionalGeneration.from_pretrained(args.actor)
    ref_actor = T5ForConditionalGeneration.from_pretrained(args.actor)
    actor_tokenizer = T5Tokenizer.from_pretrained(args.actor)

    critic = BertForSequenceClassification.from_pretrained(args.critic)
    critic_tokenizer = BertTokenizer.from_pretrained(args.critic)

    actor.to(device)
    ref_actor.to(device)
    critic.to(device)

    config = {
        "steps": 25600,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,   
        "txt_in_len": args.token_length,
        "txt_out_len": args.token_length,
        "lr": 1.41e-5,
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1, 
    }

    decoding_config = {
        "temperature": 0.5,
        "typical_p": 0.2
    }

    wandb.watch(actor, log='all')

    data = load_from_disk("data/actor_data")

    def prepare(examples):
        examples["input_ids"] = actor_tokenizer(examples["setA"],
                                                padding="max_length").input_ids
        examples["decoder_input_ids"] = actor_tokenizer(examples["setB"],
                                                padding="max_length").input_ids
        examples["query"] = actor_tokenizer.decode(examples["input_ids"],
                                                   skip_special_tokens=True)
        return examples

    data = data.map(
        prepare,
        remove_columns=["setA", "setB"],
        num_proc=16,
    )

    ppo_trainer = PPOTrainer(actor, ref_actor, **config)
    fbs = config['forward_batch_size']

    for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
        torch.cuda.empty_cache()
        data = data.shuffle()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()
        
        #### get a batch from the dataset
        batch = data["train"].select(range(config['batch_size']))
        game_data['query'] = batch['query']
        game_data["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
        
        #### get response from gpt2
        t = time.time()
        total_length = config['txt_in_len'] + config['txt_out_len']
        response  = actor.generate(game_data["input_ids"],
                                   max_length=total_length,
                                   **decoding_config)
        game_data['response'] = actor_tokenizer.batch_decode(response, skip_special_tokens=True)
        timing['time/get_response'] = time.time()-t

        #### tokenize text for sentiment analysis
        t = time.time()
        examples = critic_tokenizer(game_data['query'], game_data['response'],
                            max_length=args.token_length, return_tensors="pt",
                            padding="max_length", truncation=True)
        tokens = examples["input_ids"].to(device)
        attentions = examples["labels"].to(device)
        timing['time/build_input_sentiment'] = time.time()-t

        #### get sentiment score
        t = time.time()
        rewards = critic(tokens, attentions)
        timing['time/get_sentiment_preds'] = time.time()-t
        print(rewards)
        exit()
        
        #### Run PPO training 
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing['time/optimization'] = time.time()-t
        
        #### Log everything
        timing['time/epoch'] = time.time()-t0
        table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
        logs.update({'game_log':wandb.Table(
            columns=['query', 'response', 'reward'],
            rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        wandb.log(logs)