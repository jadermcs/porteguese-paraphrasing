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
        "txt_in_len": 5,
        "txt_out_len": 15,
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
        examples["input_ids"] = actor_tokenizer(examples["setA"], return_tensors="pt",
                                                padding="max_length").input_ids
        examples["decoder_input_ids"] = actor_tokenizer(examples["setB"], return_tensors="pt",
                                                padding="max_length").input_ids
        examples["query"] = actor_tokenizer.batch_decode(examples["input_ids"],
                                                   skip_special_tokens=True)
        return examples

    data = data.map(
        prepare,
        remove_columns=["setA", "setB"],
        num_proc=4,
        batched=True,
    )

    ppo_trainer = PPOTrainer(actor, ref_actor, **config)
    fbs = config['forward_batch_size']

    for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()
        
        #### get a batch from the dataset
        batch = data.sample(config['batch_size'])
        game_data['query'] = batch['query'].tolist()
        query_tensors = batch['tokens']
        
        #### get response from gpt2
        t = time.time()
        total_length = config['txt_in_len']+config['txt_out_len']
        response_tensors = []
        for i in range(int(config['batch_size']/fbs)):
            response  = actor.generate(query_tensors[i*fbs:(i+1)*fbs],
                                       **decoding_config)
            response_tensors.append(response)
        response_tensors = torch.cat(response_tensors)
        game_data['response'] = [actor_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
        timing['time/get_response'] = time.time()-t
        print(game_data)
        exit()

        #### tokenize text for sentiment analysis
        t = time.time()
        texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
        sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, critic_tokenizer, device)    
        timing['time/build_input_sentiment'] = time.time()-t

        #### get sentiment score
        t = time.time()
        rewards = []
        for i in range(int(config['batch_size']/fbs)):
            res = critic(sentiment_inputs[i*fbs:(i+1)*fbs],
                                        attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
            rewards.append(res)
        rewards = torch.cat(rewards)
        timing['time/get_sentiment_preds'] = time.time()-t
        
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