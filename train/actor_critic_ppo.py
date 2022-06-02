import argparse
from datasets import load_from_disk
import numpy as np
import torch
from tqdm import tqdm
import wandb
import time

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    T5Tokenizer
)
from models import T5HeadWithValue

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_critic = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = T5HeadWithValue.from_pretrained(args.actor)
    ref_actor = T5HeadWithValue.from_pretrained(args.actor)
    actor_tokenizer = T5Tokenizer.from_pretrained(args.actor)

    critic = BertForSequenceClassification.from_pretrained(args.critic)
    critic_tokenizer = BertTokenizer.from_pretrained(args.critic)

    actor.to(device)
    ref_actor.to(device)
    critic.to(device_critic)

    config = {
        "steps": 500_000,
        "batch_size": 32,
        "forward_batch_size": 16,
        "ppo_epochs": 4,   
        "txt_in_len": args.token_length,
        "txt_out_len": args.token_length,
        "lr": 3e-5,
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
        "min_length":-1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True, 
    }

    wandb.watch(actor, log='all')

    data = load_from_disk("data/ppo_data")

    ppo_trainer = PPOTrainer(actor, ref_actor, **config)

    for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
        torch.cuda.empty_cache()
        data = data.shuffle()
        logs = dict()
        rollout = dict()
        timing = dict()
        t0 = time.time()
        
        #### get a batch from the dataset
        batch = data["train"].select(range(config['batch_size']))
        rollout['query'] = batch['query']
        try: # corrigir essa gambiarra  - expected sequence of length 128 at dim 1 (got 145)
            rollout["input_ids"] = torch.LongTensor(batch["input_ids"]).to(device)
        except:
            continue
        
        #### get response from t5
        t = time.time()
        total_length = config['txt_out_len']
        response  = actor.generate(rollout["input_ids"],
                                    max_length=total_length,
                                    **decoding_config)
        # with torch.no_grad():
        actor_output = actor(input_ids=rollout["input_ids"], decoder_input_ids=response)
        rollout["values"] = actor_output.values
        rollout["logits"] = actor_output.logits
        rollout["ref_logits"] = ref_actor(input_ids=rollout["input_ids"],
                                        decoder_input_ids=response).logits
        rollout["response_ids"] = response
        rollout["response"] = actor_tokenizer.batch_decode(response, skip_special_tokens=True)
        timing["time/get_response"] = time.time()-t

        #### tokenize text for paraphrasing quality
        t = time.time()
        examples = critic_tokenizer(rollout['query'],
                                    rollout['response'], max_length=args.token_length,
                                    return_tensors="pt", padding="max_length",
                                    truncation=True)
        examples = {k:v.to(device_critic) for k,v in examples.items()}
        timing['time/build_input_sentiment'] = time.time()-t

        #### get paraphrasing score
        t = time.time()
        rewards = critic(**examples).logits.softmax(dim=-1)[:,1]
        rollout["rewards"] = rewards.to(device)
        timing['time/get_sentiment_preds'] = time.time()-t
        
        #### Run PPO training 
        t = time.time()
        stats = ppo_trainer.step(rollout)
        timing['time/optimization'] = time.time()-t

        #### Log everything
        timing['time/epoch'] = time.time()-t0
        table_rows = [list(r) for r in zip(rollout['query'], rollout['response'], rewards.cpu().tolist())]
        logs.update({'game_log':wandb.Table(
            columns=['query', 'response', 'reward'],
            rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        logs['env/reward_mean'] = torch.mean(rewards).cpu().detach().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().detach().numpy()
        logs['env/reward_dist'] = rewards.cpu().detach().numpy()
        wandb.log(logs)