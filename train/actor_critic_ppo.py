import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    T5ForConditionalGeneration, T5Tokenizer
)

@dataclass
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class ActorCritic(nn.Module):
    def __init__(self, actor, critic) -> None:
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)


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
    args = parser.parse_args(raw_args)

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    actor = T5ForConditionalGeneration.from_pretrained("models/actor")
    actor_tokenizer = T5Tokenizer.from_pretrained("models/actor")

    critic = BertForSequenceClassification.from_pretrained("models/critic")
    critic_tokenizer = BertTokenizer.from_pretrained("models/critic")


    