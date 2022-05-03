import torch
import torch.nn as nn
from torch.distributions import Categorical

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    MT5ForConditionalGeneration, MT5Tokenizer
)

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

actor = MT5ForConditionalGeneration.from_pretrained("models/paraphrasing_pt")
actor_tokenizer = MT5Tokenizer.from_pretrained("models/paraphrasing_pt")

critic = BertForSequenceClassification.from_pretrained("models/bert_fake_paraphrase_detector")
critic_tokenizer = BertTokenizer.from_pretrained("models/bert_fake_paraphrase_detector")


class ActorCritc(nn.Module):
    def __init__(self, actor, critic) -> None:
        super(ActorCritc, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)

    