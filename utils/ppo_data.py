from transformers import BertTokenizer, T5Tokenizer
from datasets import load_from_disk

actor_tokenizer = T5Tokenizer.from_pretrained("models/actor")
critic_tokenizer = BertTokenizer.from_pretrained("models/critic")
data = load_from_disk("data/actor_data").shuffle()

token_length = 128

def prepare(examples):
    examples["input_ids"] = actor_tokenizer(examples["setA"],
                                            padding="max_length",
                                            max_length=token_length).input_ids
    examples["decoder_input_ids"] = actor_tokenizer(examples["setB"],
                                            padding="max_length",
                                            max_length=token_length).input_ids
    examples["query"] = actor_tokenizer.decode(examples["input_ids"],
                                                skip_special_tokens=True)
    return examples

data = data.map(
    prepare,
    remove_columns=["setA", "setB"],
    num_proc=32,
)

data.save_to_disk("data/ppo_data")