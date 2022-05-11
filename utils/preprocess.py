from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
import random
from itertools import permutations
from utils.eda import eda

seed = 1
random.seed(seed)

def jaccard_similarity(x,y):
        """ returns the jaccard similarity between two lists """
        x = x.split()
        y = y.split()
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

def generate_data():
    data = load_dataset("tapaco", "en")
    data['train'].to_csv("data/tapaco_en.csv", index=False)

    df = pd.read_csv("data/tapaco_en.csv")
    df.drop(columns=["lists", "tags", "language"], inplace=True)
    df["paraphrase"] = df["paraphrase"].str.lower()

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

    train = Dataset.from_pandas(train_df, split="train")
    valid = Dataset.from_pandas(valid_df, split="valid")
    data = DatasetDict({"train": train, "valid": valid})

    data.save_to_disk("data/actor_data")

    del data
    del train
    del valid

    def get_other(df):
        df["other"] = None
        for index in df.index:
            examples = df.sample(1000)
            i = 0
            while df.loc[index, "other"] is None:
                if df.loc[index, "id"] != examples.iloc[i]["id"]:
                    df.loc[index, "other"] = examples.iloc[i]["setB"]
                i += 1
        return df

    train_df = get_other(train_df)
    valid_df = get_other(valid_df)

    train = Dataset.from_pandas(train_df, split="train")
    valid = Dataset.from_pandas(valid_df, split="valid")
    data = DatasetDict({"train": train, "valid": valid})

    def batched_eda(examples):
        return [eda(example, alpha_sr=.0, alpha_rs=.1, alpha_ri=.0,
                    p_rd=.2, num_aug=1)[0] for example in examples]

    def gen_examples(examples):
        result = {
            "setA": examples["setA"],
            "setB": examples["setB"],
            "other": examples["other"],
            "fake": batched_eda(examples["setA"]) # create fake paraphrasing
            }
        return result

    data = data.map(
        gen_examples,
        remove_columns=["id"],
        batched=True,
        num_proc=8,
    )

    def transform(example):
        if random.random() < .5:
            if random.random() < .3:
                example["setB"] = example["other"]
                example["labels"] = -1.5
            else:
                example["setB"] = example["fake"]
                example["labels"] = -.55
        else:
            if random.random() < .3:
                example["setB"] = example["setB"][:-1]
            example["labels"] = 1.5 - jaccard_similarity(example["setA"], example["setB"])
        return example

    data = data.map(
        transform,
        remove_columns=["other", "fake"],
    )

    data.save_to_disk("data/critic_data")

if __name__ == "__main__":
    generate_data()