from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
from tqdm import tqdm
from itertools import permutations

data = load_dataset("tapaco", "pt")
data['train'].to_csv("tapaco_pt.csv", index=False)

df = pd.read_csv("tapaco_pt.csv")

df.set_index(['paraphrase_set_id', 'sentence_id'], inplace=True)
new_df = []
for id, group in tqdm(df.groupby(level=0)):
    for seta, setb in permutations(group['paraphrase'], 2):
        new_df.append({'id': id, 'setA':seta, 'setB':setb})
new_df = pd.DataFrame(new_df, columns=['id', 'setA', 'setB'])
del df
del data

data = Dataset.from_pandas(new_df)
train = data.filter(lambda x: x["id"] % 2 == 1)
valid = data.filter(lambda x: x["id"] % 2 == 0)
data = DatasetDict({
    "train": train,
    "valid": valid,
})
del new_df

data.save_to_disk("data/mt5data")
