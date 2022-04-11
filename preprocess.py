from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from tqdm import tqdm
from itertools import permutations

data = load_dataset("tapaco", "pt")
data['train'].to_csv("tapaco_pt.csv", index=False)

df = pd.read_csv("tapaco_pt.csv")

df.set_index(['paraphrase_set_id', 'sentence_id'], inplace=True)
new_df = pd.DataFrame(columns=['id', 'setA', 'setB'])
for id, group in tqdm(df.groupby(level=0)):
    for seta, setb in permutations(group['paraphrase'], 2):
        new_df = new_df.append({'id': id, 'setA':seta, 'setB':setb}, ignore_index=True)
del df
del data

data = Dataset.from_pandas(new_df)
data = data.train_test_split(test_size=0.1)
del new_df
data.save_to_disk("data/mt5data")
