from recbole.data import create_dataset
from recbole.config import Config

DATA_PATH = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\recbole"

config = Config(
    model="RecVAE",
    dataset="movielens-1m",
    config_dict={
        "data_path": DATA_PATH
    }
)

dataset = create_dataset(config)

iid_field = dataset.iid_field

# total number of items
num_items = dataset.num(iid_field)

# generate mapping
internal_ids = list(range(num_items))
original_tokens = dataset.id2token(iid_field, internal_ids)

out_path = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\item_mapping.tsv"

with open(out_path, "w", encoding="utf-8") as f:
    f.write("internal_id\toriginal_id\n")
    for i, token in zip(internal_ids, original_tokens):
        f.write(f"{i}\t{token}\n")

print("Saved mapping to:", out_path)
print("Total items:", num_items)