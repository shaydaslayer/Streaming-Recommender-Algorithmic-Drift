import os
import pandas as pd

RAW_DIR = "data/raw/ml-1m"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

ratings_path = os.path.join(RAW_DIR, "ratings.dat")

print("Loading:", ratings_path)

ratings = pd.read_csv(
    ratings_path,
    sep="::",
    engine="python",
    names=["user_id", "item_id", "rating", "timestamp"]
)

print("Total rows:", len(ratings))

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")

#keep positive interactions
ratings = ratings[ratings["rating"] >= 3].copy()

print("Rows after rating>=3:", len(ratings))

out_path = os.path.join(OUT_DIR, "interactions.csv")
ratings.to_csv(out_path, index=False)

print("Saved:", out_path)