import os
import pandas as pd

INPUT_PATH = "data/processed/interactions.csv"
OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.70
TEST_RATIO = 0.20
MIN_INTERACTIONS = 5

USER_COL = "user_id"
TIME_COL = "timestamp"

def split_one_user(u):
    u = u.sort_values(TIME_COL)
    n = len(u)
    train_end = int(n * TRAIN_RATIO)
    test_end = int(n * (TRAIN_RATIO + TEST_RATIO))
    train = u.iloc[:train_end]
    test = u.iloc[train_end:test_end]
    val = u.iloc[test_end:]
    return train, test, val

df = pd.read_csv(INPUT_PATH)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

train_parts, test_parts, val_parts = [], [], []

for _, u in df.groupby(USER_COL):
    if len(u) < MIN_INTERACTIONS:
        continue
    tr, te, va = split_one_user(u)
    train_parts.append(tr)
    test_parts.append(te)
    val_parts.append(va)

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

total = len(train_df) + len(test_df) + len(val_df)
print("Train:", round(len(train_df) / total, 4))
print("Test:", round(len(test_df) / total, 4))
print("Val:", round(len(val_df) / total, 4))

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False)

print("Saved: data/splits/train.csv, test.csv, validation.csv")
print("DO NOT USE validation.csv until final evaluation")