import os
import csv
from collections import defaultdict

BASE_DIR = os.path.join(os.getcwd(), "data", "processed", "movielens-1m")

FULL_PATH = os.path.join(BASE_DIR, "Histories.tsv")
TRUNC_PATH = os.path.join(BASE_DIR, "Histories_for_comparison.tsv")
OUT_PATH = os.path.join(BASE_DIR, "true_holdout_last10.tsv")

def load_histories(path):
    d = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user = str(row["User"]).strip()
            video = int(row["Video"])
            d[user].append(video)
    return d

full_hist = load_histories(FULL_PATH)
trunc_hist = load_histories(TRUNC_PATH)

with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["user_id", "item_ids"])

    count = 0
    for user in full_hist:
        full_items = full_hist[user]
        trunc_items = trunc_hist[user]

        # held-out target = the tail removed from the truncated history
        heldout = full_items[len(trunc_items):]

        if len(heldout) == 10:
            writer.writerow([user, " ".join(map(str, heldout))])
            count += 1

print("Saved:", OUT_PATH)
print("Users with exact last-10 holdout:", count)

# quick debug
print("User 0 full last 15:", full_hist["0"][-15:])
print("User 0 trunc last 15:", trunc_hist["0"][-15:])
print("User 0 holdout:", full_hist["0"][len(trunc_hist["0"]):])