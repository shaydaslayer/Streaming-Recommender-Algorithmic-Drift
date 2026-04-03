import os
import csv
from collections import defaultdict

INTER_PATH = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\recbole\movielens-1m\movielens-1m.inter"
OUT_PATH = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\ground_truth_from_inter.tsv"
K = 10


def detect_columns(fieldnames):
    lower = {f.lower(): f for f in fieldnames}

    user_candidates = [
        "user_id:token", "user_id", "userid", "user", "uid"
    ]
    item_candidates = [
        "item_id:token", "item_id", "itemid", "video", "movie_id:token", "movieid", "iid"
    ]
    time_candidates = [
        "timestamp:float", "timestamp", "time", "ts"
    ]

    user_col = next((lower[c] for c in user_candidates if c in lower), None)
    item_col = next((lower[c] for c in item_candidates if c in lower), None)
    time_col = next((lower[c] for c in time_candidates if c in lower), None)

    return user_col, item_col, time_col


def main():
    with open(INTER_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames

        print("Detected columns:", fieldnames)

        user_col, item_col, time_col = detect_columns(fieldnames)

        print("Using user column:", user_col)
        print("Using item column:", item_col)
        print("Using time column:", time_col)

        if user_col is None or item_col is None:
            raise ValueError("Could not detect user/item columns in .inter file")

        rows_by_user = defaultdict(list)

        for row in reader:
            user = str(row[user_col]).strip()
            item = int(float(row[item_col]))
            if time_col is not None:
                t = float(row[time_col])
            else:
                # fallback: preserve file order if timestamp missing
                t = len(rows_by_user[user])

            rows_by_user[user].append((t, item))

    gt = {}
    for user, rows in rows_by_user.items():
        rows = sorted(rows, key=lambda x: x[0])
        items = [item for _, item in rows]
        if len(items) >= K:
            gt[user] = items[-K:]

    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["user_id", "item_ids"])
        for user, items in gt.items():
            writer.writerow([user, " ".join(map(str, items))])

    print("Saved ground truth to:", OUT_PATH)
    print("Users saved:", len(gt))


if __name__ == "__main__":
    main()