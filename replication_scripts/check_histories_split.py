import csv
import collections

paths = [
    r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\Histories.tsv",
    r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\Histories_for_comparison.tsv",
]

for p in paths:
    d = collections.defaultdict(list)
    with open(p, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            d[str(row["User"]).strip()].append(int(row["Video"]))

    print("\nFILE:", p)
    print("user 0 length =", len(d["0"]))
    print("user 0 first 10 =", d["0"][:10])
    print("user 0 last 15 =", d["0"][-15:])