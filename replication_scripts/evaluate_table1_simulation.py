import os
import math
import csv
from collections import defaultdict

BASE_DIR = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m"
HISTORIES_PATH = r"C:\Users\patel\Streaming-Recommender-Algorithmic-Drift\data\processed\movielens-1m\true_holdout_last10.tsv"

SIM_SEQ_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0",
    "sim_sequences.tsv"
)

K = 10


def load_ground_truth(histories_path, k=10):
    gt = {}
    with open(histories_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            user = str(row["user_id"]).strip()
            items = [int(x) for x in row["item_ids"].split()]
            if len(items) >= k:
                gt[user] = items[-k:]
    return gt


def load_sim_sequences(sim_seq_path):
    seqs = defaultdict(list)
    with open(sim_seq_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            b = int(row["round_b"])
            u = str(row["user_id"]).strip()
            step = int(row["step"])
            item = int(row["item_id"])
            seqs[(u, b)].append((step, item))

    final = {}
    for key, vals in seqs.items():
        vals = sorted(vals, key=lambda x: x[0])
        final[key] = [item for _, item in vals]
    return final


def hit_at_k(pred, gt, k=10):
    pred_k = pred[:k]
    gt_set = set(gt)
    return 1.0 if any(x in gt_set for x in pred_k) else 0.0


def recall_at_k(pred, gt, k=10):
    pred_k = pred[:k]
    gt_set = set(gt)
    hits = sum(1 for x in pred_k if x in gt_set)
    return hits / len(gt_set) if gt_set else 0.0


def ndcg_at_k(pred, gt, k=10):
    pred_k = pred[:k]
    gt_set = set(gt)

    dcg = 0.0
    for idx, item in enumerate(pred_k):
        if item in gt_set:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("Reading ground truth from:", HISTORIES_PATH)
    print("Reading simulation sequences from:", SIM_SEQ_PATH)

    gt = load_ground_truth(HISTORIES_PATH, K)
    seqs = load_sim_sequences(SIM_SEQ_PATH)

    first_gt_user = next(iter(gt))
    print("\nDEBUG first ground-truth user:", first_gt_user)
    print("DEBUG first ground-truth items:", gt[first_gt_user])

    shown = 0
    for (user, b), pred_seq in seqs.items():
        if user == first_gt_user:
            print("DEBUG first matching sim sequence:", pred_seq[:10])
            print("DEBUG overlap:", set(pred_seq[:10]) & set(gt[first_gt_user]))
            shown += 1
            if shown == 3:
                break

    print("Ground-truth users:", len(gt))
    print("Simulation sequences:", len(seqs))

    hr_vals = []
    rec_vals = []
    ndcg_vals = []

    for (user, b), pred_seq in seqs.items():
        if user not in gt:
            continue
        truth = gt[user]
        hr_vals.append(hit_at_k(pred_seq, truth, K))
        rec_vals.append(recall_at_k(pred_seq, truth, K))
        ndcg_vals.append(ndcg_at_k(pred_seq, truth, K))

    print("Comparable sequences:", len(hr_vals))

    if not hr_vals:
        print("No comparable user sequences found. Check file paths and user ids.")
        return

    hr = sum(hr_vals) / len(hr_vals)
    rec = sum(rec_vals) / len(rec_vals)
    ndcg = sum(ndcg_vals) / len(ndcg_vals)

    print("\nSimulation metrics (Table 1 style):")
    print(f"HR@10     = {hr:.4f}")
    print(f"Recall@10 = {rec:.4f}")
    print(f"NDCG@10   = {ndcg:.4f}")

    print("\nParent paper RecVAE simulation values:")
    print("HR@10     = 0.3393")
    print("Recall@10 = 0.0462")
    print("NDCG@10   = 0.0467")


if __name__ == "__main__":
    main()