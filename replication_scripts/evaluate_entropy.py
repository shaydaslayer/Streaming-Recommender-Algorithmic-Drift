import csv
import os

BASE_DIR = os.path.join(os.getcwd(), "data", "processed", "movielens-1m")

ENTROPY_PATH = os.path.join(
    BASE_DIR,
    "RecVAE",
    "graphs",
    "topk_10",
    "1.0_0.0_gamma1_0.0_sigmagamma1_0.01_gamma2_0.0_sigmagamma2_0.01_gamma3_0.0_sigmagamma3_0.01_eta_0.0",
    "entropy_metrics.tsv"
)

def main():
    topk_vals = []
    history_vals = []

    with open(ENTROPY_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            topk_vals.append(float(row["topk_entropy"]))
            history_vals.append(float(row["history_entropy"]))

    if not topk_vals:
        print("No entropy rows found.")
        return

    avg_topk = sum(topk_vals) / len(topk_vals)
    avg_history = sum(history_vals) / len(history_vals)

    print("Entropy summary:")
    print(f"Rows read            = {len(topk_vals)}")
    print(f"Average TopK Entropy = {avg_topk:.4f}")
    print(f"Average Hist Entropy = {avg_history:.4f}")

if __name__ == "__main__":
    main()