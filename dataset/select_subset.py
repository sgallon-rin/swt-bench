"""
Select a subset of instances from SWT-Bench Lite dataset.
For each repository, randomly select 5 instances (or all if fewer than 5).

Usage:
    python dataset/select_subset.py

Output:
    dataset/swt_bench_lite_subset.txt  (one instance_id per line)

Use the subset with inference and evaluation:

    # Inference
    python inference/inference_opencode.py \
        --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ')

    # Evaluation
    python src/main.py \
        --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
        --predictions_path <your_predictions>
"""

import random
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

SEED = 42
MAX_PER_REPO = 5
DATASET = "eth-sri/SWT-bench_Lite_bm25_27k_zsb"
OUTPUT_FILE = "dataset/swt_bench_lite_subset.txt"


def main():
    random.seed(SEED)
    ds = load_dataset(DATASET, split="test")

    repos = {}
    for ex in ds:
        iid = ex["instance_id"]
        repo = iid.split("__")[0]
        repos.setdefault(repo, []).append(iid)

    selected = []
    print(f"{'Repository':<20} {'Available':>10} {'Selected':>10}")
    print("-" * 42)
    for repo, ids in sorted(repos.items(), key=lambda x: -len(x[1])):
        chosen = random.sample(ids, min(MAX_PER_REPO, len(ids)))
        selected.extend(chosen)
        print(f"{repo:<20} {len(ids):>10} {len(chosen):>10}")

    selected.sort()
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(selected))

    print("-" * 42)
    print(f"Total selected: {len(selected)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
