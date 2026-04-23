"""
Split synthetic data into train (80%) and test (20%) sets.
Stratified — equal proportion per class in both splits.

Per Artikate brief, the test set is then hand-spot-verified.
See generation_prompt.md for full data lineage documentation.
"""
import json
import random
from pathlib import Path
import pandas as pd

random.seed(42)  # reproducibility

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "synthetic_raw.json"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]
TEST_FRACTION = 0.20  # 200 of 1000 → 40 per class


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_rows, test_rows = [], []
    for cls in CLASSES:
        cls_examples = [x for x in data if x["label"] == cls]
        random.shuffle(cls_examples)
        n_test = int(len(cls_examples) * TEST_FRACTION)
        test_rows.extend(cls_examples[:n_test])
        train_rows.extend(cls_examples[n_test:])

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    pd.DataFrame(train_rows).to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    pd.DataFrame(test_rows).to_csv(TEST_PATH, index=False, encoding="utf-8")

    print(f"Train: {len(train_rows)} examples → {TRAIN_PATH}")
    print(f"Test:  {len(test_rows)} examples → {TEST_PATH}")
    print("\nClass distribution (train / test):")
    for cls in CLASSES:
        tr = sum(1 for x in train_rows if x["label"] == cls)
        te = sum(1 for x in test_rows if x["label"] == cls)
        print(f"  {cls:18s} {tr:4d} / {te:3d}")


if __name__ == "__main__":
    main()


