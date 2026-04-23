"""
Evaluate the fine-tuned DistilBERT classifier on the held-out test set.

Reports: accuracy, per-class precision/recall/F1, confusion matrix.
Per Artikate brief, eval set is the hand-spot-verified test.csv.
"""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from predict import TicketClassifier

DATA_DIR = Path(__file__).parent / "data"
TEST_CSV = DATA_DIR / "test.csv"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"


def main():
    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} examples across {df['label'].nunique()} classes\n")

    clf = TicketClassifier()
    print("Predicting...")
    preds = clf.predict(df["text"].tolist())
    df["pred"] = preds

    # Metrics
    acc = accuracy_score(df["label"], preds)
    f1_macro = f1_score(df["label"], preds, average="macro")
    f1_weighted = f1_score(df["label"], preds, average="weighted")

    print(f"\n=== METRICS ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")

    print(f"\n=== PER-CLASS REPORT ===")
    report = classification_report(df["label"], preds, digits=3)
    print(report)

    print(f"\n=== CONFUSION MATRIX ===")
    labels_sorted = sorted(df["label"].unique())
    cm = confusion_matrix(df["label"], preds, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    print(cm_df)

    # Save results
    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": classification_report(df["label"], preds, output_dict=True, digits=3),
        "confusion_matrix": cm_df.to_dict(),
        "n_test": int(len(df)),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Most-confused pair analysis
    cm_no_diag = cm.copy()
    import numpy as np
    np.fill_diagonal(cm_no_diag, 0)
    if cm_no_diag.sum() > 0:
        i, j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
        print(f"\nMost-confused pair: '{labels_sorted[i]}' predicted as '{labels_sorted[j]}' "
              f"({cm_no_diag[i, j]} times)")


if __name__ == "__main__":
    main()


