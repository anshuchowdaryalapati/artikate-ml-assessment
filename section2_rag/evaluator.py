"""
Evaluation harness for Section 2 RAG pipeline.
Computes precision@3 against a manually-written Q&A set.

Definition of a 'hit' at rank k:
    A retrieved chunk is considered correct if BOTH:
    - Its document matches the expected document
    - Its page number is in the expected_pages list
"""
import json
from pathlib import Path
from typing import Dict, List

from retriever import HybridRetriever
from loader import load_all_pdfs
from chunker import chunk_pages

EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def is_hit(retrieved: Dict, expected_doc: str, expected_pages: List[int]) -> bool:
    """Does this retrieved chunk match the ground truth?"""
    return (
        retrieved["document"] == expected_doc
        and retrieved["page"] in expected_pages
    )


def evaluate(retriever: HybridRetriever, eval_data: List[Dict], k: int = 3) -> Dict:
    """Compute precision@k on the eval set."""
    per_question_precision = []
    hits_at_1 = 0
    hits_in_top_k = 0
    total = len(eval_data)

    print(f"\nEvaluating {total} questions at k={k}...")
    print("=" * 70)

    for i, item in enumerate(eval_data, 1):
        q = item["question"]
        expected_doc = item["expected_document"]
        expected_pages = item["expected_pages"]

        hits = retriever.retrieve(q, top_k=k)
        num_correct = sum(1 for h in hits if is_hit(h, expected_doc, expected_pages))
        precision_q = num_correct / k
        per_question_precision.append(precision_q)

        if hits and is_hit(hits[0], expected_doc, expected_pages):
            hits_at_1 += 1
        if num_correct > 0:
            hits_in_top_k += 1

        top_pages = [f"p{h['page']}" for h in hits]
        status = "✓" if num_correct > 0 else "✗"
        print(f"{status} Q{i}: {q[:55]}...")
        print(f"   expected: {expected_doc} pages {expected_pages[:4]}")
        print(
            f"   got:     "
            + ", ".join([f"{h['document']} {p}" for h, p in zip(hits, top_pages)])
        )
        print(f"   correct: {num_correct}/{k} | precision@{k}: {precision_q:.2f}")
        print()

    avg_precision = sum(per_question_precision) / total
    recall_at_k = hits_in_top_k / total  # at least one correct in top-k
    mrr_at_1 = hits_at_1 / total

    print("=" * 70)
    print(f"FINAL METRICS (n={total}, k={k})")
    print("=" * 70)
    print(f"Precision@{k}:        {avg_precision:.3f}")
    print(f"Recall@{k}:           {recall_at_k:.3f}   (at least 1 correct in top-{k})")
    print(f"Hit@1 / MRR@1:       {mrr_at_1:.3f}   (top-1 is correct)")
    return {
        "precision_at_k": avg_precision,
        "recall_at_k": recall_at_k,
        "hit_at_1": mrr_at_1,
        "n_questions": total,
        "k": k,
    }


if __name__ == "__main__":
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    retriever = HybridRetriever()
    # Load existing index (already built)
    try:
        retriever.load_index()
        print(f"Loaded {len(retriever.chunks)} chunks from disk.")
    except Exception as e:
        print(f"No saved index ({e}); building fresh...")
        pages = load_all_pdfs("section2_rag/data")
        chunks = chunk_pages(pages)
        retriever.build_index(chunks)

    metrics = evaluate(retriever, eval_data, k=3)
    print(f"\nResult JSON: {json.dumps(metrics, indent=2)}")


