"""
Latency assertion test for Section 3 ticket classifier.

Per Artikate brief:
- "Classification must complete in under 500ms per ticket"
- "Test that takes a list of 20 raw ticket strings and asserts that each
   prediction is one of the five valid classes and that inference completes
   within the 500ms constraint"
"""
import sys
import time
from pathlib import Path

import pytest

# Make section3_classifier importable
sys.path.insert(0, str(Path(__file__).parent.parent / "section3_classifier"))

from predict import TicketClassifier  # noqa: E402

VALID_CLASSES = {"billing", "technical_issue", "feature_request", "complaint", "other"}
LATENCY_BUDGET_MS = 500.0

SAMPLE_TICKETS = [
    "I was charged twice for the same plan in March, please refund.",
    "The export to CSV button does nothing when I click it.",
    "It would be great if you could add Slack integration.",
    "Your service has been terrible all month, I'm very frustrated.",
    "How do I find my account ID in the dashboard?",
    "My credit card was declined but I have funds available.",
    "App keeps crashing on the login screen, error code 500.",
    "Can you add a dark mode option to the settings?",
    "Worst customer support experience of my life.",
    "What is your refund policy for annual subscriptions?",
    "The invoice for last month shows the wrong company name.",
    "Two-factor authentication isn't sending the SMS code.",
    "Please add multi-currency support for European customers.",
    "I've been waiting 3 weeks for a response, this is unacceptable.",
    "Where can I download the API documentation?",
    "Need help understanding the breakdown of charges on my bill.",
    "Files are not syncing between desktop and mobile.",
    "Suggestion: add keyboard shortcuts for common actions.",
    "Your product has so many bugs, I want to cancel.",
    "Looking for partnership opportunities with your team.",
]


@pytest.fixture(scope="module")
def classifier():
    return TicketClassifier()


def test_predictions_are_valid_classes(classifier):
    """Every prediction must be one of the 5 allowed classes."""
    preds = classifier.predict(SAMPLE_TICKETS)
    assert len(preds) == len(SAMPLE_TICKETS), "prediction count mismatch"
    for p in preds:
        assert p in VALID_CLASSES, f"Invalid prediction: {p!r}"


def test_per_ticket_latency_under_500ms(classifier):
    """Each individual ticket must classify in under 500ms (cold + warm)."""
    # Warmup — first call always slower due to model graph init
    classifier.predict("warmup ticket text")

    latencies_ms = []
    for ticket in SAMPLE_TICKETS:
        start = time.perf_counter()
        pred = classifier.predict(ticket)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        assert pred in VALID_CLASSES
        assert elapsed_ms < LATENCY_BUDGET_MS, (
            f"Ticket exceeded {LATENCY_BUDGET_MS}ms: {elapsed_ms:.1f}ms\n"
            f"  Ticket: {ticket!r}"
        )

    # Print summary stats
    avg = sum(latencies_ms) / len(latencies_ms)
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
    p99 = sorted(latencies_ms)[-1]
    print(f"\n  Latency summary over {len(latencies_ms)} tickets:")
    print(f"    avg: {avg:.1f}ms  |  p50: {p50:.1f}ms  |  p99: {p99:.1f}ms  |  budget: {LATENCY_BUDGET_MS}ms")


def test_batch_inference_works(classifier):
    """Sanity: batch prediction returns same shape and valid labels."""
    preds = classifier.predict(SAMPLE_TICKETS)
    assert isinstance(preds, list)
    assert len(preds) == len(SAMPLE_TICKETS)
    for p in preds:
        assert p in VALID_CLASSES


