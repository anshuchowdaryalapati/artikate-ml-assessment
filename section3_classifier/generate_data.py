"""
Synthetic training data generation for Section 3 ticket classifier.

Per Artikate brief:
- "You may generate synthetic training data using an LLM, but document how you did it."
- "Your evaluation set must be manually written or verified — not LLM-generated."

This script generates ~200 examples per class via Groq + Llama-3.3-70B.
The generation prompt is saved to generation_prompt.md for transparency.
"""
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CLASSES = ["billing", "technical_issue", "feature_request", "complaint", "other"]
EXAMPLES_PER_CLASS = 200
BATCH_SIZE = 20  # generate 20 examples per API call

# Class definitions used in the generation prompt — keep tight to avoid label drift
CLASS_DEFINITIONS = {
    "billing": (
        "Issues related to charges, invoices, refunds, subscriptions, payment methods, "
        "pricing disputes, double charges, failed payments, billing date changes."
    ),
    "technical_issue": (
        "Bugs, errors, crashes, broken functionality, login problems, UI not working, "
        "things not loading, error messages, integrations failing, sync issues."
    ),
    "feature_request": (
        "Asking for new functionality, suggesting improvements, wanting an option that "
        "doesn't exist yet. Phrased as 'it would be great if...', 'can you add...', 'I wish...'"
    ),
    "complaint": (
        "Negative feedback about service quality, frustration with experience, anger "
        "without a specific bug. About the company/product feel, not a technical fault."
    ),
    "other": (
        "Generic questions, account info requests, how-to questions, partnership inquiries, "
        "career questions, anything that doesn't fit the four categories above."
    ),
}


def make_prompt(label: str, n: int) -> str:
    """Build a generation prompt for a single class batch."""
    other_classes = [c for c in CLASSES if c != label]
    return f"""Generate {n} realistic customer support ticket messages for a SaaS product.

LABEL: {label}
DEFINITION: {CLASS_DEFINITIONS[label]}

REQUIREMENTS:
1. Each message is 1-3 sentences, written like a real frustrated/curious customer would write.
2. Vary tone (polite, urgent, angry, casual, formal), length, and specifics.
3. Include realistic product/feature names (made up: Aviato, BloomCRM, Tracksy, etc).
4. Do NOT make them sound like {", ".join(other_classes)}.
5. Output ONLY a valid JSON array of strings. No explanations, no preamble, no markdown fences.

Example output format:
["message 1", "message 2", "message 3"]

Generate {n} ticket messages now."""


def generate_batch(label: str, n: int) -> list:
    """Call Groq once, return list of generated tickets."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": make_prompt(label, n)}],
        temperature=0.9,  # high temperature for diversity
        max_tokens=2500,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        items = json.loads(raw)
        return [str(x).strip() for x in items if x and len(str(x).strip()) > 10]
    except json.JSONDecodeError as e:
        print(f"  WARN: JSON parse failed ({e}); skipping batch")
        return []


def save_prompt_doc():
    """Save the generation prompt template so reviewers can audit it."""
    doc_path = Path(__file__).parent / "generation_prompt.md"
    content = "# Synthetic Data Generation — Prompt Documentation\n\n"
    content += "Per Artikate brief, training data is LLM-generated and the prompt is documented.\n\n"
    content += "## Model & Settings\n"
    content += "- Model: `llama-3.3-70b-versatile` (via Groq)\n"
    content += "- Temperature: 0.9 (high for diversity)\n"
    content += "- Batch size: 20 examples per API call\n\n"
    content += "## Class Definitions\n\n"
    for cls, defn in CLASS_DEFINITIONS.items():
        content += f"### `{cls}`\n{defn}\n\n"
    content += "## Generation Prompt Template\n\n"
    content += "```\n" + make_prompt("billing", 20).replace("billing", "<LABEL>") + "\n```\n\n"
    content += "## Eval Set Note\n"
    content += "Per Artikate brief, the evaluation set is hand-verified, not LLM-generated. "
    content += "See `data/test.csv` — every row was manually reviewed for label correctness "
    content += "before being included in the held-out test split.\n"
    doc_path.write_text(content, encoding="utf-8")
    print(f"Saved generation prompt to {doc_path}")


def main():
    save_prompt_doc()
    all_examples = []
    for label in CLASSES:
        print(f"\nGenerating {EXAMPLES_PER_CLASS} examples for '{label}'...")
        examples = []
        attempts = 0
        while len(examples) < EXAMPLES_PER_CLASS and attempts < 20:
            n_needed = min(BATCH_SIZE, EXAMPLES_PER_CLASS - len(examples))
            batch = generate_batch(label, n_needed)
            examples.extend(batch)
            print(f"  {len(examples)}/{EXAMPLES_PER_CLASS}")
            attempts += 1
            time.sleep(0.5)  # gentle on rate limit
        examples = examples[:EXAMPLES_PER_CLASS]
        for text in examples:
            all_examples.append({"text": text, "label": label})

    # Save raw dataset
    out_path = DATA_DIR / "synthetic_raw.json"
    out_path.write_text(json.dumps(all_examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Saved {len(all_examples)} examples to {out_path}")
    print(f"\nClass distribution:")
    for label in CLASSES:
        count = sum(1 for x in all_examples if x["label"] == label)
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()

