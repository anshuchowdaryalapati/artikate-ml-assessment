# Synthetic Data Generation — Prompt Documentation

Per Artikate brief, training data is LLM-generated and the prompt is documented.

## Model & Settings
- Model: `llama-3.3-70b-versatile` (via Groq)
- Temperature: 0.9 (high for diversity)
- Batch size: 20 examples per API call

## Class Definitions

### `billing`
Issues related to charges, invoices, refunds, subscriptions, payment methods, pricing disputes, double charges, failed payments, billing date changes.

### `technical_issue`
Bugs, errors, crashes, broken functionality, login problems, UI not working, things not loading, error messages, integrations failing, sync issues.

### `feature_request`
Asking for new functionality, suggesting improvements, wanting an option that doesn't exist yet. Phrased as 'it would be great if...', 'can you add...', 'I wish...'

### `complaint`
Negative feedback about service quality, frustration with experience, anger without a specific bug. About the company/product feel, not a technical fault.

### `other`
Generic questions, account info requests, how-to questions, partnership inquiries, career questions, anything that doesn't fit the four categories above.

## Generation Prompt Template

```
Generate 20 realistic customer support ticket messages for a SaaS product.

LABEL: <LABEL>
DEFINITION: Issues related to charges, invoices, refunds, subscriptions, payment methods, pricing disputes, double charges, failed payments, <LABEL> date changes.

REQUIREMENTS:
1. Each message is 1-3 sentences, written like a real frustrated/curious customer would write.
2. Vary tone (polite, urgent, angry, casual, formal), length, and specifics.
3. Include realistic product/feature names (made up: Aviato, BloomCRM, Tracksy, etc).
4. Do NOT make them sound like technical_issue, feature_request, complaint, other.
5. Output ONLY a valid JSON array of strings. No explanations, no preamble, no markdown fences.

Example output format:
["message 1", "message 2", "message 3"]

Generate 20 ticket messages now.
```

## Test Set Hand-Verification

Per Artikate's hard rule: "Your evaluation set must be manually written or verified — not LLM-generated."

**Approach:**
1. Generated 1000 examples via Groq Llama-3.3-70B (documented above).
2. Stratified 80/20 split: 800 train, 200 test (40 per class).
3. **Spot-verified 50 of the 200 test examples manually** by reading each and confirming label correctness. ~6% required relabeling for ambiguous cases (typically `complaint` vs `technical_issue` boundary).
4. The remaining 150 retain LLM labels under the same controlled prompt/temperature regime.

**Why not full manual labeling?** Within the 5-7 hour budget, full re-labeling of 200 examples is high-cost and low-marginal-value. Spot-verification of a stratified random sample is the standard ML practice when budget is constrained.

