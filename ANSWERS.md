# ANSWERS.md
# Artikate Studio — AI/ML/LLM Engineer Assessment
**Candidate:** Anusha Alapati
**Date:** April 2026

---

## Interpretations & Assumptions

Tooling: I used Claude (Anthropic) extensively as a pair-programming partner during this build — for boilerplate generation, debugging, and editing technical writing. All architectural decisions, debugging investigations, evaluation choices, and final wording are mine; the AI accelerated execution but did not replace judgment. The engineering journey log at the bottom of this document captures the specific decisions and failures encountered during the build.


- **Section 2 corpus scale:** The brief specifies 500+ legal PDFs as the production target. The same brief asks for "at least 3 sample PDF documents of your choice for testing." I treated 500 PDFs as the *architectural target* informing trade-off reasoning, and 4 sample PDFs as the *implemented test set*. The scaling answer in DESIGN.md addresses what changes at 50,000 docs.
- **Section 2 citation granularity:** The required return field is page-level (`page: int`). I implemented page-level citations as mandatory. Clause-level metadata could be added with structure-aware chunking; out of scope here.
- **Section 2 hallucination strategy:** "Hallucinated answers are unacceptable" is a directional rule, not a binary. I implemented two layers: (1) strict system prompt enforcing source-grounding, (2) confidence-threshold refusal at the retrieval boundary. A claim is treated as hallucinated if cross-encoder rerank score on the top chunk falls below the threshold.
- **Section 3 latency interpretation:** "Under 500ms per ticket" interpreted as p99 latency on a representative single-CPU server. I tested on a Windows laptop CPU (Intel multi-core) and report p99 = 20.5ms.
- **Section 3 data:** Generated 800 training + 200 test examples via Groq Llama-3.3-70B. Spot-verified 50 of 200 test examples by hand-reading; ~6% required relabeling (typically `complaint` ↔ `technical_issue` boundary cases). Documented in `section3_classifier/generation_prompt.md`.
- **Section 4:** Answered Question A (Prompt Injection) and Question C (On-Premise Deployment), as the brief permits choosing any 2 of 3.
- **Section 5 (Loom):** Recording planned for tomorrow; link will be added to README.md before submission.

---

## Section 1 — Diagnose a Failing LLM Pipeline

### Diagnosis Log 1 — Hallucinated Pricing

**Symptom:** Bot confidently gives wrong answers about product pricing.

**What I investigated first:** Whether the chatbot has a retrieval layer at all for pricing data, and if so, whether the retrieved chunk for pricing queries actually contains correct prices.

**What I ruled out:**
- *Temperature issue:* Ruled out. Temperature controls token-distribution diversity, not factuality. A temperature-0 model can still hallucinate confidently if the underlying knowledge is wrong. The "confidently wrong" symptom is consistent with parametric memory, not sampling variance.
- *Knowledge cutoff alone:* Ruled out as the *primary* cause. GPT-4o's training data does include some product pricing pages from the public web, but the model has no way of knowing whether those prices are current. The cutoff explains *why* the data is wrong; it doesn't explain *why the system relies on parametric memory in the first place*.

**Root cause:** Pricing is dynamic structured data that should never come from model weights. The most likely architecture failure is one of:
1. **No retrieval path for pricing** — the bot answers from parametric memory because pricing was never wired to a tool/RAG layer.
2. **Stale pricing index** — RAG exists but the pricing source (CMS, internal pricing DB) was indexed once at launch and never refreshed when prices changed.
3. **Retrieval fires but is ignored** — the system prompt may have implicit phrases like "use your knowledge to answer" that override retrieved context.

**How I would distinguish:** Check three logs in order: (a) does any pricing query trigger the retrieval call? (no → architectural gap, fix #1); (b) if yes, what was retrieved? (correct pricing → it's a prompt adherence problem, fix #3; stale pricing → it's an index freshness problem, fix #2).

**Fix:** Move pricing out of RAG/parametric memory entirely. Add a dedicated `get_price(product_id)` tool call to a live pricing database. Update the system prompt: *"Never state a price unless the get_price tool was called this turn. If the tool is unavailable, say so."* This forces every pricing answer to be grounded in fresh, structured data.

---

### Diagnosis Log 2 — Language Switching to English

**Symptom:** Bot occasionally responds in English even when the user writes in Hindi or Arabic.

**Mechanism causing this:** **System prompt language anchoring.** The system prompt is almost certainly written in English, and it is long, authoritative, and appears first in the context. The model's response language is heavily influenced by the dominant language across the entire input (system + user), not just the user turn alone. Two compounding factors:

1. The English system prompt is typically 200-500 tokens, while a Hindi/Arabic user message of equivalent semantic content is often 50-150 tokens (multilingual scripts tokenize less efficiently with GPT-4o's BPE tokenizer). English dominates token space.
2. Most of the model's training data on "instruction-following" demonstrates English-instruction → English-response patterns. Without an explicit override, the model defaults to its statistical prior.

**The specific prompt change to fix it (language-agnostic, testable):**


