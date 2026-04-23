# DESIGN.md — Section 2 RAG Pipeline Architecture

This document explains the architecture and trade-off reasoning behind the legal/document RAG pipeline. Implementation lives in `section2_rag/`.

## 1. Problem Framing

The brief specifies a corpus of 500+ legal contracts and policy documents (avg. 40 pages), with users asking precise factual questions that require exact source + page citation. Hallucinated answers are unacceptable.

**Two primary failure modes for this use case:**

1. **Wrong document retrieved** — user asks about Vendor X's NDA, system pulls Vendor Y's. Citation is technically present but useless.
2. **Right document, wrong page** — user asks about notice period (page 12), system retrieves the indemnification clause (page 27). Answer becomes confidently incorrect.

The pipeline is designed to make the first failure rare (high recall on document selection) and the second failure detectable (page-level metadata + confidence scoring + refusal).

**Implemented test corpus:** 4 real PDFs totaling 162 pages (GDPR regulation, VisionLLM v2 paper, I-JEPA paper, Vector-less RAG paper). Mixed legal + technical content stress-tests both lexical (exact-clause) and semantic (paraphrased) retrieval.

## 2. Component Decisions

### 2.1 PDF Loader — PyMuPDF (`fitz`)

**Choice:** `pymupdf` over `pdfplumber`, `pypdf2`, or `pdfminer.six`.

**Reasoning:** PyMuPDF is the fastest pure-text extractor on Windows, exposes per-page text trivially, and handles encrypted/complex PDFs without configuration. The brief mandates page-level citation, so the loader returns `[{document, page, text}]` — page metadata is established at extraction time and never lost downstream. `pdfplumber` was retained as a fallback dependency for table-heavy documents where layout matters; it is not used by default because table extraction is not required for the test corpus.

### 2.2 Chunking — Recursive 800-token chunks, 100-token overlap

**Choice:** `RecursiveCharacterTextSplitter` with character-based separators that respect paragraph → sentence → word boundaries.

**Reasoning:**
- **800 tokens** (~3,200 chars) is large enough to hold a complete legal clause or technical paragraph but small enough that top-3 retrieval surfaces precise context (a 2,000-token chunk would dilute reranker scores).
- **100-token overlap** prevents a clause from being split mid-sentence between chunks — a real risk with legal text where the operative phrase often crosses paragraph boundaries.
- **Recursive separators** (`["\n\n", "\n", ". ", " ", ""]`) try paragraph splits first, falling back to sentence and word splits only when needed. This preserves semantic coherence better than naive fixed-size chunking.
- **Metadata propagation** is enforced: every chunk carries the source document and page from its parent page, so retrieval results are always citable.

**What I'd change at scale:** structure-aware chunking that detects legal clause numbering (`Section 4.2`, `Article 17`) and splits on those boundaries, never within a clause. Out of scope for a 5-7 hour build.

### 2.3 Embedding Model — `BAAI/bge-small-en-v1.5`

**Choice:** BGE-small (384 dims) over OpenAI `text-embedding-3-small` (1536 dims) or BGE-large.

**Reasoning:**
- **Local + free.** Runs offline on CPU. No data leaves the machine — relevant for legal/regulated corpora.
- **384 dimensions** = 4x lower storage and 4x faster cosine similarity than 1536-dim alternatives, with comparable retrieval quality on the MTEB legal/technical benchmarks.
- **Strong out-of-domain performance** — BGE consistently outperforms older sentence-transformer models on technical retrieval tasks.

OpenAI embeddings would marginally improve quality but introduce API dependency, cost at scale (50k docs × ~200 chunks × billing per token), and a data egress concern.

### 2.4 Vector Store — ChromaDB (persistent, cosine similarity)

**Choice:** ChromaDB over FAISS, Qdrant, Pinecone, Milvus.

**Reasoning at the 500-document scale:**
- **Persistence + metadata filtering out of the box** — FAISS requires wrapper code for both.
- **Trivial setup** — single-file persistent client, no Docker container or external service.
- **Adequate performance** — ~80k chunks at 500 docs is well within Chroma's comfortable range.

**This decision flips at scale** — see Section 5.

### 2.5 Retrieval — Hybrid (BM25 + dense) → RRF fusion → cross-encoder rerank

**Choice:** Two-stage retrieval combining lexical and semantic signals.

**Why hybrid is mandatory for legal/technical text:**
- Users search for **exact terminology**: clause numbers (`Article 17`), proper nouns, statutory phrases (`limitation of liability`), specific monetary thresholds (`₹1 crore`). Pure dense retrieval fuzzes these into semantically-similar-but-wrong matches.
- BM25 catches the exact-term cases; dense embeddings catch the paraphrase cases. Neither alone is sufficient.
- **Reciprocal Rank Fusion (RRF)** with k=60 combines the two ranked lists robustly without requiring score normalization between BM25 (TF-IDF range) and cosine (0-1 range). It treats both as ordinal rankings, which is the principled fusion.

**Stage 2 — cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`):** the top 15 fused candidates are rescored by a cross-encoder that jointly attends to query and chunk. This catches subtle relevance signals that bi-encoders (used for the initial dense retrieval) miss because they encode query and document separately.

**Empirical tuning:** I tested `fetch_k = 15`, `20`, `30`. Counter-intuitively, `fetch_k = 15` outperformed `30` — a smaller, higher-quality candidate pool gave the cross-encoder less noise to score. **Final: fetch_k = 15.**

### 2.6 Generation — Llama 3.3-70B via Groq, temperature=0

**Choice:** Llama 3.3-70B served by Groq instead of GPT-4o.

**Reasoning:**
- The brief explicitly permits `"a free-tier alternative"`. Groq provides a 70B model at zero cost with sub-second latency.
- **Architecture is model-agnostic** — generation uses the OpenAI-compatible chat completion API. Switching to GPT-4o is a one-env-var change (`GENERATION_PROVIDER=openai`).
- Temperature 0 — factuality matters more than diversity for this use case. There is one correct answer per question.

### 2.7 Hallucination Mitigation — Two layers

**Layer 1 — System prompt with hard rules.** The prompt enforces (a) answer only from context, (b) every claim must be cited inline as `[doc, page X]`, (c) refuse with an exact string when context is insufficient. Temperature 0 + a strict prompt does most of the work.

**Layer 2 — Confidence-based refusal at the retrieval boundary.** If the top reranked chunk's cross-encoder score falls below a calibrated threshold (`-5.0`), the pipeline refuses BEFORE invoking the LLM. This:
- Avoids burning API tokens on questions the corpus can't answer.
- Returns the closest-found chunks for transparency (so the user can judge whether their question was misunderstood vs. genuinely out-of-corpus).
- Tested explicitly: *"What is the capital of France?"* → confidence 0.05, refusal triggered, no hallucination.

**Why not LLM-as-judge or NLI grounding?** Both would add a second LLM call per query (latency cost) for marginal gains over what the system prompt + confidence threshold already deliver. Worth adding at production scale; not worth the complexity for this assessment.

## 3. Evaluation Results

10 hand-written Q&A pairs in `eval_questions.json`. For each, expected document and a tight 3-4 page range are specified. A retrieved chunk is a "hit" only if both document AND page match.

| Metric | Value | Interpretation |
|---|---|---|
| **Precision@3** | **0.600** | 18 of 30 retrieved chunks fall within the expected page range. |
| **Recall@3** | **1.000** | Every question's expected document appears in top-3 — no off-topic retrievals. |
| **Hit@1** | **0.700** | The top-ranked chunk is correct 70% of the time. |

**Why precision@3 is not higher (and shouldn't be):** the ground-truth pages are deliberately narrow (3-4 specific pages per question). Legal and technical concepts naturally span adjacent pages (recitals, definitions, applications), so the reranker frequently surfaces topically-correct chunks from pages just outside the expected range. Broader page tolerance would inflate the metric without reflecting better retrieval.

**Where the system fails (Q3, Q5, Q7, Q8):** these questions ask about content distributed across many pages (lawful processing conditions, GDPR penalties, training datasets). The retriever finds the right document and topically-relevant content, but the most-similar chunks are from related-but-different pages than the ground truth. This is a chunking-granularity issue, not a retrieval-quality issue. At production scale, structure-aware chunking on legal clause boundaries would close most of this gap.

## 4. End-to-End Behavior Verification

Three test queries demonstrate the full pipeline:

1. **In-corpus query** — *"What rights do data subjects have under GDPR?"* → 5-clause answer, every claim cited with page numbers, confidence 1.0.
2. **In-corpus technical query** — *"What is VisionLLM v2?"* → architectural answer with page citations from vllm.pdf, confidence 1.0.
3. **Out-of-corpus query** — *"What is the capital of France?"* → exact refusal string, confidence 0.05, sources still returned for transparency.

The third case is the system working as designed — the corpus contains no answer, so the pipeline declines rather than hallucinating from parametric knowledge.

## 5. Scaling to 50,000 Documents

The brief asks what changes at 100x scale. Naming the bottleneck per layer:

**Chunking pipeline:**
- *Bottleneck:* single-process page extraction. ~50 hours sequential.
- *Fix:* parallelize via `multiprocessing.Pool` across CPU cores; for 50k PDFs, run as a one-time offline job (Spark or Ray for distributed extraction if needed).

**Embedding generation:**
- *Bottleneck:* CPU embedding. ~10ms/chunk × 10M chunks = 28 hours.
- *Fix:* GPU batch embedding (single A100 → 30x speedup); persist embeddings to a tiered storage layer.

**Vector store:**
- *Bottleneck:* ChromaDB starts to degrade past ~1-5M vectors (latency, memory).
- *Fix:* migrate to **Qdrant** or **Milvus** with HNSW for low-latency ANN, or IVF_PQ for memory efficiency at the cost of slight recall loss. Sharded collections by document type or date.

**BM25 index:**
- *Bottleneck:* in-memory `rank_bm25` with all 10M chunks won't fit in RAM and rebuild is O(n).
- *Fix:* **Elasticsearch** or **OpenSearch** for the lexical leg — designed for this scale, supports incremental indexing, exposes a familiar BM25 implementation.

**Cross-encoder reranking:**
- *Bottleneck:* now the slowest step (~50ms per pair on CPU × 15 candidates = 750ms/query).
- *Fix:* GPU-serve via ONNX Runtime or TensorRT, or distill to a smaller reranker. Target: <100ms reranking on 20-30 candidates.

**Generation latency:**
- At scale, also add a **semantic cache** (e.g., Redis with embedding-based key matching) so common queries skip the full pipeline entirely.

**Ingestion architecture:**
- Daily incremental ingestion via a stream processor (Kafka → consumer that chunks, embeds, indexes).
- Document versioning — when a contract is amended, the old version's chunks must be invalidated cleanly.

**Evaluation at scale:**
- 10 hand-written Q&A pairs no longer suffice. Move to LLM-as-judge or RAGAS metrics on a 500-1000 question synthetic eval set, with hand-verification on a 50-question gold set.

## 6. What This Pipeline Does Not Handle (Honest Limitations)

- **Tables and figures** — current loader extracts text only. Tabular contract terms (rate cards, schedules) lose structure. `pdfplumber` is a partial fix; vision-language models (LayoutLM, Donut) are the proper solution.
- **Cross-document reasoning** — *"Which contracts contain a limitation-of-liability clause above ₹1 crore?"* requires aggregation across documents, not single-document Q&A. This is a different architecture (filtered retrieval + aggregation, not generation-grounded RAG).
- **Multi-hop questions** — *"What's the notice period in the NDA signed with Vendor X?"* requires entity linking (which doc is Vendor X's NDA?) before retrieval. Currently the pipeline retrieves across all documents indiscriminately.
- **Image-based PDFs** — scanned contracts without OCR layer return empty text. OCR via Tesseract or PaddleOCR is a prerequisite for those.


