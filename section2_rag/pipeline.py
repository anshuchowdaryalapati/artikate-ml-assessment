"""
Main RAG pipeline with source-grounded generation and refusal-on-low-confidence.

Spec interface (per Artikate brief):
    result = pipeline.query(question="...")
    result = {
        "answer": str,
        "sources": [{"document": str, "page": int, "chunk": str}, ...],
        "confidence": float,  # 0-1
    }
"""
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from groq import Groq

from loader import load_all_pdfs
from chunker import chunk_pages
from retriever import HybridRetriever

load_dotenv()

# Silence chromadb telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "False"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")

DATA_DIR = "section2_rag/data"
CHROMA_DIR = "section2_rag/chroma_db"

# Refusal threshold. Cross-encoder scores below this => insufficient evidence.
# Calibrated from test runs: strong relevance scores > 0, weak < -3.
CONFIDENCE_THRESHOLD = -5.0


SYSTEM_PROMPT = """You are a precise document-grounded assistant for a legal/technical document corpus.

HARD RULES:
1. Answer ONLY using the provided context chunks. Never use outside knowledge.
2. Every factual claim must be supported by at least one chunk. Quote or paraphrase directly.
3. If the context does not contain enough information, reply exactly:
   "I cannot answer this question from the provided documents."
4. Cite sources inline using [doc_name, page X] format after each claim.
5. Do not speculate, infer beyond text, or fill gaps.
6. Keep answers concise (2-5 sentences unless the question requires more).
"""


class RAGPipeline:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not found in .env")
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.retriever = HybridRetriever()
        self._ready = False

    def setup(self, rebuild: bool = False):
        """Build or load the index. Call once before querying."""
        chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
        if chroma_exists and not rebuild:
            print("Loading existing index...")
            try:
                self.retriever.load_index()
                self._ready = True
                print(f"Loaded {len(self.retriever.chunks)} chunks from disk.")
                return
            except Exception as e:
                print(f"Load failed ({e}), rebuilding...")

        print("Building index from PDFs...")
        pages = load_all_pdfs(DATA_DIR)
        chunks = chunk_pages(pages)
        self.retriever.build_index(chunks)
        self._ready = True

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Main interface matching Artikate spec exactly.
        Returns: {answer, sources: [{document, page, chunk}], confidence}
        """
        if not self._ready:
            raise RuntimeError("Call setup() before query()")

        # Step 1: Retrieve
        hits = self.retriever.retrieve(question, top_k=top_k)

        # Step 2: Confidence = top hit's rerank score, mapped to [0, 1]
        top_score = hits[0]["score"] if hits else -10.0
        # Map cross-encoder scores to a confidence estimate.
        # Score > 0   => confident (high)
        # Score -3..0 => moderate
        # Score < -3  => refuse
        if top_score >= 0:
            confidence = min(1.0, 0.7 + top_score * 0.1)
        elif top_score >= CONFIDENCE_THRESHOLD:
            confidence = 0.3 + (top_score + 5.0) * 0.13  # maps [-3, 0] -> [0.3, 0.7]
        else:
            confidence = max(0.0, 0.3 + (top_score + 5.0) * 0.05)  # low

        # Step 3: Refuse early on low confidence
        if top_score < CONFIDENCE_THRESHOLD:
            return {
                "answer": "I cannot answer this question from the provided documents.",
                "sources": [
                    {
                        "document": h["document"],
                        "page": h["page"],
                        "chunk": h["text"][:300] + "...",
                    }
                    for h in hits
                ],
                "confidence": round(confidence, 3),
            }

        # Step 4: Build prompt with numbered citations
        context_blocks = []
        for i, h in enumerate(hits, 1):
            context_blocks.append(
                f"[CHUNK {i} | doc={h['document']} | page={h['page']}]\n{h['text']}"
            )
        context = "\n\n".join(context_blocks)
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer using ONLY the context above. Cite as [doc, page X]."

        # Step 5: Generate
        response = self.llm.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()

        # Step 6: Assemble spec-compliant response
        return {
            "answer": answer,
            "sources": [
                {
                    "document": h["document"],
                    "page": h["page"],
                    "chunk": h["text"][:500],
                }
                for h in hits
            ],
            "confidence": round(confidence, 3),
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.setup()

    test_questions = [
        "What rights do data subjects have under GDPR?",
        "What is VisionLLM v2 and what tasks can it handle?",
        "What is the capital of France?",  # Should refuse - not in corpus
    ]

for q in test_questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print('='*70)
        result = pipeline.query(q)
        print(f"ANSWER: {result['answer']}")
        print(f"CONFIDENCE: {result['confidence']}")
        print(f"SOURCES:")
        for s in result["sources"]:
            print(f"  - {s['document']} p.{s['page']}")


