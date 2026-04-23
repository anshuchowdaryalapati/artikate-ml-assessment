"""Hybrid retrieval: BM25 (lexical) + BGE embeddings (semantic) + RRF + reranking."""
from typing import List, Dict
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_DIR = "section2_rag/chroma_db"
BM25_PATH = "section2_rag/chroma_db/bm25.pkl"


class HybridRetriever:
    """
    Two-stage retrieval:
    1. Dense (BGE embeddings via Chroma) + Sparse (BM25) -> RRF fusion -> top 20
    2. Cross-encoder rerank -> top k

    Why hybrid: legal/technical docs contain exact terms (clause numbers, proper nouns,
    acronyms like "GDPR Art. 17") that dense embeddings fuzz out. BM25 catches these.
    Dense catches paraphrases BM25 misses. RRF combines both ranks robustly.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Loading reranker: {RERANK_MODEL}")
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks: List[Dict]):
        """Build Chroma + BM25 indexes from chunks."""
        self.chunks = chunks
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

        # Chroma: persistent dense index
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        # Drop + recreate to ensure clean state
        try:
            self.chroma_client.delete_collection("legal_docs")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name="legal_docs",
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Embedding {len(chunks)} chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        ).tolist()

        self.collection.add(
            ids=[str(c["chunk_id"]) for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"document": c["document"], "page": c["page"]} for c in chunks
            ],
        )

        # BM25: lexical index in memory, persisted via pickle
        print("Building BM25 index...")
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        with open(BM25_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        print(f"Index built: {len(chunks)} chunks ready")

    def load_index(self):
        """Load previously built indexes (fast startup)."""
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection("legal_docs")
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def retrieve(self, query: str, top_k: int = 3, fetch_k: int = 15) -> List[Dict]:
        """
        Hybrid retrieve with RRF fusion + cross-encoder rerank.
        Returns top_k chunks, each with a 'score' field (rerank score, higher = better).
        """
        # Dense retrieval (Chroma)
        query_emb = self.embedder.encode([query], convert_to_numpy=True).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_emb,
            n_results=fetch_k,
        )
        dense_ids = [int(i) for i in dense_results["ids"][0]]

        # Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ids = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

        # RRF fusion: score = sum(1 / (k + rank))
        rrf_k = 60
        fused = {}
        for rank, cid in enumerate(dense_ids):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)
        for rank, cid in enumerate(bm25_ids):
            fused[cid] = fused.get(cid, 0) + 1 / (rrf_k + rank)

        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:fetch_k]
        candidate_chunks = [self.chunks[cid] for cid, _ in fused_sorted]

        # Cross-encoder rerank
        pairs = [(query, c["text"]) for c in candidate_chunks]
        rerank_scores = self.reranker.predict(pairs)
        for c, s in zip(candidate_chunks, rerank_scores):
            c["score"] = float(s)
        reranked = sorted(candidate_chunks, key=lambda c: c["score"], reverse=True)
        return reranked[:top_k]


if __name__ == "__main__":
    from loader import load_all_pdfs
    from chunker import chunk_pages

    pages = load_all_pdfs("section2_rag/data")
    chunks = chunk_pages(pages)

    retriever = HybridRetriever()
    retriever.build_index(chunks)

    results = retriever.retrieve("What is personal data under GDPR?", top_k=3)
    print("\n--- TOP 3 RESULTS ---")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] doc={r['document']} page={r['page']} score={r['score']:.3f}")
        print(r["text"][:250] + "...")


