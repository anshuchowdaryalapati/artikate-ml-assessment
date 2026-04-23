"""Legal/document-aware chunking with page metadata preservation."""
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Dict]:
    """
    Split page texts into overlapping chunks, preserving document + page metadata.
    
    Why 800 tokens (~3200 chars) with 100 overlap:
    - Large enough to hold a full legal clause or paragraph
    - Small enough for precise retrieval (top-k stays focused)
    - Overlap prevents losing context at chunk boundaries
    - Recursive splitter respects paragraph/sentence structure first
    
    Returns: list of {document, page, chunk_id, text}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # rough char-to-token ratio
        chunk_overlap=chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    chunk_id = 0
    for page in pages:
        sub_chunks = splitter.split_text(page["text"])
        for sub in sub_chunks:
            if len(sub.strip()) < 50:  # drop tiny fragments
                continue
            chunks.append({
                "document": page["document"],
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": sub.strip(),
            })
            chunk_id += 1
    return chunks


if __name__ == "__main__":
    from loader import load_all_pdfs
    pages = load_all_pdfs("section2_rag/data")
    chunks = chunk_pages(pages)
    print(f"\nCreated {len(chunks)} chunks from {len(pages)} pages")
    print(f"\nSample chunk 0 (doc={chunks[0]['document']}, page={chunks[0]['page']}):")
    print(chunks[0]['text'][:300])
    print(f"\nAvg chunk length: {sum(len(c['text']) for c in chunks) // len(chunks)} chars")


