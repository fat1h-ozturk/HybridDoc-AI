"""Split page text into deterministic, page-preserving character chunks."""

from typing import Any, Dict, List


def chunk_pages(
    pages: List[Dict[str, Any]], max_chars: int = 1000, overlap: int = 100
) -> List[Dict[str, Any]]:
    """Split pages by Python characters while keeping each page's metadata."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_chars:
        raise ValueError("overlap must be an integer from 0 up to max_chars - 1")

    chunks = []
    for page in pages:
        text = page["text"]
        if not isinstance(text, str):
            raise TypeError("page text must be a string")
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append({**page, "text": text[start:end], "char_count": end - start})
            if end == len(text):
                break
            start = end - overlap

    return chunks



def chunk_formula_pages(pages, max_chars=1000, overlap=100):
    """Separate optional API: formula and warning blocks are atomic.

    Text blocks use the old character/overlap behavior. Atomic blocks may exceed
    max_chars; overlap never repeats part of a formula. Each chunk retains page
    provenance and only the formula records actually present in that chunk.
    """
    # Reuse the old API's argument validation even for an empty document.
    chunk_pages([], max_chars, overlap)
    chunks = []
    for page in pages:
        formulas = {f['id']: f for f in page['formulas']}
        for block in page['blocks']:
            metadata = {**page, 'blocks': [block], 'formulas': []}
            if block['kind'] == 'text':
                chunks.extend(chunk_pages([{**metadata, 'text': block['text']}],
                                          max_chars, overlap))
            else:
                if block['kind'] == 'formula':
                    metadata['formulas'] = [formulas[block['formula_id']]]
                chunks.append({**metadata, 'text': block['text'],
                               'char_count': len(block['text'])})
    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.ingestion.pdf_loader import load_pdf

    pdf_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf"
    )
    chunks = chunk_pages(load_pdf(pdf_path))
    print(f"Created {len(chunks)} chunks from {pdf_path.name} (maximum 1000 characters, 100-character overlap).")
    if chunks:
        example = chunks[0]
        print(f"Example source: {example['file_name']}, page {example['page_number']}")
        print(example["text"][:200])
