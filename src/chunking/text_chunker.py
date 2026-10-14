"""Split page text into deterministic, page-preserving character or word chunks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _split_text_characters(
    text: str, max_chars: int, overlap: int
) -> List[Tuple[int, int]]:
    """Split text strictly by character slices (legacy character mode)."""
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        spans.append((start, end))
        if end == n:
            break
        start = end - overlap
    return spans


def _split_text_word_boundaries(
    text: str, max_chars: int, overlap: int
) -> List[Tuple[int, int]]:
    """Split text along whitespace word boundaries within a max_chars budget.

    Avoids splitting ordinary whitespace-separated words at end and start/overlap
    boundaries where feasible, while retaining all original text and deterministic
    ordering.

    Budget and Overlap Guarantees:
        - max_chars is a hard upper bound character budget (Python len(), NOT a
          word or token count). No chunk will exceed max_chars characters.
        - If an unbroken string exceeds max_chars (e.g. very long uninterrupted
          symbols or alphanumeric sequences without whitespace), a documented
          bounded fallback splits at max_chars so text is never lost, the budget
          is strictly honored, and progress is guaranteed without infinite loops.
        - Exact overlap character counts (e.g. 100 chars) are NOT guaranteed in
          word mode because overlap boundaries align to word edges. The effective
          overlap will be aligned to the nearest word boundary.

    Args:
        text: Input page text to slice.
        max_chars: Maximum character budget per chunk.
        overlap: Target character overlap between consecutive chunks.

    Returns:
        List of (start, end) index tuples slicing `text`.
    """
    n = len(text)
    spans: List[Tuple[int, int]] = []
    start = 0

    while start < n:
        if n - start <= max_chars:
            end = n
        else:
            limit = start + max_chars
            # If limit lands on or directly after whitespace, words are naturally unbroken
            if text[limit].isspace() or text[limit - 1].isspace():
                end = limit
            else:
                # Find the last whitespace character within [start, limit)
                last_ws = -1
                for idx in range(limit - 1, start - 1, -1):
                    if text[idx].isspace():
                        last_ws = idx
                        break
                if last_ws == -1:
                    # Bounded fallback for oversized unbroken strings:
                    # cannot align to a word boundary without exceeding max_chars or dropping text,
                    # so slice at the hard character limit to guarantee budget compliance and progress.
                    end = limit
                else:
                    # Include the whitespace run in this chunk if it fits within limit,
                    # so the subsequent chunk begins cleanly at the start of the next word.
                    next_word_start = last_ws + 1
                    while next_word_start < limit and text[next_word_start].isspace():
                        next_word_start += 1
                    if next_word_start <= limit and text[next_word_start - 1].isspace():
                        end = next_word_start
                    else:
                        end = last_ws + 1

        spans.append((start, end))
        if end == n:
            break

        if overlap == 0:
            start = end
        else:
            # Calculate target overlap position
            target = max(start + 1, end - overlap)
            has_ws = any(text[i].isspace() for i in range(start, end))

            if target == 0 or (text[target - 1].isspace() and not text[target].isspace()):
                # Already at the start of a word
                next_start = target
            elif text[target].isspace():
                # Target lands in whitespace; advance to start of next word
                cand = target
                while cand < end and text[cand].isspace():
                    cand += 1
                if cand < end and cand > start:
                    next_start = cand
                else:
                    # If advancing reaches or exceeds end, retreat to start of word preceding whitespace
                    cand = target
                    while cand > start and text[cand - 1].isspace():
                        cand -= 1
                    while cand > start and not text[cand - 1].isspace():
                        cand -= 1
                    if cand > start:
                        next_start = cand
                    elif has_ws:
                        # Chunk had words/whitespace but no viable boundary in (start, end);
                        # advance to end to avoid cutting words at overlap boundary
                        next_start = end
                    else:
                        # Unbroken string fallback
                        next_start = target
            else:
                # Target lands in the middle of a word; try to align to start of that word
                word_start = target
                while word_start > start and not text[word_start - 1].isspace():
                    word_start -= 1
                if word_start > start:
                    next_start = word_start
                else:
                    # Word started at or before current start; retreating would make no progress.
                    # Look forward for the start of the next word.
                    word_end = target
                    while word_end < end and not text[word_end].isspace():
                        word_end += 1
                    while word_end < end and text[word_end].isspace():
                        word_end += 1
                    if word_end < end:
                        next_start = word_end
                    elif has_ws:
                        # Chunk had words/whitespace; advance to end to avoid cutting the word
                        next_start = end
                    else:
                        # Oversized unbroken string: character fallback to guarantee progress
                        next_start = target

            start = next_start

    return spans


def chunk_pages(
    pages: List[Dict[str, Any]],
    max_chars: int = 1000,
    overlap: int = 100,
    boundary: str = "character",
    chunk_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Split pages by Python characters or word boundaries while keeping each page's metadata.

    Args:
        pages: List of page dicts, each containing 'text' and provenance metadata.
        max_chars: Maximum character budget per chunk (Python len() count, NOT a
            word or token count). Must be a positive integer. Default: 1000.
        overlap: Overlap in characters between consecutive chunks. Note: in 'word'
            mode, exact character overlap is not guaranteed because boundaries
            align to whole words to avoid cutting words in the middle. Default: 100.
        boundary: Splitting boundary mode: 'character' (legacy default) or 'word'.
        chunk_size: Backward-compatible alias for max_chars.
    """
    if chunk_size is not None:
        max_chars = chunk_size

    if isinstance(boundary, bool) or boundary not in ("character", "word"):
        raise ValueError("boundary must be 'character' or 'word'")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < max_chars:
        raise ValueError("overlap must be an integer from 0 up to max_chars - 1")

    split_fn = _split_text_word_boundaries if boundary == "word" else _split_text_characters

    chunks = []
    for page in pages:
        text = page["text"]
        if not isinstance(text, str):
            raise TypeError("page text must be a string")
        if not text.strip():
            continue

        for start, end in split_fn(text, max_chars, overlap):
            chunks.append({**page, "text": text[start:end], "char_count": end - start})

    return chunks


def chunk_formula_pages(
    pages: List[Dict[str, Any]],
    max_chars: int = 1000,
    overlap: int = 100,
    boundary: str = "character",
    chunk_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Separate optional API: formula and warning blocks are atomic.

    Text blocks use the requested boundary behavior ('character' or 'word').
    Atomic blocks may exceed max_chars; overlap never repeats part of a formula.
    Each chunk retains page provenance and only the formula records actually
    present in that chunk.
    """
    if chunk_size is not None:
        max_chars = chunk_size

    # Reuse argument validation even for an empty document.
    chunk_pages([], max_chars, overlap, boundary=boundary)
    chunks = []
    for page in pages:
        formulas = {f['id']: f for f in page['formulas']}
        groups = {g['id']: g for g in page.get('formula_groups', [])}
        for block in page['blocks']:
            metadata = {**page, 'blocks': [block], 'formulas': []}
            if 'formula_groups' in metadata:
                metadata['formula_groups'] = []
            if block['kind'] == 'text':
                chunks.extend(
                    chunk_pages(
                        [{**metadata, 'text': block['text']}],
                        max_chars=max_chars,
                        overlap=overlap,
                        boundary=boundary,
                    )
                )
            else:
                if block['kind'] == 'formula':
                    metadata['formulas'] = [formulas[block['formula_id']]]
                elif block['kind'] == 'formula_group':
                    metadata['formulas'] = [formulas[fid] for fid in block['formula_ids']]
                    grp = block.get('group') or groups.get(block.get('group_id'))
                    if grp:
                        metadata['formula_groups'] = [grp]
                        metadata['formula_group'] = grp
                chunks.append({**metadata, 'text': block['text'],
                               'char_count': len(block['text'])})
    return chunks


def format_chunks_inspection(
    chunks: List[Dict[str, Any]],
    file_name: str,
    boundary: str,
    max_chars: int,
    overlap: int,
) -> str:
    """Format a readable, complete text export of all chunks for inspection."""
    lines: List[str] = [
        f"Document: {file_name}",
        f"Total Chunks: {len(chunks)}",
        f"Boundary Mode: {boundary}",
        f"Max Characters (Budget): {max_chars}",
        f"Target Overlap: {overlap}",
        "=" * 80,
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"=== Chunk {i} (Page {c.get('page_number')}, {c.get('char_count')} chars) ===")
        lines.append(f"Source: {c.get('file_name', file_name)}, Page {c.get('page_number')} of {c.get('total_pages', '?')}")
        lines.append("-" * 80)
        lines.append(c["text"])
        lines.append("-" * 80)
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for page-preserving PDF text chunking."""
    parser = argparse.ArgumentParser(
        description=(
            "Split PDF page text into deterministic chunks with configurable boundaries. "
            "Note: chunk size is a maximum character budget (Python len() Unicode characters), "
            "NOT a word or token count."
        )
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path("sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf"),
        help="Path to PDF document (default: sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf)",
    )
    parser.add_argument(
        "--boundary",
        choices=["character", "word"],
        default="character",
        help=(
            "Chunk boundary mode: 'character' (legacy default) or 'word' to avoid cutting "
            "ordinary whitespace-separated words at chunk edges and overlap boundaries."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        "--max-chars",
        dest="chunk_size",
        type=int,
        default=1000,
        help=(
            "Maximum character budget per chunk (Python len() Unicode characters, "
            "NOT a word/token count). Default: 1000."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help=(
            "Target overlap in characters between consecutive chunks. Note: in 'word' mode, "
            "exact overlap is not guaranteed because boundaries align to word edges. Default: 100."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output file path to write full readable chunk inspection (UTF-8 encoded).",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print all chunks with boundary details to stdout instead of only summary and first chunk.",
    )

    args = parser.parse_args(argv)

    if not args.pdf.exists():
        parser.exit(1, f"Error: PDF file not found: {args.pdf}\n")

    try:
        from src.ingestion.pdf_loader import load_pdf

        pages = load_pdf(args.pdf)
        chunks = chunk_pages(
            pages,
            max_chars=args.chunk_size,
            overlap=args.overlap,
            boundary=args.boundary,
        )

        formatted = format_chunks_inspection(
            chunks,
            file_name=args.pdf.name,
            boundary=args.boundary,
            max_chars=args.chunk_size,
            overlap=args.overlap,
        )

        if args.output:
            args.output.write_text(formatted, encoding="utf-8")

        if args.inspect:
            print(formatted, end="")
        else:
            if args.boundary == "character" and args.chunk_size == 1000 and args.overlap == 100:
                print(f"Created {len(chunks)} chunks from {args.pdf.name} (maximum 1000 characters, 100-character overlap).")
            else:
                print(
                    f"Created {len(chunks)} chunks from {args.pdf.name} "
                    f"(boundary: {args.boundary}, maximum {args.chunk_size} characters, "
                    f"{args.overlap}-character overlap target)."
                )
            if chunks:
                example = chunks[0]
                print(f"Example source: {example.get('file_name', args.pdf.name)}, page {example.get('page_number')}")
                print(example["text"][:200])

        return 0
    except (ValueError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Chunking failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
