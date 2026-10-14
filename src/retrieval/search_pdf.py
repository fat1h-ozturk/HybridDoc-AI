"""Offline keyword source-search capability for PDFs with deterministic BM25 ranking."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional
import unicodedata

import pymupdf

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

from src.ingestion.pdf_loader import load_pdf_formulas
from src.chunking.text_chunker import chunk_formula_pages
from src.ingestion.formula_signs import sign_warning_text


def normalize_text(text: str) -> str:
    """Normalize Unicode and fold Turkish and Latin case sensibly.

    This function normalizes text for tokenization and term matching only.
    It never alters the original source text in the chunks or results.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Turkish casing adjustments:
    # 'İ' (U+0130) -> 'i', 'ı' (U+0131) -> 'i', 'I' (U+0049) -> 'i'
    # This allows both dotted and dotless I in Turkish and English to match sensibly.
    text = text.replace("\u0130", "i").replace("\u0131", "i").replace("I", "i")
    text = text.lower()
    # Remove combining dot above (\u0307) that Python's lower() produces for \u0130
    text = text.replace("\u0307", "")
    return text


def tokenize(text: str) -> List[str]:
    """Extract alphanumeric tokens from normalized text."""
    normalized = normalize_text(text)
    return re.findall(r"\w+", normalized)


def extract_searchable_text(chunk: Dict[str, Any]) -> str:
    """Extract readable chunk prose and formula components without synthetic boilerplate.

    Warning strings and synthetic formula labels are omitted to prevent ranking
    on boilerplate terms like 'unresolved', 'formula', 'evidence', or ID strings.
    """
    blocks = chunk.get("blocks", [])
    if not blocks:
        return chunk.get("text", "")

    parts: List[str] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "text":
            text = block.get("text", "")
            if text:
                parts.append(text)
        elif kind == "formula":
            for f in chunk.get("formulas", []):
                orig = f.get("original_text")
                if orig:
                    parts.append(orig)
                latex = f.get("latex")
                if latex:
                    parts.append(latex)
        elif kind == "formula_group":
            for grp in chunk.get("formula_groups", []):
                orig = grp.get("original_text")
                if orig:
                    parts.append(orig)
                latex = grp.get("latex")
                if latex:
                    parts.append(latex)
            for f in chunk.get("formulas", []):
                orig = f.get("original_text")
                if orig and orig not in parts:
                    parts.append(orig)
                latex = f.get("latex")
                if latex and latex not in parts:
                    parts.append(latex)
        elif kind == "warning":
            # Synthetic warning block has no document prose; omit from search index
            pass
    return " ".join(parts)


class BM25Searcher:
    """Deterministic Okapi BM25 ranking implemented with Python standard library."""

    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.doc_freqs: Counter = Counter()

        for chunk in chunks:
            tokens = tokenize(extract_searchable_text(chunk))
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            for term in set(tokens):
                self.doc_freqs[term] += 1

        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size > 0 else 0.0

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Score chunks against query and return positive-score matches up to top_k.

        Ties are broken deterministically by within-document source_order.
        """
        if self.corpus_size == 0 or self.avgdl == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_term_counts = Counter(query_tokens)
        candidates = []

        for idx, chunk in enumerate(self.chunks):
            tokens = self.doc_tokens[idx]
            doc_len = self.doc_lens[idx]
            if doc_len == 0:
                continue

            tf = Counter(tokens)
            score = 0.0

            for q, q_count in query_term_counts.items():
                if tf[q] > 0:
                    df = self.doc_freqs[q]
                    # Okapi BM25+ / Robertson formula ensuring positive IDF
                    idf = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
                    num = tf[q] * (self.k1 + 1.0)
                    denom = tf[q] + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    term_score = idf * (num / denom)
                    score += q_count * term_score

            if score > 0.0:
                source_order = chunk.get("source_order", idx + 1)
                candidates.append((score, source_order, chunk))

        # Stable tie ordering: higher score first, lower source_order (earlier in doc) next
        candidates.sort(key=lambda item: (-item[0], item[1]))

        results = []
        for rank, (score, source_order, chunk) in enumerate(candidates[:top_k], start=1):
            res = {
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": chunk.get("chunk_id", f"chunk-{source_order}"),
                "source_order": source_order,
                "file_name": chunk.get("file_name", ""),
                "page_number": chunk.get("page_number", 1),
                "total_pages": chunk.get("total_pages", 1),
                "source_sha256": chunk.get("source_sha256", ""),
                "excerpt": chunk.get("text", ""),
                "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
                "detection_status": chunk.get("detection_status", "native_heuristic_incomplete"),
                "answer_eligible": False,  # Always honor existing flag; search never certifies answers
                "math_status": chunk.get("math_status", "no_native_candidates"),
                "review_reason": chunk.get(
                    "review_reason",
                    "native detection is incomplete; source review required for math answers",
                ),
                "unresolved_reason": chunk.get("unresolved_reason"),
                "formulas": chunk.get("formulas", []),
                "formula_groups": chunk.get("formula_groups", []),
            }
            results.append(res)

        return results


def search_pdf(
    file_path: str | Path,
    query: str,
    top_k: int = 3,
    max_chars: int = 1000,
    overlap: int = 100,
    boundary: str = "character",
) -> List[Dict[str, Any]]:
    """Load PDF natively, chunk pages, and rank chunks deterministically by BM25.

    Args:
        file_path: Path to the target PDF document.
        query: Lexical search query string.
        top_k: Positive integer specifying maximum matching chunks to return.
        max_chars: Maximum character budget per chunk for prose text.
        overlap: Overlap characters between consecutive prose chunks.
        boundary: Chunk boundary mode: 'character' (default) or 'word'.

    Returns:
        List of matching chunk result dicts with positive relevance scores.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    path = Path(file_path)
    # Native-only extraction (offline, no model weights or network needed)
    with redirect_stdout(sys.stderr):
        pages = load_pdf_formulas(path)

    raw_chunks = chunk_formula_pages(
        pages, max_chars=max_chars, overlap=overlap, boundary=boundary
    )

    chunks = []
    for i, c in enumerate(raw_chunks, start=1):
        annotated = dict(c)
        annotated["source_order"] = i
        annotated["chunk_id"] = f"chunk-{i}"
        chunks.append(annotated)

    searcher = BM25Searcher(chunks)
    return searcher.search(query, top_k=top_k)


def format_search_results(
    results: List[Dict[str, Any]],
    query: str,
    file_path: str | Path | None = None,
) -> str:
    """Render human-readable UTF-8 search results or explicit no-evidence message."""
    doc_name = (
        Path(file_path).name
        if file_path
        else (results[0]["file_name"] if results else "document")
    )

    lines: List[str] = []
    if not results:
        lines.append(f'No matching source found in {doc_name} for query: "{query}"')
        lines.append(
            "Note: This lexical keyword search found no positive matching source chunks "
            "in the extracted native text. This is NOT an assertion that the topic or fact "
            "is absent from the PDF (e.g. content may appear in unextracted raster/scanned images, "
            "use different wording, or require formula inspection)."
        )
        return "\n".join(lines) + "\n"

    lines.append(f'Search Results for query: "{query}"')
    lines.append(f"Document: {doc_name} (Total matches: {len(results)})")
    lines.append(
        "Note: Scores represent lexical relevance rankings (BM25), not confidence, accuracy, or probabilities."
    )
    lines.append("")

    for res in results:
        lines.append(f"--- Result {res['rank']} (Score: {res['score']:.4f}, Rank: {res['rank']}) ---")
        lines.append(f"Source: {res['file_name']}, Page {res['page_number']} of {res['total_pages']}")
        lines.append(f"Chunk ID: {res['chunk_id']} | Source Order: {res['source_order']}")
        lines.append(f"Source SHA-256: {res['source_sha256']}")
        lines.append(
            f"Detection Status: {res['detection_status']} | "
            f"Answer Eligible: {res['answer_eligible']}"
        )
        if res.get("review_reason"):
            lines.append(f"Review Reason: {res['review_reason']}")
        for formula in res.get('formulas', []):
            check = formula.get('sign_consistency')
            if check:
                lines.append(f"Minus check {formula['id']}: {check['status']} ({check['reason']}); not formula verification")
            warning = sign_warning_text(formula)
            if warning:
                lines.append(warning.lstrip('\n'))

        # Formula group details if present
        if res.get("formula_groups"):
            for grp in res["formula_groups"]:
                lines.append(
                    f"[Formula Group: {grp.get('id')} | Kind: {grp.get('kind')} | "
                    f"State: {grp.get('state')} | Answer Eligible: {grp.get('answer_eligible', False)}]"
                )
                lines.append(f"  Child formula IDs: {grp.get('formula_ids')}")
                for child in grp.get("formulas", []):
                    lines.append(
                        f"    - Child {child.get('id')}: original_text={child.get('original_text')!r}, "
                        f"LaTeX={child.get('latex')!r}, bbox={child.get('bbox')}, state={child.get('state')}, "
                        f"reason={child.get('reason')}"
                    )
                lines.append(
                    "  [WARNING: Indivisible heuristic wrapped relation; source-inspection context only; not answer evidence]"
                )
        elif res.get("formulas"):
            for f in res["formulas"]:
                lines.append(
                    f"[Formula: {f.get('id')} | State: {f.get('state')} | Answer Eligible: {f.get('answer_eligible', False)}]"
                )
                lines.append(
                    f"  original_text={f.get('original_text')!r}, LaTeX={f.get('latex')!r}, bbox={f.get('bbox')}, "
                    f"reason={f.get('reason')}"
                )
                lines.append(
                    "  [WARNING: Unverified formula candidate; source-inspection context only; not answer evidence]"
                )

        lines.append("Excerpt:")
        lines.append(res["excerpt"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    """CLI entry point for offline PDF source search."""
    parser = argparse.ArgumentParser(
        description="Offline keyword source-search across PDF text and formulas with BM25 ranking."
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF document")
    parser.add_argument("query", type=str, help="Search query phrase")
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=3,
        help="Maximum number of positive matching chunks to return (default: 3)",
    )
    parser.add_argument(
        "--boundary",
        choices=["character", "word"],
        default="character",
        help="Chunk boundary mode: 'character' (default) or 'word' to align prose boundaries to whitespace words.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional output file to save results (UTF-8 encoded)",
    )

    args = parser.parse_args(argv)
    try:
        results = search_pdf(
            args.pdf,
            args.query,
            top_k=args.top_k,
            boundary=args.boundary,
        )
        formatted = format_search_results(results, args.query, file_path=args.pdf)
        print(formatted, end="")
        if args.output:
            args.output.write_text(formatted, encoding="utf-8")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Search failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
