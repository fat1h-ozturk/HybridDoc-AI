import subprocess
import sys
from pathlib import Path
import pytest
import pymupdf

from src.retrieval.search_pdf import search_pdf, format_search_results


@pytest.fixture
def sample_calculus_pdf(tmp_path):
    """Synthetic PDF with multiple pages and distinct topics."""
    pdf_path = tmp_path / "calculus.pdf"
    with pymupdf.open() as doc:
        # Page 1: Introduction
        p1 = doc.new_page(width=500, height=400)
        p1.insert_text((50, 50), "Chapter 1: Limits and Continuity in real analysis.")

        # Page 2: Separable differential equations
        p2 = doc.new_page(width=500, height=400)
        p2.insert_text((50, 50), "Chapter 2: Separable Differential Equations.")
        p2.insert_text((50, 80), "A first-order separable differential equation can be written as dy/dx = g(x)h(y).")
        p2.insert_text((50, 110), "We solve separable differential equations by separating variables and integrating.")

        # Page 3: Linear algebra
        p3 = doc.new_page(width=500, height=400)
        p3.insert_text((50, 50), "Chapter 3: Matrices and Linear Algebra.")
        p3.insert_text((50, 80), "Vectors, eigenvalues, and matrix transformations in Euclidean space.")

        doc.save(pdf_path)
    return pdf_path


@pytest.fixture
def sample_competing_pdf(tmp_path):
    """Synthetic PDF to test BM25 relevance ranking between competing matches."""
    pdf_path = tmp_path / "competing.pdf"
    with pymupdf.open() as doc:
        # Page 1: Brief single mention of differential
        p1 = doc.new_page(width=500, height=400)
        p1.insert_text((50, 50), "Page 1 mentions a differential form briefly in passing.")

        # Page 2: Highly relevant section with multiple query term matches
        p2 = doc.new_page(width=500, height=400)
        p2.insert_text((50, 50), "Separable differential equations are fundamental differential systems.")
        p2.insert_text((50, 80), "This chapter focuses specifically on separable differential equations.")

        doc.save(pdf_path)
    return pdf_path


@pytest.fixture
def sample_turkish_pdf(tmp_path):
    """Synthetic PDF with portable Latin-supported Turkish terms."""
    pdf_path = tmp_path / "turkish_doc.pdf"
    with pymupdf.open() as doc:
        p1 = doc.new_page(width=500, height=400)
        p1.insert_text((50, 50), "Diferansiyel Denklemler ve Cozum")
        p1.insert_text((50, 80), "Ayrilabilir diferansiyel denklemler ve modeller.")
        doc.save(pdf_path)
    return pdf_path


@pytest.fixture
def sample_wrapped_relation_pdf(tmp_path):
    """Synthetic PDF with a wrapped relation at line end."""
    pdf_path = tmp_path / "wrapped.pdf"
    with pymupdf.open() as doc:
        p1 = doc.new_page(width=500, height=400)
        p1.insert_text((40, 60), "Integration constant:")
        p1.insert_text((420, 60), "K =")
        p1.insert_text((40, 76), "2")
        p1.insert_text((46, 73), "-", fontsize=6)
        p1.insert_text((51, 71), "2", fontsize=6)
        p1.insert_text((51, 76), "3", fontsize=6)
        p1.draw_line((51, 73), (56, 73))
        p1.insert_text((60, 76), ". Next step in the derivation.")
        doc.save(pdf_path)
    return pdf_path


def test_relevant_phrase_selects_expected_page(sample_calculus_pdf):
    """Query 'separable differential equations' selects expected Page 2."""
    results = search_pdf(sample_calculus_pdf, "separable differential equations", top_k=3)
    assert len(results) > 0
    top_result = results[0]
    assert top_result["page_number"] == 2
    assert "separable" in top_result["excerpt"].lower()
    assert top_result["score"] > 0
    assert top_result["file_name"] == sample_calculus_pdf.name
    assert top_result["total_pages"] == 3
    assert not top_result["answer_eligible"]


def test_competing_matches_rank_sensibly(sample_competing_pdf):
    """Chunks with multiple matched query terms rank higher than single-term mentions."""
    results = search_pdf(sample_competing_pdf, "separable differential equations", top_k=5)
    assert len(results) >= 2
    # Page 2 has 'separable differential equations' multiple times; Page 1 only has 'differential' once
    assert results[0]["page_number"] == 2
    assert results[0]["score"] > results[1]["score"]


def test_no_overlap_gives_no_fabricated_evidence(sample_calculus_pdf):
    """Unmatched queries return zero results and human-readable no-evidence notice."""
    results = search_pdf(sample_calculus_pdf, "quantum computing entanglement", top_k=3)
    assert results == []

    formatted = format_search_results(results, "quantum computing entanglement", file_path=sample_calculus_pdf)
    assert "No matching source found" in formatted
    assert "NOT an assertion that the" in formatted


def test_turkish_unicode_original_text_remains_intact(sample_turkish_pdf):
    """Turkish casing and diacritics match sensibly, while original source text is unmutated."""
    from src.retrieval.search_pdf import BM25Searcher

    # 1. Direct chunk API test with full Turkish Unicode diacritics and case variations
    raw_turkish_text = "AYRILABİLİR DİFERANSİYEL DENKLEMLER. Işık şiddeti ve çözümler incelenmiştir."
    chunk = {
        "text": raw_turkish_text,
        "char_count": len(raw_turkish_text),
        "page_number": 1,
        "file_name": "turkish_sample.pdf",
        "total_pages": 1,
        "source_sha256": "fakehash",
        "answer_eligible": False,
        "detection_status": "native_heuristic_incomplete",
        "blocks": [{"kind": "text", "text": raw_turkish_text}],
        "source_order": 1,
        "chunk_id": "chunk-1",
    }
    searcher = BM25Searcher([chunk])

    # Lowercase search query matches uppercase Turkish with dotted/dotless I
    results_lower = searcher.search("ayrılabilir diferansiyel", top_k=1)
    assert len(results_lower) == 1
    # Verify original text is completely intact and never mutated
    assert results_lower[0]["excerpt"] == raw_turkish_text
    assert "AYRILABİLİR DİFERANSİYEL DENKLEMLER" in results_lower[0]["excerpt"]

    # Uppercase search query with dotted I matches
    results_upper = searcher.search("AYRILABİLİR", top_k=1)
    assert len(results_upper) == 1

    # Dotless and dotted I search query matches
    results_isik = searcher.search("IŞIK", top_k=1)
    assert len(results_isik) == 1

    results_isik_lower = searcher.search("ışık", top_k=1)
    assert len(results_isik_lower) == 1

    # End-to-end PDF search with portable glyphs
    pdf_results = search_pdf(sample_turkish_pdf, "ayrilabilir diferansiyel", top_k=1)
    assert len(pdf_results) == 1
    assert "Ayrilabilir diferansiyel denklemler" in pdf_results[0]["excerpt"]


def test_deterministic_ties_preserve_source_order(tmp_path):
    """Equal-scoring chunks break ties stably by within-document source order."""
    pdf_path = tmp_path / "ties.pdf"
    with pymupdf.open() as doc:
        p1 = doc.new_page(width=500, height=400)
        p1.insert_text((50, 50), "Identifier marker alpha.")
        p2 = doc.new_page(width=500, height=400)
        p2.insert_text((50, 50), "Identifier marker alpha.")
        doc.save(pdf_path)

    results = search_pdf(pdf_path, "Identifier marker alpha", top_k=5)
    assert len(results) == 2
    assert results[0]["score"] == pytest.approx(results[1]["score"])
    assert results[0]["source_order"] < results[1]["source_order"]
    assert results[0]["page_number"] == 1
    assert results[1]["page_number"] == 2


def test_top_k_and_invalid_queries(sample_calculus_pdf):
    """Check positive top_k bounds, zero padding, and invalid query/argument rejection."""
    # top_k=1 returns only 1 result even if multiple chunks match
    results_1 = search_pdf(sample_calculus_pdf, "differential equations", top_k=1)
    assert len(results_1) == 1

    # Empty, whitespace, or punctuation-only queries return []
    assert search_pdf(sample_calculus_pdf, "") == []
    assert search_pdf(sample_calculus_pdf, "    ") == []
    assert search_pdf(sample_calculus_pdf, "???!!!,,..") == []

    # Invalid top_k types or values
    with pytest.raises(ValueError):
        search_pdf(sample_calculus_pdf, "differential", top_k=0)
    with pytest.raises(ValueError):
        search_pdf(sample_calculus_pdf, "differential", top_k=-2)
    with pytest.raises((ValueError, TypeError)):
        search_pdf(sample_calculus_pdf, "differential", top_k=True)


def test_missing_and_invalid_pdf_handling(tmp_path):
    """Missing file or non-PDF raises appropriate errors."""
    with pytest.raises(FileNotFoundError):
        search_pdf(tmp_path / "nonexistent.pdf", "differential")

    not_pdf = tmp_path / "notes.txt"
    not_pdf.write_text("Hello world", encoding="utf-8")
    with pytest.raises(ValueError):
        search_pdf(not_pdf, "differential")


def test_scanned_or_empty_pdf_no_fabrication(tmp_path):
    """Empty / scanned PDF returns 0 results with explicit note."""
    empty_pdf = tmp_path / "scanned_empty.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=500, height=400)
        doc.save(empty_pdf)

    results = search_pdf(empty_pdf, "differential", top_k=3)
    assert results == []
    formatted = format_search_results(results, "differential", file_path=empty_pdf)
    assert "No matching source found" in formatted
    assert "NOT an assertion that the" in formatted


def test_cli_full_stdout_and_file_equivalence(sample_calculus_pdf, tmp_path):
    """CLI prints to stdout and saves equivalent UTF-8 output to --output."""
    output_file = tmp_path / "results.txt"
    cmd = [
        sys.executable,
        "-m",
        "src.retrieval.search_pdf",
        str(sample_calculus_pdf),
        "separable differential equations",
        "--top-k",
        "2",
        "--output",
        str(output_file),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout_content = res.stdout.strip()
    file_content = output_file.read_text(encoding="utf-8").strip()

    assert stdout_content == file_content
    assert "Chapter 2: Separable Differential Equations" in stdout_content
    assert "Page 2" in stdout_content
    assert "Score:" in stdout_content


def test_grouped_wrapped_relation_preserves_children_and_answer_ineligible(sample_wrapped_relation_pdf):
    """Wrapped relation chunk retains both children intact and remains answer-ineligible."""
    # Query for the math constant 'K'
    results = search_pdf(sample_wrapped_relation_pdf, "K", top_k=3)
    assert len(results) > 0
    group_result = next((r for r in results if r.get("formula_groups")), None)
    assert group_result is not None

    # Verify formula group properties
    grp = group_result["formula_groups"][0]
    assert grp["kind"] == "wrapped_relation"
    assert grp["formula_ids"] == ["p1-f1", "p1-f2"]
    assert len(grp["formulas"]) == 2
    assert not grp["answer_eligible"]
    assert not group_result["answer_eligible"]

    # Child formulas retained intact
    child1, child2 = grp["formulas"]
    assert child1["id"] == "p1-f1"
    assert "K" in child1["original_text"]
    assert child2["id"] == "p1-f2"
    assert child1["state"] == "unresolved"
    assert child2["state"] == "unresolved"
    assert not child1["answer_eligible"]
    assert not child2["answer_eligible"]

    # Warning presence
    formatted = format_search_results([group_result], "K", file_path=sample_wrapped_relation_pdf)
    assert "not answer evidence" in formatted.lower() or "source-inspection" in formatted.lower()
