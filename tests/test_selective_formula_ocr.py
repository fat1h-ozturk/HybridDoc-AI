"""Tests for selective formula OCR CLI and API filters.

Proves:
- Only selected formulas' crops reach the recognizer (not filtered post-full OCR).
- Page filters, exact formula-ID filters, and combined intersection.
- Duplicate args are idempotent; source order is retained.
- Invalid page, unknown ID, group ID, and empty selection fail BEFORE any recognizer call.
- Group partial recognition keeps group unresolved, warns partial recognition, never auto-expands.
- Unselected formulas keep full provenance, not_selected_for_recognition reason, sign not_checked.
- Review HTML exporter correctly displays selective recognition results.
"""

import hashlib
import json
from pathlib import Path
import pymupdf
import pytest

from src.evaluation.evaluate_formulas import create_synthetic_benchmark_pdf
from src.evaluation.review_formulas import export_formula_review_html
from src.ingestion.formula_loader import load_pdf_formulas, main
from src.ingestion import pdf_loader


class RecordingRecognizer:
    """Test recognizer double recording exact crop bytes and call order."""

    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on or {}
        self.recorded_calls = []

    def recognize(self, png_bytes: bytes) -> str:
        h = hashlib.sha256(png_bytes).hexdigest()
        self.recorded_calls.append({
            'bytes': png_bytes,
            'sha256': h,
        })
        if h in self.fail_on:
            raise self.fail_on[h]
        if '__all_fail__' in self.fail_on:
            raise self.fail_on['__all_fail__']
        return self.responses.get(h, rf'\text{{candidate-{h[:8]}}}')


@pytest.fixture
def multi_page_math_pdf(tmp_path):
    """Creates a 3-page synthetic PDF:
    - Page 1: 2 formulas (f(x) and z = sqrt(u+v))
    - Page 2: 2 formulas forming a wrapped relation group (K = ... / 2^{-2/3})
    - Page 3: Plain text with no formulas
    """
    path = tmp_path / 'multi_page_math.pdf'
    doc = pymupdf.open()

    # Page 1
    p1 = doc.new_page(width=500, height=400)
    p1.insert_text((30, 30), 'Page 1 Prose Introduction')
    p1.insert_text((40, 75), 'f(x) =')
    p1.insert_text((72, 70), 'x')
    p1.insert_text((79, 66), '2', fontsize=7)
    p1.insert_text((85, 70), '+ 1')
    p1.insert_text((72, 85), '2y - 3')
    p1.draw_line((70, 74), (110, 74))

    p1.insert_text((40, 130), 'z =')
    writer = pymupdf.TextWriter(p1.rect)
    writer.append((60, 132), '√', font=pymupdf.Font('symbol'), fontsize=16)
    writer.write_text(p1)
    p1.draw_line((72, 120), (120, 120))
    p1.insert_text((75, 130), 'u + v')

    # Page 2: Wrapped relation group
    p2 = doc.new_page(width=500, height=400)
    p2.insert_text((30, 30), 'Page 2 Wrapped Relation Section')
    p2.insert_text((40, 60), 'g(x) = C')
    p2.insert_text((95, 60), ' continuation prose ')
    p2.insert_text((350, 60), 'K =')
    p2.insert_text((40, 83), '2')
    p2.insert_text((46, 75), '-', fontsize=6)
    p2.insert_text((51, 71), '2', fontsize=6)
    p2.insert_text((51, 79), '3', fontsize=6)
    p2.draw_line((51, 73), (56, 73), width=0.4)
    p2.insert_text((60, 83), '. Tail prose.')

    # Page 3: Pure prose, no formulas
    p3 = doc.new_page(width=500, height=400)
    p3.insert_text((30, 30), 'Page 3 purely prose with absolutely no math symbols or lines.')

    doc.save(str(path))
    doc.close()
    return path


def test_native_only_mode_zero_ocr_and_no_filters_allowed(multi_page_math_pdf):
    """Native mode without recognizer never calls OCR and rejects filters."""
    # Default native extraction works
    pages = load_pdf_formulas(multi_page_math_pdf)
    assert len(pages) == 3
    assert all(f['recognition_status'] == 'not_requested' for p in pages for f in p['formulas'])
    assert all(f['reason'] == 'recognition_not_requested' for p in pages for f in p['formulas'])

    # Supplying selection filters without a recognizer raises ValueError
    with pytest.raises(ValueError, match="require.*recognizer"):
        load_pdf_formulas(multi_page_math_pdf, pages=[1])

    with pytest.raises(ValueError, match="require.*recognizer"):
        load_pdf_formulas(multi_page_math_pdf, formula_ids=['p1-f1'])


def test_page_filter_limits_recognizer_to_selected_page(multi_page_math_pdf):
    """Page filter selects all leaf formulas on that page and calls recognizer ONLY for them."""
    rec = RecordingRecognizer()
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[1])

    p1_formulas = pages[0]['formulas']
    p2_formulas = pages[1]['formulas']

    # Page 1 formulas were selected and recognized
    assert len(p1_formulas) > 0
    assert len(rec.recorded_calls) == len(p1_formulas)
    assert all(f['recognition_status'] == 'returned' for f in p1_formulas)
    assert all(f['latex'] is not None for f in p1_formulas)

    # Page 2 formulas were NOT requested / selected
    assert len(p2_formulas) > 0
    assert all(f['recognition_status'] == 'not_requested' for f in p2_formulas)
    assert all(f['reason'] == 'not_selected_for_recognition' for f in p2_formulas)
    assert all(f['latex'] is None for f in p2_formulas)
    assert all(f['sign_consistency']['status'] == 'not_checked' for f in p2_formulas)
    assert all(len(f['warnings']) == 0 for f in p2_formulas)

    # Verify crops sent to recognizer match Page 1 formulas only
    with pymupdf.open(multi_page_math_pdf) as doc:
        p1 = doc[0]
        expected_hashes = set()
        for f in p1_formulas:
            pix = p1.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(f['bbox']))
            expected_hashes.add(hashlib.sha256(pix.tobytes('png')).hexdigest())

    call_hashes = {c['sha256'] for c in rec.recorded_calls}
    assert call_hashes == expected_hashes


def test_exact_one_formula_id_filter(multi_page_math_pdf):
    """Selecting exactly one formula ID calls recognizer exactly once for that formula."""
    # First get inventory
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    target_id = native_pages[0]['formulas'][0]['id']

    rec = RecordingRecognizer()
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=[target_id])

    assert len(rec.recorded_calls) == 1

    selected_f = next(f for p in pages for f in p['formulas'] if f['id'] == target_id)
    assert selected_f['recognition_status'] == 'returned'
    assert selected_f['latex'] is not None

    unselected = [f for p in pages for f in p['formulas'] if f['id'] != target_id]
    assert len(unselected) > 0
    for f in unselected:
        assert f['recognition_status'] == 'not_requested'
        assert f['reason'] == 'not_selected_for_recognition'
        assert f['latex'] is None
        assert f['sign_consistency']['status'] == 'not_checked'


def test_combined_page_and_id_intersection(multi_page_math_pdf):
    """Combined page and formula_id intersects valid IDs and rejects IDs outside selected pages."""
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    p1_f1 = native_pages[0]['formulas'][0]['id']
    p2_f1 = native_pages[1]['formulas'][0]['id']

    # Valid intersection: p1_f1 is on page 1
    rec = RecordingRecognizer()
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[1], formula_ids=[p1_f1])
    assert len(rec.recorded_calls) == 1

    # Rejected: p2_f1 is on page 2, not page 1
    rec2 = RecordingRecognizer()
    with pytest.raises(ValueError, match="outside selected page"):
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec2, pages=[1], formula_ids=[p2_f1])
    assert len(rec2.recorded_calls) == 0


def test_duplicate_arguments_idempotent_and_source_order(tmp_path):
    """Duplicate filters are idempotent and recognizer calls follow document source order."""
    doc_path = tmp_path / 'source_order.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=400, height=300)
        p.insert_text((40, 60), 'a = b + c')
        p.insert_text((40, 120), 'd = e + f')
        doc.save(str(doc_path))

    native_pages = load_pdf_formulas(doc_path)
    p_formulas = native_pages[0]['formulas']
    assert len(p_formulas) == 2
    f1_id = p_formulas[0]['id']
    f2_id = p_formulas[1]['id']

    # Reverse order with duplicates in arguments: f2, f1, f2
    rec = RecordingRecognizer()
    pages = load_pdf_formulas(
        doc_path,
        recognizer=rec,
        pages=[1, 1],
        formula_ids=[f2_id, f1_id, f2_id],
    )

    # Exactly 2 calls (idempotent deduplication)
    assert len(rec.recorded_calls) == 2

    # Preserves document source order (f1 before f2), NOT argument order
    with pymupdf.open(doc_path) as doc:
        p = doc[0]
        h1 = hashlib.sha256(p.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(p_formulas[0]['bbox'])).tobytes('png')).hexdigest()
        h2 = hashlib.sha256(p.get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(p_formulas[1]['bbox'])).tobytes('png')).hexdigest()

    assert rec.recorded_calls[0]['sha256'] == h1
    assert rec.recorded_calls[1]['sha256'] == h2


def test_invalid_and_out_of_range_selection_fails_before_calls(multi_page_math_pdf):
    """Out-of-range page, unknown ID, group ID, and empty selection fail with zero calls."""
    rec = RecordingRecognizer()

    # Out of range page (< 1 or > total_pages)
    with pytest.raises(ValueError, match="out of range"):
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[99])
    with pytest.raises(ValueError, match="out of range"):
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[0])

    # Unknown formula ID
    with pytest.raises(ValueError, match="Unknown formula ID"):
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=['nonexistent-f99'])

    # Empty selection (page 3 has no formulas)
    with pytest.raises(ValueError, match="Empty selection"):
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[3])

    # Group ID rejection with actionable child IDs
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    p2_groups = native_pages[1].get('formula_groups', [])
    assert len(p2_groups) > 0
    grp = p2_groups[0]
    grp_id = grp['id']
    child_ids = grp['formula_ids']

    with pytest.raises(ValueError) as excinfo:
        load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=[grp_id])
    assert grp_id in str(excinfo.value)
    for cid in child_ids:
        assert cid in str(excinfo.value)

    # Zero recognizer calls made across all failure cases
    assert len(rec.recorded_calls) == 0


def test_partial_group_recognition_and_provenance(multi_page_math_pdf):
    """Partially recognized group remains unresolved, warns partial recognition, no auto-expansion."""
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    grp = native_pages[1]['formula_groups'][0]
    child1, child2 = grp['formula_ids']

    # Select ONLY child1
    rec = RecordingRecognizer()
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=[child1])

    # Recognizer called ONLY for child1, no sibling expansion
    assert len(rec.recorded_calls) == 1

    p2 = pages[1]
    result_grp = p2['formula_groups'][0]
    assert result_grp['id'] == grp['id']
    assert result_grp['state'] == 'unresolved'
    assert result_grp['latex'] is None
    assert result_grp['answer_eligible'] is False

    # Check warning on group
    warning_codes = [w.get('code') for w in result_grp.get('warnings', [])]
    assert 'partial_group_recognition' in warning_codes

    # Child states
    f_by_id = {f['id']: f for f in p2['formulas']}
    assert f_by_id[child1]['recognition_status'] == 'returned'
    assert f_by_id[child2]['recognition_status'] == 'not_requested'
    assert f_by_id[child2]['reason'] == 'not_selected_for_recognition'

    # Prose blocks, child bboxes and geometry preserved
    assert len(result_grp['formulas']) == 2
    assert result_grp['formulas'][0]['bbox'] == f_by_id[child1]['bbox']
    assert result_grp['formulas'][1]['bbox'] == f_by_id[child2]['bbox']


def test_selected_failure_preserves_error_without_budget_reset(multi_page_math_pdf):
    """When a selected formula fails/times out, it records failed state without budget reset."""
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    f1_id = native_pages[0]['formulas'][0]['id']

    rec = RecordingRecognizer(fail_on={'__all_fail__': TimeoutError("model inference timed out")})
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=[f1_id])

    assert len(rec.recorded_calls) == 1
    failed_f = next(f for p in pages for f in p['formulas'] if f['id'] == f1_id)
    assert failed_f['recognition_status'] == 'failed'
    assert failed_f['state'] == 'unresolved'
    assert 'TimeoutError' in failed_f['reason']


def test_review_html_exporter_consumes_selective_extraction(tmp_path, multi_page_math_pdf):
    """review_formulas HTML exporter accepts selective extraction and renders accurate cards."""
    native_pages = load_pdf_formulas(multi_page_math_pdf)
    grp = native_pages[1]['formula_groups'][0]
    child1 = grp['formula_ids'][0]

    rec = RecordingRecognizer()
    pages = load_pdf_formulas(multi_page_math_pdf, recognizer=rec, formula_ids=[child1])

    json_path = tmp_path / 'selective.json'
    json_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')

    html_path = tmp_path / 'selective.html'
    export_formula_review_html(multi_page_math_pdf, html_path, extraction_path=json_path)

    assert html_path.exists()
    html_content = html_path.read_text(encoding='utf-8')

    # Verified cards rendered
    assert f'Formula <code>{child1}</code>' in html_content
    assert 'OCR candidate returned (unverified)' in html_content
    assert 'No OCR candidate (native only)' in html_content
    assert 'not_selected_for_recognition' in html_content
    assert 'partial_group_recognition' in html_content


def test_cli_flags_and_validation(tmp_path, multi_page_math_pdf, monkeypatch, capsys):
    """Test formula_loader CLI with --page, --formula-id, --output, and require --recognize."""
    from src.ingestion import formula_recognizer

    class ContextDouble:
        def __init__(self, *args, **kwargs):
            self.calls = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def recognize(self, png):
            self.calls.append(png)
            return r'\text{cli-mock}'

    monkeypatch.setattr(formula_recognizer, 'LocalFormulaRecognizer', ContextDouble)

    # 1. Selection without --recognize must fail
    with pytest.raises(SystemExit) as exc:
        main([str(multi_page_math_pdf), '--page', '1'])
    assert exc.value.code != 0

    with pytest.raises(SystemExit) as exc:
        main([str(multi_page_math_pdf), '--formula-id', 'p1-f1'])
    assert exc.value.code != 0

    # 2. Valid selective recognition via CLI
    out_json = tmp_path / 'cli_out.json'
    res = main([
        str(multi_page_math_pdf),
        '--recognize',
        '--page', '1',
        '--formula-id', 'p1-f1',
        '--output', str(out_json),
    ])
    assert res == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding='utf-8'))
    assert len(data) == 3
    p1_f1 = next(f for f in data[0]['formulas'] if f['id'] == 'p1-f1')
    assert p1_f1['recognition_status'] == 'returned'

    # Unselected formula on page 1
    other_f = next(f for f in data[0]['formulas'] if f['id'] != 'p1-f1')
    assert other_f['recognition_status'] == 'not_requested'
    assert other_f['reason'] == 'not_selected_for_recognition'


def test_pdf_loader_additive_export(multi_page_math_pdf):
    """Ensure pdf_loader.load_pdf_formulas exports pages and formula_ids parameters."""
    rec = RecordingRecognizer()
    pages = pdf_loader.load_pdf_formulas(multi_page_math_pdf, recognizer=rec, pages=[1])
    assert len(rec.recorded_calls) > 0
    assert all(f['recognition_status'] == 'not_requested' for f in pages[1]['formulas'])
