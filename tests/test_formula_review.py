"""Tests for formula review and provenance HTML export CLI and generator."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pymupdf
import pytest

from src.evaluation.evaluate_formulas import create_synthetic_benchmark_pdf
from src.evaluation.review_formulas import (
    export_formula_review_html,
    generate_formula_review_html,
    validate_extraction_data,
)
from src.ingestion.formula_loader import load_pdf_formulas

DATA_DIR = Path(__file__).resolve().parents[1] / 'eval_data'
SOURCE_PDF = DATA_DIR / 'small_formula_ocr_corrected.pdf'
SOURCE_JSON = DATA_DIR / 'small_formula_ocr_corrected.json'


class ControlledDouble:
    """Controlled recognizer double returning specific LaTeX candidates."""
    def __init__(self, mapping):
        self.mapping = mapping

    def recognize(self, png):
        h = hashlib.sha256(png).hexdigest()
        return self.mapping.get(h, r'\text{stub}')


def test_default_native_only_extraction_has_explicit_no_ocr_labels(tmp_path):
    """Native-only run generates valid HTML, PNG crops, metadata, and explicit no-OCR labels."""
    pdf_path = tmp_path / 'benchmark.pdf'
    create_synthetic_benchmark_pdf(pdf_path)
    output_html = tmp_path / 'review.html'

    export_formula_review_html(pdf_path, output_html)
    assert output_html.exists()
    content = output_html.read_text(encoding='utf-8')

    # Self-contained HTML checks
    assert content.startswith('<!DOCTYPE html>')
    assert '<meta charset="utf-8">' in content
    assert '<script' not in content.lower()  # Zero client-side JS

    # Prominent disclaimer
    assert '[UNVERIFIED NATIVE TEXT: math detection is incomplete; answer_eligible: false' in content

    # File and SHA metadata
    pdf_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert pdf_path.name in content
    assert pdf_sha in content
    assert 'Native-only heuristic extraction (no OCR requested)' in content

    # Explicit no-OCR label (no invented LaTeX)
    assert 'No OCR candidate (native only)' in content
    assert '[No OCR candidate requested or returned; native-only extraction]' in content
    assert 'Sign Not Checked' in content

    # Valid PNG data URLs embedded
    assert 'data:image/png;base64,' in content
    # Extract one data url and check it starts with valid PNG magic bytes
    start_idx = content.find('data:image/png;base64,') + len('data:image/png;base64,')
    end_idx = content.find('"', start_idx)
    raw_b64 = content[start_idx:end_idx]
    png_bytes = base64.b64decode(raw_b64)
    assert png_bytes.startswith(b'\x89PNG\r\n\x1a\n')


def test_saved_wrong_minus_candidate_warning_visible(tmp_path):
    """Saved wrong-minus candidate displays the raw candidate and conflict warning."""
    evidence = json.loads(SOURCE_JSON.read_text(encoding='utf-8'))
    call_wrong = evidence['calls'][0]  # p1-f7: 2 ^ { \cdot \frac { 2 } { 3 } }
    call_ok = evidence['calls'][1]     # p1-f1: positive fraction

    mapping = {
        call_wrong['crop_sha256']: call_wrong['raw_latex'],
        call_ok['crop_sha256']: call_ok['raw_latex'],
    }
    pages = load_pdf_formulas(SOURCE_PDF, recognizer=ControlledDouble(mapping))
    extraction_path = tmp_path / 'extraction.json'
    extraction_path.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding='utf-8')

    output_html = tmp_path / 'review_wrong_minus.html'
    export_formula_review_html(SOURCE_PDF, output_html, extraction_path)

    content = output_html.read_text(encoding='utf-8')

    # Card class and badges for mismatch
    assert 'card-formula has-mismatch' in content
    assert 'Sign Mismatch' in content

    # Conflict warning visible with actionable reason
    assert '[WARNING: source_minus_candidate_conflict]' in content
    assert 'source_minus_missing_or_changed_in_supported_candidate_scope' in content

    # Raw wrong candidate displayed without alteration or repair
    assert r'2 ^ { \cdot \frac { 2 } { 3 } }' in content

    # Positive fraction has no conflict
    assert 'has-no-conflict' in content
    assert 'No Conflict Detected (Scope-Only)' in content


def test_hash_match_succeeds_and_mismatch_or_missing_rejected(tmp_path):
    """SHA256 must match PDF exactly; missing or mismatched SHA is rejected."""
    pages = load_pdf_formulas(SOURCE_PDF)
    extraction_path = tmp_path / 'extraction.json'

    # 1. Correct SHA succeeds
    extraction_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')
    out1 = tmp_path / 'out1.html'
    export_formula_review_html(SOURCE_PDF, out1, extraction_path)
    assert out1.exists()

    # 2. Mismatched SHA is rejected with actionable error
    pages[0]['source_sha256'] = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
    extraction_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')
    out2 = tmp_path / 'out2.html'
    with pytest.raises(ValueError, match="Source SHA256 mismatch"):
        export_formula_review_html(SOURCE_PDF, out2, extraction_path)

    # 3. Missing SHA is rejected
    del pages[0]['source_sha256']
    extraction_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')
    out3 = tmp_path / 'out3.html'
    with pytest.raises(ValueError, match="Missing source_sha256"):
        export_formula_review_html(SOURCE_PDF, out3, extraction_path)


def test_malformed_page_and_bbox_metadata_rejected(tmp_path):
    """Malformed pages, inverted bboxes, and out-of-page bboxes are strictly rejected."""
    pdf_sha = hashlib.sha256(SOURCE_PDF.read_bytes()).hexdigest()
    base_pages = load_pdf_formulas(SOURCE_PDF)

    # 1. Non-list root schema
    with pytest.raises(ValueError, match="Invalid extraction format"):
        with pymupdf.open(SOURCE_PDF) as doc:
            validate_extraction_data({"pages": base_pages}, pdf_sha, doc)

    # 2. Out-of-range page number
    bad_page = json.loads(json.dumps(base_pages))
    bad_page[0]['page_number'] = 99
    with pytest.raises(ValueError, match="Malformed or out-of-order page number"):
        with pymupdf.open(SOURCE_PDF) as doc:
            validate_extraction_data(bad_page, pdf_sha, doc)

    # 3. Inverted bbox (x0 >= x1)
    bad_bbox = json.loads(json.dumps(base_pages))
    bad_bbox[0]['formulas'][0]['bbox'] = [100.0, 50.0, 80.0, 90.0]
    with pytest.raises(ValueError, match="Inverted or zero-area bbox"):
        with pymupdf.open(SOURCE_PDF) as doc:
            validate_extraction_data(bad_bbox, pdf_sha, doc)

    # 4. Out-of-page bounds bbox
    bad_bounds = json.loads(json.dumps(base_pages))
    bad_bounds[0]['formulas'][0]['bbox'] = [10.0, 20.0, 1500.0, 2000.0]
    with pytest.raises(ValueError, match="out of page bounds"):
        with pymupdf.open(SOURCE_PDF) as doc:
            validate_extraction_data(bad_bounds, pdf_sha, doc)

    # 5. Non-numeric bbox
    bad_coords = json.loads(json.dumps(base_pages))
    bad_coords[0]['formulas'][0]['bbox'] = ['a', 'b', 'c', 'd']
    with pytest.raises(ValueError, match="Non-numeric bbox coordinates"):
        with pymupdf.open(SOURCE_PDF) as doc:
            validate_extraction_data(bad_coords, pdf_sha, doc)


def test_html_injection_escaping(tmp_path):
    """All source and candidate strings must be strictly escaped into visible text."""
    pages = load_pdf_formulas(SOURCE_PDF)
    payload = '<script>alert("xss")</script><img src=x onerror=alert(1)>"\'&'
    pages[0]['formulas'][0]['original_text'] = payload
    pages[0]['formulas'][0]['latex'] = payload
    pages[0]['formulas'][0]['raw_latex'] = payload
    pages[0]['formulas'][0]['warnings'] = [
        {'code': payload, 'message': payload, 'reason': payload, 'raw_candidate': payload}
    ]

    extraction_path = tmp_path / 'injection_extraction.json'
    extraction_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')

    out_html = tmp_path / 'escaped.html'
    export_formula_review_html(SOURCE_PDF, out_html, extraction_path)
    content = out_html.read_text(encoding='utf-8')

    # Unescaped dangerous tags must not exist
    assert '<script' not in content.lower()
    assert '<img src=x' not in content.lower()

    # Escaped payloads rendered as visible text
    assert '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;' in content
    assert '&lt;img src=x onerror=alert(1)&gt;' in content


def test_group_child_provenance_preserved(tmp_path):
    """Wrapped relation groups maintain child IDs, individual bboxes, and anchor links."""
    pdf_path = tmp_path / 'wrapped_relation.pdf'
    create_synthetic_benchmark_pdf(pdf_path)

    out_html = tmp_path / 'wrapped_review.html'
    export_formula_review_html(pdf_path, out_html)
    content = out_html.read_text(encoding='utf-8')

    # Group card rendered with links to children
    assert 'Wrapped Relation Group' in content
    assert 'group-p1-f6+p1-f7' in content
    assert 'heuristic_unverified_continuation' in content

    # Links between cards
    assert 'href="#card-p1-f6"' in content
    assert 'href="#card-p1-f7"' in content
    assert 'href="#group-p1-f6+p1-f7"' in content

    # Individual bounding boxes preserved
    assert '418.0' in content  # p1-f6 x0
    assert '38.0' in content   # p1-f7 x0


def test_zero_native_formulas_scanned_case(tmp_path):
    """A PDF with zero native formulas shows clear detection-incomplete banner without claiming none exist."""
    blank_pdf = tmp_path / 'blank.pdf'
    doc = pymupdf.open()
    doc.new_page(width=500, height=400)
    doc.save(str(blank_pdf))
    doc.close()

    out_html = tmp_path / 'blank_review.html'
    export_formula_review_html(blank_pdf, out_html)
    content = out_html.read_text(encoding='utf-8')

    assert 'No Native Formulas Detected' in content
    assert 'Detection is incomplete' in content
    assert 'does NOT claim that no formulas exist in this document' in content


def test_output_input_collision_rejected(tmp_path):
    """Output path cannot alias or overwrite either input PDF or extraction JSON."""
    pdf_path = tmp_path / 'sample.pdf'
    create_synthetic_benchmark_pdf(pdf_path)
    extraction_path = tmp_path / 'extraction.json'
    extraction_path.write_text('[]', encoding='utf-8')

    with pytest.raises(ValueError, match="Output path cannot alias or overwrite input PDF"):
        export_formula_review_html(pdf_path, pdf_path)

    with pytest.raises(ValueError, match="Output path cannot alias or overwrite extraction JSON"):
        export_formula_review_html(pdf_path, extraction_path, extraction_path)


def test_no_conflict_never_says_mathematically_correct(tmp_path):
    """No conflict detected status must never claim mathematical correctness."""
    evidence = json.loads(SOURCE_JSON.read_text(encoding='utf-8'))
    call_ok = evidence['calls'][1]  # positive fraction
    mapping = {call_ok['crop_sha256']: call_ok['raw_latex']}
    pages = load_pdf_formulas(SOURCE_PDF, recognizer=ControlledDouble(mapping))
    extraction_path = tmp_path / 'extraction.json'
    extraction_path.write_text(json.dumps(pages, indent=2), encoding='utf-8')

    out_html = tmp_path / 'no_conflict.html'
    export_formula_review_html(SOURCE_PDF, out_html, extraction_path)
    content = out_html.read_text(encoding='utf-8')

    assert 'No Conflict Detected (Scope-Only)' in content
    assert 'Supported minus scopes match candidate; this does NOT verify formula correctness' in content
    assert 'mathematically correct' not in content.lower()
    assert 'verified correct' not in content.lower()


def test_cli_invocation_generates_valid_utf8_html(tmp_path):
    """CLI execution outputs valid UTF-8 HTML with zero exit status."""
    out_html = tmp_path / 'cli_review.html'
    cmd = [
        sys.executable,
        '-m',
        'src.evaluation.review_formulas',
        str(SOURCE_PDF),
        '--output',
        str(out_html),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert 'Exported formula review HTML to:' in res.stdout
    assert out_html.exists()

    content = out_html.read_text(encoding='utf-8')
    assert '<!DOCTYPE html>' in content
    assert SOURCE_PDF.name in content
