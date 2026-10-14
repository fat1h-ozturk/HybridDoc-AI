"""Source consistency regressions; saved OCR replay and controlled doubles, no inference."""
import hashlib
import json
from pathlib import Path

from src.ingestion.formula_loader import load_pdf_formulas

DATA = Path(__file__).resolve().parents[1] / 'eval_data'
EVIDENCE = json.loads((DATA / 'small_formula_ocr_corrected.json').read_text())
SOURCE = DATA / 'small_formula_ocr_corrected.pdf'


class CandidateDouble:
    """Controlled recognizer double; not a fresh accuracy measurement."""
    def __init__(self, latex):
        self.latex = latex

    def recognize(self, png):
        return self.latex


def negative_formula(pages):
    return next(f for f in pages[0]['formulas'] if f['id'] == EVIDENCE['calls'][0]['formula_id'])


def test_saved_wrong_exponent_warns_without_rewriting_or_promoting():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EVIDENCE['source_sha256']
    call = EVIDENCE['calls'][0]
    pages = load_pdf_formulas(SOURCE, recognizer=CandidateDouble(call['raw_latex']))
    formula = negative_formula(pages)
    assert formula['bbox'] == call['bbox']
    assert formula['latex'] == call['raw_latex']
    assert formula['sign_consistency']['status'] in ('mismatch', 'uncertain')
    warning = formula['warnings'][0]
    assert warning['code'] == 'source_minus_candidate_conflict'
    assert warning['raw_candidate'] == call['raw_latex']
    assert warning['source_operators'][0]['operator'] == '-'
    assert warning['source_operators'][0]['span_id'] in formula['span_ids']
    assert warning['page_number'] == 1
    assert warning['source_sha256'] == EVIDENCE['source_sha256']
    assert not formula['answer_eligible']
    assert formula['state'] == 'unresolved'
    assert pages[0]['detection_status'] == 'native_heuristic_incomplete'


def test_correct_exponent_has_no_sign_conflict_but_remains_unverified():
    pages = load_pdf_formulas(SOURCE, recognizer=CandidateDouble(EVIDENCE['calls'][0]['gold']))
    formula = negative_formula(pages)
    assert formula['sign_consistency']['status'] == 'no_conflict_detected'
    assert formula['warnings'] == []
    assert not formula['answer_eligible']
    assert formula['state'] == 'unresolved'


def test_native_only_is_explicitly_not_checked():
    formula = negative_formula(load_pdf_formulas(SOURCE))
    assert formula['sign_consistency']['status'] == 'not_checked'
    assert formula['recognition_status'] == 'not_requested'
    assert formula['latex'] is None


def test_positive_control_denominator_minus_is_not_an_exponent_minus():
    call = EVIDENCE['calls'][1]
    page = load_pdf_formulas(SOURCE, recognizer=CandidateDouble(call['raw_latex']))[0]
    formula = next(f for f in page['formulas'] if f['id'] == call['formula_id'])
    assert formula['sign_consistency']['status'] == 'no_conflict_detected'
    assert formula['warnings'] == []
    assert not formula['answer_eligible']


import pytest
import pymupdf


@pytest.mark.parametrize('latex', [
    r'2^{\frac{2}{3-1}}',
    r'2^{\cdot\frac{2}{3}} - 1',
    r'2^{\frac{2}{3}}\text{negative - exponent}',
    r'2^{\frac{2}{3}}\operatorname{minus-}',
])
def test_other_scopes_and_text_cannot_hide_missing_exponent_minus(latex):
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(latex)))
    assert formula['sign_consistency']['status'] in ('mismatch', 'uncertain')
    assert formula['warnings']


@pytest.mark.parametrize('latex', [
    r'2^{ - \frac{2}{3}}\text{annotation - only}',
    r'2\, ^ {\! - \tfrac {2}{3} }',
    '2^{−\\frac{2}{3}}',
])
def test_text_minus_and_tex_spacing_do_not_create_fake_conflicts(latex):
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(latex)))
    assert formula['sign_consistency']['status'] == 'no_conflict_detected'
    assert formula['warnings'] == []


@pytest.mark.parametrize('latex', [r'2^{ - \frac{2}{3}', r'2^{\custom{-}\frac{2}{3}}', r'2^{\sqrt{-2/3}}'])
def test_unsupported_or_unmatched_tex_is_uncertain(latex):
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(latex)))
    assert formula['sign_consistency']['status'] == 'uncertain'
    assert formula['warnings']


@pytest.mark.parametrize('error', [TimeoutError('controlled timeout'), RuntimeError('controlled unavailable')])
def test_recognition_failure_is_not_checked(error):
    class FailureDouble:
        """Controlled failure at recognizer boundary; no live OCR."""
        def recognize(self, png):
            raise error
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=FailureDouble()))
    assert formula['sign_consistency']['status'] == 'not_checked'
    assert formula['recognition_status'] == 'failed'
    assert str(error) in formula['reason']
    assert formula['warnings'] == []


def write_pdf(tmp_path, text, *, unicode=False):
    path = tmp_path / 'signs.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=200)
        if unicode:
            writer = pymupdf.TextWriter(page.rect)
            writer.append((40, 60), text, font=pymupdf.Font('helv'), fontsize=11)
            writer.write_text(page)
        else:
            page.insert_text((40, 60), text)
        page.insert_text((220, 60), 'well-known prose')
        doc.save(path)
    return path


@pytest.mark.parametrize(('text', 'latex', 'status'), [
    ('x = y - 3', r'x=y-3', 'no_conflict_detected'),
    ('x = y - 3', r'x=y\cdot3', 'mismatch'),
    ('x = y - 3', r'x=z-3', 'uncertain'),
    ('x = y - 3 - 2', r'x=y-3-2', 'uncertain'),
    ('x = 2 + 3', r'x=2\cdot3', 'not_checked'),
    ('x = 2', r'x=2\text{-}', 'not_checked'),
])
def test_inline_operator_context_and_prose_hyphen(tmp_path, text, latex, status):
    page = load_pdf_formulas(write_pdf(tmp_path, text), recognizer=CandidateDouble(latex))[0]
    formula = page['formulas'][0]
    assert formula['sign_consistency']['status'] == status
    assert all(op['span_text'] != 'well-known prose'
               for op in formula['sign_consistency']['source_operators'])
    assert not formula['answer_eligible']


def test_unicode_source_minus(tmp_path):
    page = load_pdf_formulas(write_pdf(tmp_path, 'x = y − 3', unicode=True),
                             recognizer=CandidateDouble(r'x=y\times3'))[0]
    formula = page['formulas'][0]
    assert formula['sign_consistency']['status'] == 'mismatch'
    assert formula['warnings'][0]['source_operators'][0]['operator'] == '−'


def test_no_native_text_is_explicitly_unmeasured(tmp_path):
    path = tmp_path / 'no_native.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10))
        pix.clear_with(255)
        page.insert_image(pymupdf.Rect(40, 40, 100, 100), pixmap=pix)
        doc.save(path)
    page = load_pdf_formulas(path, recognizer=CandidateDouble('x=-1'))[0]
    assert page['sign_consistency']['status'] == 'not_checked'
    assert page['formulas'] == []
    assert page['unresolved_reason'].startswith('no native text')


def test_group_chunk_retrieval_keep_child_warning_and_provenance():
    from src.chunking.text_chunker import chunk_formula_pages
    from src.retrieval.search_pdf import BM25Searcher, format_search_results
    pages = load_pdf_formulas(SOURCE, recognizer=CandidateDouble(EVIDENCE['calls'][0]['raw_latex']))
    group = pages[0]['formula_groups'][0]
    child = negative_formula(pages)
    assert group['warnings'] == child['warnings']
    chunk = next(c for c in chunk_formula_pages(pages, max_chars=110, overlap=0) if c['formula_groups'])
    assert chunk['formulas'] == group['formulas']
    assert chunk['formula_groups'][0]['warnings'] == child['warnings']
    results = BM25Searcher([chunk]).search('K')
    assert results[0]['formulas'][-1]['warnings'] == child['warnings']
    rendered = format_search_results(results, 'K')
    assert 'source_minus_candidate_conflict' in rendered
    assert 'span_id' in rendered
    assert EVIDENCE['source_sha256'] in rendered
    assert not chunk['answer_eligible']


def test_cli_preserves_warning_with_controlled_recognizer(tmp_path, monkeypatch, capsys):
    from src.ingestion.formula_loader import main
    from src.ingestion import formula_recognizer
    class ContextDouble(CandidateDouble):
        """CLI recognizer double, explicitly replaying saved output."""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(formula_recognizer, 'LocalFormulaRecognizer',
                        lambda *args: ContextDouble(EVIDENCE['calls'][0]['raw_latex']))
    output = tmp_path / 'extraction.json'
    assert main([str(SOURCE), '--recognize', '--output', str(output)]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == json.loads(output.read_text())
    assert negative_formula(parsed)['warnings'][0]['code'] == 'source_minus_candidate_conflict'
    assert main([str(SOURCE)]) == 0
    native = json.loads(capsys.readouterr().out)
    assert all(f['sign_consistency']['status'] == 'not_checked' for f in native[0]['formulas'])


def test_denominator_sign_cannot_satisfy_source_exponent_sign(tmp_path):
    path = tmp_path / 'two_scopes.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 85), '2')
        page.insert_text((46, 77), '-', fontsize=6)
        page.insert_text((51, 73), '2', fontsize=6)
        page.draw_line((51, 75), (70, 75), width=.4)
        page.insert_text((51, 81), 'y - 3', fontsize=6)
        doc.save(path)
    for latex, expected in [(r'2^{-\frac{2}{y-3}}', 'no_conflict_detected'),
                            (r'2^{\frac{2}{y-3}}', 'mismatch')]:
        page = load_pdf_formulas(path, recognizer=CandidateDouble(latex))[0]
        assert len(page['formulas']) == 1
        formula = page['formulas'][0]
        assert len(formula['sign_consistency']['source_operators']) == 2
        assert formula['sign_consistency']['status'] == expected
        assert formula['latex'] == latex


@pytest.mark.parametrize('layout', ['columns', 'table'])
def test_minus_checks_respect_formula_region_ownership(tmp_path, layout):
    path = tmp_path / 'layout.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        if layout == 'table':
            for x in (30, 180, 330):
                page.draw_line((x, 30), (x, 100))
            for y in (30, 100):
                page.draw_line((30, y), (330, y))
        page.insert_text((40, 60), 'x = y - 3')
        page.insert_text((200, 60), 'z = w + 3')
        doc.save(path)
    page = load_pdf_formulas(path, recognizer=CandidateDouble('z=w-3'))[0]
    assert len(page['formulas']) == 2
    left, right = page['formulas']
    assert left['sign_consistency']['status'] == 'uncertain'
    assert right['sign_consistency']['status'] == 'not_checked'
    assert right['warnings'] == []
    assert all(op['span_id'] in left['span_ids'] for op in left['sign_consistency']['source_operators'])
    assert page['formula_groups'] == []


@pytest.mark.parametrize('text', ['x = 2 note-minus', 'x = 2 annotation - label', 'x = 2 wrap-'])
def test_native_text_annotations_and_wrapping_hyphens_are_excluded(tmp_path, text):
    page = load_pdf_formulas(write_pdf(tmp_path, text), recognizer=CandidateDouble('x=2'))[0]
    assert page['formulas'][0]['sign_consistency']['status'] == 'not_checked'
    assert page['formulas'][0]['warnings'] == []


def test_missing_native_glyph_provenance_is_not_checked():
    from src.ingestion.formula_regions import propose_regions
    from src.ingestion.formula_signs import check_formula_signs
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble('2^{2/3}')))
    with pymupdf.open(SOURCE) as doc:
        _, spans, _ = propose_regions(doc[0])
    for s in spans:
        s.pop('chars', None)
    check_formula_signs(formula, spans)
    assert formula['sign_consistency']['status'] == 'not_checked'
    assert formula['warnings'] == []


def test_repeated_script_bases_are_uncertain(tmp_path):
    path = tmp_path / 'repeated_base.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = 2')
        page.insert_text((65, 52), '-3', fontsize=6)
        page.insert_text((73, 60), '+2')
        page.insert_text((85, 52), '4', fontsize=6)
        doc.save(path)
    page = load_pdf_formulas(path, recognizer=CandidateDouble('x=2^{3}+2^{-4}'))[0]
    assert page['formulas'][0]['sign_consistency']['status'] == 'uncertain'
    assert page['formulas'][0]['warnings']


def test_unknown_native_fraction_attachment_is_uncertain(tmp_path):
    path = tmp_path / 'ambiguous.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x =')
        page.insert_text((62, 54), '-2', fontsize=6)
        page.draw_line((62, 57), (75, 57), width=.4)
        page.insert_text((62, 63), '3', fontsize=6)
        doc.save(path)
    page = load_pdf_formulas(path, recognizer=CandidateDouble(r'x=\frac{-2}{3}'))[0]
    assert page['formulas'][0]['sign_consistency']['status'] == 'uncertain'
    assert page['formulas'][0]['warnings']


def test_default_search_remains_native_only():
    from src.retrieval.search_pdf import search_pdf, format_search_results
    results = search_pdf(SOURCE, 'K')
    assert results
    assert all(f['recognition_status'] == 'not_requested' and f['sign_consistency']['status'] == 'not_checked'
               for result in results for f in result['formulas'])
    rendered = format_search_results(results, 'K')
    assert 'source_minus_candidate_conflict' not in rendered
    assert 'source_minus_scope_uncertain' not in rendered


@pytest.mark.parametrize('latex', [r'2^{-\frac{2}{3}} + 2^{4}', r'2^{-\frac{2}{3}+\frac{4}{5}}'])
def test_repeated_candidate_scopes_cannot_prove_sign_association(latex):
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(latex)))
    assert formula['sign_consistency']['status'] == 'uncertain'
    assert formula['warnings']


def test_raw_candidate_whitespace_is_preserved():
    raw = '  ' + EVIDENCE['calls'][0]['raw_latex'] + '\n'
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(raw)))
    assert formula['raw_latex'] == raw
    assert formula['warnings'][0]['raw_candidate'] == raw


def test_retrieval_formats_structured_warning_when_excerpt_is_absent():
    from src.retrieval.search_pdf import BM25Searcher, format_search_results
    from src.chunking.text_chunker import chunk_formula_pages
    pages = load_pdf_formulas(SOURCE, recognizer=CandidateDouble(EVIDENCE['calls'][0]['raw_latex']))
    chunks = chunk_formula_pages(pages)
    results = BM25Searcher(chunks).search('K')
    result = next(r for r in results if r['formula_groups'])
    result['excerpt'] = ''
    rendered = format_search_results([result], 'K')
    assert 'source_minus_candidate_conflict' in rendered
    assert 'span_id' in rendered


@pytest.mark.parametrize(('script_y', 'latex', 'expected'), [
    (52, 'x=a^{-3}', 'no_conflict_detected'),
    (68, 'x=a_{-3}', 'no_conflict_detected'),
    (68, 'x=a^{-3}', 'mismatch'),
    (52, 'x=a^{3}', 'mismatch'),
])
def test_super_and_subscript_scopes_are_distinct(tmp_path, script_y, latex, expected):
    path = tmp_path / 'script.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = a')
        page.insert_text((65, script_y), '-3', fontsize=6)
        doc.save(path)
    page = load_pdf_formulas(path, recognizer=CandidateDouble(latex))[0]
    assert page['formulas'][0]['sign_consistency']['status'] == expected
    assert not page['formulas'][0]['answer_eligible']


@pytest.mark.parametrize('latex', ['x=a^{3}', r'x=a\cdot3'])
def test_positive_native_script_and_legitimate_multiply_are_not_sign_conflicts(tmp_path, latex):
    path = tmp_path / 'positive.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = a')
        page.insert_text((65, 52), '3', fontsize=6)
        doc.save(path)
    formula = load_pdf_formulas(path, recognizer=CandidateDouble(latex))[0]['formulas'][0]
    assert formula['sign_consistency']['status'] == 'not_checked'
    assert formula['warnings'] == []
    assert not formula['answer_eligible']


def test_different_script_terms_do_not_share_minus_evidence(tmp_path):
    path = tmp_path / 'terms.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = a')
        page.insert_text((65, 52), '-3', fontsize=6)
        page.insert_text((76, 60), '+b')
        page.insert_text((91, 52), '-4', fontsize=6)
        doc.save(path)
    for latex, expected in [('x=a^{-3}+b^{-4}', 'no_conflict_detected'),
                            ('x=a^{3}+b^{-4}', 'mismatch')]:
        formula = load_pdf_formulas(path, recognizer=CandidateDouble(latex))[0]['formulas'][0]
        assert formula['sign_consistency']['status'] == expected


def test_equally_close_native_script_bases_are_uncertain(tmp_path):
    path = tmp_path / 'ambiguous_bases.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = a')
        end = 40 + pymupdf.get_text_length('x = a')
        page.insert_text((end - pymupdf.get_text_length('b', fontname='cour'), 60),
                         'b', fontname='cour')  # distinct spans with overlapping base endpoints
        page.insert_text((65, 52), '-3', fontsize=6)
        doc.save(path)
    formula = load_pdf_formulas(path, recognizer=CandidateDouble('x=a^{-3}'))[0]['formulas'][0]
    assert formula['sign_consistency']['status'] == 'uncertain'
    assert formula['warnings']


def test_parenthesized_candidate_sign_is_uncertain_not_proven_missing():
    formula = negative_formula(load_pdf_formulas(SOURCE, recognizer=CandidateDouble(r'2^{(-\frac{2}{3})}')))
    assert formula['sign_consistency']['status'] == 'uncertain'
    assert formula['warnings'][0]['code'] == 'source_minus_scope_uncertain'


def test_same_size_raised_sign_is_uncertain_without_supported_native_attachment(tmp_path):
    path = tmp_path / 'same_size_script.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=200)
        page.insert_text((40, 60), 'x = a')
        page.insert_text((65, 54), '-3', fontsize=11)
        doc.save(path)
    formula = load_pdf_formulas(path, recognizer=CandidateDouble('x=a^{-3}'))[0]['formulas'][0]
    assert formula['sign_consistency']['status'] in ('uncertain', 'not_checked')
    assert not formula['answer_eligible']


def test_rotated_native_text_scope_is_unmeasured(tmp_path):
    path = tmp_path / 'rotated_text.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        page.insert_text((100, 200), 'x = y - 3', rotate=90)
        doc.save(path)
    formula = load_pdf_formulas(path, recognizer=CandidateDouble('x=y-3'))[0]['formulas'][0]
    assert formula['sign_consistency']['status'] == 'not_checked'
    assert formula['warnings'] == []
