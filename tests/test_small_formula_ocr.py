"""Known OCR failure replay, deliberately separate from model accuracy."""
import hashlib
import json
from pathlib import Path

from src.evaluation.evaluate_formulas import compare_latex_transcription


def test_known_negative_fractional_power_misread_is_rejected():
    # Saved CodeFormulaV2 output: a double, never a claim of live accuracy.
    data_dir = Path(__file__).resolve().parents[1] / 'eval_data'
    evidence = json.loads((data_dir / 'small_formula_ocr_density.json').read_text())
    source = data_dir / 'small_formula_ocr_original.pdf'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == evidence['source_sha256']
    raw = evidence['calls'][0]['raw_latex']
    assert raw == r'2 \cdot \frac { 2 } { 3 }'
    gold = r'2 ^ { - \frac { 2 } { 3 } }'
    assert compare_latex_transcription(raw, gold)[0] == 'FAIL'
    assert compare_latex_transcription(raw, r'2^{\frac{2}{3}}')[0] == 'FAIL'
    corrected = json.loads((data_dir / 'small_formula_ocr_corrected.json').read_text())
    assert hashlib.sha256((data_dir / 'small_formula_ocr_corrected.pdf').read_bytes()).hexdigest() == corrected['source_sha256']
    returned = corrected['calls'][0]['raw_latex']
    assert returned == r'2 ^ { \cdot \frac { 2 } { 3 } }'
    assert compare_latex_transcription(returned, gold)[0] == 'FAIL'
    positive = corrected['calls'][1]
    assert compare_latex_transcription(positive['raw_latex'], positive['gold'])[0] == 'PASS'


def test_public_fractional_exponent_has_clear_rule_and_raised_denominator(tmp_path):
    import pymupdf
    from src.evaluation.evaluate_formulas import create_synthetic_benchmark_pdf
    from src.ingestion.formula_loader import load_pdf_formulas

    path = create_synthetic_benchmark_pdf(tmp_path / 'source.pdf')
    page_data = load_pdf_formulas(path)[0]
    group = page_data['formula_groups'][0]
    rhs = group['formulas'][1]
    with pymupdf.open(path) as doc:
        page = doc[0]
        spans = [s for b in page.get_text('dict')['blocks']
                 for line in b.get('lines', []) for s in line['spans']]
        base = next(s for s in spans if s['origin'][0] == 40 and s['text'] == '2')
        denominator = next(s for s in spans if s['origin'][0] == 51 and s['text'] == '3')
        numerator = next(s for s in spans if s['origin'][0] == 51 and s['text'] == '2')
        minus = next(s for s in spans if s['origin'][0] == 46 and s['text'] == '-')
        assert denominator['origin'][1] <= base['origin'][1] - 4
        assert all(s['origin'][1] < base['origin'][1] for s in (numerator, minus, denominator))
        # A white strip below the rule must separate it from denominator ink.
        gap = page.get_pixmap(matrix=pymupdf.Matrix(20, 20),
                             clip=pymupdf.Rect(51, 293.7, 56, 294.3), alpha=False)
        assert min(gap.samples) > 240
        above = page.get_pixmap(matrix=pymupdf.Matrix(20, 20),
                               clip=pymupdf.Rect(51, 291.5, 56, 292.6), alpha=False)
        assert min(above.samples) > 240
        bbox = pymupdf.Rect(rhs['bbox'])
        assert all(bbox.contains(pymupdf.Rect(s['bbox']))
                   for s in (base, numerator, minus, denominator))
        assert bbox.x1 < 60  # adjacent prose excluded
    assert rhs['native_script_evidence']
    assert group['answer_eligible'] is False


def test_cross_block_wrap_keeps_children_prose_and_failure_state(tmp_path):
    import hashlib
    import pymupdf
    from src.chunking.text_chunker import chunk_formula_pages
    from src.evaluation.evaluate_formulas import create_synthetic_benchmark_pdf
    from src.ingestion.formula_loader import load_pdf_formulas
    from src.ingestion.formula_regions import propose_regions

    path = create_synthetic_benchmark_pdf(tmp_path / 'wrap.pdf')
    page = load_pdf_formulas(path)[0]
    group = page['formula_groups'][0]
    lhs, rhs = group['formulas']
    with pymupdf.open(path) as doc:
        _, spans, _ = propose_regions(doc[0])
    assert spans[max(lhs['span_ids'])]['block_idx'] != spans[min(rhs['span_ids'])]['block_idx']
    assert group['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not group['answer_eligible'] and group['state'] == 'unresolved'
    assert group['latex'] is None
    prose = [b['text'] for b in page['blocks'] if b['kind'] == 'text']
    assert prose[-2:] == [' on the branch of the initial point. The condition gives ', '. Cubing yields.']
    assert not any(word in rhs['original_text'] for word in ('Cubing', 'g(x)', 'condition'))
    chunks = chunk_formula_pages([page])
    chunk = next(c for c in chunks if c.get('formula_groups'))
    assert chunk['formulas'] == group['formulas']
    assert [f['bbox'] for f in chunk['formulas']] == [lhs['bbox'], rhs['bbox']]
    assert all(f['source_sha256'] == group['source_sha256'] for f in chunk['formulas'])
    assert not chunk['answer_eligible']

    class FailedRecognition:
        def recognize(self, png):
            raise TimeoutError('explicit test double; no real OCR')

    failed = load_pdf_formulas(path, recognizer=FailedRecognition())[0]['formula_groups'][0]
    assert failed['latex'] is None and failed['state'] == 'unresolved'
    assert not failed['answer_eligible']
    assert all(f['recognition_status'] == 'failed' for f in failed['formulas'])


def test_cross_block_wrap_rejects_distinct_blocks_columns_cells_and_omissions(tmp_path):
    from copy import deepcopy
    import pymupdf
    from src.evaluation.evaluate_formulas import create_synthetic_benchmark_pdf
    from src.ingestion.formula_loader import detect_wrapped_relations, load_pdf_formulas
    from src.ingestion.formula_regions import propose_regions

    path = create_synthetic_benchmark_pdf(tmp_path / 'guards.pdf')
    formulas = load_pdf_formulas(path)[0]['formulas'][-2:]
    with pymupdf.open(path) as doc:
        _, spans, _ = propose_regions(doc[0])
    assert len(detect_wrapped_relations(formulas, spans)) == 1
    a, b = max(formulas[0]['span_ids']), min(formulas[1]['span_ids'])
    for scenario in ('nonadjacent_block', 'different_column', 'left_edge_paragraph',
                     'middle_of_block', 'overlapping_blocks', 'table_cell', 'omitted_prose',
                     'complete_equation', 'large_gap', 'missing_geometry'):
        fs, ss = deepcopy(formulas), deepcopy(spans)
        if scenario == 'nonadjacent_block':
            ss[b]['block_idx'] += 1
        elif scenario == 'different_column':
            box = ss[b]['block_bbox']
            ss[b]['block_bbox'] = pymupdf.Rect(box.x0 + 100, box.y0, box.x1 + 100, box.y1)
        elif scenario == 'left_edge_paragraph':
            ss[a]['block_bbox'].x0 = ss[a]['bbox'].x0
        elif scenario == 'middle_of_block':
            ss[b]['line_idx'] = 1
        elif scenario == 'overlapping_blocks':
            ss[b]['block_bbox'].y0 = ss[a]['block_bbox'].y1 - 1
        elif scenario == 'table_cell':
            ss[b]['cell'] = {'row': 0, 'column': 1}
        elif scenario == 'omitted_prose':
            fs[0]['span_ids'] = [a - 1]  # real source K= span is now unowned
        elif scenario == 'complete_equation':
            fs[0]['original_text'] = 'K = 1'
        elif scenario == 'large_gap':
            ss[b]['line_bbox'].y0 += 100
        elif scenario == 'missing_geometry':
            ss[b]['line_bbox'] = None
        assert detect_wrapped_relations(fs, ss) == [], scenario
