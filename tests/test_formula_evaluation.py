"""Public synthetic regression fixtures and evaluation command tests.

Covers complete expression/group preservation for:
1. Fraction numerator and denominator
2. Root scope and vinculum coverage
3. Superscript and subscript attachment
4. Single absolute-value bars
5. Wrapped relation groups across line breaks

Clearly distinguishes deterministic fixture/stub tests from live OCR model accuracy.
Never uses private PDF coordinates or document paths.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pymupdf
import pytest

from src.evaluation.evaluate_formulas import (
    create_synthetic_benchmark_pdf,
    evaluate_formula_cases,
    evaluate_synthetic_benchmark,
    format_evaluation_report,
    BenchmarkStubRecognizer,
)
from src.ingestion.formula_loader import load_pdf_formulas


def test_synthetic_benchmark_covers_all_five_categories_and_passes(tmp_path):
    pdf_path = tmp_path / 'synthetic_benchmark.pdf'
    create_synthetic_benchmark_pdf(pdf_path)
    assert pdf_path.exists()

    pages = load_pdf_formulas(pdf_path, recognizer=BenchmarkStubRecognizer())
    assert len(pages) == 1
    page = pages[0]
    assert page['answer_eligible'] is False
    assert page['detection_status'] == 'native_heuristic_incomplete'

    results = evaluate_synthetic_benchmark(pages, is_stub=True)
    assert results['mode'] == 'DETERMINISTIC FIXTURE / RECOGNIZER-STUB'
    assert results['is_stub'] is True
    assert 'not certify' in results['note']

    counts = results['counts']
    assert counts['total'] == 5
    assert counts['pass'] == 5
    assert counts['fail'] == 0
    assert counts['ambiguous'] == 0
    assert counts['unmeasured'] == 0

    records_by_cat = {r['category']: r for r in results['records']}
    assert set(records_by_cat.keys()) == {
        'fraction_num_denom',
        'root_scope',
        'superscript_subscript',
        'single_abs_bars',
        'wrapped_relation',
    }

    # 1. Fraction: complete numerator, denominator, rule, and vertical span
    r_frac = records_by_cat['fraction_num_denom']
    assert r_frac['status'] == 'PASS'
    obs_frac = r_frac['observed']
    assert 'x^2 + 1' in obs_frac['original_text']
    assert '2y - 3' in obs_frac['original_text']
    assert obs_frac['rules_count'] >= 1
    assert (obs_frac['bbox'][3] - obs_frac['bbox'][1]) >= 25.0

    # 2. Root scope: radical symbol and full radicand
    r_root = records_by_cat['root_scope']
    assert r_root['status'] == 'PASS'
    obs_root = r_root['observed']
    assert '√' in obs_root['original_text']
    assert 'u + v' in obs_root['original_text']

    # 3. Superscript/subscript: base and script operands preserved
    r_script = records_by_cat['superscript_subscript']
    assert r_script['status'] == 'PASS'
    obs_script = r_script['observed']
    assert 'n-1' in obs_script['original_text']
    assert '2' in obs_script['original_text']
    assert obs_script['script_evidence'] is True

    # 4. Single absolute-value bars: both opening and closing bars preserved
    r_abs = records_by_cat['single_abs_bars']
    assert r_abs['status'] == 'PASS'
    obs_abs = r_abs['observed']
    orig_abs = obs_abs['original_text']
    assert orig_abs.startswith('|') or '|x' in orig_abs
    assert '| = 5' in orig_abs or orig_abs.endswith('|') or '|' in orig_abs.split('=')[0]

    # 5. Wrapped relation: 2 children linked in wrapped_relation group
    r_wrap = records_by_cat['wrapped_relation']
    assert r_wrap['status'] == 'PASS'
    obs_wrap = r_wrap['observed']
    assert obs_wrap['kind'] == 'wrapped_relation'
    assert len(obs_wrap['formula_ids']) == 2
    assert obs_wrap['answer_eligible'] is False


def test_single_absolute_value_bar_detection_and_preservation(tmp_path):
    path = tmp_path / 'abs_bars.pdf'
    doc = pymupdf.open()
    p = doc.new_page(width=400, height=300)
    p.insert_text((40, 50), '|x + 1| = 5')
    p.insert_text((40, 100), '|1 + y^3| = K')
    p.insert_text((40, 150), 'y = |x - 2|')
    p.insert_text((40, 200), '|a| < |b|')
    doc.save(str(path))
    doc.close()

    pages = load_pdf_formulas(path)
    assert len(pages) == 1
    formulas = pages[0]['formulas']
    # All 4 absolute value expressions are detected and preserved
    assert len(formulas) >= 4

    f_x = next(f for f in formulas if 'x + 1' in f['original_text'])
    assert '|' in f_x['original_text']
    assert f_x['original_text'].strip().startswith('|')
    assert '= 5' in f_x['original_text']
    assert f_x['answer_eligible'] is False

    f_y3 = next(f for f in formulas if '1 + y^3' in f['original_text'])
    assert '|' in f_y3['original_text']
    assert '= K' in f_y3['original_text']

    f_ineq = next(f for f in formulas if '<' in f['original_text'])
    assert '|a|' in f_ineq['original_text']
    assert '|b|' in f_ineq['original_text']


def test_fraction_complete_expression_preservation_not_substring(tmp_path):
    path = tmp_path / 'fraction_complete.pdf'
    doc = pymupdf.open()
    p = doc.new_page(width=400, height=300)
    p.insert_text((40, 60), 'f(x) =')
    p.insert_text((75, 55), 'numerator_term_alpha + beta')
    p.insert_text((75, 75), 'denominator_term_gamma + delta')
    p.draw_line((73, 62), (240, 62))
    doc.save(str(path))
    doc.close()

    pages = load_pdf_formulas(path)
    formula = pages[0]['formulas'][0]

    # Verify both full numerator AND full denominator expressions are completely preserved
    assert 'numerator_term_alpha' in formula['original_text']
    assert 'beta' in formula['original_text']
    assert 'denominator_term_gamma' in formula['original_text']
    assert 'delta' in formula['original_text']
    # Must have the division rule line recorded
    assert len(formula['native_rules']) == 1
    # Bounding box must span from above numerator to below denominator
    assert formula['bbox'][1] <= 55.0
    assert formula['bbox'][3] >= 75.0


def test_root_scope_complete_expression_preservation(tmp_path):
    path = tmp_path / 'root_scope.pdf'
    doc = pymupdf.open()
    p = doc.new_page(width=400, height=300)
    p.insert_text((40, 60), 'y =')
    writer = pymupdf.TextWriter(p.rect)
    writer.append((60, 62), '√', font=pymupdf.Font('symbol'), fontsize=16)
    writer.write_text(p)
    p.draw_line((72, 50), (160, 50))
    p.insert_text((75, 60), 'long_radicand_alpha + beta')
    doc.save(str(path))
    doc.close()

    pages = load_pdf_formulas(path)
    formula = pages[0]['formulas'][0]
    # Complete radicand expression preserved under root
    assert '√' in formula['original_text']
    assert 'long_radicand_alpha + beta' in formula['original_text']
    assert formula['bbox'][0] <= 40.0
    assert formula['bbox'][2] >= 155.0


def test_evaluation_failure_and_unmeasured_reporting(tmp_path):
    # Empty page has no formulas, should fail synthetic checks
    path = tmp_path / 'empty.pdf'
    doc = pymupdf.open()
    doc.new_page(width=300, height=300).insert_text((40, 40), 'Only plain prose here.')
    doc.save(str(path))
    doc.close()

    pages = load_pdf_formulas(path)
    results = evaluate_synthetic_benchmark(pages, is_stub=True)
    assert results['counts']['fail'] == 5
    assert results['counts']['pass'] == 0
    assert all(r['status'] == 'FAIL' for r in results['records'])


def test_evaluation_cli_execution_and_output(tmp_path):
    output_file = tmp_path / 'eval_report.json'
    result = subprocess.run(
        [sys.executable, '-m', 'src.evaluation.evaluate_formulas',
         '--output', str(output_file)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert 'FORMULA REGRESSION EVALUATION REPORT' in result.stdout
    assert 'DETERMINISTIC FIXTURE / RECOGNIZER-STUB' in result.stdout
    assert 'Total Cases: 5' in result.stdout
    assert 'PASS:        5' in result.stdout
    assert 'FAIL:        0' in result.stdout

    # Saved JSON report verified
    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding='utf-8'))
    assert data['counts']['pass'] == 5
    assert data['counts']['fail'] == 0
    assert len(data['records']) == 5
    assert data['is_stub'] is True


def test_mode_distinction_labels_fixture_stubs_vs_actual_ocr():
    # Deterministic stub mode
    stub_results = evaluate_formula_cases(is_stub=True)
    report_stub = format_evaluation_report(stub_results)
    assert 'DETERMINISTIC FIXTURE / RECOGNIZER-STUB' in report_stub
    assert 'does not certify end-to-end OCR model accuracy' in report_stub
    assert stub_results['counts']['pass'] == 5

    # Non-stub empty unmeasured mode
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        create_synthetic_benchmark_pdf(tmp_path)
        pages = load_pdf_formulas(tmp_path) # no recognizer
        results_native = evaluate_synthetic_benchmark(pages, is_stub=False)
        report_native = format_evaluation_report(results_native)
        assert 'ACTUAL OCR MODEL EVALUATION' in report_native
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
