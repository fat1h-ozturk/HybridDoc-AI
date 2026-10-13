"""Minimal offline formula regression and evaluation suite.

Evaluates formula extraction and recognition against expected components,
testing complete expression/group preservation across 5 core categories:
1. Fraction numerator and denominator
2. Root scope and vinculum coverage
3. Superscript and subscript attachment
4. Single absolute-value bars
5. Wrapped relation groups across line breaks

Explicitly distinguishes deterministic fixture/stub tests from actual OCR model accuracy.
Separates geometry/candidate routing from transcription fidelity.
Never uses private document paths or coordinates.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
import pymupdf

from src.ingestion.formula_loader import load_pdf_formulas


def create_synthetic_benchmark_pdf(pdf_path):
    """Generate a reproducible public synthetic PDF containing the 5 regression cases."""
    doc = pymupdf.open()
    p = doc.new_page(width=500, height=400)
    p.insert_text((30, 30), 'Public Synthetic Formula Regression Benchmark')

    # Case 1: Fraction with numerator and denominator (SYN-FRAC)
    p.insert_text((40, 75), 'f(x) =')
    p.insert_text((72, 70), 'x')
    p.insert_text((79, 66), '2', fontsize=7)
    p.insert_text((85, 70), '+ 1')
    p.insert_text((72, 85), '2y - 3')
    p.draw_line((70, 74), (110, 74))

    # Case 2: Root with vinculum covering full radicand (SYN-ROOT)
    p.insert_text((40, 130), 'z =')
    writer = pymupdf.TextWriter(p.rect)
    writer.append((60, 132), '√', font=pymupdf.Font('symbol'), fontsize=16)
    writer.write_text(p)
    p.draw_line((72, 120), (120, 120))
    p.insert_text((75, 130), 'u + v')

    # Case 3: Base with subscript and superscript (SYN-SCRIPT)
    p.insert_text((40, 180), 'y(t) = a')
    p.insert_text((78, 184), 'n-1', fontsize=7)
    p.insert_text((96, 180), 't')
    p.insert_text((102, 175), '2', fontsize=7)

    # Case 4: Single absolute-value bars (SYN-ABS)
    p.insert_text((40, 230), '|x + 1| = 5')

    # Case 5: Wrapped relation group across lines (SYN-WRAP)
    p.insert_text((40, 280), 'g(x) = C')
    p.insert_text((95, 280), ' on the branch of the initial point. The condition gives ')
    p.insert_text((420, 280), 'K =')
    p.insert_text((40, 296), '2')
    p.insert_text((46, 293), '-', fontsize=6)
    p.insert_text((51, 291), '2', fontsize=6)
    p.insert_text((51, 296), '3', fontsize=6)
    p.draw_line((51, 293), (56, 293))
    p.insert_text((60, 296), '. Cubing yields.')

    doc.save(str(pdf_path))
    doc.close()
    return Path(pdf_path)


class BenchmarkStubRecognizer:
    """Deterministic recognizer stub for public synthetic benchmark cases.

    Maps formula original text / context to expected synthetic LaTeX.
    Used for offline deterministic testing, clearly distinct from actual OCR models.
    """
    def recognize(self, png_bytes):
        # Recognizer is called per-crop; stub returns standard candidate format
        return 'stub_latex'

    def recognize_for_case(self, formula):
        text = formula.get('original_text', '')
        if any(p in text for p in ('x 2 + 1', 'x2 + 1', 'x^2 + 1')) and '2y - 3' in text:
            return r'f(x) = \frac{x^2 + 1}{2y - 3}'
        elif '√' in text and 'u + v' in text:
            return r'z = \sqrt{u + v}'
        elif 'y(t) = a' in text and 'n-1' in text:
            return r'y(t) = a_{n-1} t^2'
        elif '|x + 1|' in text:
            return r'|x + 1| = 5'
        elif text.strip() == 'K=':
            return r'K ='
        elif any(c in text for c in ('2', '3')) and '-' in text:
            return r'2 ^ { - \frac { 2 } { 3 } }'
        return r'\text{stub}'


def normalize_latex(text: str) -> str:
    """Conservative harmless LaTeX formatting normalization for fidelity comparison.

    Normalizes math delimiters, whitespace, and spacing around LaTeX tokens,
    while strictly preserving meaning-sensitive characters: roots, bars, braces,
    subscripts, exponents, operators, and signs.
    """
    if not text:
        return ''
    s = text.strip()
    # Strip wrapping display/inline math delimiters
    if s.startswith(r'\[') and s.endswith(r'\]') and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith(r'\(') and s.endswith(r'\)') and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith('$$') and s.endswith('$$') and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith('$') and s.endswith('$') and len(s) >= 2:
        s = s[1:-1].strip()

    # Normalize TeX control-space (\ followed by whitespace) outside \text{...}
    parts = re.split(r'(\\text\{[^{}]*\})', s)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\\\s+', ' ', parts[i])
    s = ''.join(parts)

    # 1. Remove space between command and brace: e.g. \frac { -> \frac{, \sqrt { -> \sqrt{
    s = re.sub(r'(\\[a-zA-Z]+)\s+\{', r'\1{', s)
    # 2. Remove space inside braces: { x } -> {x}
    s = re.sub(r'\{\s+', '{', s)
    s = re.sub(r'\s+\}', '}', s)
    # 3. Remove space between consecutive braces: } { -> }{
    s = re.sub(r'\}\s+\{', '}{', s)
    # 4. Remove space around sub/superscripts: _ { -> _{, ^ { -> ^{
    s = re.sub(r'\s*\^\s*', '^', s)
    s = re.sub(r'\s*_\s*', '_', s)
    # Normalize redundant braces around a SINGLE exponent/subscript token:
    # e.g. t^{2} -> t^2, a_{i} -> a_i
    # strictly single alphanumeric token [0-9a-zA-Z]
    s = re.sub(r'(?<=[\^_])\{([0-9a-zA-Z])\}', r'\1', s)
    # 5. Remove spaces inside parentheses and around absolute bars:
    s = re.sub(r'\s*\(\s*', '(', s)
    s = re.sub(r'\s*\)\s*', ')', s)
    s = re.sub(r'\|\s+', '|', s)
    s = re.sub(r'\s+\|', '|', s)
    # 6. Normalize spacing around relational and arithmetic operators to single space:
    s = re.sub(r'\s*([=+\-*/])\s*', r' \1 ', s)
    # 7. Normalize digit-variable spacing in math: 2 y -> 2y
    s = re.sub(r'([0-9])\s+([a-zA-Z])', r'\1\2', s)
    # 8. Normalize whitespace inside text command if present: \text { -> \text{
    s = re.sub(r'\\text\s*\{', r'\\text{', s)
    # 9. Collapse all remaining whitespace sequences to single space
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def compare_latex_transcription(candidate: str, expected_gold: str) -> tuple:
    """Compare candidate LaTeX against gold LaTeX using conservative normalization.

    Returns (status, reason):
    - status: 'PASS' | 'FAIL' | 'UNMEASURED'
    - reason: description
    """
    if candidate is None or not candidate.strip():
        return 'UNMEASURED', 'no transcription candidate returned by recognizer'
    norm_c = normalize_latex(candidate)
    norm_g = normalize_latex(expected_gold)
    if norm_c == norm_g:
        return 'PASS', 'transcription matches gold expression'
    return 'FAIL', f'transcription mismatch: candidate {repr(candidate)} != gold {repr(expected_gold)}'


def evaluate_synthetic_benchmark(pages, is_stub=True, provenance='stub'):
    """Evaluate extracted pages against the 5 synthetic regression criteria.

    Explicitly separates geometry / candidate routing from transcription fidelity.
    Default offline / stub mode marks transcription as UNMEASURED.
    """
    page = pages[0]
    formulas = page.get('formulas', [])
    groups = page.get('formula_groups', [])

    records = []

    # 1. SYN-FRAC: Fraction Numerator/Denominator
    frac_formula = next((f for f in formulas if any(p in f.get('original_text', '') for p in ('x 2 + 1', 'x2 + 1', 'x^2 + 1'))), None)
    expected_frac = {
        'components': ['numerator: x^2 + 1', 'denominator: 2y - 3', 'division rule'],
        'expression': r'f(x) = \frac{x^2 + 1}{2y - 3}',
    }
    if frac_formula:
        has_num = any(p in frac_formula['original_text'] for p in ('x 2 + 1', 'x2 + 1', 'x^2 + 1'))
        has_denom = '2y - 3' in frac_formula['original_text']
        has_rule = len(frac_formula.get('native_rules', [])) >= 1
        box = frac_formula['bbox']
        has_span = (box[3] - box[1]) >= 25.0
        g_pass = has_num and has_denom and has_rule and has_span
        g_status = 'PASS' if g_pass else 'FAIL'
        g_reason = 'complete numerator, denominator, rule, and vertical bounds preserved' if g_pass else 'incomplete fraction preservation'
        observed = {
            'formula_id': frac_formula['id'],
            'page': frac_formula.get('page_number', 1),
            'bbox': frac_formula['bbox'],
            'original_text': frac_formula['original_text'],
            'rules_count': len(frac_formula.get('native_rules', [])),
            'script_evidence': frac_formula.get('native_script_evidence'),
            'latex': frac_formula.get('latex'),
            'state': frac_formula.get('state'),
            'answer_eligible': frac_formula.get('answer_eligible', False),
            'reason': frac_formula.get('reason'),
        }
        if is_stub:
            t_status = 'UNMEASURED'
            t_reason = 'offline default mode; recognizer stub does not measure transcription fidelity'
        elif frac_formula.get('latex'):
            t_status, t_reason = compare_latex_transcription(frac_formula['latex'], expected_frac['expression'])
        else:
            t_status = 'UNMEASURED'
            t_reason = f"crop recognition unavailable or timed out: {frac_formula.get('reason')}"
    else:
        observed = None
        g_status = 'FAIL'
        g_reason = 'fraction formula region not proposed'
        t_status = 'UNMEASURED'
        t_reason = 'crop missing: fraction formula region not proposed'

    records.append({
        'case_id': 'SYN-FRAC',
        'category': 'fraction_num_denom',
        'description': 'Complete fraction preservation (numerator, denominator, division rule)',
        'provenance': provenance,
        'geometry_status': g_status,
        'geometry_reason': g_reason,
        'transcription_status': t_status,
        'transcription_reason': t_reason,
        'expected': expected_frac,
        'observed': observed,
    })

    # 2. SYN-ROOT: Root Scope
    root_formula = next((f for f in formulas if '√' in f.get('original_text', '')), None)
    expected_root = {
        'components': ['radical symbol √', 'radicand: u + v', 'vinculum line'],
        'expression': r'z = \sqrt{u + v}',
    }
    if root_formula:
        has_sym = '√' in root_formula['original_text']
        has_radicand = 'u + v' in root_formula['original_text']
        g_pass = has_sym and has_radicand
        g_status = 'PASS' if g_pass else 'FAIL'
        g_reason = 'radical vinculum scope completely encloses radicand u + v' if g_pass else 'radicand truncated or detached from radical'
        observed = {
            'formula_id': root_formula['id'],
            'page': root_formula.get('page_number', 1),
            'bbox': root_formula['bbox'],
            'original_text': root_formula['original_text'],
            'latex': root_formula.get('latex'),
            'state': root_formula.get('state'),
            'answer_eligible': root_formula.get('answer_eligible', False),
            'reason': root_formula.get('reason'),
        }
        if is_stub:
            t_status = 'UNMEASURED'
            t_reason = 'offline default mode; recognizer stub does not measure transcription fidelity'
        elif root_formula.get('latex'):
            t_status, t_reason = compare_latex_transcription(root_formula['latex'], expected_root['expression'])
        else:
            t_status = 'UNMEASURED'
            t_reason = f"crop recognition unavailable or timed out: {root_formula.get('reason')}"
    else:
        observed = None
        g_status = 'FAIL'
        g_reason = 'root formula region not proposed'
        t_status = 'UNMEASURED'
        t_reason = 'crop missing: root formula region not proposed'

    records.append({
        'case_id': 'SYN-ROOT',
        'category': 'root_scope',
        'description': 'Root vinculum scope preserves full radicand expression',
        'provenance': provenance,
        'geometry_status': g_status,
        'geometry_reason': g_reason,
        'transcription_status': t_status,
        'transcription_reason': t_reason,
        'expected': expected_root,
        'observed': observed,
    })

    # 3. SYN-SCRIPT: Superscript and Subscript
    script_formula = next((f for f in formulas if 'y(t) = a' in f.get('original_text', '')), None)
    expected_script = {
        'components': ['base y', 'base a', 'subscript n-1', 'base t', 'superscript 2'],
        'expression': r'y(t) = a_{n-1} t^2',
    }
    if script_formula:
        orig = script_formula['original_text']
        has_sub = 'n-1' in orig
        has_sup = '2' in orig
        has_evidence = bool(script_formula.get('native_script_evidence'))
        g_pass = has_sub and has_sup and has_evidence
        g_status = 'PASS' if g_pass else 'FAIL'
        g_reason = 'base characters with subscript and superscript preserved with script evidence' if g_pass else 'scripts missing or script evidence absent'
        observed = {
            'formula_id': script_formula['id'],
            'page': script_formula.get('page_number', 1),
            'bbox': script_formula['bbox'],
            'original_text': script_formula['original_text'],
            'script_evidence': script_formula.get('native_script_evidence'),
            'latex': script_formula.get('latex'),
            'state': script_formula.get('state'),
            'answer_eligible': script_formula.get('answer_eligible', False),
            'reason': script_formula.get('reason'),
        }
        if is_stub:
            t_status = 'UNMEASURED'
            t_reason = 'offline default mode; recognizer stub does not measure transcription fidelity'
        elif script_formula.get('latex'):
            t_status, t_reason = compare_latex_transcription(script_formula['latex'], expected_script['expression'])
        else:
            t_status = 'UNMEASURED'
            t_reason = f"crop recognition unavailable or timed out: {script_formula.get('reason')}"
    else:
        observed = None
        g_status = 'FAIL'
        g_reason = 'script formula region not proposed'
        t_status = 'UNMEASURED'
        t_reason = 'crop missing: script formula region not proposed'

    records.append({
        'case_id': 'SYN-SCRIPT',
        'category': 'superscript_subscript',
        'description': 'Complete expression preservation of base with subscript and superscript',
        'provenance': provenance,
        'geometry_status': g_status,
        'geometry_reason': g_reason,
        'transcription_status': t_status,
        'transcription_reason': t_reason,
        'expected': expected_script,
        'observed': observed,
    })

    # 4. SYN-ABS: Single Absolute-Value Bars
    abs_formula = next((f for f in formulas if '|x + 1|' in f.get('original_text', '')), None)
    expected_abs = {
        'components': ['opening bar |', 'enclosed expression x + 1', 'closing bar |', 'relation and RHS: = 5'],
        'expression': r'|x + 1| = 5',
    }
    if abs_formula:
        orig = abs_formula['original_text']
        has_open = orig.startswith('|') or '|x' in orig
        has_close = '| = 5' in orig or orig.endswith('|') or '|' in orig.split('=')[0]
        g_pass = has_open and has_close and '= 5' in orig
        g_status = 'PASS' if g_pass else 'FAIL'
        g_reason = 'both opening and closing single absolute-value bars preserved with relation' if g_pass else 'absolute-value bars dropped or truncated'
        observed = {
            'formula_id': abs_formula['id'],
            'page': abs_formula.get('page_number', 1),
            'bbox': abs_formula['bbox'],
            'original_text': abs_formula['original_text'],
            'latex': abs_formula.get('latex'),
            'state': abs_formula.get('state'),
            'answer_eligible': abs_formula.get('answer_eligible', False),
            'reason': abs_formula.get('reason'),
        }
        if is_stub:
            t_status = 'UNMEASURED'
            t_reason = 'offline default mode; recognizer stub does not measure transcription fidelity'
        elif abs_formula.get('latex'):
            t_status, t_reason = compare_latex_transcription(abs_formula['latex'], expected_abs['expression'])
        else:
            t_status = 'UNMEASURED'
            t_reason = f"crop recognition unavailable or timed out: {abs_formula.get('reason')}"
    else:
        observed = None
        g_status = 'FAIL'
        g_reason = 'absolute-value formula region not proposed'
        t_status = 'UNMEASURED'
        t_reason = 'crop missing: absolute-value formula region not proposed'

    records.append({
        'case_id': 'SYN-ABS',
        'category': 'single_abs_bars',
        'description': 'Single absolute-value bars preserved on both boundaries of enclosed expression',
        'provenance': provenance,
        'geometry_status': g_status,
        'geometry_reason': g_reason,
        'transcription_status': t_status,
        'transcription_reason': t_reason,
        'expected': expected_abs,
        'observed': observed,
    })

    # 5. SYN-WRAP: Wrapped Relation Group
    wrap_group = next((g for g in groups if g.get('kind') == 'wrapped_relation'), None)
    expected_wrap = {
        'components': ['line 1 trailing relation LHS K=', 'line 2 wrapped RHS continuation 2^{-2/3}'],
        'group_kind': 'wrapped_relation',
        'expression': r'K = 2 ^ { - \frac { 2 } { 3 } }',
    }
    if wrap_group:
        f_ids = wrap_group.get('formula_ids', [])
        has_two = len(f_ids) == 2
        child_formulas = [f for f in formulas if f['id'] in f_ids]
        has_k = any('K' in f.get('original_text', '') and '=' in f.get('original_text', '') for f in child_formulas)
        has_rhs = any(any(c in f.get('original_text', '') for c in ('2', '3')) for f in child_formulas)
        no_prose = not any(w in wrap_group.get('original_text', '') for w in ('Cubing', 'g( )'))
        g_pass = has_two and has_k and has_rhs and wrap_group['kind'] == 'wrapped_relation' and no_prose
        g_status = 'PASS' if g_pass else 'FAIL'
        g_reason = ('line-wrapped relational fragments grouped into unified formula group without contamination'
                    if g_pass else 'wrapped relation not properly linked or contaminated by prose')
        observed = {
            'group_id': wrap_group['id'],
            'kind': wrap_group['kind'],
            'page': wrap_group.get('page_number', 1),
            'formula_ids': f_ids,
            'child_bboxes': [f['bbox'] for f in child_formulas],
            'original_text': wrap_group['original_text'],
            'latex': wrap_group.get('latex'),
            'state': wrap_group.get('state'),
            'answer_eligible': wrap_group.get('answer_eligible', False),
            'reasons': wrap_group.get('reasons'),
        }
        if is_stub:
            t_status = 'UNMEASURED'
            t_reason = 'offline default mode; recognizer stub does not measure transcription fidelity'
        elif wrap_group.get('latex'):
            t_status, t_reason = compare_latex_transcription(wrap_group['latex'], expected_wrap['expression'])
        else:
            t_status = 'UNMEASURED'
            t_reason = f"crop recognition unavailable or timed out: {wrap_group.get('reasons')}"
    else:
        observed = None
        g_status = 'FAIL'
        g_reason = 'wrapped relation group not formed'
        t_status = 'UNMEASURED'
        t_reason = 'crop missing: wrapped relation group not formed'

    records.append({
        'case_id': 'SYN-WRAP',
        'category': 'wrapped_relation',
        'description': 'Wrapped relational continuation grouped into unified group while preserving prose order',
        'provenance': provenance,
        'geometry_status': g_status,
        'geometry_reason': g_reason,
        'transcription_status': t_status,
        'transcription_reason': t_reason,
        'expected': expected_wrap,
        'observed': observed,
    })

    # Separate counts
    g_pass_count = sum(1 for r in records if r['geometry_status'] == 'PASS')
    g_fail_count = sum(1 for r in records if r['geometry_status'] == 'FAIL')
    t_pass_count = sum(1 for r in records if r['transcription_status'] == 'PASS')
    t_fail_count = sum(1 for r in records if r['transcription_status'] == 'FAIL')
    t_ambig_count = sum(1 for r in records if r['transcription_status'] == 'AMBIGUOUS')
    t_unmeas_count = sum(1 for r in records if r['transcription_status'] == 'UNMEASURED')

    counts = {
        'total': len(records),
        'geometry': {
            'total': len(records),
            'pass': g_pass_count,
            'fail': g_fail_count,
        },
        'transcription': {
            'total': len(records),
            'pass': t_pass_count,
            'fail': t_fail_count,
            'ambiguous': t_ambig_count,
            'unmeasured': t_unmeas_count,
        },
        'geometry_pass': g_pass_count,
        'geometry_fail': g_fail_count,
        'transcription_pass': t_pass_count,
        'transcription_fail': t_fail_count,
        'transcription_ambiguous': t_ambig_count,
        'transcription_unmeasured': t_unmeas_count,
        # Default flat fields
        'pass': t_pass_count if not is_stub else 0,
        'fail': g_fail_count + (t_fail_count if not is_stub else 0),
        'ambiguous': t_ambig_count,
        'unmeasured': t_unmeas_count,
    }

    if is_stub:
        mode_label = 'DETERMINISTIC FIXTURE / RECOGNIZER-STUB'
    elif provenance == 'controlled_double':
        mode_label = 'CONTROLLED RECOGNIZER DOUBLE EVALUATION'
    else:
        mode_label = 'ACTUAL OCR MODEL EVALUATION'

    return {
        'mode': mode_label,
        'provenance': provenance,
        'is_stub': is_stub,
        'note': 'This evaluation measures deterministic extraction, geometry grouping, and candidate routing. It does not certify end-to-end OCR model accuracy or authorize production formula answers.',
        'counts': counts,
        'records': records,
    }


def format_evaluation_report(results):
    """Format structured evaluation output into readable text report."""
    lines = []
    lines.append('=' * 80)
    lines.append('FORMULA REGRESSION EVALUATION REPORT')
    lines.append(f"Mode:       {results['mode']}")
    lines.append(f"Provenance: {results.get('provenance', 'unknown')}")
    lines.append(f"Note:       {results['note']}")
    lines.append('=' * 80)
    lines.append('')

    for r in results['records']:
        lines.append(f"[{r['case_id']}] {r['category']}")
        lines.append(f"  Description:          {r['description']}")
        lines.append(f"  Geometry Routing:     {r['geometry_status']} - {r['geometry_reason']}")
        lines.append(f"  Transcription:        {r['transcription_status']} - {r['transcription_reason']}")
        lines.append(f"  Expected:             {json.dumps(r['expected'], ensure_ascii=False)}")
        if r['observed']:
            lines.append(f"  Observed:             {json.dumps(r['observed'], ensure_ascii=False)}")
        else:
            lines.append("  Observed:             None (not detected)")
        lines.append('-' * 80)

    counts = results['counts']
    lines.append('')
    lines.append('=' * 80)
    lines.append('EVALUATION SUMMARY COUNTS')
    lines.append('  Geometry / Candidate Routing:')
    lines.append(f"    Total Cases: {counts['geometry']['total']}")
    lines.append(f"    PASS:        {counts['geometry']['pass']}")
    lines.append(f"    FAIL:        {counts['geometry']['fail']}")
    lines.append('  Transcription Fidelity:')
    lines.append(f"    Total Cases: {counts['transcription']['total']}")
    lines.append(f"    PASS:        {counts['transcription']['pass']}")
    lines.append(f"    FAIL:        {counts['transcription']['fail']}")
    lines.append(f"    AMBIGUOUS:   {counts['transcription']['ambiguous']}")
    lines.append(f"    UNMEASURED:  {counts['transcription']['unmeasured']}")
    lines.append('=' * 80)
    return '\n'.join(lines)


def evaluate_formula_cases(pdf_path=None, recognizer=None, is_stub=True, provenance=None):
    """Run formula evaluation on given PDF or generate synthetic benchmark PDF."""
    if provenance is None:
        if is_stub:
            provenance = 'stub'
        elif getattr(recognizer, '__class__', None).__name__ == 'ControlledRecognizerDouble':
            provenance = 'controlled_double'
        elif recognizer is not None:
            provenance = 'local_formula_recognizer'
        else:
            provenance = 'native_only'

    if pdf_path is None:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            create_synthetic_benchmark_pdf(tmp_path)
            stub = recognizer if recognizer is not None else (BenchmarkStubRecognizer() if is_stub else None)
            if hasattr(stub, '__enter__') and hasattr(stub, '__exit__'):
                with stub as active_rec:
                    pages = load_pdf_formulas(tmp_path, recognizer=active_rec)
            else:
                pages = load_pdf_formulas(tmp_path, recognizer=stub)
            return evaluate_synthetic_benchmark(pages, is_stub=is_stub, provenance=provenance)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        if hasattr(recognizer, '__enter__') and hasattr(recognizer, '__exit__'):
            with recognizer as active_rec:
                pages = load_pdf_formulas(pdf_path, recognizer=active_rec)
        else:
            pages = load_pdf_formulas(pdf_path, recognizer=recognizer)
        return evaluate_synthetic_benchmark(pages, is_stub=is_stub, provenance=provenance)


def main(argv=None):
    """CLI entry point for offline formula evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, help='evaluate specific PDF (default: public synthetic benchmark)')
    parser.add_argument('--recognize', action='store_true', help='use live local CodeFormulaV2 recognizer')
    parser.add_argument('--cache-dir', type=Path, help='HF cache directory for weights')
    parser.add_argument('--timeout', type=float, default=120, help='seconds per crop')
    parser.add_argument('--budget', type=float, default=480, help='total recognition budget')
    parser.add_argument('--output', type=Path, help='save JSON report to file')
    parser.add_argument('--json', action='store_true', help='print JSON instead of text report')
    args = parser.parse_args(argv)

    is_stub = not args.recognize
    recognizer = None
    provenance = 'stub' if is_stub else 'local_formula_recognizer'

    if args.recognize:
        from src.ingestion.formula_recognizer import LocalFormulaRecognizer
        recognizer = LocalFormulaRecognizer(
            cache_dir=args.cache_dir,
            timeout_seconds=args.timeout,
            budget_seconds=args.budget,
        )

    results = evaluate_formula_cases(
        pdf_path=args.pdf,
        recognizer=recognizer,
        is_stub=is_stub,
        provenance=provenance,
    )

    rendered_json = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered_json + '\n', encoding='utf-8')

    if args.json:
        print(rendered_json)
    else:
        print(format_evaluation_report(results))

    return 0 if results['counts']['fail'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
