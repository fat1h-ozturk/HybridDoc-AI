import json
import subprocess
import sys
import time
from pathlib import Path
import pytest
from src.ingestion import formula_loader


def test_atomic_chunks_keep_latex_and_metadata():
    from src.chunking.text_chunker import chunk_formula_pages
    formula = {'id': 'p1-f1', 'latex': r'\frac{x}{y}' * 20,
               'bbox': [1, 2, 3, 4], 'state': 'unresolved', 'answer_eligible': False}
    block = {'kind': 'formula', 'formula_id': formula['id'], 'text': formula['latex']}
    page = {'page_number': 1, 'file_name': 'a.pdf', 'total_pages': 1,
            'source_sha256': 'abc', 'answer_eligible': False,
            'formulas': [formula], 'blocks': [
                {'kind': 'text', 'text': 'prose ' * 30}, block,
                {'kind': 'text', 'text': 'end ' * 30}]}
    chunks = chunk_formula_pages([page], max_chars=40, overlap=5)
    math_chunks = [c for c in chunks if c['formulas']]
    assert len(math_chunks) == 1
    assert math_chunks[0]['text'] == formula['latex']
    assert math_chunks[0]['formulas'] == [formula]
    assert math_chunks[0]['page_number'] == 1
    assert math_chunks[0]['char_count'] > 40
    assert not math_chunks[0]['answer_eligible']
    assert all(formula['latex'] not in c['text'] for c in chunks if not c['formulas'])
    with pytest.raises(ValueError):
        chunk_formula_pages([page], max_chars=0)


def slow_worker(conn, cache):
    conn.recv()
    time.sleep(10)


def failed_worker(conn, cache):
    conn.recv()
    conn.send(('error', 'weights missing'))


def test_worker_failure_does_not_restart_or_mutate_environment(monkeypatch):
    import os
    from src.ingestion import formula_recognizer as runtime
    monkeypatch.setattr(runtime, '_worker', failed_worker)
    before = dict(os.environ)
    with runtime.LocalFormulaRecognizer(timeout_seconds=3) as model:
        with pytest.raises(RuntimeError, match='weights missing'):
            model.recognize(b'png')
        with pytest.raises(RuntimeError, match='unavailable'):
            model.recognize(b'png')
    assert dict(os.environ) == before


def test_worker_timeout_and_document_budget(monkeypatch):
    from src.ingestion import formula_recognizer as runtime
    monkeypatch.setattr(runtime, '_worker', slow_worker)
    with runtime.LocalFormulaRecognizer(timeout_seconds=.05, budget_seconds=.1) as model:
        with pytest.raises(TimeoutError):
            model.recognize(b'png')
        time.sleep(.11)
        with pytest.raises(TimeoutError, match='budget'):
            model.recognize(b'png')
        assert model._process is None
    for value in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            runtime.LocalFormulaRecognizer(timeout_seconds=value)


def test_cli_full_output_and_file(tmp_path):
    import pymupdf
    path = tmp_path / 'inline.pdf'
    with pymupdf.open() as doc:
        doc.new_page().insert_text((30, 40), 'Inline equation x = 2 in prose.')
        doc.save(path)
    output = tmp_path / 'output.json'
    result = subprocess.run(
        [sys.executable, '-m', 'src.ingestion.formula_loader', str(path),
         '--output', str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert result.stdout == output.read_text(encoding='utf-8')
    assert data[0]['formulas']
    assert data[0]['answer_eligible'] is False
    readable = result.stdout
    assert 'Inline equation x = 2 in prose.' in readable
    assert 'bbox' in readable and 'unresolved' in readable
    assert 'not answer evidence' in readable


def test_missed_native_math_cannot_certify_page(tmp_path):
    import pymupdf
    path = tmp_path / 'ambiguous.pdf'
    with pymupdf.open() as doc:
        doc.new_page().insert_text((30, 40), 'sin x / tan y')
        doc.save(path)
    page = formula_loader.load_pdf_formulas(path)[0]
    assert page['answer_eligible'] is False
    assert page['detection_status'] == 'native_heuristic_incomplete'


def test_inline_latex_is_a_candidate_not_verified_truth(tmp_path):
    import pymupdf
    path = tmp_path / 'inline.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((30, 40), 'Use x = 2 here.')
        doc.save(path)
    class Recognizer:
        def recognize(self, png):
            return 'x=2'
    page = formula_loader.load_pdf_formulas(path, recognizer=Recognizer())[0]
    formula = page['formulas'][0]
    assert formula['latex'] == 'x=2'
    assert formula['state'] == 'unverified'
    assert formula['reason'] == 'recognizer_output_not_verified'
    assert not formula['answer_eligible'] and not page['answer_eligible']
