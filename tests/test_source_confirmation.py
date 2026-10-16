"""Synthetic sources and OFFLINE HTTP SIMULATION; no live providers or private data."""
import copy
import json
from pathlib import Path

import httpx
import pytest
from src import answers
from src.workbench import open_upload
from tests.form_fixtures import certificate_pdf

QUESTION = 'denizin babasinin adi'
APP = Path(__file__).resolve().parents[1] / 'apps' / 'workbench.py'


def ambiguous_evidence():
    document = open_upload('invented.pdf', certificate_pdf())
    results = document.search(QUESTION)
    father = next(r for r in results if r['excerpt'] == 'Baba Adı')
    evidence, _ = answers.build_evidence([father])
    return evidence


def test_confirmation_keeps_raw_provenance_and_requires_complete_citation():
    evidence = ambiguous_evidence()
    original = copy.deepcopy(evidence)
    assert answers.evidence_issue(evidence)
    record = answers.confirm_source_relation(evidence, evidence[0]['id'], QUESTION, 'Baba Adı', 'Orhan Örnek')
    evidence[0]['user_confirmation'] = record
    assert answers.evidence_issue(evidence, question=QUESTION) is None
    assert evidence[0]['association'] == 'not_inferred'
    assert evidence[0]['answer_eligible'] is False
    assert {'Baba Adı', 'Orhan Örnek'} <= {s['text'] for s in evidence[0]['source_spans']}
    assert all(len(s['bbox']) == 4 for s in evidence[0]['source_spans'])
    assert {k: v for k, v in evidence[0].items() if k != 'user_confirmation'} == original[0]
    reply = dict(status='answered', answer='SİMÜLASYON: Orhan Örnek.',
                 citations=[dict(source_id=evidence[0]['id'], quote=evidence[0]['text'])])
    assert answers.validate_answer(json.dumps(reply), evidence)['status'] == 'answered'
    for quote in ['Baba Adı', 'Orhan Örnek', 'not in source']:
        reply['citations'][0]['quote'] = quote
        with pytest.raises(answers.AnswerError):
            answers.validate_answer(json.dumps(reply), evidence)


@pytest.mark.parametrize('change', ['question', 'document', 'text', 'box', 'selection'])
def test_confirmation_cannot_be_reused_outside_exact_scope(change):
    evidence = ambiguous_evidence()
    evidence[0]['user_confirmation'] = answers.confirm_source_relation(
        evidence, evidence[0]['id'], QUESTION, 'Baba Adı', 'Orhan Örnek')
    question = QUESTION
    if change == 'question':
        question = 'annesinin adi'
    elif change == 'document':
        evidence[0]['source_sha256'] = '0' * 64
    elif change == 'text':
        evidence[0]['text'] += '\nOther source value'
    elif change == 'box':
        evidence[0]['source_bbox'] = [0, 0, 100, 200]
    else:
        evidence.append({**evidence[0], 'id': 'different', 'user_confirmation': None})
    assert answers.evidence_issue(evidence, question=question)
    with pytest.raises(answers.AnswerError):
        answers.generate_answer('Gemini API', 'gemini-test', question, evidence, consent=True,
                                transport=httpx.MockTransport(lambda r: pytest.fail('stale confirmation sent HTTP')))


@pytest.mark.parametrize('math', ['formula', 'group', 'syntax'])
def test_user_confirmation_cannot_override_math(math):
    evidence = ambiguous_evidence()
    evidence[0]['user_confirmation'] = answers.confirm_source_relation(
        evidence, evidence[0]['id'], QUESTION, 'Baba Adı', 'Orhan Örnek')
    if math == 'formula':
        evidence[0]['formulas'] = [{'answer_eligible': False}]
    elif math == 'group':
        evidence[0]['formula_groups'] = [{'answer_eligible': False}]
    else:
        evidence[0]['text'] += '\nx = y + 2'
    with pytest.raises(answers.AnswerError):
        answers.confirm_source_relation(evidence, evidence[0]['id'], QUESTION, 'Baba Adı', 'Orhan Örnek')
    assert 'matematik' in answers.evidence_issue(evidence, question=QUESTION)


def test_cannot_confirm_fabricated_or_empty_value():
    evidence = ambiguous_evidence()
    for value in ['', 'Invented unsupported name', 'Baba Adı']:
        with pytest.raises(answers.AnswerError):
            answers.confirm_source_relation(evidence, evidence[0]['id'], QUESTION, 'Baba Adı', value)


def test_simulated_ui_confirmation_generation_and_resets(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-simulation-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    calls = []
    def fake(request):
        calls.append(request)
        sent = json.loads(json.loads(request.content)['contents'][0]['parts'][0]['text'])
        source = sent['sources'][0]
        reply = dict(status='answered', answer='SİMÜLASYON: Baba Adı Orhan Örnek.',
                     citations=[dict(source_id=source['id'], quote=source['text'])])
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': json.dumps(reply)}]}}]})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(fake)}))
    app = AppTest.from_file(str(APP)).run()
    app.session_state['document'] = open_upload('invented.pdf', certificate_pdf())
    app.run()
    app.text_input(key='question').set_value(QUESTION)
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert app.multiselect(key='answer_sources').value == []
    father = next(r for r in app.session_state['results'] if r['excerpt'] == 'Baba Adı')
    sid = answers.build_evidence([father])[0][0]['id']
    app.multiselect(key='answer_sources').set_value([sid]).run()
    assert app.button(key='generate_answer').disabled and not calls
    assert app.get('image')
    assert app.selectbox(key='relation_label_' + sid).value is None
    assert app.selectbox(key='relation_value_' + sid).value is None
    app.selectbox(key='relation_label_' + sid).select('Baba Adı').run()
    app.selectbox(key='relation_value_' + sid).select('Orhan Örnek').run()
    app.button(key='confirm_relation_' + sid).click().run()
    assert not app.exception and not calls
    assert app.button(key='generate_answer').disabled  # consent is still independent
    app.checkbox(key='cloud_consent').check().run()
    assert not calls and not app.button(key='generate_answer').disabled
    assert any('kullanıcı onayladı' in c.value.lower() for c in app.caption)
    preview = next(t.value for t in app.text if t.value.startswith('{\n  "question"'))
    app.run()
    assert not calls
    app.button(key='generate_answer').click().run()
    assert not app.exception and len(calls) == 1
    assert json.loads(calls[0].content)['contents'][0]['parts'][0]['text'] == preview
    assert app.session_state['generated_answer']['status'] == 'answered'
    app.selectbox(key='answer_provider').select('Ollama').run()
    assert 'generated_answer' not in app.session_state
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.selectbox(key='relation_value_' + sid).select('Selin Örnek').run()
    assert app.button(key='generate_answer').disabled and not app.session_state['source_confirmations']
    app.selectbox(key='relation_value_' + sid).select('Orhan Örnek').run()
    app.button(key='confirm_relation_' + sid).click().run()
    app.checkbox(key='cloud_consent').check().run()
    app.button(key='generate_answer').click().run()
    assert 'generated_answer' in app.session_state and len(calls) == 2
    app.multiselect(key='answer_sources').set_value([]).run()
    assert 'generated_answer' not in app.session_state
    app.multiselect(key='answer_sources').set_value([sid]).run()
    assert app.button(key='generate_answer').disabled and not app.session_state['source_confirmations']
    app.selectbox(key='relation_label_' + sid).select('Baba Adı').run()
    app.selectbox(key='relation_value_' + sid).select('Orhan Örnek').run()
    app.button(key='confirm_relation_' + sid).click().run()
    app.text_input(key='question').set_value('Baba Adı').run()
    assert not app.session_state.get('source_confirmations')
    assert 'generated_answer' not in app.session_state
    assert not any(b.key == 'generate_answer' and not b.disabled for b in app.button)
    app.button(key='search').click().run()
    assert not app.session_state.get('source_confirmations')
    assert 'generated_answer' not in app.session_state and len(calls) == 2
    app.checkbox(key='generation_enabled').check().run()
    father = next(r for r in app.session_state['results'] if r['excerpt'] == 'Baba Adı')
    new_sid = answers.build_evidence([father])[0][0]['id']
    app.multiselect(key='answer_sources').set_value([new_sid]).run()
    app.selectbox(key='relation_label_' + new_sid).select('Baba Adı').run()
    app.selectbox(key='relation_value_' + new_sid).select('Orhan Örnek').run()
    app.button(key='confirm_relation_' + new_sid).click().run()
    assert app.session_state['source_confirmations']
    app.radio[0].set_value('Demo PDF').run()
    assert not app.session_state.get('source_confirmations')
    assert 'generated_answer' not in app.session_state


def test_simulated_ui_math_has_no_confirmation_override(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from tests.test_explicit_fields import native_pdf
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-simulation-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    def forbidden(request):
        pytest.fail('math UI must not contact even a simulated provider')
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(forbidden)}))
    app = AppTest.from_file(str(APP)).run()
    app.session_state['document'] = open_upload('invented.pdf', native_pdf([
        (35, 100, 'Baba Adı'), (185, 100, 'Orhan Örnek'), (35, 120, 'x = y + 2')]))
    app.run()
    app.text_input(key='question').set_value(QUESTION)
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.checkbox(key='cloud_consent').check().run()
    sid = answers.build_evidence(app.session_state['results'])[0][0]['id']
    app.multiselect(key='answer_sources').set_value([sid]).run()
    assert not app.exception and app.button(key='generate_answer').disabled
    assert not any(b.key.startswith('confirm_relation_') for b in app.button if b.key)
    assert any('matematik/formül' in w.value for w in app.warning)
