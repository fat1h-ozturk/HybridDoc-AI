"""Deterministic provider boundary tests: never call live services or read user PDFs."""
import json

import httpx
import pytest

from src import answers


def source(text='Ankara is the capital of Turkey.', **extra):
    return dict(source_sha256='a' * 64, chunk_id='chunk-1', page_number=2,
                excerpt=text, answer_eligible=False, math_status='no_native_candidates',
                detection_status='native_heuristic_incomplete', **extra)


def response_for(evidence, answer='Turkey has its capital in Ankara.'):
    return dict(status='answered', answer=answer,
                citations=[dict(source_id=evidence[0]['id'], quote=evidence[0]['text'])])


def test_evidence_preserves_complete_context_flags_and_stable_id():
    context = dict(text='Father: Invented Person', association='not_inferred', bbox=[0, 0, 80, 30])
    result = source('Father', source_context=context, review_reason='Inspect source')
    evidence, omitted = answers.build_evidence([result])
    assert omitted == 0
    assert evidence[0]['text'] == context['text']
    assert evidence[0]['page'] == 2
    assert evidence[0]['association'] == 'not_inferred'
    assert evidence[0]['answer_eligible'] is False
    assert evidence[0]['warnings']['review_reason'] == 'Inspect source'
    assert answers.build_evidence([{**result, 'rank': 5}])[0][0]['id'] == evidence[0]['id']
    assert answers.build_evidence([source('x' * 50000)])[0] == []
    assert answers.build_evidence([source('fallback', source_context={'text': 'x' * 50000})]) == ([], 1)


@pytest.mark.parametrize('mutation', ['unknown', 'quote', 'empty', 'uncited', 'oversize', 'status'])
def test_rejects_invalid_or_unsupported_output(mutation):
    evidence, _ = answers.build_evidence([source()])
    data = response_for(evidence)
    if mutation == 'unknown':
        data['citations'][0]['source_id'] = 'invented'
    elif mutation == 'quote':
        data['citations'][0]['quote'] = 'Istanbul is the capital'
    elif mutation == 'empty':
        data['answer'] = ' '
    elif mutation == 'uncited':
        data['citations'] = []
    elif mutation == 'oversize':
        data['answer'] = 'x' * 20000
    else:
        data['status'] = 'verified'
    with pytest.raises(answers.AnswerError):
        answers.validate_answer(json.dumps(data), evidence)


def test_citation_validation_does_not_claim_semantic_truth():
    evidence, _ = answers.build_evidence([source()])
    result = answers.validate_answer(json.dumps(response_for(evidence)), evidence)
    assert result['answer'] == 'Turkey has its capital in Ankara.'
    assert result['citations'][0]['quote'] == 'Ankara is the capital of Turkey.'
    abstained = dict(status='abstained', answer='Kaynakta doğum tarihi yok.', citations=[])
    assert answers.validate_answer(json.dumps(abstained), evidence)['status'] == 'abstained'


def test_configured_gemini_catalog_has_no_implicit_retired_default(monkeypatch):
    monkeypatch.delenv('HYBRIDDOC_GEMINI_MODELS', raising=False)
    assert answers.gemini_models() == []
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.8-flash')
    assert answers.gemini_models() == ['gemini-3.5-flash-lite', 'gemini-3.8-flash']
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'bad/path?key=secret')
    with pytest.raises(answers.AnswerError):
        answers.gemini_models()


def test_live_inventory_shape_uses_fixed_loopback_only():
    def handle(request):
        assert str(request.url) == 'http://127.0.0.1:11434/api/tags'
        assert request.method == 'GET'
        return httpx.Response(200, json={'models': [{'name': 'tiny:latest'}, {'name': 'other:tag'}]})
    assert answers.ollama_models(transport=httpx.MockTransport(handle)) == ['tiny:latest', 'other:tag']


@pytest.mark.parametrize('provider', ['Ollama', 'Gemini API'])
def test_provider_payload_matches_preview_and_has_no_tools_or_secret_url(provider, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    preview = answers.request_preview('Where is the capital?', evidence)
    requests = []
    def handle(request):
        if request.url.path == '/api/show':
            assert json.loads(request.content) == {'model': 'tiny:latest'}
            return httpx.Response(200, json={'model_info': {'general.architecture': 'llama', 'llama.context_length': 8192}})
        requests.append(request)
        body = json.loads(request.content)
        assert 'test-secret' not in str(request.url)
        if provider == 'Ollama':
            assert str(request.url) == 'http://127.0.0.1:11434/api/chat'
            assert body['model'] == 'tiny:latest'
            assert body['stream'] is False
            assert body['messages'][1]['content'] == preview
            assert 'untrusted' in body['messages'][0]['content']
            assert 'tools' not in body
            return httpx.Response(200, json={'done': True, 'done_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(response_for(evidence))}})
        assert request.headers['x-goog-api-key'] == 'test-secret'
        assert str(request.url) == 'https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent'
        assert body['contents'][0]['parts'][0]['text'] == preview
        assert 'tools' not in body
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{'text': json.dumps(response_for(evidence))}]}}]})
    result = answers.generate_answer(provider, 'tiny:latest' if provider == 'Ollama' else 'gemini-test',
                                     'Where is the capital?', evidence, consent=True,
                                     transport=httpx.MockTransport(handle))
    assert len(requests) == 1
    assert result['status'] == 'answered'


@pytest.mark.parametrize('kind', ['none', 'consent', 'key', 'model', 'math', 'form', 'question'])
def test_guards_prevent_any_provider_call(kind, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    consent, model, question = True, 'gemini-test', 'capital?'
    if kind == 'none':
        evidence = []
    elif kind == 'consent':
        consent = False
    elif kind == 'key':
        monkeypatch.delenv('GEMINI_API_KEY')
    elif kind == 'model':
        model = 'not-configured'
    elif kind == 'math':
        evidence, _ = answers.build_evidence([source('Solve x + 1 = 2', formulas=[{'id': 'f1', 'answer_eligible': False}])])
    elif kind == 'form':
        evidence, _ = answers.build_evidence([source('Father', source_context={'text': 'Father\nMother\nInvented Person\nOther Person', 'association': 'not_inferred'})])
    else:
        question = 'x' * 3000
    def forbidden(request):
        pytest.fail('guard must run before HTTP')
    with pytest.raises(answers.AnswerError):
        answers.generate_answer('Gemini API', model, question, evidence, consent=consent,
                                transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize('code', [400, 401, 403, 404, 429, 500])
def test_safe_http_errors_never_echo_raw_body_or_credentials(code, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    def handle(request):
        return httpx.Response(code, text='test-secret private text')
    with pytest.raises(answers.AnswerError) as error:
        answers.generate_answer('Gemini API', 'gemini-test', 'capital?', evidence, consent=True,
                                transport=httpx.MockTransport(handle))
    assert 'test-secret' not in str(error.value)
    assert 'private text' not in str(error.value)
    if code == 429:
        assert 'kota' in str(error.value)


@pytest.mark.parametrize('payload', [None, {}, {'candidates': []}, {'candidates': [{'finishReason': 'SAFETY'}]}, {'candidates': [{'finishReason': 'MAX_TOKENS', 'content': {'parts': [{'text': '{}'}]}}]}])
def test_refusal_malformed_and_incomplete_responses_are_rejected(payload, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    with pytest.raises(answers.AnswerError):
        answers.generate_answer('Gemini API', 'gemini-test', 'capital?', evidence, consent=True,
                                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))


def test_prose_on_unresolved_formula_page_can_be_cited_but_formula_band_cannot():
    prose = source('Source passages are exact quotes from the document.',
                   source_context={'text': 'Keyword search\nSource passages are exact quotes from the document.',
                                   'association': 'not_inferred'})
    prose['math_status'] = 'unresolved'
    evidence, _ = answers.build_evidence([prose])
    assert answers.evidence_issue(evidence) is None
    prose['source_context']['text'] += '\nx = y - 2'
    evidence, _ = answers.build_evidence([prose])
    assert 'matematik' in answers.evidence_issue(evidence)


def test_inventory_filters_remote_and_cloud_aliases_before_selection():
    payload = {'models': [{'name': 'tiny:latest'}, {'name': 'foo:cloud'},
                          {'name': 'remote:latest', 'remote_model': 'upstream'},
                          {'name': 'alias:tag', 'remote_host': 'https://ollama.com'},
                          {'name': 'big:120b-cloud'}]}
    assert answers.ollama_models(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))) == ['tiny:latest']


@pytest.mark.parametrize('info', [{'remote_model': 'upstream'}, {'remote_host': 'https://ollama.com'}, {}, {'model_info': {}}])
def test_ollama_checks_model_is_local_without_sending_question_first(info):
    evidence, _ = answers.build_evidence([source()])
    def handle(request):
        assert str(request.url) == 'http://127.0.0.1:11434/api/show'
        assert json.loads(request.content) == {'model': 'tiny:latest'}
        return httpx.Response(200, json=info)
    with pytest.raises(answers.AnswerError):
        answers.generate_answer('Ollama', 'tiny:latest', 'capital?', evidence,
                                transport=httpx.MockTransport(handle))


@pytest.mark.parametrize('payload', [{'candidates': [None]}, {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [None]}}]},
                                     {'candidates': [{'finishReason': 'STOP', 'content': {'parts': ['text']}}]}])
def test_nested_malformed_response_is_safe(payload, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    with pytest.raises(answers.AnswerError):
        answers.generate_answer('Gemini API', 'gemini-test', 'capital?', evidence, consent=True,
                                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))


@pytest.mark.parametrize('kind', ['timeout', 'disconnect', 'oversize', 'redirect', 'invalid-json', 'deep-json'])
def test_transport_limits_and_errors(kind, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-secret')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    evidence, _ = answers.build_evidence([source()])
    attempts = []
    def handle(request):
        attempts.append(request)
        if kind == 'timeout':
            raise httpx.ReadTimeout('private test-secret', request=request)
        if kind == 'disconnect':
            raise httpx.ConnectError('private test-secret', request=request)
        if kind == 'oversize':
            return httpx.Response(200, content=b'x' * 70000)
        if kind == 'redirect':
            return httpx.Response(302, headers={'Location': 'https://elsewhere.invalid/?key=test-secret'})
        if kind == 'deep-json':
            return httpx.Response(200, content=b'[' * 3000 + b']' * 3000)
        return httpx.Response(200, content=b'not json private test-secret')
    with pytest.raises(answers.AnswerError) as error:
        answers.generate_answer('Gemini API', 'gemini-test', 'capital?', evidence, consent=True,
                                transport=httpx.MockTransport(handle))
    assert len(attempts) == 1
    assert 'test-secret' not in str(error.value)


def test_context_preview_does_not_include_excerpt_outside_the_selected_source_crop():
    result = source('Outside cropped band. Do not transmit this.', source_context={
        'text': 'Inside cropped band.', 'association': 'not_inferred'})
    evidence, _ = answers.build_evidence([result])
    preview = answers.request_preview('Where?', evidence)
    assert 'Inside cropped band.' in preview
    assert 'Outside cropped band' not in preview


def test_local_model_context_budget_prevents_silent_evidence_truncation():
    evidence, _ = answers.build_evidence([source()])
    def handle(request):
        assert request.url.path == '/api/show', 'oversize input must not reach inference'
        return httpx.Response(200, json={'model_info': {'llama.context_length': 2048}})
    with pytest.raises(answers.AnswerError, match='bağlam'):
        answers.generate_answer('Ollama', 'tiny:latest', 'capital?', evidence,
                                transport=httpx.MockTransport(handle))
