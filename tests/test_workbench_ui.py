"""Executable UI tests; optional UI dependencies are required only for these checks."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "apps" / "workbench.py"


def test_demo_question_complete_source_report_and_clear():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert not app.text_input
    app.radio[0].set_value("Demo PDF").run()
    assert not app.exception
    app.text_input(key="question").set_value("source passages")
    app.button(key="search").click().run()
    assert not app.exception
    assert app.session_state["results"][0]["page_number"] == 2
    assert app.session_state["results"][0]["answer_eligible"] is False
    assert any("Source passages are exact quotes" in item.value for item in app.text)
    app.selectbox(key="page").select(2).run()
    assert any("END OF DEMO PAGE TWO" in item.value for item in app.text)
    app.button(key="report").click().run()
    assert not app.exception
    assert "answer_eligible: false" in app.session_state["report_html"]
    app.button(key="clear").click().run()
    assert not app.exception
    assert "document" not in app.session_state
    assert not app.text_input


def test_no_match_empty_question_and_independent_sessions():
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("Demo PDF").run()
    app.text_input(key="question").set_value("quasar")
    app.button(key="search").click().run()
    assert app.session_state["results"] == []
    assert app.info
    app.text_input(key="question").set_value("??")
    app.button(key="search").click().run()
    assert app.error
    assert not app.exception
    other = AppTest.from_file(str(APP)).run()
    assert "document" not in other.session_state
    assert not other.text_input


def test_synthetic_form_shows_readable_source_band_and_crop():
    from src.workbench import open_upload
    from tests.form_fixtures import certificate_pdf

    app = AppTest.from_file(str(APP)).run()
    # AppTest cannot upload; inject the actual upload backend result into this session.
    app.session_state['document'] = open_upload('invented.pdf', certificate_pdf(order='reverse'))
    app.run()
    app.text_input(key='question').set_value('denizin babasinin adi')
    app.button(key='search').click().run()
    assert not app.exception
    father = next(r for r in app.session_state['results'] if r['excerpt'] == 'Baba Adı')
    context = father['source_context']['text']
    assert 'Orhan Örnek' in context
    assert any(item.value == context for item in app.text)
    assert any('Sayfa 1' in item.value and 'invented.pdf' in item.value for item in app.text)
    assert any('kesin cevap çıkarılamaz' in item.value for item in app.caption)
    assert app.get('image')


def test_readable_result_cards_expander_metadata_warnings_and_download():
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("Demo PDF").run()
    app.text_input(key="question").set_value("source passages")
    app.button(key="search").click().run()

    assert not app.exception
    assert len(app.session_state["results"]) == 1
    res = app.session_state["results"][0]

    # File and page first
    assert any(
        f"Kaynak 1 — Sayfa {res['page_number']} / {res['total_pages']}" in s.value
        for s in app.subheader
    )

    # Complete passage text visible
    passage_text = res["source_context"]["text"] if res.get("source_context") else res["excerpt"]
    assert any(passage_text == item.value for item in app.text)
    assert any("Source passages are exact quotes from the document." in item.value for item in app.text)

    # Diagnostic raw dump string not in top-level passage view
    assert not any("--- Result 1 (Score:" in item.value for item in app.text)

    # Metadata accessible separately in expander
    assert len(app.expander) >= 1
    assert app.expander[0].label == "Teknik ayrıntılar — Kaynak 1"
    exp_texts = [item.value for item in app.expander[0].text]
    assert any(f"BM25 skoru: {res['score']:.4f}" in t for t in exp_texts)
    assert not any("%" in t for t in exp_texts)  # Raw score, not confidence percentage
    assert any(f"Parça kimliği (Chunk ID): {res['chunk_id']}" in t for t in exp_texts)
    assert any(f"Kaynak sırası (Source Order): {res['source_order']}" in t for t in exp_texts)
    assert any(f"Kaynak SHA-256: {res['source_sha256']}" in t for t in exp_texts)
    assert any(f"Tespit durumu: {res['detection_status']}" in t for t in exp_texts)
    assert any(f"Cevap uygunluğu (Answer Eligible): {res['answer_eligible']}" in t for t in exp_texts)
    assert any(f"Matematik durumu: {res['math_status']}" in t for t in exp_texts)

    # Warnings not hidden
    assert any(res["review_reason"] in w.value for w in app.warning)
    assert any("İnceleme uyarısı" in w.value for w in app.warning)
    assert not any("İnceleme uyarısı" in t for t in exp_texts)

    # Download content retained
    download_btn = next((b for b in app.download_button if "Kaynak sonuçlarını indir" in b.label), None)
    assert download_btn is not None
    download_text = app.session_state.get("result_text", "")
    assert 'Search Results for query: "source passages"' in download_text
    assert f"Document: {res['file_name']}" in download_text
    assert f"--- Result 1 (Score: {res['score']:.4f}" in download_text
    assert f"Source SHA-256: {res['source_sha256']}" in download_text
    assert f"Chunk ID: {res['chunk_id']}" in download_text
    assert res["excerpt"] in download_text


def test_formula_warnings_and_metadata_in_results():
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value("Demo PDF").run()
    app.text_input(key="question").set_value("formula")
    app.button(key="search").click().run()

    assert not app.exception
    assert len(app.session_state["results"]) == 2

    # Formula candidate warning visible
    assert any("Doğrulanmamış formül adayı" in w.value for w in app.warning)
    # Child formula metadata accessible in expander
    assert any("p2-f1" in t.value for t in app.expander[0].text)


def test_generation_off_local_inventory_and_explicit_request_only(monkeypatch):
    import json
    import httpx
    from src import answers

    calls = []
    def handle(request):
        calls.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json={'models': [{'name': 'demo:tiny'}, {'name': 'demo:other'}]})
        if request.url.path == '/api/show':
            assert json.loads(request.content) == {'model': 'demo:other'}
            return httpx.Response(200, json={'model_info': {'general.architecture': 'llama', 'llama.context_length': 8192}})
        body = json.loads(request.content)
        sent = json.loads(body['messages'][1]['content'])
        evidence = sent['sources']
        return httpx.Response(200, json={'done': True, 'done_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps({
            'status': 'answered', 'answer': '<script>alert(1)</script> Kaynak pasajları alıntılardır.',
            'citations': [{'source_id': evidence[0]['id'], 'quote': 'Source passages are exact quotes from the document.'}]})}})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(handle)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    assert calls == []
    assert not app.checkbox(key='generation_enabled').value
    app.checkbox(key='generation_enabled').check().run()
    assert app.selectbox(key='answer_provider').value == 'Ollama'
    assert calls == []
    app.button(key='refresh_models').click().run()
    assert len(calls) == 1 and calls[0].method == 'GET'
    assert app.selectbox(key='ollama_model').options == ['demo:tiny', 'demo:other']
    app.selectbox(key='ollama_model').select('demo:other').run()
    assert len(calls) == 1
    preview = next(t.value for t in app.text if '"question": "source passages"' in t.value)
    app.button(key='generate_answer').click().run()
    assert not app.exception
    assert len(calls) == 3
    payload = json.loads(calls[-1].content)
    assert payload['model'] == 'demo:other'
    assert payload['messages'][1]['content'] == preview
    assert any(t.value.startswith('<script>') for t in app.text)
    assert any('anlamsal doğruluğu' in item.value for item in app.caption)
    app.selectbox(key='page').select(2).run()
    assert len(calls) == 3
    assert 'generated_answer' in app.session_state
    app.selectbox(key='ollama_model').select('demo:tiny').run()
    assert 'generated_answer' not in app.session_state
    app.button(key='search').click().run()
    assert len(calls) == 3
    app.button(key='clear').click().run()
    assert len(calls) == 3 and 'generated_answer' not in app.session_state


def test_gemini_requires_document_consent_preview_and_resets(monkeypatch):
    import json
    import httpx
    from src import answers

    monkeypatch.setenv('GEMINI_API_KEY', 'fake-ui-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test,gemini-other')
    sent = []
    def handle(request):
        sent.append(request)
        data = {'status': 'abstained', 'answer': 'Kaynakta doğum tarihi yok; kaynağı inceleyin.', 'citations': []}
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{'text': json.dumps(data)}]}}]})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(handle)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert not sent
    assert app.button(key='generate_answer').disabled
    assert any('Google' in item.value for item in app.warning)
    assert app.selectbox(key='gemini_model').options == ['gemini-test', 'gemini-other']
    app.checkbox(key='cloud_consent').check().run()
    assert not sent
    preview = next(t.value for t in app.text if '"question": "source passages"' in t.value)
    app.button(key='generate_answer').click().run()
    assert len(sent) == 1
    assert json.loads(sent[0].content)['contents'][0]['parts'][0]['text'] == preview
    assert any('doğum tarihi yok' in item.value for item in app.text)
    app.run()
    assert len(sent) == 1
    app.radio[0].set_value('PDF yükle').run()
    assert 'generated_answer' not in app.session_state
    assert 'cloud_consent' not in app.session_state
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    if not app.checkbox(key='generation_enabled').value:
        app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert not app.checkbox(key='cloud_consent').value
    assert len(sent) == 1


def test_no_evidence_form_math_and_service_error_in_ui(monkeypatch):
    from src import answers
    import httpx

    calls = []
    def offline(request):
        calls.append(request)
        raise httpx.ConnectError('raw-private-provider-text', request=request)
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(offline)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('quasar')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    assert app.button(key='generate_answer').disabled
    assert any('kanıtı yok' in item.value for item in app.warning)
    assert not calls
    app.button(key='refresh_models').click().run()
    assert len(calls) == 1
    assert any('erişilemiyor' in item.value for item in app.error)
    assert not any('raw-private' in item.value for item in app.error)
    app.text_input(key='question').set_value('formula')
    app.button(key='search').click().run()
    assert app.button(key='generate_answer').disabled
    assert any('matematik' in item.value for item in app.warning)
    assert len(calls) == 1


def test_source_selection_no_evidence_and_answer_error_need_new_explicit_click(monkeypatch):
    import httpx
    from src import answers

    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(429, text='private raw body fake-key')
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(handle)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.checkbox(key='cloud_consent').check().run()
    selected = app.multiselect(key='answer_sources').value
    app.multiselect(key='answer_sources').set_value([]).run()
    assert app.button(key='generate_answer').disabled
    assert not requests
    app.multiselect(key='answer_sources').set_value(selected).run()
    app.button(key='generate_answer').click().run()
    assert not app.exception and len(requests) == 1
    assert any('kota' in item.value for item in app.error)
    assert not any('private raw' in item.value or 'fake-key' in item.value for item in app.error)
    assert 'generated_answer' not in app.session_state
    app.run()
    assert len(requests) == 1
    app.checkbox(key='generation_enabled').uncheck().run()
    app.checkbox(key='generation_enabled').check().run()
    assert app.selectbox(key='answer_provider').value == 'Ollama'
    assert len(requests) == 1


def test_ambiguous_synthetic_form_abstains_in_ui_without_network(monkeypatch):
    from src.workbench import open_upload
    from src import answers
    from tests.form_fixtures import certificate_pdf
    import httpx

    def forbidden(request):
        pytest.fail('ambiguous source inspection must not contact a provider')
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(forbidden)}))
    app = AppTest.from_file(str(APP)).run()
    app.session_state['document'] = open_upload('invented.pdf', certificate_pdf(columns=2))
    app.run()
    app.text_input(key='question').set_value('Baba Adı')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    assert not app.exception
    assert app.button(key='generate_answer').disabled
    assert any('Alan ilişkisi belirsiz' in item.value for item in app.warning)
    assert app.get('image')


def test_gemini_missing_configuration_does_not_call_network(monkeypatch):
    from src import answers
    import httpx

    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('HYBRIDDOC_GEMINI_MODELS', raising=False)
    def forbidden(request):
        pytest.fail('missing configuration must not contact Google')
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(forbidden)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert app.button(key='generate_answer').disabled
    assert any('GEMINI_API_KEY eksik' in item.value for item in app.info)
    assert any('HYBRIDDOC_GEMINI_MODELS' in item.value for item in app.info)


def test_direct_document_replacement_resets_consent_and_answer(monkeypatch):
    import json
    import httpx
    import pymupdf
    from src import answers

    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    real_read = Path.read_bytes
    content = [real_read(APP.parents[1] / 'sample_docs' / 'workbench_demo.pdf')]
    def demo_read(path):
        if path == APP.parents[1] / 'sample_docs' / 'workbench_demo.pdf':
            return content[0]
        return real_read(path)
    monkeypatch.setattr(Path, 'read_bytes', demo_read)
    calls = []
    def handle(request):
        calls.append(request)
        reply = {'status': 'abstained', 'answer': 'Kaynakta yok.', 'citations': []}
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(reply)}]}}]})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(handle)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.checkbox(key='cloud_consent').check().run()
    app.button(key='generate_answer').click().run()
    assert 'generated_answer' in app.session_state
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 70), 'Source passages are changed in this document.')
        content[0] = pdf.tobytes()
    app.run()
    assert not app.exception
    assert 'generated_answer' not in app.session_state
    assert 'cloud_consent' not in app.session_state
    assert len(calls) == 1
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert not app.checkbox(key='cloud_consent').value
    assert app.button(key='generate_answer').disabled
    assert len(calls) == 1


def test_invalid_model_citation_is_never_displayed_as_answer(monkeypatch):
    import json
    import httpx
    from src import answers

    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-test')
    def handle(request):
        reply = {'status': 'answered', 'answer': 'UNSUPPORTED-SENTINEL',
                 'citations': [{'source_id': 'unknown-source', 'quote': 'invented quote'}]}
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': json.dumps(reply)}]}}]})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(handle)}))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Demo PDF').run()
    app.text_input(key='question').set_value('source passages')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.checkbox(key='cloud_consent').check().run()
    app.button(key='generate_answer').click().run()
    assert not app.exception
    assert 'generated_answer' not in app.session_state
    assert any('alıntı denetimi geçersiz' in item.value for item in app.error)
    assert not any('UNSUPPORTED-SENTINEL' in item.value for item in app.text)


def test_simulated_split_field_mixed_sources_explicit_action_and_reset(monkeypatch):
    """OFFLINE SIMULATION: real PDF/search/UI, invented response via fake HTTP."""
    import json
    import httpx
    import pymupdf
    from src import answers
    from src.workbench import open_upload
    from tests.form_fixtures import certificate_pdf

    monkeypatch.setenv('GEMINI_API_KEY', 'offline-simulation-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-3.8-flash')
    with pymupdf.open(stream=certificate_pdf(colon=True), filetype='pdf') as pdf:
        pdf[0].insert_text((35, 300), 'adi ambiguous label')
        data = pdf.tobytes()
    calls = []
    def fake(request):
        calls.append(request)
        sent = json.loads(json.loads(request.content)['contents'][0]['parts'][0]['text'])
        father = next(s for s in sent['sources'] if s.get('field_relation',{}).get('label') == 'Baba Adı')
        assert not any('ambiguous label' in s['text'] for s in sent['sources'])
        reply = {'status':'answered','answer':'SİMÜLASYON: Baba Adı alanı Orhan Örnek.',
                 'citations':[{'source_id':father['id'],'quote':father['field_relation']['quote']}]}
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(reply)}]}}]})
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx,'Client',lambda **kw:real_client(**{**kw,'transport':httpx.MockTransport(fake)}))
    app = AppTest.from_file(str(APP)).run()
    app.session_state['document'] = open_upload('synthetic.pdf', data)
    app.run()
    app.text_input(key='question').set_value('denizin babasinin adi')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    assert not calls and app.button(key='generate_answer').disabled
    app.checkbox(key='cloud_consent').check().run()
    assert not calls and not app.button(key='generate_answer').disabled
    assert any('Varsayılan seçim dışında' in w.value and 'Alan ilişkisi belirsiz' in w.value for w in app.warning)
    selected = app.multiselect(key='answer_sources').value
    assert 0 < len(selected) < len(app.multiselect(key='answer_sources').options)
    preview = next(t.value for t in app.text if t.value.startswith('{\n  "question"'))
    app.button(key='generate_answer').click().run()
    assert not app.exception and len(calls) == 1
    assert json.loads(calls[0].content)['contents'][0]['parts'][0]['text'] == preview
    assert app.session_state['generated_answer']['status'] == 'answered'
    app.selectbox(key='answer_provider').select('Ollama').run()
    assert 'generated_answer' not in app.session_state and len(calls) == 1
    app.multiselect(key='answer_sources').set_value([s['id'] for s in answers.build_evidence(app.session_state['results'])[0]]).run()
    assert app.button(key='generate_answer').disabled
    assert len(calls) == 1


def test_simulated_configured_no_colon_duplicate_values_cannot_be_selected_as_prose(monkeypatch):
    """OFFLINE SIMULATION: configured Gemini UI with invented PDF and fake HTTP."""
    import json
    import httpx
    from src import answers
    from src.workbench import open_upload
    from tests.test_explicit_fields import native_pdf

    monkeypatch.setenv('GEMINI_API_KEY', 'offline-simulation-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-3.8-flash')
    def forbidden(request):
        pytest.fail('ambiguous no-colon form must not contact simulated provider')
    real_client = httpx.Client
    monkeypatch.setattr(answers.httpx, 'Client', lambda **kw: real_client(**{**kw, 'transport': httpx.MockTransport(forbidden)}))
    data = native_pdf([(35,100,'Baba Adı'),(185,100,'Deniz.'),
                       (35,130,'Baba Adı'),(185,130,'Orhan.')])
    app = AppTest.from_file(str(APP)).run()
    app.session_state['document'] = open_upload('synthetic.pdf', data)
    app.run()
    app.text_input(key='question').set_value('Baba Adı Deniz')
    app.button(key='search').click().run()
    app.checkbox(key='generation_enabled').check().run()
    app.selectbox(key='answer_provider').select('Gemini API').run()
    app.checkbox(key='cloud_consent').check().run()
    assert not app.exception
    assert app.selectbox(key='gemini_model').value == 'gemini-3.8-flash'
    assert any(r['excerpt'] == 'Deniz.' for r in app.session_state['results'])
    assert app.multiselect(key='answer_sources').value == []
    assert app.button(key='generate_answer').disabled
    assert any('Alan ilişkisi belirsiz' in w.value for w in app.warning)
    evidence, _ = answers.build_evidence(app.session_state['results'])
    for item in evidence:
        app.multiselect(key='answer_sources').set_value([item['id']]).run()
        assert not app.exception and app.button(key='generate_answer').disabled
        preview = next(t.value for t in app.text if t.value.startswith('{\n  "question"'))
        assert json.loads(preview)['sources'] == [item]
    assert 'generated_answer' not in app.session_state
