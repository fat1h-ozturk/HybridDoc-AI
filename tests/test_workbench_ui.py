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
        f"Kaynak 1: {res['file_name']} — Sayfa {res['page_number']} / {res['total_pages']}" in s.value
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
