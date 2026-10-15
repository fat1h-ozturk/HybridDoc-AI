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
    app.radio(key="source_mode").set_value("Demo PDF").run()
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
    app.radio(key="source_mode").set_value("Demo PDF").run()
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
