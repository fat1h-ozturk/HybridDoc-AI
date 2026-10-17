"""SIMULATED offline milestone demonstration, NOT live model inference.

Run: python -m pytest tests/test_answer_demo.py -q -s
All PDF content is invented here; HTTPX MockTransport never accesses a service.
"""
import json

import httpx
import pymupdf
import pytest

from src import answers
from src.workbench import open_upload
from tests.form_fixtures import certificate_pdf


def invented_pdf(text):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40, 70), text)
        return pdf.tobytes()


@pytest.mark.parametrize('text,question,reply,quote', [
    ('Ankara is the capital of Turkey.', 'What is the capital of Turkey?', 'Türkiye’nin başkenti Ankara’dır.', 'Ankara is the capital of Turkey.'),
    ('Father: Orhan Example', 'Who is Father?', 'Kaynakta Father alanı Orhan Example olarak yazılıdır.', 'Father: Orhan Example'),
    ('Ankara is the capital of Turkey.', 'What is the population of Ankara?', None, None),
])
def test_simulated_prose_explicit_field_and_missing_fact(text, question, reply, quote, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-simulated')
    document = open_upload('invented-demo.pdf', invented_pdf(text))
    evidence, omitted = answers.build_evidence(document.search(question))
    assert evidence and not omitted
    if evidence[0]['association'] == 'not_inferred':
        assert answers.evidence_issue(evidence)
        evidence[0]['user_confirmation'] = answers.confirm_source_passage(evidence, evidence[0]['id'], question)
    def fake(request):
        sent = json.loads(json.loads(request.content)['contents'][0]['parts'][0]['text'])
        assert sent['question'] == question
        assert sent['sources'][0]['text'] == text
        output = {'status': 'answered' if reply else 'abstained',
                  'answer': reply or 'Kaynakta nüfus bilgisi yok. Kaynağı inceleyin.',
                  'citations': [{'source_id': evidence[0]['id'], 'quote': quote}] if quote else []}
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{'text': json.dumps(output)}]}}]})
    result = answers.generate_answer('Gemini API', 'gemini-simulated', question, evidence, consent=True,
                                     transport=httpx.MockTransport(fake))
    assert result['status'] == ('answered' if reply else 'abstained')
    assert result['answer'] == (reply or 'Kaynakta nüfus bilgisi yok. Kaynağı inceleyin.')
    print(f"SIMÜLASYON — {question}: {result['answer']}")


def test_ambiguous_form_and_unverified_math_are_source_only():
    cases = [
        (certificate_pdf(columns=2), 'Baba Adı', 'Alan ilişkisi belirsiz'),
        (invented_pdf('Unverified formula: x = y - 2'), 'formula', 'matematik'),
    ]
    for data, question, reason in cases:
        document = open_upload('invented.pdf', data)
        evidence, _ = answers.build_evidence(document.search(question))
        def forbidden(request):
            pytest.fail('abstention must happen before HTTP')
        with pytest.raises(answers.AnswerError, match=reason) as error:
            answers.generate_answer('Ollama', 'simulated', question, evidence,
                                    transport=httpx.MockTransport(forbidden))
        print(f'SİMÜLASYON — {question}: {error.value}')
