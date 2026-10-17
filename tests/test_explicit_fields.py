"""Synthetic native-geometry regressions; provider checks are OFFLINE SIMULATION."""
import json

import httpx
import pymupdf
import pytest

from src import answers
from src.workbench import open_upload
from tests.form_fixtures import certificate_pdf, mixed_font_prose_pdf, wrapped_prose_pdf

QUESTION = 'denizin babasinin adi'


def native_pdf(entries, *, rotation=0, text_rotation=0, pages=1, table=False):
    with pymupdf.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page(width=650, height=500)
            page.insert_font(fontname='fixture', fontbuffer=pymupdf.Font('cjk').buffer)
            for x, y, text in entries:
                page.insert_text((x, y), text, fontsize=11, fontname='fixture', rotate=text_rotation)
            if table:
                for x in (25, 170, 330):
                    page.draw_line((x, 70), (x, 150))
                for y in (70, 110, 150):
                    page.draw_line((25, y), (330, y))
            page.set_rotation(rotation)
        pdf.subset_fonts()
        return pdf.tobytes(garbage=4, deflate=True)


def evidence_for(data, query=QUESTION):
    doc = open_upload('synthetic.pdf', data)
    results = doc.search(query)
    evidence, _ = answers.build_evidence(results)
    return doc, results, evidence


@pytest.mark.parametrize('inline,order', [(False,'columns'), (False,'reverse'), (True,'reverse')])
def test_native_explicit_field_has_label_value_original_geometry_and_quote(inline, order):
    doc, results, evidence = evidence_for(certificate_pdf(colon=True, inline=inline, order=order))
    father = next(r for r in results if r['excerpt'].startswith('Baba Adı'))
    context = father['source_context']
    assert context['association'] == 'explicit_same_row'
    relation = context['field_relation']
    assert relation['label'] == 'Baba Adı'
    assert relation['value'] == 'Orhan Örnek'
    assert relation['quote'] in context['text']
    assert 'Baba Adı:' in relation['quote'] and 'Orhan Örnek' in relation['quote']
    assert relation['spans'] and all(s['text'] and len(s['bbox']) == 4 for s in relation['spans'])
    assert father['answer_eligible'] is False
    item = next(s for s in evidence if s.get('field_relation', {}).get('label') == 'Baba Adı')
    assert item['source_sha256'] == father['source_sha256']
    assert item['page'] == father['page_number'] == 1
    assert item['field_relation'] == relation
    assert answers.evidence_issue([item]) is None
    assert pymupdf.Pixmap(doc.source_crop(father)).width > 0


@pytest.mark.parametrize('entries', [
    [(35,100,'Baba Adı:'), (185,130,'Other Row Value')],
    [(35,100,'Baba Adı:'), (185,105,'Wrong Baseline')],
    [(35,100,'Baba Adı:'), (185,100,'One'), (250,100,'Two')],
    [(35,100,'Baba Adı:'), (185,100,'One'), (350,100,'Baba Adı:'), (500,100,'Two')],
    [(35,100,'Baba Adı: One'), (35,140,'Baba Adı: Two')],
    [(35,100,'Baba Adı: One'), (35,140,'Baba Adı: One')],
    [(35,100,'Baba Adı: One.'), (35,140,'Baba Adı: Two.')],
    [(35,100,'Baba Adı: One'), (35,140,'Baba Adı:')],
    [(35,85,'Baba'), (35,100,'Adı:'), (185,100,'One')],
    [(35,100,'Baba Adı:'), (35,140,'Anne Adı: Other')],
])
def test_ambiguous_geometry_never_becomes_field_evidence(entries):
    _, results, evidence = evidence_for(native_pdf(entries), 'Baba Adı')
    assert results
    assert all(r['source_context']['association'] == 'not_inferred' for r in results
               if r['excerpt'].startswith(('Baba', 'Adı')))
    fathers = [e for e in evidence if e.get('association') == 'not_inferred']
    assert fathers and answers.evidence_issue(fathers)


@pytest.mark.parametrize('rotation', [0,90,180,270])
def test_page_rotation_preserves_native_same_row_relation(rotation):
    _, _, evidence = evidence_for(native_pdf([(35,100,'Baba Adı:'),(185,100,'One')],rotation=rotation))
    assert evidence[0]['association'] == 'explicit_same_row'


def test_separate_page_text_and_vertical_text_do_not_pair():
    _, _, evidence = evidence_for(native_pdf([(200,350,'Baba Adı:'),(350,350,'One')],text_rotation=90))
    assert all(e['association'] == 'not_inferred' for e in evidence)
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((35,100),'Father:')
        pdf.new_page().insert_text((185,100),'Value')
        data = pdf.tobytes()
    _, _, evidence = evidence_for(data, 'Father')
    assert answers.evidence_issue(evidence)


def test_duplicate_label_across_pages_blocks_and_equal_values_on_distinct_fields_work():
    _, _, evidence = evidence_for(native_pdf([(35,100,'Baba Adı: One')],pages=2))
    assert answers.evidence_issue(evidence)
    _, _, evidence = evidence_for(certificate_pdf(colon=True, father='Selin Örnek'))
    fields = {e['field_relation']['label']:e['field_relation']['value'] for e in evidence}
    assert fields['Baba Adı'] == fields['Anne Adı'] == 'Selin Örnek'


def test_colon_formula_retains_math_guard():
    _, _, evidence = evidence_for(native_pdf([(35,100,'Result: x = y - 2')]), 'Result')
    assert 'matematik' in answers.evidence_issue(evidence)


def test_simulated_natural_question_and_full_field_citation(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS','gemini-simulated')
    _, _, evidence = evidence_for(certificate_pdf(colon=True))
    source = next(e for e in evidence if e.get('field_relation',{}).get('label') == 'Baba Adı')
    calls = []
    def fake(request):
        calls.append(request)
        assert json.loads(request.content)['contents'][0]['parts'][0]['text'] == answers.request_preview(QUESTION,[source])
        output = {'status':'answered','answer':'Baba Adı alanında Orhan Örnek yazıyor.',
                  'citations':[{'source_id':source['id'],'quote':source['field_relation']['quote']}]}
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(output)}]}}]})
    assert not calls
    result = answers.generate_answer('Gemini API','gemini-simulated',QUESTION,[source],consent=True,transport=httpx.MockTransport(fake))
    assert len(calls) == 1 and result['status'] == 'answered'
    for quote in ['Baba Adı:', 'Orhan Örnek']:
        result['citations'][0]['quote'] = quote
        with pytest.raises(answers.AnswerError):
            answers.validate_answer(json.dumps(result), [source])


def test_unknown_label_is_not_a_document_specific_rule_and_unknown_query_has_no_evidence():
    _, _, evidence = evidence_for(native_pdf([(35,100,'Reference Code:'),(185,100,'ABC')]), 'Reference')
    assert evidence[0]['field_relation']['label'] == 'Reference Code'
    assert answers.evidence_issue(evidence) is None
    _, results, evidence = evidence_for(certificate_pdf(colon=True), 'quasar')
    assert results == [] and evidence == []
    assert 'kaynak' in answers.evidence_issue(evidence)


def test_simulated_missing_father_abstains_with_other_clear_sources(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','offline-fake-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS','gemini-simulated')
    _, results, evidence = evidence_for(certificate_pdf(colon=True, father=''))
    father = next(r for r in results if r['excerpt'] == 'Baba Adı:')
    assert father['source_context']['association'] == 'not_inferred'
    usable = [e for e in evidence if not answers.evidence_issue([e])]
    assert usable and not any(e['field_relation']['label'] == 'Baba Adı' for e in usable)
    def fake(request):
        reply = {'status':'abstained','answer':'Baba Adı alanında değer yok; kaynak kesitini inceleyin.', 'citations':[]}
        return httpx.Response(200,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps(reply)}]}}]})
    result = answers.generate_answer('Gemini API','gemini-simulated',QUESTION,usable,consent=True,transport=httpx.MockTransport(fake))
    assert result['status'] == 'abstained' and 'değer yok' in result['answer']


def test_ambiguous_duplicate_elsewhere_cannot_leave_one_clear_conflicting_field():
    entries = [(35,100,'Baba Adı: One'), (350,100,'Baba Adı: Two'), (35,150,'Baba Adı: Three')]
    _, results, evidence = evidence_for(native_pdf(entries))
    assert all(r['source_context']['association'] == 'not_inferred' for r in results)
    assert all(answers.evidence_issue([e]) for e in evidence)


def test_colon_in_its_own_native_span_is_explicit():
    _, _, evidence = evidence_for(native_pdf([(35,100,'Baba Adı'),(100,100,':'),(185,100,'One')]))
    assert evidence[0]['association'] == 'explicit_same_row'


@pytest.mark.parametrize('entries', [
    [(35,100,'Baba Adı:'),(185,100,'Unrelated.')],
    [(35,100,'Baba Adı'),(100,100,':'),(185,100,'Unrelated.')],
])
def test_blank_table_cell_cannot_take_adjacent_value(entries):
    data = native_pdf(entries, table=True)
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        assert len(pdf[0].find_tables().tables[0].rows[0].cells) == 2
    _, results, evidence = evidence_for(data, 'Baba Adı Unrelated')
    assert results and evidence
    assert all(r['source_context']['association'] == 'not_inferred' for r in results)
    assert all(answers.evidence_issue([e]) for e in evidence)


def test_same_cell_field_still_works():
    _, _, evidence = evidence_for(native_pdf([(35,100,'Baba Adı: One')], table=True))
    assert evidence[0]['association'] == 'explicit_same_row'
    assert answers.evidence_issue(evidence) is None


def test_separate_colon_competitors_count_against_third_field():
    entries = [(35,100,'Baba Adı'),(100,100,':'),(185,100,'One'),
               (350,100,'Baba Adı'),(420,100,':'),(500,100,'Two'),
               (35,150,'Baba Adı: Three')]
    _, results, evidence = evidence_for(native_pdf(entries), 'Baba Adı Three')
    assert results and evidence
    assert all(r['source_context']['association'] == 'not_inferred' for r in results)
    assert all(answers.evidence_issue([e]) for e in evidence)


@pytest.mark.parametrize('provider', ['Ollama', 'Gemini API'])
def test_punctuated_ambiguous_value_is_blocked_before_simulated_http(provider):
    entries = [(35,100,'Baba Adı:'),(185,100,'Deniz.'),
               (35,130,'Baba Adı:'),(185,130,'Orhan.')]
    _, results, evidence = evidence_for(native_pdf(entries), 'Baba Adı Deniz')
    assert any(r['excerpt'] == 'Deniz.' for r in results)
    assert evidence and all(answers.evidence_issue([e]) for e in evidence)
    assert all(e['requires_source_review'] for e in evidence)
    def forbidden(request):
        pytest.fail('ambiguous value must not reach simulated HTTP transport')
    for item in evidence:
        with pytest.raises(answers.AnswerError, match='association=not_inferred'):
            answers.generate_answer(provider, 'simulated', 'Baba Adı Deniz', [item],
                                    consent=True, transport=httpx.MockTransport(forbidden))
        assert answers.evidence_issue([{**item, 'requires_source_review': False}])


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('value_on_left', [False, True])
@pytest.mark.parametrize('value', ['Deniz', 'Orhan'])
def test_no_colon_duplicate_values_stay_ambiguous(rotation, value_on_left, value):
    import hashlib

    label_x, value_x = (185, 35) if value_on_left else (35, 185)
    entries = [(label_x,100,'Baba Adı'),(value_x,100,'Deniz.'),
               (label_x,130,'Baba Adı'),(value_x,130,'Orhan.')]
    data = native_pdf(entries, rotation=rotation)
    doc, results, evidence = evidence_for(data, f'Baba Adı {value}')
    assert any(r['excerpt'] == value + '.' for r in results)
    assert evidence and all(e['requires_source_review'] for e in evidence)
    assert all(e['association'] == 'not_inferred' for e in evidence)
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        page = pdf[0]
        page.set_rotation(0)
        for result, item in zip(results, evidence):
            context = result['source_context']
            assert context['text'] == page.get_text('text', clip=pymupdf.Rect(context['bbox']), sort=True).strip()
            assert item['text'] == context['text']
            assert item['page'] == result['page_number'] == 1
            assert item['source_sha256'] == hashlib.sha256(data).hexdigest()
            assert item['answer_eligible'] is False
            assert pymupdf.Pixmap(doc.source_crop(result)).width > 0
    def forbidden(request):
        pytest.fail('ambiguous no-colon value must not reach simulated HTTP')
    for item in evidence:
        assert answers.evidence_issue([item])
        for provider in ('Ollama', 'Gemini API'):
            with pytest.raises(answers.AnswerError, match='association=not_inferred'):
                answers.generate_answer(provider, 'simulated', f'Baba Adı {value}', [item],
                                        consent=True, transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize('rotation', [0,90,180,270])
@pytest.mark.parametrize('provider', ['Ollama', 'Gemini API'])
def test_split_square_root_field_blocks_before_simulated_http(rotation, provider):
    data = native_pdf([(35,100,'Result:'),(185,100,'√2')], rotation=rotation)
    doc, results, evidence = evidence_for(data, 'Result')
    result = next(r for r in results if r['excerpt'] == 'Result:')
    context = result['source_context']
    assert context['association'] == 'explicit_same_row'
    assert context['field_relation']['value'] == '√2'
    assert '√2' in context['text']
    original = doc.formula_pages[0]['formulas']
    assert len(original) == 1 and original[0]['state'] == 'unresolved'
    assert original[0]['answer_eligible'] is False
    assert result['formulas'] == original
    item = next(e for e in evidence if e['text'] == context['text'])
    assert item['formulas'] == original
    assert item['answer_eligible'] is False
    assert item['page'] == 1 and item['source_sha256'] == original[0]['source_sha256']
    assert json.loads(answers.request_preview('Result', [item]))['sources'][0]['text'] == context['text']
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        page = pdf[0]
        page.set_rotation(0)
        assert context['text'] == page.get_text('text', clip=pymupdf.Rect(context['bbox']), sort=True).strip()
    assert pymupdf.Pixmap(doc.source_crop(result)).width > 0
    assert 'matematik' in answers.evidence_issue([item])
    def forbidden(request):
        pytest.fail('unresolved formula must not reach simulated HTTP transport')
    with pytest.raises(answers.AnswerError, match='matematik'):
        answers.generate_answer(provider, 'simulated', 'Result', [item], consent=True,
                                transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('value_first', [False, True])
@pytest.mark.parametrize('value', ['Deniz', 'Orhan'])
@pytest.mark.parametrize('gap', [20, 40, 65])
@pytest.mark.parametrize('reverse', [False, True])
def test_stacked_no_colon_values_stay_ambiguous(rotation, value_first, value, gap, reverse):
    offset = -gap if value_first else gap
    second_label = 100 + 3 * gap
    entries = [(35,100,'Baba Adı'),(35,100 + offset,'Deniz.'),
               (35,second_label,'Baba Adı'),(35,second_label + offset,'Orhan.')]
    if reverse:
        entries.reverse()
    data = native_pdf(entries, rotation=rotation)
    doc, results, evidence = evidence_for(data, f'Baba Adı {value}')
    assert any(r['excerpt'] == value + '.' for r in results)
    assert evidence and all(e['requires_source_review'] for e in evidence)
    assert all(e['association'] == 'not_inferred' and 'field_relation' not in e for e in evidence)
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        pdf[0].set_rotation(0)
        for result, item in zip(results, evidence):
            context = result['source_context']
            assert item['text'] == context['text'] == pdf[0].get_text(
                'text', clip=pymupdf.Rect(context['bbox']), sort=True).strip()
            assert pymupdf.Pixmap(doc.source_crop(result)).width > 0
    def forbidden(request):
        pytest.fail('stacked form must not reach simulated HTTP')
    for item in evidence:
        for provider in ('Ollama', 'Gemini API'):
            with pytest.raises(answers.AnswerError, match='association=not_inferred'):
                answers.generate_answer(provider, 'simulated', f'Baba Adı {value}', [item],
                                        consent=True, transport=httpx.MockTransport(forbidden))


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('query', ['Turkey', 'Ankara'])
def test_contiguous_mixed_font_prose_is_usable_from_either_span(rotation, reverse, query):
    import hashlib

    data = mixed_font_prose_pdf(rotation=rotation, reverse=reverse)
    doc, results, evidence = evidence_for(data, query)
    assert len(results) == len(evidence) == 1
    assert results[0]['excerpt'] in ('Turkey.', 'Ankara is the capital of ')
    item = evidence[0]
    assert answers.evidence_issue(evidence)
    item['user_confirmation'] = answers.confirm_source_passage(evidence, item['id'], query)
    assert answers.evidence_issue(evidence) is None
    assert item['association'] == 'not_inferred' and 'field_relation' not in item
    assert item['page'] == 1 and item['source_sha256'] == hashlib.sha256(data).hexdigest()
    assert item['answer_eligible'] is False
    assert 'Ankara is the capital of' in item['text'] and 'Turkey.' in item['text']
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        pdf[0].set_rotation(0)
        assert item['text'] == pdf[0].get_text(
            'text', clip=pymupdf.Rect(results[0]['source_context']['bbox']), sort=True).strip()
        spans = [s for b in pdf[0].get_text('dict')['blocks'] if b['type'] == 0
                 for line in b['lines'] for s in line['spans']]
        assert len(spans) == 2 and len({s['font'] for s in spans}) == 2
        assert sorted(b['bbox'] for b in doc.formula_pages[0]['blocks'] if b['kind'] == 'text') == sorted(
            list(s['bbox']) for s in spans)
    assert pymupdf.Pixmap(doc.source_crop(results[0])).width > 0


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('reverse,native_paragraph,single_word_end', [
    (False, False, False), (True, False, False), (False, True, False), (False, True, True)])
def test_wrapped_prose_retains_source_and_supported_citation(rotation, reverse, native_paragraph, single_word_end, monkeypatch):
    import hashlib

    data = wrapped_prose_pdf(rotation=rotation, reverse=reverse, native_paragraph=native_paragraph,
                             single_word_end=single_word_end)
    doc, results, evidence = evidence_for(data, 'city')
    assert len(results) == len(evidence) == 1
    assert results[0]['excerpt'] == ('city.' if single_word_end else 'a major city.')
    item = evidence[0]
    assert answers.evidence_issue(evidence)
    item['user_confirmation'] = answers.confirm_source_passage(evidence, item['id'], 'city')
    assert answers.evidence_issue(evidence) is None
    assert item['association'] == 'not_inferred' and 'field_relation' not in item
    assert item['page'] == 1 and item['source_sha256'] == hashlib.sha256(data).hexdigest()
    assert item['answer_eligible'] is False
    assert 'Ankara is the capital of Turkey and' in item['text']
    assert ('city.' if single_word_end else 'a major city.') in item['text']
    with pymupdf.open(stream=data, filetype='pdf') as pdf:
        pdf[0].set_rotation(0)
        assert item['text'] == results[0]['source_context']['text'] == pdf[0].get_text(
            'text', clip=pymupdf.Rect(results[0]['source_context']['bbox']), sort=True).strip()
    assert pymupdf.Pixmap(doc.source_crop(results[0])).width > 0
    monkeypatch.setenv('GEMINI_API_KEY', 'offline-simulation-key')
    monkeypatch.setenv('HYBRIDDOC_GEMINI_MODELS', 'gemini-simulated')
    calls = []
    def fake(request):
        calls.append(request)
        assert json.loads(request.content)['contents'][0]['parts'][0]['text'] == answers.request_preview('city', evidence)
        reply = {'status': 'answered', 'answer': 'SİMÜLASYON: Ankara büyük bir şehirdir.',
                 'citations': [{'source_id': item['id'], 'quote': item['text']}]}
        return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
            'parts': [{'text': json.dumps(reply)}]}}]})
    answer = answers.generate_answer('Gemini API', 'gemini-simulated', 'city', evidence,
                                     consent=True, transport=httpx.MockTransport(fake))
    assert len(calls) == 1 and answer['status'] == 'answered'


@pytest.mark.parametrize('formula_y,blocked', [(80,True),(120,True),(300,False)])
def test_only_formula_records_overlapping_full_source_band_are_retained(formula_y, blocked):
    doc, results, evidence = evidence_for(native_pdf([
        (35,100,'Reference Code: ABC'), (350,formula_y,'√2')]), 'Reference')
    assert doc.formula_pages[0]['formulas']
    assert bool(results[0]['formulas']) is blocked
    assert bool(evidence[0]['formulas']) is blocked
    assert bool(answers.evidence_issue(evidence)) is blocked


def test_source_band_retains_overlapping_wrapped_formula_group():
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=400)
        page.insert_text((40,60), 'f(x) = C')
        page.insert_text((110,60), ' on the branch of the initial point. The condition gives ')
        page.insert_text((420,60), 'K =')
        page.insert_text((40,76), '2')
        page.insert_text((46,73), '-', fontsize=6)
        page.insert_text((51,71), '2', fontsize=6)
        page.insert_text((51,76), '3', fontsize=6)
        page.draw_line((51,73), (56,73))
        page.insert_text((60,76), '. More prose.')
        data = pdf.tobytes()
    doc, results, evidence = evidence_for(data, 'branch')
    groups = doc.formula_pages[0]['formula_groups']
    assert len(groups) == 1
    assert results[0]['formula_groups'] == groups
    assert evidence[0]['formula_groups'] == groups
    assert groups[0]['answer_eligible'] is False
    assert all(f['answer_eligible'] is False for f in evidence[0]['formulas'])
    assert 'matematik' in answers.evidence_issue(evidence)
