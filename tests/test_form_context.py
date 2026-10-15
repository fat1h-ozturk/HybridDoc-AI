"""Exercise real PDF upload/search: source context must survive isolated spans."""
import hashlib
from pathlib import Path

import pymupdf
import pytest

from src.workbench import open_upload
from src.retrieval.search_pdf import search_pdf, format_search_results
from tests.form_fixtures import certificate_pdf

FIXTURE = Path(__file__).parent / 'fixtures' / 'synthetic_student_certificate.pdf'


@pytest.mark.parametrize('order', ['columns', 'rows', 'reverse'])
@pytest.mark.parametrize('query', ['denizin babasinin adi', 'Deniz’in babasının adı', 'Baba Adı', 'baba adi'])
def test_parent_source_context_survives_drawing_order(order, query):
    data = certificate_pdf(order=order)
    document = open_upload('invented.pdf', data)
    results = document.search(query)
    father = next(r for r in results if r['excerpt'] == 'Baba Adı')
    context = father['source_context']
    assert 'Baba Adı' in context['text']
    assert 'Orhan Örnek' in context['text']
    # A source band is evidence for inspection, never an inferred parent answer.
    assert context['association'] == 'not_inferred'
    assert father['answer_eligible'] is False
    assert father['source_sha256'] == hashlib.sha256(data).hexdigest()
    assert father['file_name'] == 'invented.pdf'
    assert father['page_number'] == 1
    assert 'Orhan Örnek' in format_search_results([father], query)
    crop = document.source_crop(father)
    pixmap = pymupdf.Pixmap(crop)
    assert pixmap.width > pixmap.height > 0
    assert pixmap.width <= 1200


@pytest.mark.parametrize('label,value,other', [
    ('Adı / Soyadı', 'Deniz Örnek', 'Orhan Örnek'),
    ('Anne Adı', 'Selin Örnek', 'Deniz Örnek'),
    ('Baba Adı', 'Orhan Örnek', 'Selin Örnek'),
])
def test_rows_are_readable_and_not_reassigned(label, value, other):
    document = open_upload('invented.pdf', FIXTURE.read_bytes())
    result = next(r for r in document.search(label) if r['excerpt'] == label)
    lines = result['source_context']['text'].splitlines()
    row = next(line for line in lines if label in line)
    assert value in row
    assert other not in row


@pytest.mark.parametrize('order', ['columns', 'reverse'])
def test_two_columns_remain_visible_without_choosing_a_name(order):
    document = open_upload('invented.pdf', certificate_pdf(columns=2, order=order))
    result = next(r for r in document.search('baba adi') if r['excerpt'] == 'Baba Adı')
    context = result['source_context']
    row = next(line for line in context['text'].splitlines() if 'Orhan Örnek' in line)
    assert row.count('Baba Adı') == 2
    assert row.index('Baba Adı') < row.index('Orhan Örnek') < row.rindex('Baba Adı') < row.index('Şükrü Kurgu')
    assert context['association'] == 'not_inferred'


@pytest.mark.parametrize('father', ['', 'Selin Örnek', 'Deniz Örnek'])
def test_empty_or_same_names_do_not_infer_a_father(father):
    document = open_upload('invented.pdf', certificate_pdf(father=father, second_page=True))
    result = next(r for r in document.search('baba adi') if r['excerpt'] == 'Baba Adı')
    context = result['source_context']
    assert 'Uzak Sayfa' not in context['text']
    row = next(line for line in context['text'].splitlines() if 'Baba Adı' in line)
    if father:
        assert father in row
    else:
        assert 'Selin Örnek' not in row
        assert 'Deniz Örnek' not in row
        assert 'Orhan Örnek' not in context['text']
    assert context['association'] == 'not_inferred'
    assert result['answer_eligible'] is False


def test_stacked_value_uses_bounded_context_not_next_string_guess():
    document = open_upload('invented.pdf', certificate_pdf(stacked=True, order='reverse'))
    result = next(r for r in document.search('baba adi') if r['excerpt'] == 'Baba Adı')
    context = result['source_context']
    assert 'Orhan Örnek' in context['text']
    assert 'Selin Örnek' not in context['text']
    assert context['bbox'][3] - context['bbox'][1] <= 160
    assert context['association'] == 'not_inferred'


def test_existing_prose_no_match_and_public_search_contract():
    document = open_upload('workbench_demo.pdf', (FIXTURE.parents[2] / 'sample_docs/workbench_demo.pdf').read_bytes())
    result = document.search('source passages')[0]
    assert result['page_number'] == 2
    assert 'Source passages are exact quotes' in result['excerpt']
    assert document.search('quasar') == []
    results = search_pdf(FIXTURE, 'baba adi')
    assert results[0]['excerpt'] == 'Baba Adı'
    assert results[0]['answer_eligible'] is False
    assert 'source_context' not in results[0]


def test_accent_contract_is_lexical_not_silent_stemming_or_translation():
    document = open_upload('invented.pdf', FIXTURE.read_bytes())
    assert [r['chunk_id'] for r in document.search('Baba Adı')] == [
        r['chunk_id'] for r in document.search('baba adi')]
    assert document.search('ÖĞRENCİ')
    assert document.search('ogrenci') == []
    assert document.search('babasinin') == []
    assert document.search('babasının') == []


def test_native_context_does_not_promote_nearby_math():
    with pymupdf.open(stream=FIXTURE.read_bytes(), filetype='pdf') as pdf:
        pdf[0].insert_text((35, 193), 'x = y - 2')
        data = pdf.tobytes()
    document = open_upload('mixed.pdf', data)
    results = document.search('baba adi')
    father = next(r for r in results if r['excerpt'] == 'Baba Adı')
    assert 'x = y - 2' in father['source_context']['text']
    assert all(r['answer_eligible'] is False for r in results)
    formulas = document.search('x y')
    assert formulas
    assert all(r['answer_eligible'] is False for r in formulas)
    assert all('source_context' not in r for r in formulas)
    assert all(f['answer_eligible'] is False for r in formulas for f in r['formulas'])


def test_rotated_pdf_uses_same_unrotated_source_geometry():
    with pymupdf.open(stream=FIXTURE.read_bytes(), filetype='pdf') as pdf:
        pdf[0].set_rotation(90)
        data = pdf.tobytes()
    document = open_upload('rotated.pdf', data)
    father = next(r for r in document.search('baba adi') if r['excerpt'] == 'Baba Adı')
    assert 'Orhan Örnek' in father['source_context']['text']
    image = pymupdf.Pixmap(document.source_crop(father))
    assert image.width > image.height
