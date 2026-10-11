"""Public synthetic geometry, not coordinates copied from a private document."""
import pymupdf
import pytest
from src.ingestion import pdf_loader


def optional_load(path, **kwargs):
    load = getattr(pdf_loader, 'load_pdf_formulas', None)
    assert callable(load), 'optional formula-aware loader is missing'
    return load(path, **kwargs)


@pytest.fixture
def math_pdf(tmp_path):
    path = tmp_path / 'synthetic.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=330)
        p.insert_text((30, 35), 'Plain prose stays readable.')
        p.insert_text((40, 90), "y' =")
        p.insert_text((100, 84), 'y')
        p.insert_text((100, 105), 'x')
        p.draw_line((97, 91), (111, 91))
        writer = pymupdf.TextWriter(p.rect)
        writer.append((80, 93), '√', font=pymupdf.Font('symbol'), fontsize=20)
        writer.write_text(p)
        p.draw_line((93, 73), (114, 73))
        p.insert_text((40, 150), 'f(x) = x + a')
        p.insert_text((77, 143), '2', fontsize=7)
        p.insert_text((101, 155), 'n-1', fontsize=7)
        for x in (30, 150, 270):
            p.draw_line((x, 210), (x, 300))
        for y in (210, 245, 300):
            p.draw_line((30, y), (270, y))
        p.insert_text((40, 230), 'Equation')
        p.insert_text((165, 230), 'Type')
        p.insert_text((40, 270), "z' =")
        p.insert_text((85, 264), 'z')
        p.insert_text((85, 285), 't')
        p.draw_line((82, 271), (96, 271))
        p.insert_text((165, 275), 'ODE')
        doc.save(path)
    return path


def test_native_regions_keep_fraction_root_and_scripts_with_provenance(math_pdf):
    pages = optional_load(math_pdf)
    assert len(pages) == 1
    page = pages[0]
    assert page['page_number'] == 1 and page['total_pages'] == 1
    assert page['file_name'] == 'synthetic.pdf'
    assert len(page['source_sha256']) == 64
    assert 'Plain prose stays readable.' in page['text']
    assert page['original_text'] == pdf_loader.load_pdf(math_pdf)[0]['text']
    root = next(f for f in page['formulas'] if "y'" in f['original_text'])
    assert root['bbox'][0] <= 40 and root['bbox'][2] >= 111
    assert root['bbox'][1] <= 73 and root['bbox'][3] >= 105
    assert 'y' in root['original_text'] and 'x' in root['original_text']
    assert root['state'] == 'unresolved' and not root['answer_eligible']
    scripts = next(f for f in page['formulas'] if 'f(x)' in f['original_text'])
    assert scripts['bbox'][1] < 143 and scripts['bbox'][3] >= 155
    assert 'n-1' in scripts['original_text']
    assert "y' =" not in page['text']  # no silently trusted flat replacement


def test_table_candidate_keeps_its_cell_without_neighboring_classification(math_pdf):
    page = optional_load(math_pdf)[0]
    formula = next(f for f in page['formulas'] if "z'" in f['original_text'])
    assert formula['table_cell']['row'] == 1
    assert formula['table_cell']['column'] == 0
    assert formula['bbox'][2] <= 150
    assert 'ODE' not in formula['original_text']
    assert 'ODE' in page['text']


@pytest.mark.parametrize('outside_xy,inside_xy,row,column', [
    ((80, 165), (110, 165), 1, 0),
    ((190, 165), (160, 165), 1, 1),
    ((110, 95), (110, 115), 0, 0),
    ((110, 195), (110, 175), 1, 0),
])
@pytest.mark.parametrize('cell_text', ['y=2', 'tag'])
def test_outside_formula_does_not_absorb_adjacent_table_spans(
        tmp_path, outside_xy, inside_xy, row, column, cell_text):
    path = tmp_path / 'adjacent.pdf'
    crops = []

    class Recognizer:
        def recognize(self, png):
            crops.append(png)
            return 'candidate'

    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        for x in (100, 140, 180):
            page.draw_line((x, 100), (x, 180))
        for y in (100, 140, 180):
            page.draw_line((100, y), (180, y))
        page.insert_text(outside_xy, 'x=1')
        page.insert_text(inside_xy, cell_text)
        page.insert_text((150 if column == 0 else 110,
                          125 if row == 1 else 165), 'ODE')
        doc.save(path)

    native = optional_load(path)[0]
    result = optional_load(path, recognizer=Recognizer())[0]
    for extracted in (native, result):
        outside = next(f for f in extracted['formulas'] if 'x=1' in f['original_text'])
        assert outside['original_text'] == 'x=1'
        assert 'table_cell' not in outside
        assert 'ODE' in extracted['text']
        cells = [f for f in extracted['formulas'] if 'table_cell' in f]
        if cell_text == 'y=2':
            assert len(extracted['formulas']) == 2
            assert len(cells) == 1
        elif not cells:
            assert 'tag' in extracted['text']
        for cell in cells:
            assert cell['original_text'] == cell_text
            assert cell['table_cell'] == {'table': 0, 'row': row, 'column': column}
        assert all(not f['answer_eligible'] for f in extracted['formulas'])
    assert len(crops) == len(result['formulas'])
    with pymupdf.open(path) as doc:
        for formula, crop in zip(result['formulas'], crops):
            expected = doc[0].get_pixmap(
                matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(formula['bbox']))
            assert pymupdf.Pixmap(crop).samples == expected.samples


@pytest.mark.parametrize('in_table', [False, True])
def test_fraction_bar_groups_distant_operands_for_recognition(tmp_path, in_table):
    path = tmp_path / 'fraction.pdf'
    crops = []

    class Recognizer:
        def recognize(self, png):
            crops.append(png)
            return r'\frac{x}{y}'

    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=300)
        if in_table:
            for x in (90, 130, 170):
                p.draw_line((x, 60), (x, 180))
            for y in (60, 130, 180):
                p.draw_line((90, y), (170, y))
            p.insert_text((140, 110), 'tag')
        p.insert_text((100, 85), 'x')
        p.insert_text((100, 110), 'y')
        p.draw_line((98, 94), (110, 94))
        doc.save(path)

    native = optional_load(path)[0]
    result = optional_load(path, recognizer=Recognizer())[0]
    for page in (native, result):
        assert len(page['formulas']) == 1
        formula = page['formulas'][0]
        assert formula['original_text'].split() == ['x', 'y']
        assert formula['bbox'][0] <= 98 and formula['bbox'][2] >= 110
        assert formula['bbox'][1] <= 73 and formula['bbox'][3] >= 113
        assert formula['native_rules'] == [[98, 94, 110, 94]]
        assert not formula['answer_eligible']
        if in_table:
            assert formula['table_cell'] == {'table': 0, 'row': 0, 'column': 0}
            assert 'tag' in page['text']
        else:
            assert 'table_cell' not in formula
    assert len(crops) == 1
    formula = result['formulas'][0]
    assert formula['recognition_status'] == 'returned'
    assert formula['state'] == 'unresolved'
    with pymupdf.open(path) as doc:
        expected = doc[0].get_pixmap(
            matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(formula['bbox']))
        assert pymupdf.Pixmap(crops[0]).samples == expected.samples


@pytest.mark.parametrize('baseline', [95, 195])
@pytest.mark.parametrize('reverse', [False, True])
def test_table_borders_do_not_join_outside_equation_crops(tmp_path, baseline, reverse):
    path = tmp_path / 'border.pdf'
    crops = []

    class Recognizer:
        def recognize(self, png):
            crops.append(png)
            return 'candidate'

    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=300)
        for x in (100, 140, 180):
            p.draw_line((x, 100), (x, 180))
        for y in (100, 140, 180):
            start, end = ((180, y), (100, y)) if reverse else ((100, y), (180, y))
            p.draw_line(start, end)
        p.insert_text((110, baseline), 'x=1')
        p.insert_text((160, baseline), 'z=2')
        p.insert_text((110, 115 if baseline == 95 else 175), 'tag')
        p.insert_text((150, 165 if baseline == 95 else 125), 'ODE')
        doc.save(path)

    native = optional_load(path)[0]
    result = optional_load(path, recognizer=Recognizer())[0]
    for page in (native, result):
        assert [f['original_text'] for f in page['formulas']] == ['x=1', 'z=2']
        assert 'tag' in page['text'] and 'ODE' in page['text']
        for formula in page['formulas']:
            assert formula['native_rules'] == []
            assert formula['bbox'][2] - formula['bbox'][0] < 30
            assert 'table_cell' not in formula
            assert not formula['answer_eligible']
    assert len(crops) == 2
    with pymupdf.open(path) as doc:
        for formula, crop in zip(result['formulas'], crops):
            expected = doc[0].get_pixmap(
                matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(formula['bbox']))
            assert pymupdf.Pixmap(crop).samples == expected.samples


def test_returned_latex_remains_explicitly_unverified(math_pdf):
    class Recognizer:
        def recognize(self, png):
            assert png.startswith(b'\x89PNG')
            return r"y'=\sqrt{\frac{y}{x}}"
    page = optional_load(math_pdf, recognizer=Recognizer())[0]
    root = next(f for f in page['formulas'] if "y'" in f['original_text'])
    assert root['latex'] == r"y'=\sqrt{\frac{y}{x}}"
    assert root['state'] == 'unresolved'  # scope still requires source verification
    assert root['recognition_status'] == 'returned'
    assert not root['answer_eligible']
    block = next(b for b in page['blocks'] if b.get('formula_id') == root['id'])
    assert root['latex'] in block['text']
    assert 'UNRESOLVED' in block['text']


@pytest.mark.parametrize('error', [TimeoutError('crop timed out'), RuntimeError('weights missing')])
def test_recognition_failure_retains_source_and_explicit_unresolved(math_pdf, error):
    class Broken:
        def recognize(self, png):
            raise error
    page = optional_load(math_pdf, recognizer=Broken())[0]
    assert page['formulas']
    for f in page['formulas']:
        assert f['state'] == 'unresolved' and not f['answer_eligible']
        assert f['latex'] is None
        assert f['original_text'] and f['reason']
    assert page['math_status'] == 'unresolved'


def test_image_only_page_is_not_reported_as_safe_empty_text(tmp_path):
    path = tmp_path / 'image.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(255)
        p.insert_image(pymupdf.Rect(30, 30, 200, 200), stream=pix.tobytes('png'))
        doc.save(path)
    page = optional_load(path)[0]
    assert page['math_status'] == 'unresolved'
    assert 'image' in page['unresolved_reason'].lower()
    assert 'UNRESOLVED' in page['text']


@pytest.mark.parametrize('candidate', ['', '   ', None, 42])
def test_invalid_recognizer_output_cannot_become_trusted_text(math_pdf, candidate):
    class Invalid:
        def recognize(self, png):
            return candidate
    page = optional_load(math_pdf, recognizer=Invalid())[0]
    assert all(f['state'] == 'unresolved' and f['latex'] is None for f in page['formulas'])
    assert all(not f['answer_eligible'] for f in page['formulas'])
    assert "y' =" not in page['text']


@pytest.mark.parametrize('region_kind', ['table', 'outside', 'fraction'])
@pytest.mark.parametrize('script_first', [True, False])
@pytest.mark.parametrize('script_text,body_text', [
    ('n=2', 'x=1'), ('2', 'x=1'), ('n=2', 'x'),
])
def test_unflagged_script_evidence_is_independent_of_span_order(
        tmp_path, region_kind, script_first, script_text, body_text):
    path = tmp_path / 'ordered-script.pdf'

    class Recognizer:
        def recognize(self, png):
            return 'candidate'

    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=300)
        if region_kind == 'table':
            for x in (90, 150, 210):
                p.draw_line((x, 60), (x, 180))
            for y in (60, 130, 180):
                p.draw_line((90, y), (210, y))
        elif region_kind == 'fraction':
            p.draw_line((98, 94), (120, 94))
        parts = [((100, 90), script_text, 7), ((100, 107), body_text, 11)]
        if not script_first:
            parts.reverse()
        for position, text, size in parts:
            p.insert_text(position, text, fontsize=size)
        spans = [s for b in p.get_text('dict')['blocks']
                 for line in b.get('lines', []) for s in line['spans']]
        assert [s['text'] for s in spans] == [part[1] for part in parts]
        assert all(not s['flags'] & 1 for s in spans)
        doc.save(path)

    native = optional_load(path)[0]
    recognized = optional_load(path, recognizer=Recognizer())[0]
    for page in (native, recognized):
        assert len(page['formulas']) == 1
        formula = page['formulas'][0]
        assert script_text in formula['original_text']
        assert body_text in formula['original_text']
        assert formula['native_script_evidence'] is True
        assert formula['state'] == 'unresolved'
        assert not formula['answer_eligible'] and not page['answer_eligible']
        assert ('table_cell' in formula) == (region_kind == 'table')
    assert recognized['formulas'][0]['recognition_status'] == 'returned'
    assert recognized['formulas'][0]['reason'] == 'layout_scope_or_script_attachment_requires_source_review'


def test_native_script_attachment_stays_unresolved_even_when_model_returns(math_pdf):
    class Recognizer:
        def recognize(self, png):
            return 'f(x)=x2+a n-1'  # plausible flat output is not verified attachment
    page = optional_load(math_pdf, recognizer=Recognizer())[0]
    scripts = next(f for f in page['formulas'] if 'f(x)' in f['original_text'])
    assert scripts['state'] == 'unresolved'
    assert scripts['native_script_evidence'] is True
    assert not scripts['answer_eligible']


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('width,height', [(300, 600), (600, 300)])
def test_rotated_pages_keep_native_geometry_text_and_recognition_crops(
        tmp_path, rotation, width, height):
    path = tmp_path / 'rotated.pdf'
    crops = []

    class Recognizer:
        def recognize(self, png):
            crops.append(png)
            return 'x=1'

    with pymupdf.open() as doc:
        page = doc.new_page(width=width, height=height)
        page.insert_text((width - 100, height - 100), 'x = 1')
        page.insert_text((30, 70), 'numerator')
        page.insert_text((30, 92), 'denominator')
        page.draw_line((25, 75), (285, 75))
        for x in (30, 130, 230):
            page.draw_line((x, 120), (x, 200))
        for y in (120, 160, 200):
            page.draw_line((30, y), (230, y))
        page.insert_text((40, 145), 'Equation')
        page.insert_text((140, 145), 'Type')
        page.insert_text((40, 185), 'z = 2')
        page.insert_text((140, 185), 'ODE')
        page.set_rotation(rotation)
        doc.save(path)

    before = path.read_bytes()
    ordinary = pdf_loader.load_pdf(path)
    native = optional_load(path)[0]
    result = optional_load(path, recognizer=Recognizer())[0]
    assert result['original_text'] == ordinary[0]['text']
    assert native['formulas']
    formula = next(f for f in result['formulas'] if 'x = 1' in f['original_text'])
    assert formula['bbox'][0] > width - 110
    assert formula['bbox'][1] > height - 120
    assert formula['recognition_status'] == 'returned'
    table = next(f for f in result['formulas'] if 'z = 2' in f['original_text'])
    assert table['table_cell']['row'] == 1
    assert table['table_cell']['column'] == 0
    assert 'ODE' not in table['original_text']
    fractions = [f for f in result['formulas'] if 'numerator' in f['original_text']]
    assert bool(fractions) == (width == 600)
    assert len(crops) == len(result['formulas'])
    with pymupdf.open(path) as doc:
        assert doc[0].rotation == rotation
        doc[0].set_rotation(0)
        ordered = sorted(result['formulas'], key=lambda f: (
            '√' not in f['original_text'], '=' not in f['original_text']))
        for f, crop in zip(ordered, crops):
            expected = doc[0].get_pixmap(
                matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(f['bbox']))
            actual = pymupdf.Pixmap(crop)
            assert (actual.width, actual.height) == (expected.width, expected.height)
            assert actual.samples == expected.samples
            assert min(actual.samples) < 255
            assert not f['answer_eligible']
            assert f['state'] in ('unresolved', 'unverified')
            assert f['source_sha256'] == result['source_sha256']
    assert path.read_bytes() == before
    assert pdf_loader.load_pdf(path) == ordinary
