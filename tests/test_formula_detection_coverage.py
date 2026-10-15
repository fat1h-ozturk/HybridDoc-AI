"""Executable regressions for bounded native formula detection coverage.

Tests coverage of digital/native formula candidates:
1. Display fractions with rectangular rules (filled rects)
2. Fraction numerators with raised superscripts and denominator subscripts
3. Radical expressions (roots) with rectangular vinculums
4. Inline roots with parenthesized radicands containing spaces
5. Polynomial exponent continuations preserving trailing terms with spaces
6. Negative controls: Turkish/English prose, dates, student IDs, page numbers, hyphenated text, table labels
7. Image / raster content trust boundaries: explicit unmeasured / unresolved warnings preserved.
"""
from pathlib import Path
import pymupdf
import pytest

from src.ingestion.formula_loader import load_pdf_formulas
from src.ingestion.formula_regions import propose_regions, _is_math_operand


def test_display_fraction_with_rectangular_rule(tmp_path):
    """Fractions with rules drawn as filled rectangles ('re') must be detected."""
    pdf_path = tmp_path / 'rect_fraction.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((40, 75), 'f(x) =')
        p.insert_text((75, 70), 'x + 1')
        p.insert_text((75, 85), '2y - 3')
        # Horizontal fraction bar drawn as filled rect (item[0] == 're')
        p.draw_rect(pymupdf.Rect(72, 73.6, 115, 74.4), fill=(0, 0, 0), width=0)
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    page = pages[0]
    assert len(page['formulas']) == 1

    f = page['formulas'][0]
    assert 'f(x) =' in f['original_text']
    assert 'x + 1' in f['original_text']
    assert '2y - 3' in f['original_text']
    # Rule geometry preserved in native_rules
    assert len(f['native_rules']) == 1
    assert abs(f['native_rules'][0][1] - 74.0) < 0.5
    # Bbox spans full fraction height and width
    assert f['bbox'][0] <= 40.0
    assert f['bbox'][2] >= 115.0
    assert f['bbox'][1] <= 70.0
    assert f['bbox'][3] >= 85.0
    # Honest trust boundaries
    assert f['state'] == 'unresolved'
    assert f['answer_eligible'] is False


def test_fraction_numerator_raised_superscript_and_denominator_subscript(tmp_path):
    """Raised superscripts in numerator and lowered subscripts in denominator must not be stripped."""
    pdf_path = tmp_path / 'scripted_fraction.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((40, 75), 'f(x) =')
        p.insert_text((72, 70), 'x')
        p.insert_text((79, 63), '2', fontsize=7)  # raised superscript
        p.insert_text((85, 70), '+ 1')
        p.insert_text((72, 85), '2y')
        p.insert_text((85, 88), '1', fontsize=7)  # lowered subscript
        p.insert_text((91, 85), '- 3')
        p.draw_line((70, 74), (110, 74))
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    f = pages[0]['formulas'][0]
    orig = f['original_text']
    assert 'x' in orig
    assert '2' in orig
    assert '+ 1' in orig
    assert '2y' in orig
    assert '1' in orig
    assert '- 3' in orig
    assert f['native_script_evidence'] is True
    # Bbox includes raised superscript at y~63
    assert f['bbox'][1] <= 60.0


def test_root_with_rectangular_vinculum_and_radicand(tmp_path):
    """Radical vinculums drawn as filled rectangles must preserve full radicand scope."""
    pdf_path = tmp_path / 'rect_root.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((40, 130), 'z =')
        writer = pymupdf.TextWriter(p.rect)
        writer.append((60, 132), '√', font=pymupdf.Font('symbol'), fontsize=16)
        writer.write_text(p)
        # Vinculum drawn as filled rectangle
        p.draw_rect(pymupdf.Rect(72, 119.6, 120, 120.4), fill=(0, 0, 0), width=0)
        p.insert_text((75, 130), 'u + v')
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    f = pages[0]['formulas'][0]
    assert '√' in f['original_text']
    assert 'u + v' in f['original_text']
    assert len(f['native_rules']) == 1
    assert abs(f['native_rules'][0][1] - 120.0) < 0.5
    assert f['bbox'][2] >= 120.0


def test_inline_root_with_parenthesized_radicand_with_spaces(tmp_path):
    """Inline native text root with spaces in radicand must not drop radicand."""
    pdf_path = tmp_path / 'inline_root.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((40, 50), 'z =')
        writer = pymupdf.TextWriter(p.rect)
        writer.append((60, 50), '√', font=pymupdf.Font('symbol'), fontsize=12)
        writer.write_text(p)
        p.insert_text((72, 50), '(x + 1)')
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    f = pages[0]['formulas'][0]
    assert '√' in f['original_text']
    assert '(x + 1)' in f['original_text']
    assert f['bbox'][2] >= 100.0


def test_polynomial_exponent_continuation_with_operators_and_spaces(tmp_path):
    """Polynomial terms following a raised exponent must not be truncated."""
    pdf_path = tmp_path / 'polynomial.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.insert_text((40, 50), 'f(x) = x')
        p.insert_text((85, 45), '3', fontsize=7)
        p.insert_text((92, 50), '- 2x + 1')
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    f = pages[0]['formulas'][0]
    orig = f['original_text']
    assert 'f(x) = x' in orig
    assert '3' in orig
    assert '- 2x + 1' in orig
    assert f['bbox'][2] >= 130.0


def test_negative_controls_are_not_detected_as_formulas(tmp_path):
    """Negative controls must yield zero formula proposals."""
    pdf_path = tmp_path / 'negative_controls.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=500, height=400)
        p.insert_text((30, 30), 'Türkçe normal metin: Bu araştırma raporunda modeller incelenmiştir.')
        p.insert_text((30, 60), 'English normal prose: The experiment was conducted with default parameters.')
        p.insert_text((30, 90), 'Tarih: 2024-10-15  Öğrenci No: 211101042')
        p.insert_text((30, 120), 'Date: 01/02/2023, Reference ID: 987654321')
        p.insert_text((200, 380), 'Sayfa 1 / 15')
        p.insert_text((30, 150), 'Bu sistem retrieval-augmented generation ve state-of-the-art yöntemleri kullanır.')
        p.insert_text((30, 180), 'Tablo 1: Deney Sonuçları ve Başarım Ölçütleri')
        doc.save(pdf_path)

    with pymupdf.open(pdf_path) as check_doc:
        regions, spans, table_error = propose_regions(check_doc[0])
        assert len(regions) == 0, f'Expected 0 formulas in negative controls, got {len(regions)}'

    pages = load_pdf_formulas(pdf_path)
    assert len(pages[0]['formulas']) == 0


def test_is_math_operand_precision():
    """Verify operand helper precision: admits math tokens, rejects natural prose."""
    # Math expressions that may contain spaces around operators/delimiters
    assert _is_math_operand('+ 1') is True
    assert _is_math_operand('- 2x + 1') is True
    assert _is_math_operand('(x + 1)') is True
    assert _is_math_operand('u + v') is True
    assert _is_math_operand('2y - 3') is True
    assert _is_math_operand('n-1') is True
    assert _is_math_operand('x') is True
    assert _is_math_operand('2') is True
    assert _is_math_operand('|x|') is True

    # Natural language prose controls
    assert _is_math_operand('retrieval-augmented') is False
    assert _is_math_operand('use-case') is False
    assert _is_math_operand('first-order') is False
    assert _is_math_operand('Tarih: 2024-10-15') is False
    assert _is_math_operand('Cubing yields.') is False
    assert _is_math_operand('on the branch') is False
    assert _is_math_operand('The condition gives') is False
    assert _is_math_operand('where x = 5') is False
    assert _is_math_operand('Sayfa 1 / 15') is False
    assert _is_math_operand('Tablo 1: Deney') is False


def test_image_page_retains_incomplete_warning_and_does_not_classify_image_as_formula(tmp_path):
    """Pages with raster images must retain explicit unverified warning and not classify image as formula."""
    pdf_path = tmp_path / 'image_page.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=400, height=300)
        p.insert_text((30, 30), 'Document header with organization logo.')
        # Insert a 100x100 dummy pixmap
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
        pix.clear_with(200)
        p.insert_image(pymupdf.Rect(40, 50, 140, 150), pixmap=pix)
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    page = pages[0]
    # Warning must be retained
    assert page['unresolved_reason'] == 'image content is not covered by native math detection'
    warnings = [b['text'] for b in page['blocks'] if b['kind'] == 'warning']
    assert any('image content is not covered' in w for w in warnings)
    assert any('math detection is incomplete' in w for w in warnings)
    # The image must NOT be classified as a formula
    assert len(page['formulas']) == 0


def test_scanned_page_remains_explicitly_unmeasured(tmp_path):
    """Scanned/raster-only page with no native text remains explicitly unmeasured."""
    pdf_path = tmp_path / 'scanned_page.pdf'
    with pymupdf.open() as doc:
        p = doc.new_page(width=400, height=300)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300), False)
        pix.clear_with(240)
        p.insert_image(pymupdf.Rect(0, 0, 400, 300), pixmap=pix)
        doc.save(pdf_path)

    pages = load_pdf_formulas(pdf_path)
    assert len(pages) == 1
    page = pages[0]
    assert len(page['formulas']) == 0
    assert page['unresolved_reason'] is not None
    assert 'image content' in page['unresolved_reason'] or 'no native text' in page['unresolved_reason']
