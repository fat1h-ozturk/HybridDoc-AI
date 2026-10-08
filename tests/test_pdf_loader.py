from pathlib import Path
import pytest
from src.ingestion.pdf_loader import PDFDocumentLoader, load_pdf

SAMPLE_PDF = Path("sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf")


def test_pdf_loader_valid_file():
    """Geçerli bir PDF dosyasının doğru şekilde yüklendiğini test eder."""
    assert SAMPLE_PDF.exists(), "Örnek PDF dosyası mevcut olmalıdır."
    
    pages = load_pdf(SAMPLE_PDF)
    assert len(pages) == 15, "Örnek doküman 15 sayfa olmalıdır."
    
    # 1. sayfa kontrolleri
    first_page = pages[0]
    assert first_page["page_number"] == 1
    assert "MULTIMODAL PDF" in first_page["text"]
    assert first_page["char_count"] > 0
    assert first_page["file_name"] == SAMPLE_PDF.name


def test_pdf_loader_nonexistent_file():
    """Olmayan bir dosya verildiğinde FileNotFoundError fırlatıldığını doğrular."""
    with pytest.raises(FileNotFoundError):
        load_pdf("olmayan_dosya.pdf")


def test_pdf_loader_invalid_extension(tmp_path):
    """PDF olmayan bir uzantı verildiğinde ValueError fırlatıldığını doğrular."""
    fake_txt = tmp_path / "test.txt"
    fake_txt.write_text("merhaba dunya")
    with pytest.raises(ValueError):
        load_pdf(fake_txt)
