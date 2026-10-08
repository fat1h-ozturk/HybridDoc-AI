"""
PDF Ingestion Module.
Bu modül, PDF belgelerini PyMuPDF (fitz) ile ayrıştırır, her sayfanın metnini
ve sayfa numarası gibi temel metadata bilgilerini çıkarır.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf  # Modern PyMuPDF API


class PDFDocumentLoader:
    """PDF dokümanlarını okumak ve sayfa bazında metin çıkarmak için kullanılan sınıf."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Geçersiz dosya formatı. PDF bekleniyor: {self.file_path}")

    def load(self) -> List[Dict[str, Any]]:
        """
        PDF dosyasını sayfa sayfa açar ve her sayfanın metnini ve metaverilerini döner.

        Returns:
            List[Dict[str, Any]]: Her sayfa için sözlük listesi:
                [
                    {
                        "page_number": 1,
                        "text": "Sayfa metni...",
                        "char_count": 450,
                        "file_name": "dokuman.pdf",
                        "total_pages": 15
                    },
                    ...
                ]
        """
        pages_data: List[Dict[str, Any]] = []

        try:
            # Belgeyi pymupdf ile açıyoruz
            doc = pymupdf.open(self.file_path)
            total_pages = len(doc)

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_number = page_idx + 1  # 1-indexed sayfa numarası
                text = page.get_text("text").strip()

                pages_data.append({
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text),
                    "file_name": self.file_path.name,
                    "total_pages": total_pages,
                })

            doc.close()
            return pages_data

        except Exception as e:
            raise RuntimeError(f"PDF okunurken hata oluştu ({self.file_path.name}): {e}") from e


def load_pdf(file_path: str | Path) -> List[Dict[str, Any]]:
    """Hızlı kullanım için yardımcı fonksiyon."""
    loader = PDFDocumentLoader(file_path)
    return loader.load()


if __name__ == "__main__":
    import sys

    # Terminalden test amaçlı çalıştırma
    test_file = sys.argv[1] if len(sys.argv) > 1 else "sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf"
    print(f"📄 Okunuyor: {test_file}")

    try:
        pages = load_pdf(test_file)
        print(f"✅ Başarılı! Toplam sayfa sayısı: {len(pages)}")
        print("-" * 50)
        
        # 1. sayfanın ilk 300 karakterini önizleme yapalım
        if pages:
            first_page = pages[0]
            print(f"📖 1. Sayfa Önizleme (Karakter: {first_page['char_count']}):")
            print(first_page["text"][:300] + "...\n")
            
            # Sayfa metin istatistikleri
            empty_pages = [p["page_number"] for p in pages if p["char_count"] == 0]
            if empty_pages:
                print(f"⚠️ Dikkat: Metin içermeyen (taranmış/görsel olabilir) sayfalar: {empty_pages}")
            else:
                print("✨ Tüm sayfalar dijital metin içeriyor.")
    except Exception as err:
        print(f"❌ Hata: {err}")
