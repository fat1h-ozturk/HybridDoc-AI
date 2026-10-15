"""Session-local integration of the existing native PDF inspection/search tools."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import pymupdf

from src.chunking.text_chunker import chunk_pages, chunk_formula_pages, format_chunks_inspection
from src.evaluation.review_formulas import generate_formula_review_html
from src.ingestion.pdf_loader import load_pdf, load_pdf_formulas
from src.retrieval.search_pdf import BM25Searcher, tokenize

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200


@dataclass
class WorkbenchDocument:
    file_name: str
    pdf_bytes: bytes
    native_pages: list
    native_chunks: list
    formula_pages: list
    search_chunks: list

    def search(self, question: str) -> list:
        if not tokenize(question):
            raise ValueError("Soruya en az bir sözcük yazın.")
        return BM25Searcher(self.search_chunks).search(question, top_k=5)

    def native_export(self) -> str:
        return "\n\n".join(
            f"Source: {self.file_name}, Page {p['page_number']} of {p['total_pages']}\n"
            f"{p['text']}" for p in self.native_pages
        ) + "\n"

    def chunks_export(self) -> str:
        return format_chunks_inspection(self.native_chunks, self.file_name, "word", 1000, 100)

    def formula_report(self) -> str:
        # Reuse the validated saved-extraction path, rather than extracting again.
        with tempfile.TemporaryDirectory(prefix="hybriddoc-") as directory:
            pdf_path = Path(directory) / self.file_name
            pdf_path.write_bytes(self.pdf_bytes)
            extraction_path = Path(directory) / "extraction.json"
            extraction_path.write_text(json.dumps(self.formula_pages, ensure_ascii=False), encoding="utf-8")
            return generate_formula_review_html(pdf_path, extraction_path)


def open_upload(file_name: str, data: bytes) -> WorkbenchDocument:
    """Validate an upload and remove private disk copies before returning or raising."""
    if (not file_name or len(file_name) > 200 or any(c in file_name for c in '/\\\x00')
            or Path(file_name).suffix.lower() != ".pdf"):
        raise ValueError("Geçerli bir PDF dosya adı gerekli (dizin yolu içeremez).")
    if not data:
        raise ValueError("PDF dosyası boş. Başka bir PDF seçin.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("PDF boyut sınırı 20 MiB. Daha küçük bir PDF seçin.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("Dosya PDF içeriği taşımıyor. Geçerli bir PDF seçin.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise ValueError("Şifreli PDF desteklenmiyor. Yerelde şifresiz bir kopya oluşturun.")
            if not 0 < len(doc) <= MAX_PAGES:
                raise ValueError("PDF 1–200 sayfa içermeli. Daha küçük bir belge seçin.")
        with tempfile.TemporaryDirectory(prefix="hybriddoc-") as directory:
            path = Path(directory) / file_name
            path.write_bytes(data)
            native_pages = load_pdf(path)
            formula_pages = load_pdf_formulas(path)
        native_chunks = chunk_pages(native_pages, boundary="word")
        search_chunks = chunk_formula_pages(formula_pages, boundary="word")
        for i, chunk in enumerate(search_chunks, 1):
            chunk.update(source_order=i, chunk_id=f"chunk-{i}")
            # The BM25 block index must see the actual slice, not the parent span.
            if chunk['blocks'][0]['kind'] == 'text':
                chunk['blocks'] = [{**chunk['blocks'][0], 'text': chunk['text']}]
        return WorkbenchDocument(file_name, data, native_pages, native_chunks,
                                 formula_pages, search_chunks)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF okunamadı. Bozuk/eksik dosyayı yeniden dışa aktarın.") from exc
