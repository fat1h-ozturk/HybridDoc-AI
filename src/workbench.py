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
from src.field_evidence import explicit_fields, relation_for_hit

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
        results = BM25Searcher(self.search_chunks).search(question, top_k=5)
        # Expand source evidence; explicit colon relations require native geometry.
        with pymupdf.open(stream=self.pdf_bytes, filetype="pdf") as pdf:
            fields = explicit_fields(pdf)
            for result in results:
                chunk = self.search_chunks[result['source_order'] - 1]
                block = chunk['blocks'][0]
                if block['kind'] != 'text' or not block.get('bbox'):
                    continue
                page = pdf[result['page_number'] - 1]
                page.set_rotation(0)
                hit = pymupdf.Rect(block['bbox'])
                height = min(160, max(48, hit.height * 5))
                center = (hit.y0 + hit.y1) / 2
                band = pymupdf.Rect(page.rect.x0, center - height / 2,
                                    page.rect.x1, center + height / 2) & page.rect
                result['source_context'] = {
                    'text': page.get_text('text', clip=band, sort=True).strip(),
                    'bbox': list(band),
                    'association': 'not_inferred',
                    'note': ('Bu kesit çevredeki satır ve sütunları gösterir; etiket–değer '
                             'ilişkisi otomatik seçilmez. Değer boş veya ilişki belirsizse '
                             'bu kaynaktan kesin cevap çıkarılamaz. Kırpımı inceleyin.'),
                }
                formula_page = self.formula_pages[page.number]
                result['formulas'] = [f for f in formula_page['formulas']
                                      if pymupdf.Rect(f['bbox']).intersects(band)]
                formula_ids = {f['id'] for f in result['formulas']}
                result['formula_groups'] = [g for g in formula_page.get('formula_groups', [])
                                            if formula_ids.intersection(g['formula_ids'])]
                context = result['source_context']
                relation = relation_for_hit(fields.get(page.number, []), hit, context['text'])
                if relation:
                    context.update(association='explicit_same_row', field_relation=relation,
                                   note=('Native metinde tek satırda açık etiket: değer ilişkisi bulundu. '
                                         'Bu bir çıkarım ilişkisi; anlamsal doğruluk veya kimlik doğrulaması değildir. '
                                         'Özgün kesiti ve alıntıyı inceleyin.'))
        return results

    def source_crop(self, result: dict) -> bytes:
        """Render only the bounded source context from this session's PDF."""
        with pymupdf.open(stream=self.pdf_bytes, filetype="pdf") as pdf:
            page = pdf[result['page_number'] - 1]
            page.set_rotation(0)
            band = pymupdf.Rect(result['source_context']['bbox']) & page.rect
            scale = min(1.5, 1200 / band.width)
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                   clip=band, alpha=False).tobytes('png')

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
