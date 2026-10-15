"""Workbench integration contracts: upload boundaries, source fidelity and cleanup."""
import hashlib
from pathlib import Path

import pymupdf
import pytest

from src import workbench


def pdf_bytes(*texts, encrypted=False):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_textbox((40, 40, 550, 800), text)
        if encrypted:
            return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                              owner_pw="owner", user_pw="secret")
        return doc.tobytes()


@pytest.mark.parametrize("name,data", [
    ("notes.txt", b"%PDF-1.7"), ("../notes.pdf", b"%PDF-1.7"),
    ("a\\notes.pdf", b"%PDF-1.7"), ("bad\x00.pdf", b"%PDF-1.7"),
    ("empty.pdf", b""), ("fake.pdf", b"hello"),
    ("broken.pdf", b"%PDF-1.7\nnot a document"),
])
def test_invalid_upload_rejected_without_files(name, data, tmp_path, monkeypatch):
    monkeypatch.setattr(workbench.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError):
        workbench.open_upload(name, data)
    assert list(tmp_path.iterdir()) == []


def test_upload_limits_and_encryption(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(workbench, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ValueError, match="boyut"):
        workbench.open_upload("large.pdf", pdf_bytes("large"))
    monkeypatch.setattr(workbench, "MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    with pytest.raises(ValueError, match="ifre"):
        workbench.open_upload("locked.pdf", pdf_bytes("locked", encrypted=True))
    monkeypatch.setattr(workbench, "MAX_PAGES", 1)
    with pytest.raises(ValueError, match="sayfa"):
        workbench.open_upload("long.pdf", pdf_bytes("one", "two"))
    assert list(tmp_path.iterdir()) == []


def test_complete_source_keyword_page_and_private_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench.tempfile, "tempdir", str(tmp_path))
    text = "Native source passages preserve the entire original text. " * 50
    data = pdf_bytes("Privacy: files stay on this computer.", text)
    original = tmp_path / "source.pdf"
    original.write_bytes(b"user-owned sentinel")
    document = workbench.open_upload("source.pdf", data)
    assert original.read_bytes() == b"user-owned sentinel"
    assert list(tmp_path.iterdir()) == [original]
    assert document.native_pages[1]["text"].endswith("original text.")
    results = document.search("source passages")
    assert results[0]["page_number"] == 2
    assert results[0]["file_name"] == "source.pdf"
    assert results[0]["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert all(r["answer_eligible"] is False for r in results)
    assert document.search("quasar") == []
    with pytest.raises(ValueError, match="sözcük"):
        document.search("??")
    full = document.native_export()
    assert document.native_pages[1]["text"] in full
    assert "Page 2 of 2" in full
    assert full.encode("utf-8").decode("utf-8") == full
    assert len(document.native_chunks) > 2
    assert all(c["char_count"] <= 1000 for c in document.native_chunks)
    for chunk in document.native_chunks:
        assert chunk["text"] in document.chunks_export()
    other = workbench.open_upload("source.pdf", pdf_bytes("replacement nebula"))
    assert other.search("passages") == []
    assert document.search("passages")


def test_empty_native_pdf_is_honest():
    document = workbench.open_upload("scanned.pdf", pdf_bytes(""))
    assert document.search("anything") == []
    assert document.native_pages[0]["text"] == ""
    assert document.formula_pages[0]["unresolved_reason"]


def test_report_reuses_extraction_and_escapes_untrusted_text(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench.tempfile, "tempdir", str(tmp_path))
    document = workbench.open_upload("<evil>.pdf", pdf_bytes("x = y - 2\n<script>alert('evil')</script>"))
    report = document.formula_report()
    assert "&lt;evil&gt;.pdf" in report
    assert "<script>" not in report
    assert "answer_eligible: false" in report
    assert "data:image/png;base64," in report
    assert list(tmp_path.iterdir()) == []


def test_extraction_failure_also_removes_temporary_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench.tempfile, "tempdir", str(tmp_path))
    def fail(_):
        raise RuntimeError("native failure")
    monkeypatch.setattr(workbench, "load_pdf_formulas", fail)
    with pytest.raises(ValueError, match="okunamadı"):
        workbench.open_upload("valid.pdf", pdf_bytes("valid"))
    assert list(tmp_path.iterdir()) == []
