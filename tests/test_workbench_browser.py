"""Optional real-browser regressions: install playwright and Chrome to run."""
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytest.importorskip("streamlit")
fitz = pytest.importorskip("pymupdf")
playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]


def test_clear_demo_then_upload_uses_uploaded_document(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        pytest.skip("Chrome or Chromium is required for the browser regression")
    upload = tmp_path / "replacement.pdf"
    with fitz.open() as document:
        document.new_page().insert_text((40, 70), "Replacement nebula document")
        document.save(upload)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "apps/workbench.py",
         "--server.address", "127.0.0.1", "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urlopen(f"{url}/_stcore/health", timeout=1) as response:
                    assert response.status == 200
                break
            except (URLError, TimeoutError):
                if server.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("Streamlit did not start")
                time.sleep(0.1)
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            try:
                page = browser.new_page()
                page.goto(url)
                page.get_by_text("Demo PDF", exact=True).click()
                playwright.expect(page.get_by_text("Dosya: workbench_demo.pdf", exact=False)).to_be_visible()
                page.get_by_role("button", name="Oturumu temizle", exact=True).click()
                playwright.expect(page.locator('input[type="file"]')).to_be_attached()
                playwright.expect(page.get_by_text("Dosya:", exact=False)).to_have_count(0)
                page.locator('input[type="file"]').set_input_files(upload)
                playwright.expect(page.get_by_text("Dosya: replacement.pdf | 1 sayfa", exact=True)).to_be_visible()
                playwright.expect(page.get_by_role("radio", name="PDF yükle", exact=True)).to_be_checked()
                playwright.expect(page.get_by_role("tabpanel", name="Tam sayfa metni").get_by_text(
                    "Replacement nebula document", exact=True)).to_be_visible()
            finally:
                browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
