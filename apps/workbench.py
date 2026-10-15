"""Run with: python -m streamlit run apps/workbench.py --server.address 127.0.0.1 --browser.gatherUsageStats false --client.toolbarMode minimal"""
import hashlib
from pathlib import Path
import sys

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.search_pdf import format_search_results
from src.workbench import open_upload

st.set_page_config(page_title="HybridDoc — Yerel PDF", layout="wide")
st.title("HybridDoc — Yerel PDF çalışma masası")
st.caption("PDF → soru → eksiksiz kaynak pasajları. İşlem bu bilgisayarda yapılır.")
st.info("Bu arayüz belgenin söylediği metni gösterir: sözcük eşleşmesine dayalı kaynak aramasıdır. "
        "LLM cevabı üretmez; çeviri veya anlamsal arama yapmaz. Aynı dildeki belge sözcüklerini kullanın.")

if st.button("Oturumu temizle", key="clear"):
    epoch = st.session_state.get("upload_epoch", 0) + 1
    st.session_state.clear()
    st.session_state["upload_epoch"] = epoch
    st.rerun()

mode = st.radio("Belge kaynağı", ["PDF yükle", "Demo PDF"], horizontal=True,
                key=f"source_mode-{st.session_state.get('upload_epoch', 0)}")
if mode == "Demo PDF":
    name = "workbench_demo.pdf"
    data = (ROOT / "sample_docs" / name).read_bytes()
    st.caption("Demo araması: source passages — beklenen ilk kaynak sayfa 2. Metin CC0-1.0.")
else:
    upload = st.file_uploader("PDF seçin (en çok 20 MiB / 200 sayfa)", type=["pdf"],
                              max_upload_size=20,
                              key=f"upload-{st.session_state.get('upload_epoch', 0)}")
    name, data = (upload.name, upload.getvalue()) if upload else (None, None)

identity = (name, hashlib.sha256(data).hexdigest()) if data is not None else None
if identity != st.session_state.get("identity"):
    for key in ("document", "results", "result_question", "report_html", "load_error", "question"):
        st.session_state.pop(key, None)
    st.session_state["identity"] = identity
    if data is not None:
        try:
            with st.spinner("Yerel metin ve formül envanteri çıkarılıyor…"):
                st.session_state["document"] = open_upload(name, data)
        except ValueError as exc:
            st.session_state["load_error"] = str(exc)

if st.session_state.get("load_error"):
    st.error(st.session_state["load_error"])
if "document" not in st.session_state:
    st.caption("Başlamak için PDF yükleyin veya Demo PDF seçin. API anahtarı gerekmez.")
    st.stop()

document = st.session_state["document"]
st.text(f"Dosya: {document.file_name} | {len(document.native_pages)} sayfa")
st.warning("Formül tespiti eksiktir; native metin ve OCR adayları matematik cevabı için doğrulanmış "
           "kanıt değildir. answer_eligible=false korunur. Formülleri kaynak görüntüsünden inceleyin.")
empty_pages = [p['page_number'] for p in document.native_pages if not p['text'].strip()]
if empty_pages:
    st.warning(f"Native metin bulunmayan sayfalar: {empty_pages}. Taranmış/görsel olabilir. "
               "Metin katmanı olan bir PDF seçin veya mevcut açıkça isteğe bağlı yerel OCR yolunu inceleyin. "
               "Bu uygulama OCR çalıştırmaz.")

with st.form("question_form"):
    question = st.text_input("Soru / belge sözcükleri", key="question")
    submitted = st.form_submit_button("Kaynakları ara", key="search")
if submitted:
    st.session_state.pop("results", None)
    try:
        st.session_state["results"] = document.search(question)
        st.session_state["result_question"] = question
    except ValueError as exc:
        st.error(str(exc))

if "results" in st.session_state:
    results = st.session_state["results"]
    st.subheader("Belgenin söylediği — kaynak pasajları")
    st.caption("En çok 5 eşleşme. BM25 sırası güven, doğruluk veya olasılık değildir. "
               "Pasajlar yorumlanmadan tam olarak gösterilir.")
    if not results:
        st.info("Çıkarılan native metinde eşleşen kaynak bulunamadı. Bu, konunun PDF'te olmadığı "
                "anlamına gelmez. Belgedeki başka sözcükleri deneyin, sayfa metnini ve formül raporunu inceleyin.")
    result_text = format_search_results(results, st.session_state["result_question"], document.file_name)
    st.text(result_text)
    st.download_button("Kaynak sonuçlarını indir (UTF-8)", result_text.encode("utf-8"),
                       file_name="sources.txt", mime="text/plain; charset=utf-8")

native_tab, chunk_tab, formula_tab = st.tabs(["Tam sayfa metni", "Sözcük sınırı parçaları", "Formül inceleme"])
with native_tab:
    page_number = st.selectbox("Sayfa", range(1, len(document.native_pages) + 1), key="page")
    st.text(f"{document.file_name} — Sayfa {page_number}")
    st.text(document.native_pages[page_number - 1]["text"] or "[Native metin yok]")
    st.download_button("Tüm sayfa metinlerini indir (UTF-8)", document.native_export().encode("utf-8"),
                       file_name="native-pages.txt", mime="text/plain; charset=utf-8")
with chunk_tab:
    st.caption("Native sayfa metni: 1000 karakter bütçesi, hedef 100 karakter örtüşme, word modu. "
               "Aşırı uzun kesintisiz dizelerde karakter kesme uygulanır. Arama ayrıca mevcut formül bloklarını korur.")
    if document.native_chunks:
        index = st.selectbox("Parça", range(len(document.native_chunks)),
                             format_func=lambda i: f"{i + 1} — Sayfa {document.native_chunks[i]['page_number']}",
                             key="chunk")
        chunk = document.native_chunks[index]
        st.text(f"{document.file_name} — Sayfa {chunk['page_number']} | {chunk['char_count']} karakter")
        st.text(chunk['text'])
    else:
        st.info("Native metin olmadığı için sözcük parçası üretilemedi.")
    st.download_button("Tüm parçaları indir (UTF-8)", document.chunks_export().encode("utf-8"),
                       file_name="word-chunks.txt", mime="text/plain; charset=utf-8")
with formula_tab:
    st.caption("Mevcut HTML exporter yeniden kullanılır: native kırpımlar, tam metin, bbox ve çözülemeyen "
               "formül uyarıları. Model indirme, OCR veya formül çözme yapılmaz. "
               "Önceki OCR JSON çıktılarının incelenmesi için README'deki review_formulas komutu kullanılabilir.")
    for page in document.formula_pages:
        if page.get("unresolved_reason"):
            st.text(f"Sayfa {page['page_number']} — {page['unresolved_reason']}")
    if st.button("Formül inceleme raporunu aç", key="report"):
        try:
            with st.spinner("Yerel HTML raporu hazırlanıyor…"):
                st.session_state["report_html"] = document.formula_report()
        except (ValueError, RuntimeError) as exc:
            st.error(f"Rapor üretilemedi: {exc}")
    if "report_html" in st.session_state:
        st.download_button("Formül HTML raporunu indir / yeniden aç", st.session_state["report_html"].encode("utf-8"),
                           file_name="formula-review.html", mime="text/html; charset=utf-8")
        components.html(st.session_state["report_html"], height=700, scrolling=True)
