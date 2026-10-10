# HybridDoc AI: Multimodal PDF Intelligence & RAG Platform

> **RAG, OCR ve Görsel Doküman Anlama Üzerine Kapsamlı Doküman Zekâsı Projesi**

HybridDoc AI; dijital PDF'lerin yanı sıra taranmış sayfalar, ekran görüntüleri, tablolar ve şemalar içeren dokümanları işlemeyi, kaynak kanıtlarını (sayfa numarası, içerik türü, bounding box) doğrulamayı ve şeffaf referanslar sunmayı hedefleyen açık kaynaklı bir doküman zekâsı projesidir.

Şu anda dijital PDF'lerden sayfa metni ve kaynak metaverisi çıkarma ile aşağıda açıklanan metin parçalama bileşeni mevcuttur. Embeddings, Chroma retrieval, 20 doküman sorusuyla değerlendirme, LLM ve kullanıcı arayüzü sonraki geliştirme adımlarıdır; OCR ve multimodal işleme henüz uygulanmamıştır.

---

## 🎯 Proje Hedefleri ve Aşamalar

- **Aşama 1: Temel RAG Web Uygulaması (*Ask My PDF*)**
  - Çekirdek RAG akışı (ingestion, chunking, embeddings, vector store, retrieval, LLM prompting).
  - Harici framework bağımlılığı olmadan saf mekanizmanın inşası.
  - Kaynak sayfa referansları ve hallucination engelleme (*no-evidence policy*).
- **Aşama 2: OCR ve Multimodal Doküman Anlama**
  - Adaptif sayfa analizi: dijital metin yoksa OCR (PaddleOCR / Docling) tetikleme.
  - VLM destekli görsel/grafik semantik açıklamaları.
  - Modality metadata takibi (`native_text`, `ocr_text`, `table`, `image_caption`).
- **Aşama 3: HybridDoc AI (Final Ürünü)**
  - Hibrit arama (Dense Semantic + BM25/Sparse) ve Reranking (Cross-Encoder).
  - Next.js modern kullanıcı arayüzü ve PDF viewer senkronizasyonu.
  - Detaylı benchmark ve evaluation metrikleri (Hit@k, MRR, Groundedness).

---

## 📁 Proje Klasör Mimarisi

Aşağıdaki yapı hedeflenen sorumlulukları gösterir; mevcut bileşenler dışındaki klasörler şimdilik yer tutucudur.

```text
RAG-OCR-proje/
├── apps/
│   ├── api/              # FastAPI backend servisi
│   └── web/              # Next.js frontend kullanıcı arayüzü
├── src/
│   ├── ingestion/        # PDF ayrıştırma, metin ve görsel çıkarma, OCR
│   ├── chunking/         # Metin parçalama ve overlap algoritmaları
│   ├── embeddings/       # Embedding modelleri ve vektörleştirme
│   ├── retrieval/        # Dense, sparse ve hibrit arama, reranking
│   ├── generation/       # Prompt şablonları, LLM entegrasyonu
│   ├── citations/        # Kaynak doğrulama ve referans eşleştirme
│   └── evaluation/       # Hit@k, MRR, doğruluk metrikleri ve testler
├── sample_docs/          # Test belgeleri (taranmış, tablolu, grafikli, dijital PDF)
├── eval_data/            # Altın veri seti (golden dataset) ve değerlendirme soruları
├── tests/                # Otomatik pytest birim ve entegrasyon testleri
├── docs/                 # Mimari diyagramlar ve dokümantasyon
├── requirements.txt      # Python bağımlılıkları
├── .env.example          # Ortam değişkenleri şablonu
└── README.md
```

---

## 🚀 Hızlı Başlangıç (Geliştirici Kurulumu)

1. **Sanal Ortamı Oluşturun ve Aktif Edin:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ortam Değişkenlerini Tanımlayın:**
   ```bash
   cp .env.example .env
   ```

## Sayfa Kaynaklı Metin Parçalama

PDF okuyucusunun sayfa kayıtlarını karakter sınırlarıyla parçalamak için proje kökünden şu komutu çalıştırın:

```bash
python -m src.chunking.text_chunker sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf
```

Komut varsayılan olarak en fazla 1000 Python karakteri ve 100 karakter örtüşme kullanır; örnek bir parçanın dosya adını ve kaynak sayfasını gösterir.

Boyut birimi tokenizer token'ı değildir; Python `len()` ile sayılan Unicode karakter konumlarıdır.

`chunk_pages(pages, max_chars=1000, overlap=100)` fonksiyonu her sayfayı ayrı işler, boş veya yalnızca boşluk içeren sayfaları atlar ve parçalarda kaynak `page_number`, `file_name` ve `total_pages` alanlarını korur. Her parçanın `text` alanı ilgili metin dilimini, `char_count` alanı ise o parçanın karakter sayısını içerir. `max_chars` pozitif bir tam sayı, `overlap` ise `0 <= overlap < max_chars` koşulunu sağlayan bir tam sayı olmalıdır; boolean değerler kabul edilmez.
