# HybridDoc AI: Multimodal PDF Intelligence & RAG Platform

> **RAG, OCR ve Görsel Doküman Anlama Üzerine Kapsamlı Doküman Zekâsı Projesi**

HybridDoc AI; dijital PDF'lerin yanı sıra taranmış sayfalar, ekran görüntüleri, tablolar ve şemalar içeren dokümanları işlemeyi, kaynak kanıtlarını (sayfa numarası, içerik türü, bounding box) doğrulamayı ve şeffaf referanslar sunmayı hedefleyen açık kaynaklı bir doküman zekâsı projesidir.

Şu anda dijital PDF'lerden sayfa metni ve kaynak metaverisi çıkarma ile aşağıda açıklanan metin parçalama bileşeni mevcuttur. Embeddings, Chroma retrieval, 20 doküman sorusuyla değerlendirme, LLM ve kullanıcı arayüzü sonraki geliştirme adımlarıdır. İsteğe bağlı yerel formül çıkarma aşağıda açıklanmıştır; genel sayfa OCR ve multimodal cevaplama henüz uygulanmamıştır.

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

## İsteğe Bağlı Yerel Formül Çıkarma

Dijital PDF için ayrı bir yol vardır: PyMuPDF yazı tipleri, semboller, script
konumları, çizgiler ve tablo hücrelerinden aday bölgeler önerir. Yerel
CodeFormulaV2 bu bölgelerin görüntülerinden aday LaTeX üretir. Eski `load_pdf`
ve `chunk_pages` davranışı değişmez. Bu yol genel taranmış sayfa OCR'ı değildir.

Model olmadan tam, okunabilir JSON dökümü:

```bash
python -m src.ingestion.formula_loader belge.pdf --output extraction.json
```

Model kurulumu isteğe bağlıdır. Ayrı bir sanal ortam kullanın; mevcut ortamı
veya sistem Python'unu değiştirmek gerekmez. Test edilen runtime Python 3.14,
PyMuPDF 1.28.2, Docling 2.137.0 ve CPU PyTorch 2.14.1 kullanır:

```bash
python3 -m venv .venv-formulas
source .venv-formulas/bin/activate
python -m pip install pymupdf==1.28.2
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-formulas.txt
```

İlk model indirme yalnızca açıkça aşağıdaki hazırlık komutunu çalıştırdığınızda
olur. Bu komut PDF okumaz/göndermez; sadece model dosyalarını indirir.
Ağırlıklar zaten önbellekteyse bu adımı atlayın:

```bash
HF_HOME="$PWD/.cache/formulas" python -c "from huggingface_hub import snapshot_download; snapshot_download('docling-project/CodeFormulaV2', revision='ecedbe111d15c2dc60bfd4a823cbe80127b58af4')"
```

Worker aynı sabit model revision'ını kullanır; önbellekteki başka bir modele
geçiş yapmaz.

Çıkarma sırasında worker çevrimdışıdır; eksik ağırlık/dependency, hata veya
zaman aşımı `unresolved` olarak kaydedilir, otomatik indirme yapılmaz:

```bash
python -m src.ingestion.formula_loader belge.pdf --recognize \
  --cache-dir "$PWD/.cache/formulas" --timeout 120 --budget 480 \
  --output extraction.json
```

Standart çıktı tüm sayfaları, özgün metni, bbox'ları, aday LaTeX'i ve nedenleri
tek bir JSON belgesi olarak gösterir; bağımlılık tanıları standart hataya
(stderr) yönlendirilir. `--output` aynı JSON'u UTF-8 dosyaya da yazar. Başarılı dosya çıkarma
(exit 0), formüllerin doğru olduğu anlamına gelmez. Tanıma her bölgede CPU
float32, iki thread ve batch bir kullanır; süre sınırı model yüklemesini de
kapsar. Bir worker hatasından sonra belge boyunca yeni worker başlatılmaz.
Varsayılan toplam tanıma bütçesi 480 saniyedir; native analiz bu bütçeye dahil
değildir. Tüm sayfalar için hız veya doğruluk garantisi yoktur. Model
ağırlığı yaklaşık 631 MB'dır; önceki tek-crop CPU ölçümü yaklaşık 2.4 GiB RSS
kullanmıştır. Docling MIT, model CDLA-Permissive-2.0 lisanslıdır; bulut/ücretli
API kullanılmaz, yerel CPU/RAM ve depolama gerekir.

Python kullanımı:

```python
from src.ingestion.pdf_loader import load_pdf_formulas
from src.ingestion.formula_recognizer import LocalFormulaRecognizer
from src.chunking.text_chunker import chunk_formula_pages

with LocalFormulaRecognizer(cache_dir=".cache/formulas") as recognizer:
    pages = load_pdf_formulas("belge.pdf", recognizer=recognizer)
chunks = chunk_formula_pages(pages, max_chars=1000, overlap=100)
```

`chunk_formula_pages` formül ve uyarı bloklarını atomik tutar; bir atomik blok sınırdan
uzunsa tek, daha büyük parça üretir. Düz yazı blokları karakter/örtüşme
kurallarını kullanır; formül örtüşmesi yapılmaz. Parçalar kaynak sayfa, hash,
bbox, özgün metin ve belirsizlik alanlarını korur.

**Güven sözleşmesi:** Her formül `original_text`, `latex`, `page_number`,
`bbox`, `source_sha256`, `recognition_status`, `state` ve `reason` taşır.
Formül bbox'ları döndürülmemiş sayfanın koordinat sistemindedir; özgün bölge
metni ve tanıma görüntüsü de aynı geometriyle çıkarılır. PDF dosyasının kayıtlı
sayfa dönüşü değiştirilmez.
`unverified` modelin aday döndürdüğünü; `unresolved` tanıma yapılmadığını,
başarısız olduğunu veya çizgili kapsam/script bağlantısının kaynak incelemesi gerektirdiğini
belirtir. İkisi de doğrulanmış gerçek değildir. Yerel detector işaretsiz inline
matematiği kaçırabilir; bu yüzden tüm sayfalar
`detection_status=native_heuristic_incomplete` ve `answer_eligible=false`
taşır. Sayfa görüntüsü kaynak kanıtıdır, aday LaTeX'in doğruluk sertifikası
değildir. Sonraki cevap katmanı kaynakla ayrıca doğrulanmamış formüle dayanarak
cevap vermemeli; belirsiz/kaçmış matematikte kaynak incelemesi istemeli veya
cevaptan kaçınmalıdır. Bu değişiklik henüz cevap/retrieval katmanı kurmaz.

Kontrol: `python -m pytest -q`. Testler herkese açık sentetik PDF geometrisi
kullanır; özel PDF veya altın koordinatlar repoya dahil değildir.
