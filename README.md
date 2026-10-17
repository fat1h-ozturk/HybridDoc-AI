# HybridDoc AI: Multimodal PDF Intelligence & RAG Platform

> **RAG, OCR ve Görsel Doküman Anlama Üzerine Kapsamlı Doküman Zekâsı Projesi**

HybridDoc AI; dijital PDF'lerin yanı sıra taranmış sayfalar, ekran görüntüleri, tablolar ve şemalar içeren dokümanları işlemeyi, kaynak kanıtlarını (sayfa numarası, içerik türü, bounding box) doğrulamayı ve şeffaf referanslar sunmayı hedefleyen açık kaynaklı bir doküman zekâsı projesidir.

Şu anda dijital PDF'lerden native metin çıkarma, sözcük sınırı parçalama, BM25 kaynak araması ve yerel PDF çalışma masası kullanılabilir. İsteğe bağlı, kaynak alıntılarını denetleyen Ollama/Gemini cevap üretimi ilk entegrasyon olarak kullanılabilir. Embeddings, Chroma retrieval ve 20 doküman sorusuyla değerlendirme sonraki geliştirme adımlarıdır. İsteğe bağlı yerel formül çıkarma aşağıda açıklanmıştır; genel sayfa OCR ve multimodal cevaplama henüz uygulanmamıştır.

Sistemin mimarisini, uçtan uca veri akışını ve koordinat temelli çalışma prensiplerini adım adım incelemek için [Mimari Rehber (Architecture Walkthrough)](docs/architecture_walkthrough.md) dokümanına göz atabilirsiniz.

---

## Yerel PDF çalışma masası — hemen çalıştırın

Python **3.11 veya üstü** ile, bu repoyu klonladığınız dizinde:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-workbench.txt
python -m streamlit run apps/workbench.py --server.address 127.0.0.1 --server.maxUploadSize 20 --browser.gatherUsageStats false --client.toolbarMode minimal
```

Son satır arayüzü başlatır; tarayıcı otomatik açılmazsa terminaldeki
**http://127.0.0.1:8501** adresini açın. Port doluysa aynı komuta
`--server.port 8502` ekleyin. Durdurmak için terminalde `Ctrl+C` kullanın.
Bu arayüz için büyük `requirements.txt`, `.env`, API anahtarı, model indirme,
vektör veritabanı veya OCR kurulumu gerekmez. Streamlit isteğe bağlıdır;
mevcut CLI komutları aynı şekilde çalışır.

1. **PDF yükle** ile kendi PDF'inizi seçin veya **Demo PDF** seçin.
2. Soru veya belge sözcüklerini yazıp **Kaynakları ara** düğmesine basın.
   Demo sorgusu ve beklenen çıktılar için [demo kılavuzunu](sample_docs/workbench_demo.md) izleyin.
3. Dosya/sayfa, hash, tam kaynak pasajı ve uyarıları okuyun. **Tam sayfa metni**
   sekmesinden istediğiniz sayfayı seçin.
   **Sözcük sınırı parçaları** sekmesi parça metnini eksiksiz gösterir.
   Metin eşleşmelerinde etiketin çevresindeki aynı sayfa kesiti, konuma göre
   sıralanmış native metin ve kaynak görüntüsüyle gösterilir. Böylece ayrı PDF
   metin parçalarında duran form etiketleri ve yakındaki değerler birlikte
   incelenebilir. Kesit komşu satır/sütunları da içerir; alan ilişkileri ve
   belirsiz kaynakların kullanıcı onayı için [kanıt ve alıntı sınırlarını](docs/grounded-answers.md#kanıt-alıntı-ve-sınırlar) izleyin.
4. Tam sonuçları, tüm native sayfaları ve tüm parçaları **UTF-8** indirin.
   **Formül inceleme** sekmesinde raporu açın; kaynak kırpımları ve çözülemeyen
   formül uyarıları mevcut HTML exporter ile gösterilir. HTML'i indirip daha
   sonra çevrimdışı açabilirsiniz.
5. **Oturumu temizle** belgeyi, sonuçları ve raporu arayüz oturumundan kaldırır.

**Gösterilen şey:** belgenin söylediği özgün metne dayalı, çıkarımsız kaynak
araması. En fazla beş tam pasaj BM25 ile sıralanır. Sıralama puanı güven,
doğruluk veya olasılık değildir. Sorudaki aynı sözcükleri arar; Türkçe–İngilizce
çeviri, anlamsal arama veya formül çözümü yapmaz. İsteğe bağlı LLM cevabı ayrı ve açık bir eylemdir; aşağıdaki kılavuza bakın. İngilizce demo için
İngilizce sözcük kullanın. Eşleşme bulunmaması konunun belge
resimlerinde bulunmadığını kanıtlamaz. Boş soru için sözcük girin; native
metinsiz/taranmış PDF için metin katmanı olan bir kopya seçin. Genel sayfa OCR'ı
yoktur. Aşağıdaki isteğe bağlı yerel formül OCR yolu ayrıca ve açıkça çalıştırılır.

Türkçe `Adı` / `adi` gibi I/ı/İ farkları eşleşir; iyelik ekleri çözülmez
(`babasının` otomatik olarak `baba` olmaz). Alan ararken `Baba Adı` gibi
belgedeki sözcükleri kullanın. Diğer aksanlar kaldırılmaz (`öğrenci` ile
`ogrenci` aynı sözcük sayılmaz). Sıralı sonuç bir kişi adı cevabı değildir.
Kaynak kesiti en çok 160 PDF noktası yüksekliğinde ve bulunduğu sayfayla
sınırlıdır; uzaktaki bir değeri kapsamayabilir. Tam sayfa metni ayrıca açıktır.

**Formül sınırı:** `answer_eligible=false` korunur. Native formül tespiti eksiktir;
LaTeX adayı veya eksi işareti kontrolü formül doğruluğunu kanıtlamaz. Çözülemeyen
bölge ve formül uyarıları kaynak sonucunda ve raporda kalır. Önceden üretilmiş
OCR JSON'unu aynı PDF ile yeniden incelemek için
[kaydedilmiş çıkarım raporu komutunu](#2-kaydedilmiş-çıkarım-jsonu-ile-inceleme-saved-candidates--warnings) kullanın.

**Yerellik ve saklama:** yukarıdaki başlatma komutu yalnızca `127.0.0.1`'e bağlanır; bunu ağ adresine
çevirmeyin. Varsayılan kaynak incelemesinde PDF/soru harici servislere gönderilmez. İsteğe bağlı Gemini cevabında yalnızca açık onay ve düğmeyle soru/seçili kaynaklar Google’a gider; [cevap üretimi kılavuzunu](docs/grounded-answers.md) okuyun. Telemetri yukarıdaki komutta
kapalıdır. En çok **20 MiB / 200 sayfa**, şifresiz PDF kabul edilir. Kaynak dosya
asla değiştirilmez. Yüklemeler paylaşılan cache'e yazılmaz; her tarayıcı oturumu
kendi belleğini kullanır. Ayrıştırma ve rapor sırasında rastgele, özel geçici
dizin kullanılır ve işlem başarılı ya da hatalı bittiğinde silinir. Uygulama
kapanınca bellekteki içerik gider; indirdiğiniz dosyalar size aittir.
Normal silme güvenli disk silme garantisi vermez; zorla süreç öldürmede işletim
sisteminin geçici dizininde bir kopya kalabilir. Büyük/grafik yoğun dosyalarda
native analiz ve HTML kırpımları zaman ve RAM kullanabilir; doğruluk garantisi yoktur.

Tekrarlanabilir yerel duman testi ve tüm testler (model/ağ gerekmez):

```bash
python -m pytest tests/test_workbench.py tests/test_workbench_ui.py -q
python -m pytest -q
```

---

## İsteğe bağlı cevap üretimi

**Ollama (yerel varsayılan) ve Gemini API**, her biri içinde model seçimiyle
kullanılabilir. Varsayılan kapalıdır; yalnızca **Cevap üret** düğmesi istek
gönderir. Google için belgeye özel açık onay gerekir.
[Türkçe kurulum, kaynak önizlemesi, sınırlar ve çevrimdışı simülasyon](docs/grounded-answers.md).

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
│   ├── workbench.py      # Mevcut yerel Streamlit PDF çalışma masası
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

PDF okuyucusunun sayfa kayıtlarını parçalamak için proje kökünden metin parçalayıcı komutunu çalıştırabilirsiniz.

Varsayılan karakter tabanlı parçalama:

```bash
python -m src.chunking.text_chunker sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf
```

Sözcükleri ve bağlamı koruyan opt-in sözcük sınırı (`--boundary word`) parçalama:

```bash
python -m src.chunking.text_chunker sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf \
  --boundary word \
  --chunk-size 1000 \
  --overlap 100 \
  --output chunks.txt
```

Tüm parçaları doğrudan terminalde eksiksiz incelemek için `--inspect` bayrağı eklenebilir.

### Karakter Bütçesi ve Sınır Kuralları

- **Maksimum Karakter Bütçesi (`chunk_size` / `max_chars`):** Boyut birimi tokenizer token'ı veya kelime sayısı değildir; Python `len()` ile sayılan Unicode karakter konumlarıdır. Hiçbir parça belirtilen `chunk_size` sınırını aşamaz.
- **Sözcük Sınırları (`--boundary word`):** Düz yazı metninde boşlukla ayrılmış sıradan sözcüklerin hem parça bitişlerinde hem de örtüşme (overlap) başlangıçlarında ortadan bölünmesini engeller. Tüm özgün metin ve sayfa metaverileri korunur.
- **Örtüşme Davranışı:** Sözcük modunda sözcük kenarlarına hizalama yapıldığından tam 100 karakterlik sabit örtüşme garanti edilmez; hedef örtüşmeye en yakın sözcük sınırına hizalanır.
- **Aşırı Uzun Kesintisiz Dizeler (Bounded Fallback):** Boşluk içermeyen ve bütçeyi aşan kesintisiz uzun simge/sözcük dizilerinde (unbroken strings) deterministik karakter kesme devreye girer; böylece metin kaybı yaşanmaz, azami bütçe kesinlikle aşılmaz ve dar bütçe/geniş örtüşmelerde sonsuz döngü engellenir.
- **Geriye Dönük Uyumluluk:** Varsayılan mod `--boundary character` (eski karakter tabanlı dilimleme) olarak kalır. `chunk_pages(pages, max_chars=1000, overlap=100, boundary="word")` fonksiyonu API düzeyinde doğrudan kullanılabilir; `chunk_size` parametresi `max_chars` için geriye dönük uyumlu bir alias'tır.

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

`chunk_formula_pages` formül, formül grubu ve uyarı bloklarını atomik tutar; bir atomik blok sınırdan
uzunsa tek, daha büyük parça üretir. Düz yazı blokları karakter/örtüşme
kurallarını kullanır; formül örtüşmesi yapılmaz. Parçalar kaynak sayfa, hash,
bbox, özgün metin ve belirsizlik alanlarını korur.

### Seçici Yerel Formül Tanıma (Selective Formula OCR)

Tüm belgedeki onlarca veya yüzlerce formül (örneğin 188 formül) için işlem bütçesini tüketmek yerine, kullanıcının yalnızca incelemek istediği tek bir formülü veya belirli sayfaları seçip tanıma bütçesini sınırlayarak çalıştırmasını sağlar.

#### 1. Formül ID'lerini Keşfetme (Native Inventory)
Önce sıfır OCR harcaması ve model yükü olmadan belgedeki tüm aday bölgelerin ve grupların listesini (envanter) çıkarın:

```bash
.venv/bin/python -m src.ingestion.formula_loader belge.pdf --output envanter.json
```

Bu JSON çıktısında her adayın `id` (ör. `p1-f7`) ve sayfa numarası yer alır. İstenirse `.venv/bin/python -m src.evaluation.review_formulas belge.pdf --output envanter.html` ile tarayıcıda görsel olarak incelenebilir.

#### 2. Seçici Tanıma Komutu (Selective OCR)
Opsiyonel formül ortamı ve model önbellek dizini kullanılarak (modelleri içermeyen standart `.venv` yerine opsiyonel formül ortamı ile) yalnızca ilgilenilen formül tanınır:

```bash
# Tek bir formül ID seçimi
env PYTHONPATH=".venv/lib/python3.14/site-packages" /home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/ocr-venv/bin/python -m src.ingestion.formula_loader belge.pdf \
  --recognize \
  --formula-id p1-f7 \
  --cache-dir /home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/model-cache \
  --timeout 120 \
  --budget 150 \
  --output selected.json
```

Sayfa bazında filtreleme için tekrarlanabilir `--page` parametresi kullanılabilir:

```bash
# Yalnızca 1. sayfadaki formüller
env PYTHONPATH=".venv/lib/python3.14/site-packages" /home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/ocr-venv/bin/python -m src.ingestion.formula_loader belge.pdf \
  --recognize \
  --page 1 \
  --cache-dir /home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/model-cache \
  --timeout 120 \
  --budget 150 \
  --output selected.json
```

#### 3. İnceleme Raporunu Dışa Aktarma
Ardından standart ortamdaki `review_formulas` ile çıkarım incelenir:

```bash
.venv/bin/python -m src.evaluation.review_formulas belge.pdf \
  --extraction selected.json \
  --output selected.html
```

#### Seçici Tanıma Semantiği ve Kuralları
- **Yalnızca Tanıma Çağrılarını Sınırlar:** `--page` ve `--formula-id` filtreleri yalnızca model çağrılarını (`recognizer.recognize`) sınırlar. Belgenin tam native bağlamı, tüm sayfaları, düz yazı blokları, formülleri ve grupları çıktıda eksiksiz korunur.
- **Seçilmeyen Formüller:** Seçilmeyen formüller başarısız (`failed`) sayılmaz; açıkça `recognition_status="not_requested"` ve `reason="not_selected_for_recognition"` durumunu alır. İşaret tutarlılık tanıları `not_checked` kalır, `answer_eligible=false` korunur.
- **Kısmi Gruplar (Partial Groups):** Bir formül grubu içindeki çocuklardan yalnızca biri seçildiğinde kardeş formüllere otomatik genişletilmez (`no auto-expansion`). Kısmen tanınan grup `state="unresolved"` kalır, asla birleşik tam formül uydurulmaz (`latex=null`) ve açıkça `partial_group_recognition` uyarısı verilir.
- **Grup ID Reddi:** Filtreye doğrudan grup ID'si (ör. `p1-f6+p1-f7`) verilemez; sistem bunu reddederek seçilebilecek çocuk yaprak ID'leri (`p1-f6`, `p1-f7`) açıkça bildirir.
- **Hata ve Bütçe Koruma:** Filtreler `--recognize` olmadan çalıştırılamaz. Geçersiz/aralık dışı sayfa, bilinmeyen ID, seçilen sayfa dışındaki ID veya boş seçim durumunda herhangi bir OCR modeli çalıştırılmadan önce hata verilir. Filtreler zaman aşımı veya toplam tanıma bütçesini sıfırlamaz.

### Satır Sonu İlişki Devamları (Wrapped Relations)

Aynı paragraf ve sayfa bölümünde satır sonundaki eksik bir bağıntı parçası (ör. `K=`)
fiziksel bir sonraki satırın başında formül devamıyla (ör. `2^{-2/3}`) birleştiğinde,
kaynak okuma sırası kanıtlarıyla bu iki parça isteğe bağlı `formula_group` bloğu ve
tek bir atomik parçada bir arada tutulur. Her iki alt formül kaydının bağımsız
ID'si, bbox'ı, SHA-256'sı, özgün metni ve aday LaTeX'i eksiksiz korunur; uzak kutular
fiziksel olarak birleştirilmez. Aradaki anlamlı düz yazı, farklı paragraf, sütun veya
tablo hücreleri bu gruplamayı kesin olarak reddeder.

PyMuPDF yükseltilmiş simgeler nedeniyle devamı ayrı bir ham metin bloğuna ayırabilir.
Kaynakta ardışık iki blok, aynı sol kenar, son/ilk satır ve normal satır aralığı
kanıtı taşıyorsa satır sonundan sol kenara dönen eksik bağıntı yine gruplanabilir.
Bu bağlantı sezgiseldir; cevap kanıtı olarak onaylanmaz.

**Güven sözleşmesi:** Her formül `original_text`, `latex`, `page_number`,
`bbox`, `source_sha256`, `recognition_status`, `state` ve `reason` taşır.
Formül bbox'ları döndürülmemiş sayfanın koordinat sistemindedir; özgün bölge
metni ve tanıma görüntüsü de aynı geometriyle çıkarılır. HTML inceleme raporu,
kaydedilmiş JSON dahil, kutuları aynı geometride doğrular ve kaynak kırpımlarını
bu geometride üretir; grup çocuklarının kutuları da korunur. Sayfa dönüşü yalnızca
bellekte sıfırlanır; kaynak PDF'nin baytları, kayıtlı dönüşü ve SHA-256 özeti değişmez.

`unverified` modelin aday döndürdüğünü; `unresolved` tanıma yapılmadığını,
başarısız olduğunu veya çizgili kapsam/script bağlantısının kaynak incelemesi gerektirdiğini
belirtir. İkisi de doğrulanmış gerçek değildir. Model mutlak değer çubukları yerine çift çubuk
(`\|`) üretebilir veya düz yazı sözcüklerini math-mode harfleri olarak aktarabilir; bu çıktılar
kusursuz doğru kabul edilmemelidir. Formül grupları da sezgisel aday ilişkilerdir; doğrulanmış denklem
değildir ve alt tanıma başarısızlığında birleşik geçerli LaTeX uydurulmaz. Yerel detector işaretsiz inline
matematiği kaçırabilir; bu yüzden tüm sayfalar, formüller, gruplar ve parçalar
`detection_status=native_heuristic_incomplete` ve `answer_eligible=false`
taşır. Sayfa görüntüsü kaynak kanıtıdır, aday LaTeX'in doğruluk sertifikası
değildir. Bu özellik yalnızca ilişkisel kaynak bağlamını korur; eksiksiz bir PDF soru-cevap
sistemi kurmaz.

### Kaynak Eksi İşareti Tutarlılık Uyarısı

Native glif kanıtı ve döndürülmüş LaTeX birlikte varsa `sign_consistency`,
eksi işaretinin desteklenen kapsamda ve komşu operandlarla korunup korunmadığını
sezgisel olarak kontrol eder. Üs/alt indis, tek kesrin pay/payda kapsamı ve
basit satır içi işaretler ayrılır; paydadaki eksi, eksik üs eksisini karşılamaz.
`warnings` kaynak işareti, span/karakter konumu, glif ve formül bbox'ı, sayfa,
SHA-256 ve değiştirilmemiş `raw_candidate` taşır. Gruplar, atomik parçalar ve
arama gösterimi bu uyarıları korur. Mevcut tanıma `reason`/hata durumları değişmez.

Örnek: [kaydedilmiş açık kaynak ölçümü](eval_data/small_formula_ocr_corrected.json)
`2 ^ { \\cdot \\frac { 2 } { 3 } }` döndürmüştür. Aynı PDF ile bu çıktıyı
**kontrollü recognizer double** üzerinden tekrar oynatınca beklenen sonuç:

```json
{
  "sign_consistency": {"status": "mismatch"},
  "warnings": [{
    "code": "source_minus_candidate_conflict",
    "page_number": 1,
    "source_operators": [{"operator": "-", "span_id": 18,
      "scope": [["superscript", "2"]]}],
    "raw_candidate": "2 ^ { \\cdot \\frac { 2 } { 3 } }"
  }],
  "answer_eligible": false
}
```

Bu kısaltılmış örnek yeni OCR ölçümü değildir; kalan model hatası düzeltilmez.
`no_conflict_detected` yalnızca desteklenen eksi kanıtında çatışma bulunmadığını
belirtir; kök, üs, alt indis veya bütün formülün doğruluğunu kanıtlamaz.
Tanıma yapılmamış/başarısızsa, native glif yoksa, tarama/görselde veya desteklenen
eksi kanıtı çıkarılamıyorsa `not_checked` kalır. Native yazı sözcükleri,
satır sonu tireleri ve çizilmiş kesir çizgileri eksi olarak sayılmaz.
Yatay native yerleşim, tek atomlu script tabanları ve sınırlı TeX sözdizimi desteklenir; metin
komutlarının içindeki işaretler matematik işareti sayılmaz. Birden çok kesir,
tekrarlanan kapsam/taban, belirsiz script bağlantısı, bozuk veya desteklenmeyen
TeX `uncertain` uyarısı üretir. Genel LaTeX/algebra doğrulaması yapılmaz;
`native_heuristic_incomplete`, doğrulanmamış durumlar ve `answer_eligible=false`
korunur. Varsayılan CLI/arama native-only kalır ve OCR çalıştırıldığını iddia etmez.

### Çevrimdışı Formül Regresyon ve Değerlendirme

Çevrimdışı, model indirme veya harici API gerektirmeyen deterministik formül değerlendirme komutu mevcuttur. 5 temel regresyon kategorisini (kesir pay/payda tam kapsamı, kök vinculum kapsamı, üst/alt simge, tekil mutlak değer çubukları ve satır sonu ilişkisel formül devamları) geometrik aday yönlendirmesi ve metin kirlenmesi açısından test eder.

Komut satırı kullanımı (varsayılan çevrimdışı modda herkese açık sentetik benchmark çalışır; geometri yönlendirmesi ölçülür, transkripsiyon UNMEASURED kalır):

```bash
.venv/bin/python -m src.evaluation.evaluate_formulas
```

JSON rapor çıktısı alma veya belirli bir PDF çalıştırma:

```bash
.venv/bin/python -m src.evaluation.evaluate_formulas --output eval_report.json
.venv/bin/python -m src.evaluation.evaluate_formulas --pdf belge.pdf --output eval_report.json
.venv/bin/python -m src.evaluation.evaluate_formulas --json
```

İsteğe bağlı olarak yerel Docling `CodeFormulaV2` modeliyle gerçek OCR transkripsiyon değerlendirmesi çalıştırma (PyTorch ve Docling içeren opsiyonel formül ortamı gereklidir; sistem Python'unda bu kütüphanelerin bulunduğu varsayılmaz):

```bash
# Örnek: var olan formül ortamı ve önbellek dizini ile çalıştırma
/home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/ocr-venv/bin/python -m src.evaluation.evaluate_formulas \
  --recognize \
  --cache-dir /home/fatih/projects/firstmate/data/hybriddoc-formula-fidelity/model-cache \
  --timeout 120 \
  --budget 480 \
  --output live_eval_report.json
```

Python kullanımı:

```python
from src.evaluation.evaluate_formulas import evaluate_formula_cases, format_evaluation_report

results = evaluate_formula_cases()
print(format_evaluation_report(results))
```

#### Güven Sözleşmesi ve Sınırlamalar
- **Geometri / Aday Yönlendirmesi ile Transkripsiyon Sadakati Ayrımı:** Varsayılan çevrimdışı komut ve stub testleri yalnızca deterministik çıkarma geometrisini, sınırları ve grup bağlantılarını doğrular (`Geometry Routing: PASS`). Varsayılan modda stub LaTeX çıktıları transkripsiyon doğruluğu iddiasında bulunamaz ve transkripsiyon sadakati açıkça `UNMEASURED` olarak etiketlenir.
- **Canlı Tanıma ve Muhafazakâr LaTeX Normalizasyonu:** Gerçek OCR sadakati yalnızca `--recognize` bayrağı ile ölçülür. Tanıma sonuçları altın LaTeX dizeleriyle karşılaştırılırken yalnızca iki bilinen zararsız biçimlendirme kuralı ve temel boşluklar normalize edilir:
  1. `\text{...}` dışındaki TeX kontrol boşluğu (`\ `) normal boşluğa indirgenir;
  2. Tekil üs/alt indis simgelerindeki gereksiz parantezler (`t^{2}` vs `t^2`, `a_{i}` vs `a_i`) eşitlenir.
  Çok simgeli üsler/indisler (`t^{2n}`), kesirli ve negatif üsler (`2^{-2/3}`), kök kapsamları (`\sqrt{u+v}`), mutlak değer çubukları (`|`), işlem işaretleri ve sembolik küme parantezleri (`\{`) semantik olarak kesinlikle korunur. Desteklenmeyen veya farklı yapılar açık uyumsuzluk (`FAIL`) olarak kaydedilir; keyfi semantik onarım veya genel LaTeX parse motoru kullanılmaz.
- **Sentetik Benchmark Font ve Geometri Doğruluğu:** `SYN-FRAC` sentetik fikstürü, kaynakta ASCII şapka (`^`) yerine PDF yerleşik fontlarıyla yükseltilmiş matematiksel üs (`x`, `2`, `+ 1`) geometrisi çizer; böylece modelin fiziksel glifleri doğru okuması ve altın veriyle hizalanması sağlanır.
- **Negatif Kesirli Üs Sınırlaması:** `SYN-WRAP` kaynağındaki payda/çizgi çakışması ve üs konumu düzeltildi. Düzeltilmiş açık kaynakta model hâlâ eksi yerine `\cdot` döndürdü; sonuç `FAIL` olarak korunur. [Kaynaklar, ayrı ölçüm bütçeleri ve yeniden üretilebilir regresyon](docs/small_formula_ocr.md) gerçek OCR doğruluğunu deterministik testlerden ayırır.
- **Ayrı Sayım Metrikleri:** Komut çıktısı Geometri Yönlendirmesi ve Transkripsiyon Sadakati için ayrı ayrı `PASS`, `FAIL`, `AMBIGUOUS` ve `UNMEASURED` sayaçları üretir.
- **Cevap Uygunluğu Değişmezliği:** Sentetik benchmark'ta tüm adaylar `answer_eligible=false` ve `detection_status=native_heuristic_incomplete` sözleşmesini korur. Başarılı sentetik eşleşmeler üretim formüllerini kesin gerçek olarak onaylamaz.
- **İzole Sentetik Veri:** Değerlendirme yalnızca açık sentetik test geometrileri kullanır; özel PDF yolları veya koordinatları repoya veya CI'a dahil edilmez.

## Çevrimdışı Formül İnceleme ve Kaynak Kanıtı Görselleştirme (Formula Review Export)

Çıkarılan formül bölgelerini, kaynak PNG kırpıntılarını (PyMuPDF 3x render), ham aday LaTeX metinlerini ve işaret/eylem tutarlılık tanılarını bağımsız ve tek parça statik bir HTML raporu olarak dışa aktarmak için:

### 1. Varsayılan Çevrimdışı Mod (Yalnızca Yerel Çıkarım - Native-Only)
Herhangi bir OCR modeli veya harici bağımlılık çalıştırmadan, yalnızca PDF'teki dijital formül adaylarını ve kaynak kırpıntılarını görselleştirir (aday LaTeX üretilmez, açıkça `No OCR candidate` olarak etiketlenir):

```bash
.venv/bin/python -m src.evaluation.review_formulas belge.pdf --output review.html
```

### 2. Kaydedilmiş Çıkarım JSON'u ile İnceleme (Saved Candidates & Warnings)
Daha önce `src.ingestion.formula_loader` ile üretilmiş aday LaTeX ve işaret uyarılarını içeren JSON dışa aktarımı ile birlikte görselleştirme:

```bash
# Adım 1: formula_loader ile JSON çıkarımı oluşturma (model isteğe bağlıdır)
.venv/bin/python -m src.ingestion.formula_loader belge.pdf --output extraction.json

# Adım 2: İnceleme HTML raporunu üretme
.venv/bin/python -m src.evaluation.review_formulas belge.pdf --output review.html --extraction extraction.json
```

Oluşturulan HTML dosyasını tarayıcınızda açarak inceleyebilirsiniz:

```bash
# Google Chrome veya varsayılan sistem tarayıcısı ile açma
google-chrome review.html
# veya
xdg-open review.html
```

### Özellikler ve Güven Sözleşmesi
- **Kendi Kendine Yeten Statik HTML (Zero JS / Zero Network):** Üretilen rapor tamamen tek parça UTF-8 HTML'dir. Sunucu, webapp çatısı, Google Fonts, CDN veya JavaScript çalıştırması içermez. Tüm kaynak formül kırpıntıları doğrudan `data:image/png;base64` veri URL'leri olarak gömülür.
- **Garantili SHA-256 Eşleşmesi:** `--extraction` parametresi verildiğinde, JSON içerisindeki `source_sha256` değeri sağlanan PDF dosyasının SHA-256 özetiyle tam olarak eşleşmek zorundadır. Eksik, uyumsuz veya tahrif edilmiş özetler ve sayfa sınırları dışındaki (`out-of-bounds`) koordinatlar açıklayıcı hata ile kesinlikle reddedilir.
- **İki Sütunlu Karşılaştırma Düzeni:** Masaüstünde sol sütunda yerel kaynak PNG kırpıntısı, özgün metin ve koordinatlar; sağ sütunda ise ham aday LaTeX, tanıma durumu ve işaret tutarlılığı tanıları sergilenir (küçük ekranlarda duyarlı olarak tek sütuna katlanır).
- **Satır Sonu İlişkisel Gruplar (Wrapped Groups):** Satır sonu devam ilişkisi bulunan formüllerde (`p1-f6+p1-f7`), çocuk formüllerin tekil sınır kutuları (`bbox`), sırası ve kırpıntıları kesinlikle korunur; yapay bileşik kırpıntı üretilmez ve kartlar arasında iki yönlü bağlantı bağlantıları (`#card-id` ve `#group-id`) sunulur.
- **Doğruluk İddiası Yoktur:** Desteklenen eksi kapsamlarında çelişki bulunmaması (`no_conflict_detected`), formülün matematiksel olarak doğrulanmış olduğu anlamına gelmez. Adaylar kesinlikle onarılmaz; ham hali ve varsa uyarılarıyla birlikte gösterilir. `answer_eligible=false` ve eksik tespit uyarıları raporun en başında belirgin şekilde vurgulanır.
- **Girdi Koruma Garantisi:** Çıktı HTML yolu, kaynak PDF veya çıkarım JSON dosyalarının üzerine yazılamaz ya da bu yollarla eşlenemez (alias oluşturulamaz).

## Çevrimdışı Anahtar Kelime Kaynak Arama (BM25 Source Search)

Harici model indirme, API veya vektör veritabanı gerektirmeyen, Python standart kütüphanesiyle çalışan deterministik Okapi BM25 sözlüksel arama bileşeni mevcuttur. Belgedeki düz yazı parçalarını ve formül bileşen metinlerini indeksleyerek ilgili kaynak sayfaları ve parçaları sıralar.

Kurulumdan bağımsız komut örneği:

```bash
python -m src.retrieval.search_pdf sample_docs/Fatih_RAG_OCR_Multimodal_PDF_Use_Case.pdf "ayrılabilir diferansiyel" --top-k 3 --output sonuclar.txt
```

Python kullanımı:

```python
from src.retrieval.search_pdf import search_pdf, format_search_results

results = search_pdf("belge.pdf", "separable differential equations", top_k=3)
formatted_output = format_search_results(results, "separable differential equations", file_path="belge.pdf")
print(formatted_output)
```

### Özellikler ve Güven Sözleşmesi

- **Deterministik BM25 Sıralaması:** Okapi BM25 formülü standart kütüphane ile işletilir; eşit puanlarda kaynak belge sırası (`source_order`) deterministik olarak korunur. Puanlar bağıl alaka sıralamasıdır; olasılık, kesinlik veya doğruluk skoru değildir.
- **Sadece Pozitif Eşleşmeler:** Yalnızca pozitif puanlı parçalar döndürülür; eşleşme olmayan sorgularda ilgisiz parçalar listelenmez.
- **Kanıt Bulunamadı Politikası (*No-Evidence*):** Eşleşme bulunamadığında açıkça eşleşen kaynak bulunamadığı belirtilir. Bu durum, aranan bilginin dokümanda kesinlikle bulunmadığı anlamına gelmez (ör. taranmış görsel sayfalar veya farklı sözcük kullanımı).
- **Cevap Uygunluğu (*answer_eligible=false*):** Arama bir LLM cevaplayıcı veya formül doğrulayıcı değildir; yalnızca kaynak bağlamını bulur. Tüm parçalarda `answer_eligible=false` ve `detection_status=native_heuristic_incomplete` korunur; arama hiçbir bayrağı tersine çevirmez.
- **Formül Grupları:** Satır sonu ilişkisel formül devamları (`formula_group`) bölünmez; sıralı çocuk ID'leri, özgün metinleri, bbox'ları ve durumları eksiksiz listelenir.
- **Çıktı Alanları:** Her sonuç `rank`, `score`, `file_name`, `page_number`, `total_pages`, `chunk_id`, `source_order`, `source_sha256`, `detection_status`, `answer_eligible`, `review_reason`, formül/grup metaverileri ve tam alıntıyı (`excerpt`) taşır. Standart çıktı ile `--output` dosya içeriği birebir eşdeğerdir.

### Sınırlamalar

- **Sözlüksel Eşleştirme:** Çevrimdışı BM25 araması Türkçe sorguları İngilizce belge terimlerine çevirmez ve eş anlamlıları algılamaz. Sorgular belgenin kendi sözcükleriyle yapılmalıdır.
- **Taranmış Sayfalar:** Yalnızca taranmış/görsel içeren ve dijital metin katmanı olmayan sayfalarda yerel metin çıkarılamadığından eşleşme bulunamaz.

Kontrol: `python -m pytest -q`. Testler herkese açık sentetik PDF geometrisi
kullanır; özel PDF veya altın koordinatlar repoya dahil değildir.
