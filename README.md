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

`chunk_formula_pages` formül, formül grubu ve uyarı bloklarını atomik tutar; bir atomik blok sınırdan
uzunsa tek, daha büyük parça üretir. Düz yazı blokları karakter/örtüşme
kurallarını kullanır; formül örtüşmesi yapılmaz. Parçalar kaynak sayfa, hash,
bbox, özgün metin ve belirsizlik alanlarını korur.

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
metni ve tanıma görüntüsü de aynı geometriyle çıkarılır. PDF dosyasının kayıtlı
sayfa dönüşü değiştirilmez.

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
