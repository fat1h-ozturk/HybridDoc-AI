# Kaynaklara dayalı cevap üretimi — ilk entegrasyon

Arama ve kaynak inceleme varsayılan olarak model kullanmaz. Cevap üretimi
isteğe bağlıdır: **Kaynakları ara → Cevap üretimini etkinleştir → sağlayıcı ve
model seç → gönderilecek kaynakları/önizlemeyi incele → Cevap üret**.
Etkinleştirme sırasında varsayılan sağlayıcı **Ollama** olur. Sağlayıcı/model
seçimi, sayfa seçimi, normal yeniden çizim veya arama cevap isteği göndermez.
Buluta otomatik geçiş yoktur. Her yeni cevap için düğmeye tekrar basmak gerekir.

## Ortak kurulum

Python 3.11+ ile:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-workbench.txt
python -m streamlit run apps/workbench.py --server.address 127.0.0.1 --server.maxUploadSize 20 --browser.gatherUsageStats false --client.toolbarMode minimal
```

Tarayıcıda http://127.0.0.1:8501 adresini açın. Arayüzü internete açmayın.
HTTPX doğrudan bağımlılıktır; ayrıca bir sağlayıcı SDK’sı gerekmez.
Mevcut kaynak kartları, görüntü kırpımları, sayfa/parça/formül incelemesi ve
kaynak indirmeleri aynı şekilde kullanılabilir. Model cevabı kaynak indirmesine
veya HTML raporuna eklenmez.

## Yerel Ollama

Ollama ayrıca kurulmuş, `127.0.0.1:11434` üzerinde çalışıyor ve bir **yerel**
model önceden indirilmiş olmalı. Kurulum/model indirme bu uygulamanın işi
değildir; resmi [Ollama belgelerini](https://docs.ollama.com/quickstart) izleyin.
Model seçimini kendi RAM/VRAM kapasitenize göre yapın; bu proje herhangi bir
modelin donanıma sığdığını veya iyi Türkçe cevap verdiğini ölçmüş değildir.

1. Belgeyi aradıktan sonra cevap üretimini etkinleştirin; **Ollama** seçili kalır.
2. **Yerel model listesini oku / yenile** düğmesine basın. Bu yalnızca
   `GET /api/tags` envanteridir; soru, belge veya üretim isteği içermez.
3. Listelenen kurulu modellerden birini seçin. Liste boşsa uygun yerel model
   olmayabilir veya yalnızca uzak modeller bulunabilir. Bağlantı hatası ise
   model yokluğu kanıtı değildir; servis/adres durumunu yerelde kontrol edin.
4. Kaynak önizlemesini inceleyip **Cevap üret** düğmesine basın. Önce
   `POST /api/show` yalnızca model adıyla yerel metadata’yı denetler; sonra
   `POST /api/chat` önizlemedeki soru/kaynakları gönderir.

Adres sabittir; uzak endpoint veya proxy ortam değişkeni kabul edilmez.
Ollama’nın yerel API’si bulut modellerini de sunabildiğinden `remote_host`,
`remote_model` ve cloud etiketli modeller reddedilir; metadata yerel
model ağırlıklarını göstermiyorsa soru gönderilmez. Ek sunucu güvencesi için
Ollama’yı kendi yapılandırmanızda `OLLAMA_NO_CLOUD=1` ile yerel moda alın
([resmi açıklama](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features)).
Yerel modelin metadata’sında bağlam kapasitesi yoksa üretim durur. Girdi için
UTF-8 bayt sayısıyla muhafazakâr bir token üst sınırı, 1500 çıktı tokenı ve
256 şablon tokenı payı kullanılır; istek en çok 8192 veya modelin bildirdiği
daha küçük bağlamla sınırlanır. Fazla büyükse tam kaynakları azaltın;
arka planda kesme yapılmaz. Bu kontrol donanıma sığma garantisi değildir.
Uygulama hiçbir servisi başlatmaz/durdurmaz ve model indirmez.

## Gemini API

Google’a gönderim isteğe bağlı ve belgeye özel onaylıdır. Soru ve önizlemede
seçili metinler Google’a gider; PDF dosyası/görüntüler/tüm belge yüklenmez.
Google veri koşulları ve proje ücret/kotası geçerlidir. Hassas belge için bu
seçeneği kullanmadan önce bu aktarımı değerlendirin.

1. Gemini API erişimi olan kendi Google projenizi ve anahtarınızı yerelde
   yapılandırın. **AGY aboneliği API erişimi/kotası kanıtı değildir.** Anahtarı
   sohbete, Git’e veya komut satırı argümanına yapıştırmayın. Örneğin Bash’te
   anahtarı ekrana ve geçmişe yazmadan ortam değişkenine almak için:

   ```bash
   read -r -s -p 'Gemini API anahtarı (yerel): ' GEMINI_API_KEY
   export GEMINI_API_KEY
   ```

2. [Güncel resmi model kataloğundan](https://ai.google.dev/gemini-api/docs/models)
   hesabınızın desteklediği metin modellerini seçip aynı terminalde tanımlayın.
   15 Ekim 2026’da katalogda bulunan kimliklerle yapılandırma örneği:

   ```bash
   export HYBRIDDOC_GEMINI_MODELS='gemini-3.5-flash-lite,gemini-3.8-flash'
   ```

   Bu yalnızca yapılandırma örneğidir; model erişimi/ücreti canlı test edilmedi.
   Liste uygulamaya gömülü değildir; katalog değişirse ortam listesini güncelleyin.
   Liste boşsa otomatik veya eski bir model seçilmez. Uygulama anahtar/model
   erişimini otomatik sorgulamaz; `GEMINI_API_KEY` eksikse açık yönlendirme gösterir.
3. Ortam değişkenlerini alan terminalden yukarıdaki Streamlit komutunu çalıştırın.
4. Arayüzde **Gemini API**, katalogdaki model ve kaynakları seçin. Google
   gönderim açıklamasını okuyun; **bu belge için** onay kutusunu işaretleyin.
   Onay vermek tek başına istek göndermez. **Cevap üret** düğmesine basın.

Anahtar sadece sunucuda `x-goog-api-key` başlığına eklenir; URL/önizleme/indirilen
kaynaklarda bulunmaz. Ham servis hata gövdeleri gösterilmez. Otomatik yeniden
deneme yoktur. Hız/kota sınırı, erişim reddi, eksik model, zaman aşımı,
bozuk/eksik JSON ve model reddi için güvenli hata gösterilir. Bağlantı zaman
sınırı 2 saniye; envanter okuma sınırı 5, üretim okuma sınırı 45 saniyedir.
Yanıt gövdesi en çok 64 KiB; üretim çıktısı en çok 1500 token istenir.

## Kanıt, alıntı ve sınırlar

- En çok beş kaynak seçilebilir. Varsa **tam** `source_context`, yoksa özgün
  `excerpt` kullanılır. Kaynak başına 4000, metadata dahil toplam 16000 karakter
  bütçesi uygulanır. Bütçeyi aşan kaynak **bütün olarak çıkarılır** ve sayı
  gösterilir; kesilerek alan ilişkisi değiştirilmez. Soru 1–2000 karakterdir.
- Önizlemedeki JSON her iki sağlayıcıya da aynı kullanıcı içeriği olarak gider;
  sabit model talimatı ayrıca açılabilir. Her kaynağın belge özeti/parça/sayfadan
  türetilen sabit kimliği ve sayfası vardır. Dosya adı bulut içeriğine eklenmez.
- Model cevabı `answered` veya `abstained`, cevap metni ve kaynak kimliği +
  **birebir destekleyici alıntı** döndürmelidir. Bilinmeyen kaynak, metinde olmayan
  alıntı, boş/alıntısız cevap veya bozuk yapı reddedilir. Alıntı denetimi,
  cevabın anlamsal doğruluğunu ya da alıntının iddiayı desteklediğini kanıtlamaz.
  Yanıt hiçbir zaman “doğrulanmış” olarak sunulmaz; kaynak incelemesi gerekir.
- Hiç kanıt yoksa çağrı yapılmaz. Eşleşen kaynakta sorulan bilgi yoksa modelden
  çekimser kalması istenir. Bu istem politikasıdır, model doğruluğu garantisi değildir.
- `association=not_inferred` korunur. Ayrı etiket ve yakın isimlerden
  Father/Mother/Student eşleştirmesi yapılmaz. İlk entegrasyon muhafazakârdır:
  özgün eşleşmedeki satırlar açık `Alan: Değer` biçiminde veya noktalama ile
  biten cümle görünümünde değilse üretim durur ve kaynak kesiti gösterilir.
  Bu yapısal kontrol bir semantik form çözümleyici değildir; bazı düz metinleri
  de incelemeye bırakabilir. Açık alan cümlesi bile ancak alıntı kanıtı sunar.
- Formül adayları/grupları veya seçili kesitte matematik gösterimi varsa üretim
  yapılmaz. OCR/formül görüntüleri güvenilir zemin hâline gelmez. Sayfaya ait
  `math_status`, `detection_status`, `answer_eligible=false`, formül ve inceleme
  uyarıları korunur. Formülsüz bir metin pasajı aynı sayfadaki başka formül
  yüzünden matematik cevabı olarak sunulmaz; modelden formül çözmesi istenmez.
- Kaynak metin ve soru güvenilmeyen veri kabul edilir. İstem kaynak içindeki
  talimatların uygulanmasını yasaklar; hiçbir araç/çalıştırma yetkisi yoktur.
  Model cevabı ve alıntılar düz metin gösterilir.
- Belge değişince onay ve cevap silinir. Yeni arama, kaynak/model/sağlayıcı
  değişikliği eski cevabı kaldırır. Oturumu temizlemek tüm oturum verisini siler.
- BM25 hâlâ sözcük aramasıdır; Türkçe iyelik ekleri ve genel etiket eşitlikleri
  düzeltilmiş değildir. LLM, getirilmeyen bir bilgiyi bulmuş sayılmaz. Güvenilir
  genel RAG, form ilişkileri ve formül doğruluğu bu kilometre taşının iddiası değildir.

## Çevrimdışı, açıkça simüle edilmiş gösterim ve testler

```bash
python -m pytest tests/test_answer_demo.py -q -s
python -m pytest tests/test_answers.py tests/test_workbench_ui.py -q
python -m pytest -q
```

İlk komut **SİMÜLASYON** etiketli çıktı verir: Ankara/başkent düz metni, açık
`Father: Orhan Example` alanı, eksik nüfus bilgisi, iki sütunlu tamamen uydurma
form ve doğrulanmamış formül. PDF’ler testte üretilir. Cevap metinleri test
fikstürüdür; **gerçek çıkarım değildir**. HTTPX `MockTransport` kullanılır;
Ollama/Gemini’ye bağlantı, API kotası, model indirme veya özel öğrenci belgesi yoktur.
Streamlit AppTest; seçim, tam gönderim içeriği, onay/reset, normal yeniden
çizimde çağrı olmaması, alıntılar, hata ve çekimserlik yollarını çalıştırır.
Donanım performansı ve gerçek model kalitesi bu testlerle ölçülmez.

## Resmi kaynaklar (15 Ekim 2026’da kontrol edildi)

API şekilleri için [Ollama chat](https://docs.ollama.com/api/chat),
[model listesi](https://docs.ollama.com/api/tags),
[Ollama model metadata tipleri](https://github.com/ollama/ollama/blob/main/api/types.go),
[Gemini generateContent](https://ai.google.dev/api/generate-content),
[JSON yanıtları](https://ai.google.dev/gemini-api/docs/structured-output),
[model kataloğu](https://ai.google.dev/gemini-api/docs/models) ve
[anahtar yapılandırması](https://ai.google.dev/gemini-api/docs/api-key) kullanıldı.
Eski scout raporundaki model adları veya donanıma sığma tahminleri bu uygulamanın
dayanağı değildir. Bu geliştirmede canlı bulut isteği/hesap probu ve canlı yerel
çıkarım yapılmadı; tüm üretim yanıtları testlerde simüle edildi.
