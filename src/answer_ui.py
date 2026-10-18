"""Small opt-in Streamlit panel; network only inside explicit button handlers."""
import os

import streamlit as st

from src.answers import (AnswerError, SYSTEM_PROMPT, build_evidence, evidence_issue,
                         confirm_source_passage, confirm_source_relation, evidence_fingerprint, math_evidence_issue,
                         gemini_models, generate_answer, ollama_models, relation_choices, request_preview)


def _format_step_checklist(
    *,
    document_name: str | None,
    results_count: int,
    generation_enabled: bool,
    provider: str | None = None,
    model: str | None = None,
    configured: bool = False,
    needs_confirmation: bool = False,
    confirmation_done: bool = False,
    consent: bool = False,
    can_generate: bool = False,
) -> str:
    lines = ["**Adım Akışı / Kontrol Listesi:**"]
    lines.append(f"1. **Belge:** ✓ Hazır ({document_name or 'Belge yüklendi'})")
    if results_count > 0:
        lines.append(f"2. **Kaynaklar:** ✓ {results_count} kaynak bulundu (Yerel BM25 metin araması)")
    else:
        lines.append("2. **Kaynaklar:** ○ Eşleşen kaynak bulunamadı (Farklı sözcüklerle arayın)")

    if not generation_enabled:
        lines.append("3. **Sağlayıcı / model kurulumu:** ○ Kapalı (İsteğe bağlı model cevabı için 'Cevap üretimini etkinleştir' kutusunu işaretleyin)")
    else:
        if provider == "Ollama":
            if model:
                lines.append(f"3. **Sağlayıcı / model kurulumu:** ✓ Ollama yerel model hazır ({model})")
            else:
                lines.append("3. **Sağlayıcı / model kurulumu:** ○ Yerel model seçilmedi ('Yerel model listesini oku / yenile' düğmesini kullanın)")
        else:
            if configured:
                lines.append(f"3. **Sağlayıcı / model kurulumu:** ✓ Ortam hazır ({model}) — API bağlantısı/yetki otomatik sınanmaz")
            else:
                lines.append("3. **Sağlayıcı / model kurulumu:** ○ Ortam yapılandırması eksik (GEMINI_API_KEY veya HYBRIDDOC_GEMINI_MODELS)")

    step_num = 4
    if needs_confirmation:
        if confirmation_done:
            lines.append(f"{step_num}. **Kaynak onayı:** ✓ Kullanıcı onayı tamamlandı")
        else:
            lines.append(f"{step_num}. **Kaynak onayı:** ○ Onay bekleniyor (Aşağıdaki kaynak onayı formunu doldurun)")
        step_num += 1

    if generation_enabled and provider == "Gemini API":
        if consent:
            lines.append(f"{step_num}. **Bulut veri onayı:** ✓ Google aktarım onayı verildi")
        else:
            lines.append(f"{step_num}. **Bulut veri onayı:** ○ Google aktarım onayı bekleniyor (Onay kutusunu işaretleyin)")
        step_num += 1

    if can_generate:
        lines.append(f"{step_num}. **Cevap üret:** ✓ Hazır ('Cevap üret' düğmesine basın)")
    else:
        lines.append(f"{step_num}. **Cevap üret:** ○ Bekleniyor (Eksik adımları tamamlayın)")

    return "\n".join(lines)


def render_answer_panel(results: list, question: str, document=None):
    st.subheader('Kaynaklara dayalı cevap — isteğe bağlı')
    doc_name = document.file_name if document is not None else None
    if not st.checkbox('Cevap üretimini etkinleştir', key='generation_enabled'):
        st.session_state.pop('generated_answer', None)
        st.session_state.pop('source_confirmations', None)
        st.markdown(_format_step_checklist(
            document_name=doc_name,
            results_count=len(results),
            generation_enabled=False,
        ))
        return

    provider = st.selectbox('Sağlayıcı', ['Ollama', 'Gemini API'], key='answer_provider')
    evidence, omitted = build_evidence(results)
    if omitted:
        st.warning(f'{omitted} kaynak boş veya kanıt bütçesini aştığı için gönderimden çıkarıldı. '
                   'Metin kesilmedi; tam kaynağı yukarıdan inceleyin.')
    by_id = {item['id']: item for item in evidence}
    usable = []
    for sid, item in by_id.items():
        reason = evidence_issue([item])
        if reason:
            st.warning(f"Varsayılan seçim dışında: {sid} — Sayfa {item['page']}: {reason} "
                       'Kaynak yukarıda görünür; seçili sıradan metin için ayrıca kaynak veya ilişki onayı gerekir.')
        else:
            usable.append(sid)
    selected = st.multiselect('Gönderilecek kaynaklar', list(by_id), default=usable,
                             format_func=lambda sid: f"{sid} — Sayfa {by_id[sid]['page']} — {by_id[sid].get('field_relation', {}).get('label', 'Kaynak')}",
                             key='answer_sources')
    evidence = [by_id[sid] for sid in selected if sid in by_id]
    confirmation_scope = (question, evidence_fingerprint(evidence))
    if confirmation_scope != st.session_state.get('confirmation_scope'):
        st.session_state['source_confirmations'] = {}
        st.session_state['confirmation_scope'] = confirmation_scope
        for key in list(st.session_state):
            if key.startswith(('relation_label_', 'relation_value_', 'confirmation_kind_')):
                st.session_state.pop(key, None)
    records = st.session_state.setdefault('source_confirmations', {})
    for item in evidence:
        if not evidence_issue([item]) or math_evidence_issue(item):
            continue
        sid = item['id']
        st.text(f"Kaynak onayı — {sid} — Sayfa {item['page']}\n{item['text']}")
        crop_visible = False
        if document is not None and item.get('source_bbox'):
            try:
                st.image(document.source_crop({'page_number': item['page'],
                                               'source_context': {'bbox': item['source_bbox']}}),
                         caption='Onaylanacak kaynağın özgün kırpımı')
                crop_visible = True
            except Exception:
                st.warning('Kaynak kırpımı gösterilemedi; kaynak onayı kullanılamaz.')
        kind = st.selectbox('Onay türü', ['relation', 'passage'],
                            format_func=lambda k: 'Etiket–değer ilişkisi' if k == 'relation' else 'Tam pasajı kaynak olarak kullan',
                            key='confirmation_kind_' + sid)
        record = records.get(sid)
        if record and record.get('kind') != kind:
            records.pop(sid)
        if kind == 'relation':
            lines = relation_choices(item)
            label = st.selectbox('Kaynakta görünen etiket', lines, index=None,
                                 key='relation_label_' + sid)
            value = st.selectbox('Bu etikete ait kaynakta görünen değer', lines, index=None,
                                 key='relation_value_' + sid)
            record = records.get(sid)
            if record and (record['label'], record['value']) != (label, value):
                records.pop(sid)
            st.caption(f"Seçilen ilişki: {label or 'Etiket seçilmedi'} → {value or 'Değer seçilmedi'}. "
                       'Yalnızca bu kaynak ve bu soru için kullanıcı beyanıdır; uygulama doğrulaması değildir.')
            if st.button('Gösterilen etiket–değer ilişkisini onaylıyorum', key='confirm_relation_' + sid,
                         disabled=not crop_visible or not label or not value or label == value):
                records[sid] = confirm_source_relation(evidence, sid, question, label, value)
        else:
            st.caption('Tam pasajı bu soru için kaynak olarak kullanmayı onaylayın. '
                       'Bu onay etiket–değer ilişkisi kurmaz; belirsiz alanlar için ayrı ilişki onayı gerekir.')
            if st.button('Gösterilen tam pasajı kaynak olarak onaylıyorum', key='confirm_passage_' + sid,
                         disabled=not crop_visible):
                records[sid] = confirm_source_passage(evidence, sid, question)
        if sid in records:
            item['user_confirmation'] = records[sid]
            st.caption(('Bu ilişkiyi kullanıcı onayladı' if kind == 'relation' else 'Bu tam pasajı kullanıcı onayladı')
                       + '; uygulama doğrulamadı. association=not_inferred korunur.')

    st.caption('Önizleme: yalnızca aşağıdaki soru ve seçili kaynak JSON’u gönderilir. '
               'PDF dosyası, görseller, API anahtarı ve diğer sayfalar bu içeriğe eklenmez. '
               'En çok 5 kaynak, kaynak başına 4000 karakter; toplam kanıt 16000 karakter.')
    preview = request_preview(question, evidence)
    with st.expander('İstek gövdesi önizlemesi (gönderilecek tam JSON)', expanded=False):
        st.text(preview)
    with st.expander('Sabit model talimatı', expanded=False):
        st.text(SYSTEM_PROMPT)

    model, configured, consent = None, False, False
    models = []
    if provider == 'Ollama':
        st.caption('Sabit yerel adres: http://127.0.0.1:11434. Model indirme/servis başlatma yapılmaz. '
                   'Yerel envanter düğmesi yalnızca model adlarını okur; belge göndermez. '
                   'Cloud/uzak modeller kullanılmaz; Cevap üret önce yerel model metadata’sını denetler.')
        if st.button('Yerel model listesini oku / yenile', key='refresh_models'):
            st.session_state.pop('generated_answer', None)
            try:
                st.session_state['local_models'] = ollama_models()
                st.session_state.pop('inventory_error', None)
            except AnswerError as exc:
                st.session_state['local_models'] = []
                st.session_state['inventory_error'] = str(exc)
        if st.session_state.get('inventory_error'):
            st.error(st.session_state['inventory_error'])
        models = st.session_state.get('local_models', [])
        if models:
            model = st.selectbox('Yerel model', models, key='ollama_model')
            configured = True
        else:
            st.info('Yerel envanteri okuyun. Uygun yerel model listesi boş olabilir veya yalnızca cloud '
                    'modelleri bulunabilir; erişim hatası modelin bulunmadığını kanıtlamaz. '
                    'Kurulum için Türkçe kılavuza bakın.')
        with st.expander('Ollama yerel servis kurulum rehberi', expanded=not configured):
            st.markdown(
                "**Yerel Ollama Kurulumu (Terminal):**\n\n"
                "Ollama bu makinede (`http://127.0.0.1:11434`) çalışmalıdır. Uygulama otomatik servis başlatmaz veya model indirmez.\n\n"
                "```zsh\n"
                "# Ollama servisini başlatın:\n"
                "ollama serve\n\n"
                "# Mevcut modelleri listeleyin:\n"
                "ollama list\n\n"
                "# Gerekirse yerel model indirin:\n"
                "ollama pull llama3.2:3b\n"
                "```\n"
                "Servis çalıştıktan sonra yukarıdaki **'Yerel model listesini oku / yenile'** düğmesine basın."
            )
    else:
        st.warning('Gemini API seçilirse soru ve önizlemede gösterilen seçili kaynak alıntıları '
                   'Google’a gönderilir. Ücret/kota ve Google veri koşulları geçerlidir. '
                   'Sağlayıcı seçmek gönderim yapmaz; bu belge için onay ve Cevap üret gerekir.')
        consent = st.checkbox('Bu belge için önizlemedeki soru ve kaynakların Google’a gönderilmesine onay veriyorum.',
                              key='cloud_consent')
        try:
            models = gemini_models()
        except AnswerError as exc:
            models = []
            st.error(str(exc))
        if models:
            model = st.selectbox('Gemini modeli (yapılandırılmış katalog)', models, key='gemini_model')
        else:
            st.info('HYBRIDDOC_GEMINI_MODELS eksik/boş: resmi katalogdaki desteklenen model kimliklerini '
                    'ortamda virgülle ayırın. Otomatik bulut model seçimi yapılmaz.')
        configured = bool(model and os.getenv('GEMINI_API_KEY', '').strip())
        if not os.getenv('GEMINI_API_KEY', '').strip():
            st.info('GEMINI_API_KEY eksik. Anahtarı yerel ortamda ayarlayın; sohbetten paylaşmayın. '
                    'AGY aboneliği API erişimi anlamına gelmez.')
        with st.expander('Gemini API anahtarı ve ortam kurulum rehberi (yerel terminal)', expanded=not configured):
            st.markdown(
                "**Başlangıç Rehberi: Gemini API Kurulumu (Yerel Terminal)**\n\n"
                "Güvenlik gereği API anahtarı web arayüzüne **yazılmaz** ve kaydedilmez. Anahtar yalnızca uygulamayı başlattığınız terminal oturumundaki ortam değişkenlerinden okunur.\n\n"
                "**Zsh / Bash terminalinizde çalıştırın:**\n"
                "```zsh\n"
                "# 1. API anahtarını tanımlayın (geçmişe yazmadan güvenle girmek için):\n"
                'read -s "GEMINI_API_KEY?Gemini API anahtarı (yerel): " && export GEMINI_API_KEY\n\n'
                "# Veya doğrudan atayın:\n"
                'export GEMINI_API_KEY="AIzaSy..."\n\n'
                "# 2. Desteklenen Gemini model kataloğunu virgülle ayırarak tanımlayın:\n"
                'export HYBRIDDOC_GEMINI_MODELS="gemini-2.5-flash,gemini-3.5-flash-lite,gemini-3.8-flash"\n\n'
                "# 3. Ardından bu terminalde uygulamayı yeniden başlatın:\n"
                "python -m streamlit run apps/workbench.py --server.address 127.0.0.1 --browser.gatherUsageStats false --client.toolbarMode minimal\n"
                "```\n\n"
                "> **Önemli Hatırlatma:** Ortam değişkenleri tanımlandığında arayüzde 'Yapılandırma hazır' görünür. "
                "Ancak Google sunucusuna önceden bağlantı veya yetki testi yapılmaz (kota/yetki otomatik prob edilmez). "
                "Gerçek API çağrısı yalnızca siz açıkça **Cevap üret** düğmesine bastığınızda ve onay verdiğinizde yapılır."
            )

    if configured:
        st.caption('Yapılandırma hazır. Sağlayıcı bağlantısı, API yetkisi ve model erişimi henüz '
                   'doğrulanmış değildir; Cevap üret isteği sırasında denetlenir. Kaynak uygunluğu ayrı denetlenir.')

    scope = (provider, model, preview, consent)
    if scope != st.session_state.get('answer_scope'):
        st.session_state.pop('generated_answer', None)
        st.session_state.pop('answer_error', None)
        st.session_state['answer_scope'] = scope

    issue = evidence_issue(evidence, question=question)
    if issue:
        st.warning(issue)

    disabled_reasons = []
    next_actions = []

    if not configured:
        if provider == 'Ollama':
            disabled_reasons.append('Yerel model seçilmedi veya Ollama servisine erişilemedi.')
            next_actions.append("'Yerel model listesini oku / yenile' düğmesine basın (servis çalışmıyorsa terminalde `ollama serve` başlatın).")
        else:
            if not os.getenv('GEMINI_API_KEY', '').strip():
                disabled_reasons.append('GEMINI_API_KEY ortam değişkeni eksik.')
                next_actions.append("Terminalde API anahtarını tanımlayın: `export GEMINI_API_KEY=\"...\"` (Aşağıdaki 'Gemini API anahtarı ve ortam kurulum rehberi' açılır alanına bakın).")
            if not models:
                disabled_reasons.append('HYBRIDDOC_GEMINI_MODELS ortam değişkeni eksik veya boş.')
                next_actions.append('Terminalde model listesini tanımlayın: `export HYBRIDDOC_GEMINI_MODELS="gemini-2.5-flash"`')
            elif not model:
                disabled_reasons.append('Gemini modeli seçilmedi.')
                next_actions.append('Yukarıdaki model açılır kutusundan bir model seçin.')

    if provider == 'Gemini API' and not consent:
        disabled_reasons.append('Bulut veri aktarımı onayı verilmedi.')
        next_actions.append("'Bu belge için önizlemedeki soru ve kaynakların Google’a gönderilmesine onay veriyorum' kutusunu işaretleyin.")

    if issue:
        disabled_reasons.append(f'Kaynak kanıt engeli: {issue}')
        needing_conf = [item for item in evidence if evidence_issue([item]) and not math_evidence_issue(item)]
        if needing_conf:
            next_actions.append("Yukarıdaki 'Kaynak onayı' bölümünden belirsiz kaynaklar için etiket–değer ilişkisini veya tam pasajı onaylayın ya da kaynağı seçimden çıkarın.")
        elif any(math_evidence_issue(item) for item in evidence):
            next_actions.append("Doğrulanmamış matematik/formül içeren kaynağı 'Gönderilecek kaynaklar' listesinden çıkarın; formülü kaynak kesitinden inceleyin.")
        elif not evidence:
            if any(evidence_issue([item]) and not math_evidence_issue(item) for item in by_id.values()):
                next_actions.append("'Gönderilecek kaynaklar' listesinden kaynak seçin (belirsiz kaynaklar için yukarıdaki 'Kaynak onayı' adımı uygulanacaktır).")
            else:
                next_actions.append("'Gönderilecek kaynaklar' listesinden en az bir kaynak seçin.")
        else:
            next_actions.append("Soru sözcüklerini değiştirerek 'Kaynakları ara' yapın.")

    is_disabled = bool(disabled_reasons)

    needs_conf = any(evidence_issue([item]) and not math_evidence_issue(item) for item in evidence)
    conf_done = bool(needs_conf and all(item['id'] in records for item in evidence if evidence_issue([item]) and not math_evidence_issue(item)))

    st.markdown(_format_step_checklist(
        document_name=doc_name,
        results_count=len(results),
        generation_enabled=True,
        provider=provider,
        model=model,
        configured=configured,
        needs_confirmation=needs_conf,
        confirmation_done=conf_done,
        consent=consent,
        can_generate=not is_disabled,
    ))

    if is_disabled:
        st.info(
            "**Cevap üretimi şu an devre dışı:**\n"
            + "\n".join(f"- **Neden:** {r}" for r in disabled_reasons)
            + "\n\n"
            + "**Sonraki desteklenen adım:**\n"
            + "\n".join(f"- {a}" for a in next_actions)
        )

    if st.button('Cevap üret', key='generate_answer', disabled=is_disabled):
        st.session_state.pop('generated_answer', None)
        try:
            with st.spinner('Seçili sağlayıcıdan cevap bekleniyor…'):
                st.session_state['generated_answer'] = generate_answer(provider, model, question, evidence, consent=consent)
            st.session_state.pop('answer_error', None)
        except AnswerError as exc:
            st.session_state['answer_error'] = str(exc)
    if st.session_state.get('answer_error'):
        st.error(st.session_state['answer_error'])
    if 'generated_answer' in st.session_state:
        answer = st.session_state['generated_answer']
        st.caption('Model cevabı: ' + ('kaynak yetersiz / çekimser' if answer['status'] == 'abstained' else 'alıntı denetiminden geçti'))
        if any(item.get('user_confirmation') for item in evidence):
            st.caption('Kaynak kullanımını veya ilişkisini kullanıcı onayladı; uygulama doğrulaması veya anlamsal doğruluk garantisi değildir.')
        st.text(answer['answer'])
        for citation in answer['citations']:
            source = by_id[citation['source_id']]
            st.text(f"{citation['source_id']} — Sayfa {source['page']}\n{citation['quote']}")
        st.caption('Denetim yalnızca kaynak kimliğini ve alıntının gönderilen metinde bulunduğunu kontrol eder; '
                   'cevabın anlamsal doğruluğu veya alıntının iddiayı desteklemesi garanti edilmez. '
                   'answer_eligible=false ve tüm formül/kaynak uyarıları korunur. Kaynağı inceleyin.')
