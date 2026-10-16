"""Invented Turkish form PDFs; no real student data or external font files."""
import pymupdf


def mixed_font_prose_pdf(*, rotation=0, reverse=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=650, height=500)
        prefix = 'Ankara is the capital of '
        entries = [(35, prefix, 'helv'),
                   (35 + pymupdf.get_text_length(prefix, fontname='helv', fontsize=11),
                    'Turkey.', 'hebo')]
        for x, text, font in reversed(entries) if reverse else entries:
            page.insert_text((x, 100), text, fontname=font, fontsize=11)
        page.set_rotation(rotation)
        return pdf.tobytes()


def certificate_pdf(*, order='columns', father='Orhan Örnek', columns=1,
                    stacked=False, second_page=False, inline=False, colon=False):
    with pymupdf.open() as doc:
        page = doc.new_page(width=650, height=500)
        page.insert_font(fontname='fixture', fontbuffer=pymupdf.Font('cjk').buffer)
        def put(x, y, text):
            page.insert_text((x, y), text, fontsize=11, fontname='fixture')
        put(35, 45, 'SENTETİK ÖĞRENCİ BELGESİ — tüm kişiler uydurmadır')
        entries = []
        for col in range(columns):
            x = 35 + col * 320
            values = ['Deniz Örnek', 'Selin Örnek', father] if col == 0 else ['Ekin Kurgu', 'İpek Kurgu', 'Şükrü Kurgu']
            for row, (label, value) in enumerate(zip(['Adı / Soyadı', 'Anne Adı', 'Baba Adı'], values)):
                y = 110 + row * (65 if stacked else 30)
                if inline:
                    entries.append((x, y, f'{label}: {value}'))
                else:
                    entries.append((x, y, label + (":" if colon else "")))
                    if value:
                        entries.append((x if stacked else x + 150, y + 20 if stacked else y, value))
        if order == 'columns':
            entries.sort(key=lambda e: (e[0], e[1]))
        elif order == 'reverse':
            entries.reverse()
        for x, y, text in entries:
            put(x, y, text)
        if second_page:
            page = doc.new_page(width=650, height=500)
            page.insert_font(fontname='fixture', fontbuffer=pymupdf.Font('cjk').buffer)
            put(35, 110, 'Baba Adı: Uzak Sayfa')
        doc.subset_fonts()
        return doc.tobytes(garbage=4, deflate=True)
