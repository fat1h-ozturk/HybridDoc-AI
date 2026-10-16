"""Native source geometry for prose runs and narrow same-row colon relations."""
from collections import Counter
import re

import pymupdf

from src.retrieval.search_pdf import normalize_text
from src.ingestion.formula_regions import native_table_cells


def _native_rows(page) -> list:
    spans = [dict(span, direction=line['dir'], wmode=line['wmode'])
             for block in page.get_text('dict')['blocks'] if block['type'] == 0
             for line in block['lines'] for span in line['spans'] if span['text'].strip()]
    rows = []
    for span in sorted(spans, key=lambda s: (s['origin'][1], s['bbox'][0])):
        if not rows or abs(span['origin'][1] - rows[-1][0]['origin'][1]) > 1:
            rows.append([])
        rows[-1].append(span)
    return rows


def prose_run_for_hit(page, hit) -> str:
    """Use native paragraphs and contiguous font runs, excluding isolated word fragments."""
    runs = []
    for row in _native_rows(page):
        run = None
        for span in sorted(row, key=lambda s: s['bbox'][0]):
            box = pymupdf.Rect(span['bbox'])
            horizontal = tuple(span['direction']) == (1, 0) and span['wmode'] == 0
            if (run and horizontal and run['horizontal']
                    and -.5 <= box.x0 - run['bbox'].x1 <= max(run['size'], span['size']) / 2):
                run['bbox'] |= box
                run['text'] += span['text']
                run['size'] = max(run['size'], span['size'])
            else:
                run = dict(bbox=box, text=span['text'], size=span['size'], horizontal=horizontal)
                runs.append(run)
    matches = [run for run in runs if run['bbox'].intersects(hit)]
    if len(matches) != 1 or not matches[0]['horizontal']:
        return ''
    selected = matches[0]
    box = selected['bbox']
    for other in runs:
        if other is selected:
            continue
        peer = other['bbox']
        if peer.y0 < box.y1 and box.y0 < peer.y1:
            return ''
    paragraphs = [block for block in page.get_text('dict')['blocks']
                  if block['type'] == 0 and pymupdf.Rect(block['bbox']).contains(box)]
    text = selected['text']
    if len(paragraphs) == 1:
        lines = paragraphs[0]['lines']
        if any(tuple(line['dir']) != (1, 0) or line['wmode'] != 0 for line in lines):
            return ''
        text = ' '.join(''.join(span['text'] for span in line['spans']).strip()
                        for line in sorted(lines, key=lambda line: (line['bbox'][1], line['bbox'][0])))
    return text if len(text.split()) > 1 else ''


def explicit_fields(pdf) -> dict:
    """Keep original native spans; reject multi-column, duplicate and wrapped fields."""
    candidates, counts = [], Counter()
    for page in pdf:
        rows = _native_rows(page)
        spans = [span for row in rows for span in row]
        cells, table_error = native_table_cells(page)
        for row in rows:
            row.sort(key=lambda s: s['bbox'][0])
            text = ' '.join(s['text'].strip() for s in row)
            for prefix in text.split(':')[:-1]:
                words = prefix.split()
                counts.update({normalize_text(' '.join(words[i:]))
                               for i in range(len(words))
                               if re.fullmatch(r'[^\W\d_][\w /()\-]{0,99}', ' '.join(words[i:]))})
            if table_error:
                continue
            if any(tuple(s['direction']) != (1, 0) or s['wmode'] != 0 for s in row):
                continue
            if any(a['bbox'][2] > b['bbox'][0] + .5 for a, b in zip(row, row[1:])):
                continue
            if text.count(':') != 1:
                continue
            label, value = (part.strip() for part in text.split(':'))
            if not re.fullmatch(r'[^\W\d_][\w /()\-]{0,99}', label):
                continue
            if not value:
                continue
            # A value must be a single native span, not arbitrary neighboring text.
            colon = next(i for i, s in enumerate(row) if ':' in s['text'])
            if len(row) - colon > (1 if row[colon]['text'].split(':')[1].strip() else 2):
                continue
            box = pymupdf.Rect(row[0]['bbox'])
            for span in row[1:]:
                box |= pymupdf.Rect(span['bbox'])
            if (any(c['bbox'].intersects(box) for c in cells)
                    and not any(c['bbox'].contains(box) for c in cells)):
                continue
            label_right = row[colon]['bbox'][2]
            # shortcut: wrapped labels need a richer form parser; leave them for source review.
            if any(s not in row and s['bbox'][0] < label_right and s['bbox'][2] > box.x0
                   and 0 <= box.y0 - s['bbox'][3] < box.height / 2 for s in spans):
                continue
            candidates.append((page.number, {'label': label, 'value': value,
                                             'bbox': list(box), 'spans': row}))
    fields = {}
    for page, field in candidates:
        label = normalize_text(field['label'])
        if counts[label] == 1:
            fields.setdefault(page, []).append(field)
    return fields


def relation_for_hit(fields: list, hit, context_text: str) -> dict | None:
    matches = [f for f in fields if pymupdf.Rect(f['bbox']).intersects(hit)]
    if len(matches) != 1:
        return None
    field = matches[0]
    expected = (field['label'], field['value'])
    quote = next((line.strip() for line in context_text.splitlines()
                  if line.count(':') == 1 and tuple(' '.join(part.split())
                     for part in line.split(':')) == expected), None)
    return dict(field, quote=quote) if quote else None
