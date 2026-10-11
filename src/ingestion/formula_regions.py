"""Conservative native math region proposals, without reconstructing equations."""
import re
import pymupdf


def _math_text(text, font):
    return ('math' in font.lower() or any(0x1D400 <= ord(c) <= 0x1D7FF for c in text)
            or any(c in text for c in '√∂∫∑∏′″‴')
            or bool(re.search(r"[A-Za-z0-9)](?:['′″])*\s*[=≠]", text)))


def _near(a, b, size):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return dx <= size * 1.5 and dy <= size * .65


def _merge_regions(regions):
    changed = True
    while changed:
        changed = False
        for ai, a in enumerate(regions):
            for bi in range(ai + 1, len(regions)):
                b = regions[bi]
                if a['cell'] is None and b['cell'] is None and _near(a['bbox'], b['bbox'], max(a['size'], b['size'])):
                    a['bbox'] |= b['bbox']
                    a['span_ids'] |= b['span_ids']
                    a['size'] = max(a['size'], b['size'])
                    regions.pop(bi)
                    changed = True
                    break
            if changed:
                break


def propose_regions(page):
    """Return candidate boxes, native span inventory and table cell metadata.

    Proposals deliberately overflag; boxes/LaTeX are never semantic verification.
    """
    spans = []
    for block in page.get_text('rawdict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                text = ''.join(c['c'] for c in span['chars'])
                if text.strip():
                    spans.append({'text': text, 'bbox': pymupdf.Rect(span['bbox']),
                                  'size': span['size'], 'script': bool(span['flags'] & 1), 'math': _math_text(text, span['font'])
                                  or bool(span['flags'] & 1)})
    cells = []
    table_error = None
    try:
        for ti, table in enumerate(page.find_tables().tables):
            for row, rr in enumerate(table.rows):
                for col, box in enumerate(rr.cells):
                    if box:
                        cells.append({'bbox': pymupdf.Rect(box), 'table': ti,
                                      'row': row, 'column': col})
    except Exception as exc:
        table_error = f'table detection failed: {exc}'
    rules = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1:]
                if abs(a.y - b.y) < .1 and abs(a.x - b.x) > 2:
                    rules.append(pymupdf.Rect(min(a.x, b.x), a.y, max(a.x, b.x), a.y))
    # A bar with glyphs on both sides is evidence of math, not its interpretation.
    for rule in rules:
        above, below = [], []
        for i, s in enumerate(spans):
            box = s['bbox']
            if box.x0 < rule.x1 and box.x1 > rule.x0:
                cy = (box.y0 + box.y1) / 2
                if abs(cy - rule.y0) < 1.5 * s['size']:
                    (above if cy < rule.y0 else below).append(i)
        if above and below and rule.width < .8 * page.rect.width:
            for i in above + below:
                if len(spans[i]['text'].strip()) <= 12:
                    spans[i]['math'] = True
    regions = []
    for i, s in enumerate(spans):
        cell = s['cell'] = next((c for c in cells if c['bbox'].contains(
            pymupdf.Point((s['bbox'].x0+s['bbox'].x1)/2,
                          (s['bbox'].y0+s['bbox'].y1)/2))), None)
        if not s['math']:
            continue
        if cell:
            found = next((r for r in regions if r['cell'] == cell), None)
            if found:
                found['span_ids'].add(i)
                continue
        regions.append({'bbox': pymupdf.Rect(cell['bbox'] if cell else s['bbox']),
                        'span_ids': {i}, 'size': s['size'], 'cell': cell})
    _merge_regions(regions)
    # Attach nearby small operands/scripts from other native blocks.
    for r in regions:
        for i, s in enumerate(spans):
            if i in r['span_ids']:
                continue
            if r['cell']:
                inside = r['bbox'].contains(s['bbox'])
            else:
                inside = (s['cell'] is None and len(s['text'].strip()) <= 8 and
                          _near(r['bbox'], s['bbox'], r['size']))
            if inside:
                r['span_ids'].add(i)
                if not r['cell']:
                    r['bbox'] |= s['bbox']
                    r['size'] = max(r['size'], s['size'])
    _merge_regions(regions)
    for r in regions:
        related = []
        for rule in rules:
            if r['cell'] and (abs(rule.y0-r['bbox'].y0) < 1 or abs(rule.y0-r['bbox'].y1) < 1):
                continue
            if (rule.x0 < r['bbox'].x1 and rule.x1 > r['bbox'].x0
                    and r['bbox'].y0-r['size']*.5 <= rule.y0 <= r['bbox'].y1+r['size']*.5):
                related.append(list(rule))
                if not r['cell']:
                    r['bbox'] |= pymupdf.Rect(rule.x0, rule.y0-.05, rule.x1, rule.y0+.05)
        r['rules'] = related
        r['script_evidence'] = any(spans[i]['script'] or spans[i]['size'] < .9*r['size']
                                   for i in r['span_ids'])
        box = r['bbox']
        if not r['cell']:
            r['bbox'] = pymupdf.Rect(box.x0-2, box.y0-2, box.x1+2, box.y1+2) & page.rect
    return sorted(regions, key=lambda r: min(r['span_ids'])), spans, table_error
