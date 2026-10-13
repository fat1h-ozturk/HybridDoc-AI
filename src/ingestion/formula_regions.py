"""Conservative native math region proposals, without reconstructing equations."""
import re
import pymupdf


def _math_text(text, font):
    return ('math' in font.lower() or any(0x1D400 <= ord(c) <= 0x1D7FF for c in text)
            or any(c in text for c in '√∂∫∑∏′″‴')
            or bool(re.search(r"[A-Za-z0-9)|\]\}](?:['′″])*\s*[=≠≤≥≈<>]", text)))


def _near(a, b, size):
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return dx <= size * 1.5 and dy <= size * .65


def _has_omitted_interval(a, b, spans):
    sids_a = a['span_ids']
    sids_b = b['span_ids']
    min_sid = min(min(sids_a), min(sids_b))
    max_sid = max(max(sids_a), max(sids_b))
    all_sids = sids_a | sids_b
    merged_box = a['bbox'] | b['bbox']
    for k in range(min_sid + 1, max_sid):
        if k not in all_sids:
            s = spans[k]
            if not merged_box.contains(s['bbox']):
                return True
            if not s['math'] and len(s['text'].strip()) > 8:
                return True
    return False


def _merge_regions(regions, spans):
    changed = True
    while changed:
        changed = False
        for ai, a in enumerate(regions):
            for bi in range(ai + 1, len(regions)):
                b = regions[bi]
                if (a['cell'] is None and b['cell'] is None
                        and _near(a['bbox'], b['bbox'], max(a['size'], b['size']))
                        and not _has_omitted_interval(a, b, spans)):
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
    for b_idx, block in enumerate(page.get_text('rawdict')['blocks']):
        if block.get('type') != 0:
            continue
        block_bbox = pymupdf.Rect(block['bbox'])
        for l_idx, line in enumerate(block.get('lines', [])):
            line_bbox = pymupdf.Rect(line['bbox'])
            for span in line['spans']:
                text = ''.join(c['c'] for c in span['chars'])
                if text.strip():
                    spans.append({'text': text, 'bbox': pymupdf.Rect(span['bbox']),
                                  'size': span['size'], 'script': bool(span['flags'] & 1),
                                  'math': _math_text(text, span['font'])
                                  or bool(span['flags'] & 1),
                                  'block_idx': b_idx, 'line_idx': l_idx,
                                  'line_bbox': line_bbox, 'block_bbox': block_bbox})
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
    for s in spans:
        s['cell'] = next((c for c in cells if c['bbox'].contains(
            pymupdf.Point((s['bbox'].x0+s['bbox'].x1)/2,
                          (s['bbox'].y0+s['bbox'].y1)/2))), None)
    rules = []
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0] == 'l':
                a, b = item[1:]
                if abs(a.y - b.y) < .1 and abs(a.x - b.x) > 2:
                    rule = pymupdf.Rect(min(a.x, b.x), a.y, max(a.x, b.x), a.y)
                    if not any(rule.x0 < c['bbox'].x1 and rule.x1 > c['bbox'].x0
                               and (abs(rule.y0-c['bbox'].y0) < 1
                                    or abs(rule.y0-c['bbox'].y1) < 1) for c in cells):
                        cell = next((c for c in cells if c['bbox'].contains(
                            pymupdf.Point((rule.x0+rule.x1)/2, rule.y0))), None)
                        rules.append((rule, cell))
    regions = []
    # A bar with glyphs on both sides is evidence of math, not its interpretation.
    math_rules = []
    for rule, cell in rules:
        above, below = [], []
        for i, s in enumerate(spans):
            if s['cell'] != cell or (not s['math'] and len(s['text'].strip()) > 35):
                continue
            box = s['bbox']
            if box.x0 < rule.x1 and box.x1 > rule.x0:
                cy = (box.y0 + box.y1) / 2
                if abs(cy - rule.y0) < 1.5 * s['size']:
                    (above if cy < rule.y0 else below).append(i)
        if above and below and rule.width < .8 * page.rect.width:
            min_dist_above = min(rule.y0 - (spans[i]['bbox'].y0 + spans[i]['bbox'].y1) / 2 for i in above)
            above = [i for i in above if (rule.y0 - (spans[i]['bbox'].y0 + spans[i]['bbox'].y1) / 2) <= min_dist_above + 3.0]
            min_dist_below = min((spans[i]['bbox'].y0 + spans[i]['bbox'].y1) / 2 - rule.y0 for i in below)
            below = [i for i in below if ((spans[i]['bbox'].y0 + spans[i]['bbox'].y1) / 2 - rule.y0) <= min_dist_below + 3.0]
            for i in above + below:
                spans[i]['math'] = True
            if cell is None:
                box = pymupdf.Rect(rule.x0, rule.y0-.05, rule.x1, rule.y0+.05)
                for i in above + below:
                    box |= spans[i]['bbox']
                regions.append({'bbox': box, 'span_ids': set(above + below),
                                'size': max(spans[i]['size'] for i in above + below),
                                'cell': None})
            math_rules.append((rule, cell, set(above + below)))
    for i, s in enumerate(spans):
        cell = s['cell']
        if not s['math'] or any(i in r['span_ids'] for r in regions):
            continue
        if cell:
            found = next((r for r in regions if r['cell'] == cell), None)
            if found:
                found['span_ids'].add(i)
                found['size'] = max(found['size'], s['size'])
                continue
        regions.append({'bbox': pymupdf.Rect(cell['bbox'] if cell else s['bbox']),
                        'span_ids': {i}, 'size': s['size'], 'cell': cell})
    _merge_regions(regions, spans)
    # Attach nearby small operands/scripts from other native blocks.
    for r in regions:
        for i, s in enumerate(spans):
            if i in r['span_ids']:
                continue
            if r['cell']:
                inside = r['bbox'].contains(s['bbox'])
            else:
                inside = (s['cell'] is None and (s['math'] or len(s['text'].strip()) <= 30) and
                          _near(r['bbox'], s['bbox'], r['size']) and
                          not _has_omitted_interval(r, {'span_ids': {i}, 'bbox': s['bbox']}, spans))
            if inside:
                r['span_ids'].add(i)
                r['size'] = max(r['size'], s['size'])
                if not r['cell']:
                    r['bbox'] |= s['bbox']
    _merge_regions(regions, spans)
    for r in regions:
        related = []
        for rule, cell, rule_spans in math_rules:
            if cell != r['cell']:
                continue
            if rule_spans & r['span_ids']:
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
