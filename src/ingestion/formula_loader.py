"""Optional local formula ingestion: source evidence and candidates, never verified truth."""
import hashlib
from pathlib import Path
import pymupdf
from src.ingestion.pdf_loader import PDFDocumentLoader
from src.ingestion.formula_regions import propose_regions

_RELATIONAL_OPS = ('=', '≠', '≈', '≤', '≥', '<', '>')


def detect_wrapped_relations(formulas, spans):
    """Identify contiguous line-wrapped relational continuations.

    Returns a list of tuples: (left_formula, right_formula).
    Must strictly satisfy:
    1. Left formula ends with an incomplete relational operator (e.g. 'K=').
    2. Has a non-empty LHS before the relational operator.
    3. Right formula does not start with a relational operator.
    4. Consecutive in source spans (no intervening text or omissions).
    5. Same raw block, or adjacent blocks with aligned paragraph margins.
    6. Consecutive lines, or the final/first lines of those adjacent blocks.
    7. Neither formula is in a table cell.
    8. Normal line spacing (no excessive vertical gap).
    9. Right formula wraps to the left margin (horizontal wrap: B.x0 < A.x1).
    10. Unambiguous (1-to-1 linkage).
    """
    groups = []
    used_ids = set()
    for i in range(len(formulas) - 1):
        f_a = formulas[i]
        f_b = formulas[i + 1]
        if f_a['id'] in used_ids or f_b['id'] in used_ids:
            continue
        if f_a.get('table_cell') is not None or f_b.get('table_cell') is not None:
            continue
        sids_a = f_a.get('span_ids', [])
        sids_b = f_b.get('span_ids', [])
        if not sids_a or not sids_b:
            continue
        if min(sids_b) != max(sids_a) + 1:
            continue

        span_a_last = spans[max(sids_a)]
        span_b_first = spans[min(sids_b)]

        if span_a_last.get('cell') is not None or span_b_first.get('cell') is not None:
            continue

        block_a = span_a_last.get('block_idx')
        block_b = span_b_first.get('block_idx')
        if block_a is None or block_b is None:
            continue

        line_a = span_a_last.get('line_idx')
        line_b = span_b_first.get('line_idx')
        if line_a is None or line_b is None:
            continue
        if block_a == block_b:
            if line_b != line_a + 1:
                continue
        else:
            # Raised scripts can split a genuine continuation into a new raw block.
            a_box = span_a_last.get('block_bbox')
            b_box = span_b_first.get('block_bbox')
            size = max(span_a_last.get('size', 10), span_b_first.get('size', 10))
            if (block_b != block_a + 1 or line_b != 0
                    or line_a != max(s['line_idx'] for s in spans if s.get('block_idx') == block_a)
                    or a_box is None or b_box is None
                    or span_a_last.get('line_bbox') is None or span_b_first.get('line_bbox') is None
                    or b_box.y0 < a_box.y1
                    or abs(a_box.x0 - b_box.x0) > size / 2
                    or abs(span_b_first['bbox'].x0 - a_box.x0) > size / 2
                    or span_a_last['bbox'].x0 - a_box.x0 < 2 * size):
                continue

        line_a_box = span_a_last.get('line_bbox')
        line_b_box = span_b_first.get('line_bbox')
        if line_a_box is not None and line_b_box is not None:
            pitch = line_b_box.y0 - line_a_box.y0
            gap = line_b_box.y0 - line_a_box.y1
            max_size = max(span_a_last.get('size', 10), span_b_first.get('size', 10))
            if pitch <= 0 or pitch > 3.0 * max_size or gap > 25.0:
                continue

        if span_b_first['bbox'].x0 >= span_a_last['bbox'].x1:
            continue

        orig_a = f_a['original_text'].strip()
        trailing_op = next((op for op in _RELATIONAL_OPS if orig_a.endswith(op)), None)
        if not trailing_op:
            continue
        lhs = orig_a[:-len(trailing_op)].strip()
        if not lhs:
            continue

        orig_b = f_b['original_text'].strip()
        if not orig_b:
            continue
        if any(orig_b.startswith(op) for op in _RELATIONAL_OPS):
            continue

        groups.append((f_a, f_b))
        used_ids.add(f_a['id'])
        used_ids.add(f_b['id'])
    return groups


def load_pdf_formulas(file_path, *, recognizer=None):
    """Preserve provenance and mask native math with explicitly uncertain blocks.

    Without a recognizer this performs native region analysis only. A recognizer
    exposes recognize(png_bytes)->str; errors retain unresolved source regions.
    """
    loader = PDFDocumentLoader(file_path)
    source_hash = hashlib.sha256(loader.file_path.read_bytes()).hexdigest()
    pages, pending = [], []
    with pymupdf.open(loader.file_path) as doc:
        for pn, page in enumerate(doc, 1):
            page.set_rotation(0)
            regions, spans, table_error = propose_regions(page)
            formulas, owners = [], {}
            blocks = [{'kind': 'warning', 'text':
                       '[UNVERIFIED NATIVE TEXT: math detection is incomplete; source review required]'}]
            for n, region in enumerate(regions, 1):
                box = region['bbox']
                original = page.get_text('text', clip=box).strip()
                formula = {'id': f'p{pn}-f{n}', 'page_number': pn, 'bbox': list(box),
                           'original_text': original, 'latex': None, 'state': 'unresolved',
                           'answer_eligible': False, 'recognition_status': 'not_requested',
                           'reason': 'recognition_not_requested', 'source_sha256': source_hash,
                           'native_rules': region['rules'],
                           'native_script_evidence': region['script_evidence'],
                           'span_ids': sorted(region['span_ids'])}
                if region['cell']:
                    formula['table_cell'] = {k: v for k, v in region['cell'].items() if k != 'bbox'}
                formulas.append(formula)
                for i in region['span_ids']:
                    owners[i] = formula
                pending.append((formula, pn-1, bool(region['rules']) or region['script_evidence']))

            wrapped_pairs = detect_wrapped_relations(formulas, spans)
            group_by_f_id = {}
            formula_groups = []
            for f_a, f_b in wrapped_pairs:
                grp = {
                    'id': f"{f_a['id']}+{f_b['id']}",
                    'page_number': pn,
                    'kind': 'wrapped_relation',
                    'state': 'unresolved',
                    'answer_eligible': False,
                    'linkage': 'heuristic_unverified_continuation',
                    'formula_ids': [f_a['id'], f_b['id']],
                    'formulas': [f_a, f_b],
                    'original_text': f"{f_a['original_text']} {f_b['original_text']}".strip(),
                    'latex': None,
                    'reasons': [f_a['reason'], f_b['reason']],
                    'source_sha256': source_hash,
                }
                group_by_f_id[f_a['id']] = grp
                group_by_f_id[f_b['id']] = grp
                formula_groups.append(grp)

            emitted = set()
            for i, span in enumerate(spans):
                f = owners.get(i)
                if f:
                    grp = group_by_f_id.get(f['id'])
                    if grp:
                        if grp['id'] not in emitted:
                            blocks.append({
                                'kind': 'formula_group',
                                'group_id': grp['id'],
                                'formula_ids': list(grp['formula_ids']),
                                'group': grp,
                                'text': '',
                            })
                            emitted.add(grp['id'])
                            for fid in grp['formula_ids']:
                                emitted.add(fid)
                    else:
                        if f['id'] not in emitted:
                            blocks.append({'kind': 'formula', 'formula_id': f['id'], 'text': ''})
                            emitted.add(f['id'])
                else:
                    blocks.append({'kind': 'text', 'text': span['text']})
            original = page.get_text('text').strip()
            unresolved_reason = table_error
            if page.get_images(full=True):
                unresolved_reason = 'image content is not covered by native math detection'
            if not original:
                unresolved_reason = 'no native text; image/scanned page requires review'
            if unresolved_reason:
                blocks.append({'kind': 'warning', 'text': f'[UNRESOLVED PAGE: {unresolved_reason}]'})
            pages.append({'page_number': pn, 'file_name': loader.file_path.name,
                          'total_pages': len(doc), 'source_sha256': source_hash,
                          'original_text': original, 'blocks': blocks, 'formulas': formulas,
                          'formula_groups': formula_groups,
                          'unresolved_reason': unresolved_reason})
        # Prioritize radicals before the bounded worker budget is spent on small inline references.
        if recognizer:
            for f, page_idx, review_required in sorted(pending, key=lambda p: ('√' not in p[0]['original_text'], '=' not in p[0]['original_text'])):
                stripped = f['original_text'].strip()
                if f['id'] not in group_by_f_id and (len(stripped) < 2 or (len(stripped) < 3 and not any(op in stripped for op in _RELATIONAL_OPS))):
                    f['reason'] = 'isolated_math_reference_requires_review'
                    continue
                try:
                    png = doc[page_idx].get_pixmap(matrix=pymupdf.Matrix(3, 3), clip=pymupdf.Rect(f['bbox'])).tobytes('png')
                    candidate = recognizer.recognize(png)
                    if not isinstance(candidate, str) or not candidate.strip():
                        raise ValueError('recognizer returned empty/non-text LaTeX')
                    f.update(latex=candidate.strip(), recognition_status='returned',
                             state='unresolved' if review_required else 'unverified',
                             reason='layout_scope_or_script_attachment_requires_source_review' if review_required else 'recognizer_output_not_verified')
                except Exception as exc:
                    f.update(state='unresolved', recognition_status='failed',
                             reason=f'{type(exc).__name__}: {exc}')
    for page in pages:
        by_id = {f['id']: f for f in page['formulas']}
        for grp in page.get('formula_groups', []):
            child_formulas = [by_id[fid] for fid in grp['formula_ids']]
            grp['formulas'] = child_formulas
            grp['reasons'] = [f.get('reason') for f in child_formulas]
            grp['original_text'] = ' '.join(f['original_text'] for f in child_formulas).strip()
            has_unresolved = any(f.get('state') == 'unresolved' for f in child_formulas)
            all_returned = all(f.get('recognition_status') == 'returned' and f.get('latex') for f in child_formulas)
            if all_returned:
                grp['latex'] = ' '.join(f['latex'].strip() for f in child_formulas)
                grp['state'] = 'unresolved' if has_unresolved else 'unverified'
            else:
                grp['latex'] = None
                grp['state'] = 'unresolved'
            grp['answer_eligible'] = False

        for block in page['blocks']:
            if block['kind'] == 'formula':
                f = by_id[block['formula_id']]
                label = f"[{f['state'].upper()} FORMULA {f['id']}; not answer evidence]"
                block['text'] = label + (f"\n\\[ {f['latex']} \\]" if f['latex'] else '')
            elif block['kind'] == 'formula_group':
                grp = block['group']
                label = f"[{grp['state'].upper()} FORMULA GROUP {grp['id']}; heuristic wrapped relation; not answer evidence]"
                block['text'] = label + (f"\n\\[ {grp['latex']} \\]" if grp['latex'] else '')
        page['text'] = '\n'.join(b['text'] for b in page['blocks'])
        page['char_count'] = len(page['text'])
        page['math_status'] = ('unresolved' if page['unresolved_reason'] or any(f['state']=='unresolved' for f in page['formulas'])
                               else 'unverified' if page['formulas'] else 'no_native_candidates')
        # Native evidence can miss unmarked inline or raster math. No page is certified.
        page['detection_status'] = 'native_heuristic_incomplete'
        page['answer_eligible'] = False
        page['review_reason'] = 'native detection is incomplete; source review required for math answers'
    return pages


def main(argv=None):
    """Readable full JSON, optionally saved; native-only unless explicitly enabled."""
    import argparse
    import json
    import sys
    from contextlib import nullcontext, redirect_stdout
    from src.ingestion.formula_recognizer import LocalFormulaRecognizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path)
    parser.add_argument('--recognize', action='store_true', help='use cached local CodeFormulaV2')
    parser.add_argument('--cache-dir', type=Path, help='HF_HOME containing hub/; never downloads')
    parser.add_argument('--timeout', type=float, default=120, help='seconds per crop, including model load')
    parser.add_argument('--budget', type=float, default=480, help='total recognition seconds')
    parser.add_argument('--output', type=Path, help='also save full UTF-8 JSON')
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            context = (LocalFormulaRecognizer(args.cache_dir, args.timeout, args.budget)
                       if args.recognize else nullcontext(None))
            with context as recognizer:
                pages = load_pdf_formulas(args.pdf, recognizer=recognizer)
        rendered = json.dumps(pages, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(rendered + '\n', encoding='utf-8')
        print(rendered)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'formula extraction failed: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
