"""Optional local formula ingestion: source evidence and candidates, never verified truth."""
import hashlib
from pathlib import Path
import pymupdf
from src.ingestion.pdf_loader import PDFDocumentLoader
from src.ingestion.formula_regions import propose_regions


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
                           'native_script_evidence': region['script_evidence']}
                if region['cell']:
                    formula['table_cell'] = {k: v for k, v in region['cell'].items() if k != 'bbox'}
                formulas.append(formula)
                for i in region['span_ids']:
                    owners[i] = formula
                pending.append((formula, pn-1, bool(region['rules']) or region['script_evidence']))
            emitted = set()
            for i, span in enumerate(spans):
                f = owners.get(i)
                if f:
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
                          'unresolved_reason': unresolved_reason})
        # Prioritize radicals before the bounded worker budget is spent on small inline references.
        if recognizer:
            for f, page_idx, review_required in sorted(pending, key=lambda p: ('√' not in p[0]['original_text'], '=' not in p[0]['original_text'])):
                if len(f['original_text'].strip()) < 3:
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
        for block in page['blocks']:
            if block['kind'] == 'formula':
                f = by_id[block['formula_id']]
                label = f"[{f['state'].upper()} FORMULA {f['id']}; not answer evidence]"
                block['text'] = label + (f"\n\\[ {f['latex']} \\]" if f['latex'] else '')
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
