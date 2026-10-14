"""Generate self-contained static HTML formula review and provenance inspection reports.

Strictly offline, self-contained UTF-8 HTML with embedded PNG crops as data URLs.
No server, external CSS/fonts/CDNs, or client-side JavaScript execution.
Explicitly highlights recognition candidates, sign consistency, and warnings
while preserving child formula provenance and geometry for wrapped relation groups.
"""

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from src.ingestion.formula_loader import load_pdf_formulas


CSS_STYLES = """
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --color-bg: #f8fafc;
  --color-surface: #ffffff;
  --color-text: #0f172a;
  --color-muted: #475569;
  --color-border: #cbd5e1;
  --color-danger: #dc2626;
  --color-danger-bg: #fef2f2;
  --color-danger-border: #f87171;
  --color-warning: #d97706;
  --color-warning-bg: #fffbeb;
  --color-warning-border: #fcd34d;
  --color-success: #16a34a;
  --color-success-bg: #f0fdf4;
  --color-success-border: #86efac;
  --color-info: #2563eb;
  --color-info-bg: #eff6ff;
  --color-info-border: #93c5fd;
  --color-neutral: #64748b;
  --color-neutral-bg: #f1f5f9;
  --color-neutral-border: #cbd5e1;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  background-color: var(--color-bg);
  color: var(--color-text);
  line-height: 1.5;
  padding: 1.5rem;
}

.container {
  max-width: 1280px;
  margin: 0 auto;
}

header.review-header {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

header.review-header h1 {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1rem;
  color: var(--color-text);
}

.disclaimer-banner {
  background-color: var(--color-danger-bg);
  border: 2px solid var(--color-danger-border);
  color: #7f1d1d;
  padding: 1rem 1.25rem;
  border-radius: 6px;
  margin-bottom: 1.25rem;
  font-size: 0.95rem;
}

.disclaimer-banner strong {
  display: block;
  font-size: 1rem;
  margin-bottom: 0.25rem;
  color: var(--color-danger);
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem 1.5rem;
  font-size: 0.9rem;
  background: var(--color-neutral-bg);
  padding: 1rem;
  border-radius: 6px;
  border: 1px solid var(--color-border);
}

.meta-item {
  word-break: break-all;
}

.meta-item .meta-label {
  font-weight: 600;
  color: var(--color-muted);
  display: block;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.meta-item .meta-value {
  font-family: var(--font-mono);
  font-size: 0.85rem;
}

.page-section {
  margin-bottom: 2.5rem;
}

.page-heading {
  font-size: 1.35rem;
  font-weight: 700;
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
  border-bottom: 2px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  margin-bottom: 1.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.card-formula {
  border-left: 6px solid var(--color-neutral);
}

.card-formula.has-mismatch {
  border-left: 6px solid var(--color-danger);
}

.card-formula.has-uncertain {
  border-left: 6px solid var(--color-warning);
}

.card-formula.has-no-conflict {
  border-left: 6px solid var(--color-success);
}

.card-group {
  border-left: 6px solid var(--color-info);
  background: #fcfdfe;
}

.card-header {
  padding: 0.85rem 1.25rem;
  background: #f8fafc;
  border-bottom: 1px solid var(--color-border);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
}

.card-title {
  font-size: 1.05rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.card-title code {
  font-family: var(--font-mono);
}

.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  align-items: center;
}

.badge {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  display: inline-block;
  white-space: nowrap;
}

.badge-neutral {
  background: var(--color-neutral-bg);
  color: var(--color-neutral);
  border: 1px solid var(--color-neutral-border);
}

.badge-danger {
  background: var(--color-danger-bg);
  color: var(--color-danger);
  border: 1px solid var(--color-danger-border);
}

.badge-warning {
  background: var(--color-warning-bg);
  color: var(--color-warning);
  border: 1px solid var(--color-warning-border);
}

.badge-success {
  background: var(--color-success-bg);
  color: var(--color-success);
  border: 1px solid var(--color-success-border);
}

.badge-info {
  background: var(--color-info-bg);
  color: var(--color-info);
  border: 1px solid var(--color-info-border);
}

.card-body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  padding: 1.25rem;
}

@media (max-width: 860px) {
  .card-body {
    grid-template-columns: 1fr;
    gap: 1.25rem;
  }
}

.column {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.field-label {
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--color-muted);
  text-transform: uppercase;
  letter-spacing: 0.03em;
  margin-bottom: 0.35rem;
}

.crop-wrapper {
  background: #ffffff;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 80px;
}

.source-crop {
  max-width: 100%;
  height: auto;
  image-rendering: -webkit-optimize-contrast;
  display: block;
}

.code-box {
  background: var(--color-neutral-bg);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 0.75rem;
  font-family: var(--font-mono);
  font-size: 0.85rem;
  white-space: pre-wrap;
  word-break: break-all;
  overflow-wrap: anywhere;
}

.code-box.latex-candidate {
  background: #fdfefe;
  border-color: #cbd5e1;
}

.code-box.empty-candidate {
  font-family: var(--font-sans);
  color: var(--color-muted);
  font-style: italic;
}

.warning-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.warning-alert {
  background: var(--color-danger-bg);
  border: 1px solid var(--color-danger-border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  font-size: 0.85rem;
  color: #7f1d1d;
}

.warning-alert strong {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--color-danger);
}

.warning-detail {
  margin-top: 0.35rem;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  word-break: break-all;
  white-space: pre-wrap;
}

.info-note {
  font-size: 0.8rem;
  color: var(--color-muted);
  line-height: 1.4;
}

.group-links {
  margin-top: 0.5rem;
  font-size: 0.85rem;
}

.group-links a, .card a {
  color: var(--color-info);
  text-decoration: none;
  font-weight: 600;
}

.group-links a:hover, .card a:hover {
  text-decoration: underline;
}

.empty-page-banner {
  background: var(--color-neutral-bg);
  border: 1px dashed var(--color-border);
  padding: 2rem;
  text-align: center;
  border-radius: 8px;
  color: var(--color-muted);
}

.empty-page-banner h3 {
  font-size: 1.15rem;
  color: var(--color-text);
  margin-bottom: 0.5rem;
}

.empty-page-banner p {
  font-size: 0.9rem;
  max-width: 600px;
  margin: 0 auto;
}
"""


def render_crop_data_url(page: pymupdf.Page, bbox: List[float], matrix_scale: float = 3.0) -> Optional[str]:
    """Render high-resolution PNG crop for a formula bbox and return data URL."""
    rect = pymupdf.Rect(bbox) & page.rect
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    pix = page.get_pixmap(matrix=pymupdf.Matrix(matrix_scale, matrix_scale), clip=rect)
    png_bytes = pix.tobytes('png')
    b64_data = base64.b64encode(png_bytes).decode('ascii')
    return f"data:image/png;base64,{b64_data}"


def validate_extraction_data(extraction_data: Any, pdf_sha: str, doc: pymupdf.Document) -> List[Dict[str, Any]]:
    """Validate extraction JSON strictly against PDF geometry, SHA256, and formula schema.

    Rejects missing/mismatched SHA, out-of-order pages, malformed bounding boxes,
    or out-of-bounds coordinates with actionable explicit errors.
    """
    if not isinstance(extraction_data, list):
        raise ValueError("Invalid extraction format: expected JSON array of page objects produced by src.ingestion.formula_loader")

    total_pages = len(doc)
    if len(extraction_data) != total_pages:
        raise ValueError(f"Page count mismatch: extraction has {len(extraction_data)} pages, PDF has {total_pages} pages")

    for p_idx, page in enumerate(extraction_data):
        if not isinstance(page, dict):
            raise ValueError(f"Invalid extraction schema: item at index {p_idx} is not a page object")

        pn = page.get('page_number')
        if not isinstance(pn, int) or pn != p_idx + 1:
            raise ValueError(f"Malformed or out-of-order page number: {pn} (expected {p_idx + 1})")

        page_sha = page.get('source_sha256')
        if not page_sha:
            raise ValueError(f"Missing source_sha256 on extraction page {pn}")
        if page_sha != pdf_sha:
            raise ValueError(f"Source SHA256 mismatch: PDF is {pdf_sha}, but extraction page {pn} has {page_sha}")

        formulas = page.get('formulas')
        if not isinstance(formulas, list):
            raise ValueError(f"Malformed 'formulas' field on page {pn}: expected list")

        doc_page = doc[pn - 1]
        p_rect = doc_page.rect
        formula_ids = set()

        for f_idx, f in enumerate(formulas):
            if not isinstance(f, dict):
                raise ValueError(f"Malformed formula item at index {f_idx} on page {pn}: expected object")

            fid = f.get('id')
            if not fid or not isinstance(fid, str):
                raise ValueError(f"Missing or invalid formula 'id' at index {f_idx} on page {pn}")
            if fid in formula_ids:
                raise ValueError(f"Duplicate formula ID '{fid}' on page {pn}")
            formula_ids.add(fid)

            f_pn = f.get('page_number')
            if f_pn != pn:
                raise ValueError(f"Formula '{fid}' specifies page_number {f_pn}, expected {pn}")

            bbox = f.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Malformed bbox for formula '{fid}': {bbox}")

            try:
                x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            except (ValueError, TypeError):
                raise ValueError(f"Non-numeric bbox coordinates for formula '{fid}': {bbox}")

            if x0 >= x1 or y0 >= y1:
                raise ValueError(f"Inverted or zero-area bbox for formula '{fid}': {bbox}")

            # Bounds check with a small 1.0pt margin for renderer rounding
            if x0 < -1.0 or y0 < -1.0 or x1 > p_rect.width + 1.0 or y1 > p_rect.height + 1.0:
                raise ValueError(
                    f"Formula '{fid}' bbox {bbox} is out of page bounds (0, 0, {p_rect.width:.1f}, {p_rect.height:.1f})"
                )

        groups = page.get('formula_groups', [])
        if not isinstance(groups, list):
            raise ValueError(f"Malformed 'formula_groups' field on page {pn}: expected list")

        for g_idx, g in enumerate(groups):
            if not isinstance(g, dict):
                raise ValueError(f"Malformed formula group at index {g_idx} on page {pn}")
            gid = g.get('id')
            if not gid or not isinstance(gid, str):
                raise ValueError(f"Missing or invalid group 'id' at index {g_idx} on page {pn}")
            child_ids = g.get('formula_ids')
            if not isinstance(child_ids, list) or not child_ids:
                raise ValueError(f"Group '{gid}' missing or empty 'formula_ids'")
            for cid in child_ids:
                if cid not in formula_ids:
                    raise ValueError(f"Group '{gid}' references non-existent formula ID '{cid}' on page {pn}")

    return extraction_data


def generate_formula_review_html(
    pdf_path: Path,
    extraction_path: Optional[Path] = None,
) -> str:
    """Generate self-contained UTF-8 HTML string reviewing PDF formulas and source provenance."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()

    with pymupdf.open(pdf_path) as doc:
        total_pages = len(doc)
        if extraction_path is not None:
            extraction_file = Path(extraction_path)
            if not extraction_file.exists() or not extraction_file.is_file():
                raise FileNotFoundError(f"Extraction file not found: {extraction_file}")
            try:
                raw_json = json.loads(extraction_file.read_text(encoding='utf-8'))
            except Exception as exc:
                raise ValueError(f"Failed to parse extraction JSON from {extraction_file}: {exc}") from exc

            pages = validate_extraction_data(raw_json, pdf_sha, doc)
            extraction_mode = f"Saved formula_loader export ({html.escape(extraction_file.name)})"
        else:
            pages = load_pdf_formulas(pdf_path, recognizer=None)
            extraction_mode = "Native-only heuristic extraction (no OCR requested)"

        # Collect metrics
        total_formulas = sum(len(p.get('formulas', [])) for p in pages)
        total_groups = sum(len(p.get('formula_groups', [])) for p in pages)

        # Build map of formula_id -> list of group dicts
        groups_by_formula = {}
        for p in pages:
            for g in p.get('formula_groups', []):
                for fid in g.get('formula_ids', []):
                    groups_by_formula.setdefault(fid, []).append(g)

        # Pre-render PNG crops
        crop_data_urls = {}
        for p in pages:
            pn = p['page_number']
            doc_page = doc[pn - 1]
            for f in p.get('formulas', []):
                fid = f['id']
                crop_url = render_crop_data_url(doc_page, f['bbox'])
                if crop_url:
                    crop_data_urls[fid] = crop_url

    # Assemble HTML pieces
    html_parts = []
    html_parts.append("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n")
    html_parts.append("<meta charset=\"utf-8\">\n")
    html_parts.append("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
    html_parts.append(f"<title>Formula Review & Provenance - {html.escape(pdf_path.name)}</title>\n")
    html_parts.append(f"<style>{CSS_STYLES}</style>\n")
    html_parts.append("</head>\n<body>\n<div class=\"container\">\n")

    # Header & Prominent Disclaimer
    html_parts.append("<header class=\"review-header\">\n")
    html_parts.append(f"  <h1>Formula Review &amp; Provenance Inspection</h1>\n")
    html_parts.append("  <div class=\"disclaimer-banner\">\n")
    html_parts.append("    <strong>[UNVERIFIED NATIVE TEXT: math detection is incomplete; answer_eligible: false; source review required for math answers]</strong>\n")
    html_parts.append("    <p>This report displays native digital PDF source regions alongside optional OCR candidates and operator consistency diagnostics. Heuristic matching or absence of conflicts does NOT certify mathematical correctness. Unresolved, failed, or unmeasured statuses are explicitly separated from verified truth.</p>\n")
    html_parts.append("  </div>\n")

    # Metadata Grid
    html_parts.append("  <div class=\"meta-grid\">\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">File Name</span><span class=\"meta-value\">{html.escape(pdf_path.name)}</span></div>\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">Source SHA-256</span><span class=\"meta-value\">{html.escape(pdf_sha)}</span></div>\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">Total Pages</span><span class=\"meta-value\">{total_pages}</span></div>\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">Extraction Mode</span><span class=\"meta-value\">{extraction_mode}</span></div>\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">Detected Formulas</span><span class=\"meta-value\">{total_formulas}</span></div>\n")
    html_parts.append(f"    <div class=\"meta-item\"><span class=\"meta-label\">Wrapped Groups</span><span class=\"meta-value\">{total_groups}</span></div>\n")
    html_parts.append("  </div>\n")
    html_parts.append("</header>\n")

    # Content Pages
    if total_formulas == 0:
        html_parts.append("<section class=\"page-section\">\n")
        html_parts.append("  <div class=\"empty-page-banner\">\n")
        html_parts.append("    <h3>No Native Formulas Detected</h3>\n")
        html_parts.append("    <p><strong>Detection is incomplete:</strong> PyMuPDF digital text and vector line analysis found no candidate math formulas. Scanned pages, pure images, or non-standard fonts require visual source review; absence of native candidates does NOT claim that no formulas exist in this document.</p>\n")
        html_parts.append("  </div>\n")
        html_parts.append("</section>\n")
    else:
        for p in pages:
            pn = p['page_number']
            page_formulas = p.get('formulas', [])
            page_groups = p.get('formula_groups', [])

            html_parts.append(f"<section class=\"page-section\" id=\"page-{pn}\">\n")
            html_parts.append(f"  <h2 class=\"page-heading\">Page {pn} of {total_pages} ")
            html_parts.append(f"<span class=\"badge badge-neutral\">{len(page_formulas)} formula(s), {len(page_groups)} group(s)</span></h2>\n")

            # Check if page has unresolved reason (e.g. image/scanned)
            if p.get('unresolved_reason'):
                html_parts.append(f"  <div class=\"warning-alert\" style=\"margin-bottom: 1.5rem;\"><strong>[PAGE REVIEW REASON]</strong> {html.escape(p['unresolved_reason'])}</div>\n")

            if not page_formulas:
                html_parts.append("  <div class=\"empty-page-banner\">\n")
                html_parts.append("    <h3>No Native Formulas on this Page</h3>\n")
                html_parts.append("    <p>Detection is incomplete: no native math regions detected on page. Scanned or image formulas require manual source review.</p>\n")
                html_parts.append("  </div>\n")
                html_parts.append("</section>\n")
                continue

            # Render Formula Cards
            for f in page_formulas:
                fid = f['id']
                raw_latex = f.get('raw_latex') or f.get('latex')
                has_latex = bool(raw_latex and raw_latex.strip())
                sign_consistency = f.get('sign_consistency', {})
                sign_status = sign_consistency.get('status', 'not_checked')
                rec_status = f.get('recognition_status', 'not_requested')
                warnings = f.get('warnings', [])

                card_class = "card card-formula"
                if sign_status == 'mismatch' or any(w.get('code') == 'source_minus_candidate_conflict' for w in warnings):
                    card_class += " has-mismatch"
                elif sign_status == 'uncertain' or warnings:
                    card_class += " has-uncertain"
                elif sign_status == 'no_conflict_detected':
                    card_class += " has-no-conflict"

                html_parts.append(f"  <div class=\"{card_class}\" id=\"card-{html.escape(fid)}\">\n")
                html_parts.append("    <div class=\"card-header\">\n")
                html_parts.append(f"      <div class=\"card-title\">Formula <code>{html.escape(fid)}</code></div>\n")
                html_parts.append("      <div class=\"badge-row\">\n")

                # State badge
                state = f.get('state', 'unresolved')
                html_parts.append(f"        <span class=\"badge badge-neutral\">{html.escape(state)}</span>\n")
                html_parts.append(f"        <span class=\"badge badge-danger\">answer_eligible: false</span>\n")

                # Recognition status badge
                if rec_status == 'not_requested':
                    html_parts.append("        <span class=\"badge badge-neutral\">No OCR candidate (native only)</span>\n")
                elif rec_status == 'returned':
                    html_parts.append("        <span class=\"badge badge-info\">OCR candidate returned (unverified)</span>\n")
                elif rec_status == 'failed':
                    html_parts.append("        <span class=\"badge badge-danger\">OCR recognition failed</span>\n")
                else:
                    html_parts.append(f"        <span class=\"badge badge-neutral\">{html.escape(rec_status)}</span>\n")

                # Sign status badge
                if sign_status == 'mismatch':
                    html_parts.append("        <span class=\"badge badge-danger\">Sign Mismatch</span>\n")
                elif sign_status == 'uncertain':
                    html_parts.append("        <span class=\"badge badge-warning\">Sign Uncertain</span>\n")
                elif sign_status == 'no_conflict_detected':
                    html_parts.append("        <span class=\"badge badge-success\">No Conflict Detected (Scope-Only)</span>\n")
                elif sign_status == 'not_checked':
                    html_parts.append("        <span class=\"badge badge-neutral\">Sign Not Checked</span>\n")
                elif sign_status == 'unmeasured':
                    html_parts.append("        <span class=\"badge badge-neutral\">Sign Unmeasured</span>\n")
                elif sign_status == 'unsupported':
                    html_parts.append("        <span class=\"badge badge-warning\">Sign Unsupported Orientation</span>\n")

                html_parts.append("      </div>\n")
                html_parts.append("    </div>\n")

                # Card Body: Left Column (Native evidence) and Right Column (OCR & Diagnostics)
                html_parts.append("    <div class=\"card-body\">\n")
                html_parts.append("      <div class=\"column\">\n")
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Native Source Crop (PyMuPDF 3x Render)</div>\n")
                html_parts.append("          <div class=\"crop-wrapper\">\n")

                crop_url = crop_data_urls.get(fid)
                if crop_url:
                    html_parts.append(f"            <img src=\"{crop_url}\" alt=\"Source crop for {html.escape(fid)}\" class=\"source-crop\" />\n")
                else:
                    html_parts.append("            <span class=\"info-note\">Crop rendering unavailable</span>\n")

                html_parts.append("          </div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Original Extracted Native Text</div>\n")
                orig_text = f.get('original_text', '')
                html_parts.append(f"          <div class=\"code-box\">{html.escape(orig_text)}</div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Source Geometry &amp; Metadata</div>\n")
                bbox_str = json.dumps(f.get('bbox', []))
                html_parts.append(f"          <div class=\"info-note\">\n")
                html_parts.append(f"            <strong>BBox:</strong> <code>{html.escape(bbox_str)}</code><br>\n")
                html_parts.append(f"            <strong>Rules:</strong> <code>{html.escape(str(f.get('native_rules', [])))}</code><br>\n")
                html_parts.append(f"            <strong>Script Evidence:</strong> <code>{html.escape(str(f.get('native_script_evidence', False)))}</code><br>\n")
                html_parts.append(f"            <strong>Reason:</strong> <code>{html.escape(str(f.get('reason', '')))}</code>\n")
                html_parts.append("          </div>\n")
                html_parts.append("        </div>\n")

                # Group link if formula is part of a wrapped relation
                parent_groups = groups_by_formula.get(fid, [])
                if parent_groups:
                    html_parts.append("        <div class=\"group-links\">\n")
                    html_parts.append("          <div class=\"field-label\">Wrapped Relation Group</div>\n")
                    for pg in parent_groups:
                        pg_id = pg['id']
                        html_parts.append(f"          <div>Continuation link: <a href=\"#group-{html.escape(pg_id)}\">Group <code>{html.escape(pg_id)}</code></a> ({html.escape(pg.get('kind', 'group'))})</div>\n")
                    html_parts.append("        </div>\n")

                html_parts.append("      </div>\n")

                # Right Column: OCR Candidate & Warnings
                html_parts.append("      <div class=\"column\">\n")
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Raw Candidate LaTeX</div>\n")
                if has_latex:
                    html_parts.append(f"          <div class=\"code-box latex-candidate\"><code>{html.escape(raw_latex)}</code></div>\n")
                else:
                    html_parts.append("          <div class=\"code-box empty-candidate\">[No OCR candidate requested or returned; native-only extraction]</div>\n")
                html_parts.append("        </div>\n")

                # Sign Consistency Details
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Sign Consistency Diagnostic</div>\n")
                html_parts.append("          <div class=\"info-note\">\n")
                html_parts.append(f"            <strong>Status:</strong> <code>{html.escape(sign_status)}</code><br>\n")
                sign_reason = sign_consistency.get('reason', '')
                if sign_reason:
                    html_parts.append(f"            <strong>Reason:</strong> <code>{html.escape(sign_reason)}</code><br>\n")
                if sign_status == 'no_conflict_detected':
                    html_parts.append("            <em>Scope note: Supported minus scopes match candidate; this does NOT verify formula correctness.</em><br>\n")

                ops = sign_consistency.get('source_operators', [])
                if ops:
                    html_parts.append(f"            <strong>Detected Minus Operators ({len(ops)}):</strong>\n")
                    html_parts.append(f"            <pre class=\"code-box\" style=\"margin-top:0.35rem; font-size:0.75rem;\">{html.escape(json.dumps(ops, indent=2))}</pre>\n")
                html_parts.append("          </div>\n")
                html_parts.append("        </div>\n")

                # Warnings
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Warnings &amp; Conflicts</div>\n")
                if warnings:
                    html_parts.append("          <div class=\"warning-container\">\n")
                    for w in warnings:
                        w_code = w.get('code', 'warning')
                        w_msg = w.get('message', '')
                        w_reason = w.get('reason', '')
                        w_cand = w.get('raw_candidate', '')
                        html_parts.append("            <div class=\"warning-alert\">\n")
                        html_parts.append(f"              <strong>[WARNING: {html.escape(w_code)}]</strong>\n")
                        if w_msg:
                            html_parts.append(f"              <div>{html.escape(w_msg)}</div>\n")
                        if w_reason:
                            html_parts.append(f"              <div class=\"warning-detail\">reason: {html.escape(w_reason)}</div>\n")
                        if w_cand:
                            html_parts.append(f"              <div class=\"warning-detail\">raw_candidate: <code>{html.escape(str(w_cand))}</code></div>\n")
                        html_parts.append("            </div>\n")
                    html_parts.append("          </div>\n")
                else:
                    html_parts.append("          <div class=\"info-note\">No recognition or consistency warnings recorded.</div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("      </div>\n")
                html_parts.append("    </div>\n")
                html_parts.append("  </div>\n")

            # Render Group Cards (if any)
            for g in page_groups:
                gid = g['id']
                child_ids = g.get('formula_ids', [])
                g_kind = g.get('kind', 'wrapped_relation')
                g_linkage = g.get('linkage', 'heuristic_continuation')
                g_orig = g.get('original_text', '')
                g_latex = g.get('latex')
                g_sign = g.get('sign_consistency', {})
                g_warnings = g.get('warnings', [])

                html_parts.append(f"  <div class=\"card card-group\" id=\"group-{html.escape(gid)}\">\n")
                html_parts.append("    <div class=\"card-header\">\n")
                html_parts.append(f"      <div class=\"card-title\">Wrapped Relation Group <code>{html.escape(gid)}</code></div>\n")
                html_parts.append("      <div class=\"badge-row\">\n")
                html_parts.append(f"        <span class=\"badge badge-info\">{html.escape(g_kind)}</span>\n")
                html_parts.append("        <span class=\"badge badge-danger\">answer_eligible: false</span>\n")
                html_parts.append(f"        <span class=\"badge badge-neutral\">{html.escape(g.get('state', 'unresolved'))}</span>\n")
                html_parts.append("      </div>\n")
                html_parts.append("    </div>\n")

                html_parts.append("    <div class=\"card-body\">\n")
                html_parts.append("      <div class=\"column\">\n")
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Combined Extracted Text</div>\n")
                html_parts.append(f"          <div class=\"code-box\">{html.escape(g_orig)}</div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Child Formulas &amp; Preserved Geometry</div>\n")
                html_parts.append("          <div class=\"info-note\">\n")
                html_parts.append("            <p>Composite crops are not invented; individual bounding boxes and reading order are strictly preserved:</p>\n")
                html_parts.append("            <ul style=\"margin-left: 1.25rem; margin-top: 0.35rem;\">\n")
                for cid in child_ids:
                    child_f = next((f for f in page_formulas if f['id'] == cid), None)
                    child_bbox = child_f['bbox'] if child_f else 'unknown'
                    html_parts.append(f"              <li><a href=\"#card-{html.escape(cid)}\">Formula <code>{html.escape(cid)}</code></a> - bbox: <code>{html.escape(json.dumps(child_bbox))}</code></li>\n")
                html_parts.append("            </ul>\n")
                html_parts.append(f"            <div style=\"margin-top:0.35rem;\"><strong>Linkage:</strong> <code>{html.escape(g_linkage)}</code></div>\n")
                html_parts.append("          </div>\n")
                html_parts.append("        </div>\n")
                html_parts.append("      </div>\n")

                html_parts.append("      <div class=\"column\">\n")
                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Combined Candidate LaTeX</div>\n")
                if g_latex:
                    html_parts.append(f"          <div class=\"code-box latex-candidate\"><code>{html.escape(g_latex)}</code></div>\n")
                else:
                    html_parts.append("          <div class=\"code-box empty-candidate\">[No combined LaTeX candidate available; child recognition incomplete]</div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("        <div>\n")
                html_parts.append("          <div class=\"field-label\">Group Diagnostics &amp; Warnings</div>\n")
                html_parts.append("          <div class=\"info-note\">\n")
                html_parts.append(f"            <strong>Group Sign Status:</strong> <code>{html.escape(g_sign.get('status', 'not_checked'))}</code><br>\n")
                if g_warnings:
                    html_parts.append("            <div class=\"warning-container\" style=\"margin-top:0.5rem;\">\n")
                    for w in g_warnings:
                        html_parts.append(f"              <div class=\"warning-alert\"><strong>[WARNING: {html.escape(w.get('code', ''))}]</strong> {html.escape(w.get('message', ''))}</div>\n")
                    html_parts.append("            </div>\n")
                else:
                    html_parts.append("            <div>No group warnings recorded.</div>\n")
                html_parts.append("          </div>\n")
                html_parts.append("        </div>\n")

                html_parts.append("      </div>\n")
                html_parts.append("    </div>\n")
                html_parts.append("  </div>\n")

            html_parts.append("</section>\n")

    html_parts.append("</div>\n</body>\n</html>\n")
    return "".join(html_parts)


def export_formula_review_html(
    pdf_path: Path,
    output_path: Path,
    extraction_path: Optional[Path] = None,
) -> Path:
    """Export review HTML to output_path while strictly guarding against input overwrites."""
    pdf_path = Path(pdf_path).resolve()
    output_path = Path(output_path).resolve()

    if output_path == pdf_path:
        raise ValueError(f"Output path cannot alias or overwrite input PDF: {output_path}")

    if extraction_path is not None:
        extraction_path = Path(extraction_path).resolve()
        if output_path == extraction_path:
            raise ValueError(f"Output path cannot alias or overwrite extraction JSON: {output_path}")

    html_content = generate_formula_review_html(pdf_path, extraction_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    return output_path


def main(argv=None) -> int:
    """CLI entrypoint for formula review HTML generation."""
    parser = argparse.ArgumentParser(
        description="Generate self-contained static HTML review report for extracted PDF formulas."
    )
    parser.add_argument('pdf', type=Path, help="Input PDF document path")
    parser.add_argument('-o', '--output', type=Path, required=True, help="Destination HTML review file path")
    parser.add_argument('-e', '--extraction', type=Path, default=None, help="Optional saved formula_loader JSON extraction file")

    args = parser.parse_args(argv)

    try:
        exported = export_formula_review_html(args.pdf, args.output, args.extraction)
        print(f"Exported formula review HTML to: {exported}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"Formula review export failed: {exc}\n")


if __name__ == '__main__':
    raise SystemExit(main())
