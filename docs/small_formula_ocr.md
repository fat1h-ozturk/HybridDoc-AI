# Small negative fractional exponent regression

The public SYN-WRAP fixture now typesets a clearly raised `2^{-2/3}`. OCR is
**still incorrect** on this source: CodeFormulaV2 returned
`2 ^ { \cdot \frac { 2 } { 3 } }`, replacing the minus with multiplication.
This change corrects the source and preserves its wrapped group; it does not
repair model output or certify math answers.

## Source diagnosis and correction

The original source put the base and denominator on the same baseline (y=296)
and drew a 1pt fraction rule at y=293 through denominator ink. Its crop was
complete, but its typography was ambiguous. Earlier claims that this was solely
a model error on unambiguous pixels were too strong.

The corrected source keeps the numerator baseline at y=291, moves the minus to
y=295 and denominator to y=299, uses a 0.4pt rule at y=293, and moves the base
and following prose to y=303. The negative sign is raised and aligned with the
fraction rule. White gaps separate the rule from numerator and denominator.
The intended gold remains `2 ^ { - \frac { 2 } { 3 } }`.

Moving the base creates a new PyMuPDF raw block. Native grouping previously
rejected the valid continuation solely because its block identifier differed.
The narrow exception requires adjacent source blocks, consecutive owned spans,
final/first block lines, nonoverlapping block heights, matching left margins,
a return to that margin, and an incomplete fragment positioned away from the
left edge of the preceding block. Existing line spacing, omitted prose,
column, cell, complete equation, and one-to-one guards remain active.

Children retain their source bboxes/hash, raw candidates, unresolved states,
and `answer_eligible=False`. The group stays heuristic and unverified. Exact
surrounding prose order is tested. Renderer and recognizer settings are unchanged.
Evaluator CLI diagnostics go to stderr so the JSON stdout/file contract remains
valid when PyMuPDF prints its advisory.

## Measured evidence (separate budgets)

Model: `docling-project/CodeFormulaV2`, revision
`ecedbe111d15c2dc60bfd4a823cbe80127b58af4`; CPU float32, batch one, two
threads, offline cached weights. Docling 2.137.0, Transformers 5.19.0,
PyTorch 2.14.1+cpu, PyMuPDF 1.28.2. Formula prompt, temperature zero,
2048 maximum new tokens. No retries, downloads, API fallbacks, or model changes.

| Source / control | Rendering | Full raw returned LaTeX | Intended gold | Seconds |
|---|---|---|---|---:|
| Original negative | 216 DPI | `2 \cdot \frac { 2 } { 3 }` | FAIL | 55.0423 |
| Original negative | 600 DPI | `2 - \frac { 2 } { 3 }` | FAIL | 44.2550 |
| Original positive fraction | 216 DPI | `f ( x ) = \frac { x ^ { 2 } + 1 } { 2 y - 3 }` | PASS | 34.7971 |
| Original positive fraction | 600 DPI | `f ( x ) = \frac { x ^ { 2 } + 1 } { 2 y - 3 }` | PASS | 34.8086 |
| Corrected negative | Matrix(3,3) | `2 ^ { \cdot \frac { 2 } { 3 } }` | FAIL | 54.5548 |
| Corrected positive fraction | Matrix(3,3) | `f ( x ) = \frac { x ^ { 2 } + 1 } { 2 y - 3 }` | PASS | 34.4195 |

Positive control gold is `f(x) = \frac{x^2 + 1}{2y - 3}`. It distinguishes
positive superscript attachment and fraction scope from the negative case.

The original four-call density experiment took 168.9441s including initialization
under a 360s overall / 120s per-call cap. The separately authorized corrected
source verification took 89.1848s including initialization under a new
240s overall / 120s per-call cap, exactly two calls. It ran only after source
geometry, grouping tests, and PyMuPDF/Poppler visual inspection passed.
The first budget was exhausted and never reset. This is two measured controls,
not a new global benchmark or an end-to-end PDF accuracy result.

The processor resizes the longest edge to 2048, then uses 512px image tiles;
raising PDF DPI changes raster detail rather than simply increasing final model
scale. The density experiment did not improve the negative case. No production
input setting is justified by these measurements.

## Reusable public inputs and deterministic replay

[Original PDF](../eval_data/small_formula_ocr_original.pdf) preserves the exact
invalid source once; [corrected PDF](../eval_data/small_formula_ocr_corrected.pdf)
preserves the exact source used for verification. PDF IDs may change when the
fixture generator runs again; these archived files retain the measured hashes.
[Original evidence](../eval_data/small_formula_ocr_density.json) and
[corrected evidence](../eval_data/small_formula_ocr_corrected.json) include full
raw output, complete gold, source/crop hashes, source bboxes, dimensions,
render parameters, model identity, and timings. DPI-tagged PNGs and Matrix(3,3)
PNGs have different metadata; Matrix(3,3) has a 216 DPI geometric scale even
though its pixmap DPI tags default to 96.

Run the portable regression with:

```sh
python -m pytest tests/test_small_formula_ocr.py
```

It replays recorded failures and **expects FAIL** while checking the positive
control; CI makes no real model calls. It also checks corrected source pixels,
complete script crop geometry, prose exclusion, cross-block grouping, child
provenance and failure states, and counterexamples for distinct blocks,
columns, cells, omitted spans, complete equations, missing geometry and spacing.

For future explicitly budgeted model work, render the two archived corrected
inputs from their native detector bboxes using `Matrix(3,3)` and feed each once
to the existing `LocalFormulaRecognizer` with the recorded cached revision.
Select the RHS of the wrapped group and the separate numerator/denominator
fraction control. Preserve the returned strings, hashes, elapsed time and full
negative gold; do not run the whole benchmark to reproduce these two cases.

Remaining limitation: the negative sign is visibly present in the corrected
crop, but the measured model replaces it with `\cdot`. The comparator rejects
that meaning change. No OCR success or production answer eligibility is claimed.
