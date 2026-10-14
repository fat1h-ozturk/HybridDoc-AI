"""Bounded native-minus / candidate diagnostic. Never repairs or verifies math."""
import re
from collections import Counter

_MINUS = '-−'
_SPACING = {r'\,', r'\;', r'\:', r'\!', '\\ ', r'\quad', r'\qquad', r'\/', r'\left', r'\right'}
_TEXT = {r'\text', r'\textrm', r'\textnormal', r'\textbf', r'\textit', r'\mbox', r'\operatorname'}


def _candidate_scopes(latex):
    """Walk only braces, single-atom scripts and fractions; reject other syntax.

    This is an operator inventory, not a general LaTeX parser. Scope paths plus
    immediate operands prevent another term's minus from satisfying a sign.
    """
    # shortcut: single-atom scripts/one fraction only; extend with reliable native scope evidence.
    tokens = [t for t in re.findall(r'\\[A-Za-z]+|\\.|[^\s]', latex) if t not in _SPACING]
    scopes = {}
    index = 0

    def walk(path, argument=False):
        nonlocal index
        grouped = argument and index < len(tokens) and tokens[index] == '{'
        if grouped:
            index += 1
        values = scopes.setdefault(path, [])
        consumed = 0
        while index < len(tokens):
            t = tokens[index]
            if t == '}':
                if not grouped:
                    break
                index += 1
                return
            if argument and not grouped and consumed:
                return
            index += 1
            consumed += 1
            if t == '{':
                index -= 1
                walk(path, True)
            elif t in ('^', '_'):
                if not values or not values[-1].isalnum() or len(values[-1]) != 1:
                    raise ValueError('script base is not a supported single atom')
                child = path + (('superscript' if t == '^' else 'subscript', values[-1]),)
                if child in scopes:
                    raise ValueError('repeated candidate script scope')
                walk(child, True)
            elif t in (r'\frac', r'\dfrac', r'\tfrac'):
                values.append('fraction')
                if path + (('numerator', ''),) in scopes:
                    raise ValueError('repeated candidate fraction scope')
                walk(path + (('numerator', ''),), True)
                walk(path + (('denominator', ''),), True)
            elif t in _TEXT:
                if index >= len(tokens) or tokens[index] != '{':
                    raise ValueError('text command without a group')
                depth = 1
                index += 1
                while index < len(tokens) and depth:
                    depth += (tokens[index] == '{') - (tokens[index] == '}')
                    index += 1
                if depth:
                    raise ValueError('unclosed text group')
                values.append('annotation')
            elif t in (r'\cdot', r'\times', r'\ast', '·', '×', '*'):
                values.append('multiply')
            elif t in _MINUS:
                values.append('-')
            elif t.startswith('\\') or t in ('%', '#', '&', '$', '~'):
                raise ValueError('unsupported TeX syntax')
            else:
                values.append(t)
        if grouped or (argument and not consumed):
            raise ValueError('missing or unclosed argument')

    try:
        walk(())
        if index != len(tokens):
            raise ValueError('unmatched closing brace')
    except (ValueError, RecursionError) as exc:
        return {}, str(exc)
    return scopes, None


def _source_operators(formula, spans):
    glyphs = []
    for sid in formula['span_ids']:
        s = spans[sid]
        if tuple(s.get('direction', ())) != (1.0, 0.0):
            return [], False  # Unsupported native orientation stays unmeasured.
        # Prose/annotation words and wrapping hyphens are outside this release.
        if re.search(r'[A-Za-z]{2,}', s['text']):
            continue
        for offset, c in enumerate(s.get('chars', [])):
            if c['c'].strip() and not c.get('synthetic'):
                glyphs.append({**c, 'span_id': sid, 'char_offset': offset,
                               'size': s['size'], 'span_text': s['text'], 'math': s['math'],
                               'native_script': s['script']})
    rules = formula['native_rules']
    ambiguous_attachment = False

    def script(g):
        nonlocal ambiguous_attachment
        x, y = g['origin']
        bases = [b for b in glyphs if b['c'].isalnum() and b['size'] > g['size'] / .9
                 and -.25 * b['size'] <= x - b['bbox'][2] <= 1.5 * b['size']
                 and .25 * b['size'] <= abs(y - b['origin'][1]) <= 1.5 * b['size']]
        if not bases:
            return (), None
        bases.sort(key=lambda b: x - b['bbox'][2])
        b = bases[0]
        if len(bases) > 1 and abs(b['bbox'][2] - bases[1]['bbox'][2]) < .1 * b['size']:
            ambiguous_attachment = True
        path = (('superscript' if y < b['origin'][1] else 'subscript', b['c']),)
        return path, (b['span_id'], b['char_offset'])

    scopes, anchors = {}, {}
    for g in glyphs:
        path, anchor = script(g)
        if anchor:
            anchors.setdefault(path, set()).add(anchor)
        inside = [r for r in rules if r[0] <= g['origin'][0] < r[2]]
        if inside:
            r = inside[0]
            part = (('numerator' if g['origin'][1] < r[1] else 'denominator', ''),)
            base = next((b for b in glyphs if (b['span_id'], b['char_offset']) == anchor), None)
            if base and r[0] <= base['origin'][0] < r[2]:
                # A numerator glyph is not a script base for the denominator.
                path = part + (path if (base['origin'][1] < r[1]) == (g['origin'][1] < r[1]) else ())
            else:
                path += part
        g['scope'] = path
        scopes.setdefault(path, []).append((g['origin'][0], g['c'], g))
    # The fraction is an operand in its parent scope, never a minus rule line.
    ambiguous = False
    for r in rules:
        members = [g for g in glyphs if r[0] <= g['origin'][0] < r[2]]
        if members:
            largest = max(g['size'] for g in members)
            paths = {script(g)[0] for g in members if g['size'] == largest}
            if len(paths) == 1:
                scopes.setdefault(paths.pop(), []).append((r[0], 'fraction', None))
            else:
                ambiguous = True
    operators = []
    for path, values in scopes.items():
        values.sort(key=lambda v: v[0])
        for i, (_, char, g) in enumerate(values):
            if char not in _MINUS or not (g['math'] or path):
                continue
            operators.append({'operator': char, 'span_id': g['span_id'],
                              'char_offset': g['char_offset'], 'span_text': g['span_text'],
                              'bbox': list(g['bbox']), 'origin': list(g['origin']),
                              'scope': [list(part) for part in path],
                              'left_operand': values[i-1][1] if i else None,
                              'right_operand': values[i+1][1] if i+1 < len(values) else None})
            if g['native_script'] and not path:
                ambiguous = True
    ambiguous = (ambiguous or ambiguous_attachment or len(rules) > 1 or any(len(a) > 1 for a in anchors.values())
                 or any(o['right_operand'] is None for o in operators)
                 or any(n > 1 for n in Counter(tuple(map(tuple, o['scope'])) for o in operators).values()))
    return operators, ambiguous


def check_formula_signs(formula, spans):
    """Add an explicitly limited check outcome and actionable source warnings."""
    operators, ambiguous = _source_operators(formula, spans)
    result = {'status': 'not_checked', 'reason': 'recognition_not_returned',
              'source_operators': operators}
    formula['sign_consistency'] = result
    formula['warnings'] = []
    if formula['recognition_status'] != 'returned' or not formula['latex']:
        return
    if not operators:
        result['reason'] = 'no_supported_native_minus_evidence'
        return
    scopes, error = _candidate_scopes(formula['latex'])
    if error or ambiguous:
        result.update(status='uncertain', reason=error or 'native_operator_scope_is_ambiguous')
    else:
        unmatched = []
        candidate_association_uncertain = False
        for op in operators:
            path = tuple(map(tuple, op['scope']))
            values = scopes.get(path, [])
            matches = [i for i, v in enumerate(values) if v == '-'
                       and (values[i-1] if i else None) == op['left_operand']
                       and (values[i+1] if i+1 < len(values) else None) == op['right_operand']]
            if len(matches) != 1:
                unmatched.append(op)
                candidate_association_uncertain |= '-' in values
        if not unmatched:
            result.update(status='no_conflict_detected', reason='supported_minus_scopes_only_not_formula_verification')
            return
        result.update(status='uncertain' if candidate_association_uncertain else 'mismatch',
                      reason='candidate_minus_term_association_is_ambiguous' if candidate_association_uncertain
                      else 'source_minus_missing_or_changed_in_supported_candidate_scope')
    warning = {'code': 'source_minus_candidate_conflict' if result['status'] == 'mismatch'
               else 'source_minus_scope_uncertain',
               'reason': result['reason'], 'formula_id': formula['id'],
               'page_number': formula['page_number'], 'bbox': formula['bbox'],
               'source_sha256': formula['source_sha256'], 'source_operators': operators,
               'raw_candidate': formula.get('raw_latex', formula['latex']),
               'message': 'Review source minus operators against raw candidate; heuristic sign check only, not formula verification.'}
    formula['warnings'].append(warning)


def summarize_sign_checks(formulas):
    statuses = [f['sign_consistency']['status'] for f in formulas]
    status = next((s for s in ('mismatch', 'uncertain') if s in statuses),
                  'no_conflict_detected' if statuses and all(s == 'no_conflict_detected' for s in statuses)
                  else 'not_checked')
    return {'status': status, 'reason': 'child_minus_checks_only_not_formula_verification',
            'formula_statuses': {f['id']: f['sign_consistency']['status'] for f in formulas}}


def sign_warning_text(record):
    import json
    return ''.join('\n[WARNING: ' + json.dumps(w, ensure_ascii=False) + ']'
                   for w in record.get('warnings', []))
