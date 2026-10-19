"""Conservative finite typed interventions; raw text never passes through QA EM.

Unsupported or conflicting answers are explicitly excluded, never coerced to an
institution. This is a declared operational protocol, not a universal NER system.
"""
import calendar
import hashlib
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation

WORDS = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'
SMALL = {w: i for i, w in enumerate(WORDS.split())}
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
MONTHS = {v.lower(): i for i, v in enumerate(calendar.month_name) if v}
MONTHS.update({v.lower(): i for i, v in enumerate(calendar.month_abbr) if v})
MONTH_PATTERN = '(?:' + '|'.join(sorted(MONTHS, key=len, reverse=True)) + ')'
POOLS = {
    'person': ['Ada Lovelace', 'Charles Darwin', 'Marie Curie', 'Isaac Newton', 'Katherine Johnson'],
    'country': ['Brazil', 'Norway', 'Kenya', 'Portugal', 'Canada', 'Japan'],
    'continent': ['Asia', 'Africa', 'Europe', 'Australia', 'Antarctica'],
    'us_region': ['New Mexico', 'Arizona', 'Colorado', 'Utah', 'Nevada'],
    'color': ['white', 'black', 'red', 'blue', 'green', 'yellow', 'orange', 'purple'],
    'organ': ['liver', 'kidney', 'pancreas', 'spleen', 'stomach', 'heart'],
    'meningeal_layer': ['pia mater', 'arachnoid mater', 'dura mater'],
    'body_region': ['trunk', 'forearm', 'neck', 'abdomen'],
    'tissue': ['muscles', 'tendons', 'ligaments', 'fascia'],
    'brain_structure': ['hypothalamus', 'hippocampus', 'cerebellum', 'brainstem'],
    'gland': ['pituitary gland', 'thyroid gland', 'adrenal gland', 'pineal gland'],
    'lobe': ['frontal lobe', 'temporal lobe', 'parietal lobe', 'occipital lobe'],
}
COUNTRIES = set(('Brazil Norway Kenya Portugal Canada Japan Nepal Ukraine China India Russia Germany Spain France '
    'Italy Greece Australia Poland Sweden Finland Denmark Belgium Austria Switzerland Mexico Argentina Chile Peru '
    'Colombia Egypt Nigeria Ethiopia Sudan Uganda Ghana Angola Zambia Zimbabwe Pakistan Bangladesh Thailand Vietnam '
    'Cambodia Laos Myanmar Malaysia Indonesia Singapore Philippines Korea Taiwan Mongolia Iran Iraq Israel Jordan '
    'Lebanon Syria Turkey Yemen Oman Qatar Kuwait Bahrain Afghanistan Iceland Ireland Luxembourg Hungary Romania '
    'Bulgaria Croatia Serbia Slovakia Slovenia Estonia Latvia Lithuania Belarus Moldova Georgia Armenia Azerbaijan '
    'Cyprus Malta Albania Andorra Monaco Liechtenstein SanMarino VaticanCity').lower().split())
COUNTRY_NAMES = {'united states', 'united states of america', 'united kingdom', 'great britain',
    'new zealand', 'south africa', 'south korea', 'north korea', 'costa rica', 'sri lanka', 'vatican city'}
STATES = {'new mexico', 'arizona', 'colorado', 'utah', 'nevada', 'california', 'texas', 'florida', 'alaska', 'hawaii'}
_NUM = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'


def compact(text):
    return re.sub(r'\s+', ' ', text.replace('\u00a0', ' ')).strip()


def entity_key(text):
    # Entity comparison only, never a numeric/date parsing input.
    return compact(re.sub(r'\b(?:a|an|the)\b', ' ', text.casefold())).strip()


def word_number(text):
    words = compact(text.casefold().replace('-', ' ')).split()
    if len(words) == 1:
        return SMALL.get(words[0], TENS.get(words[0]))
    if len(words) == 2 and words[0] in TENS and words[1] in SMALL and 0 < SMALL[words[1]] < 10:
        return TENS[words[0]] + SMALL[words[1]]
    return None


def number_words(value):
    if value != int(value) or not 0 <= int(value) < 100:
        return None
    n = int(value)
    if n < 20:
        return WORDS.split()[n]
    tens = next(k for k, v in TENS.items() if v == n // 10 * 10)
    return tens if not n % 10 else tens + '-' + WORDS.split()[n % 10]


def parse_date(raw):
    text = compact(raw)
    m = re.fullmatch(r'(\d{1,2})\s*[-–—]\s*(\d{1,2})\s+(' + MONTH_PATTERN + r')', text, re.I)
    if m:
        start, end, month = int(m[1]), int(m[2]), MONTHS[m[3].lower()]
        if not 1 <= start <= end <= calendar.monthrange(2000, month)[1]:
            return {'kind': 'invalid', 'reason': 'invalid_or_reversed_date_range'}
        return dict(kind='day_range', value=(month, start, end), raw=raw)
    patterns = [
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+(' + MONTH_PATTERN + r')\s*,?\s*(\d{4})', 'dmy'),
        (r'(' + MONTH_PATTERN + r')\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})', 'mdy'),
        (r'(\d{4})-(\d{2})-(\d{2})', 'iso'),
    ]
    for pattern, style in patterns:
        m = re.fullmatch(pattern, text, re.I)
        if not m:
            continue
        if style == 'dmy': y, mo, d = int(m[3]), MONTHS[m[2].lower()], int(m[1])
        elif style == 'mdy': y, mo, d = int(m[3]), MONTHS[m[1].lower()], int(m[2])
        else: y, mo, d = map(int, m.groups())
        try: date(y, mo, d)
        except ValueError: return dict(kind='invalid', reason='invalid_calendar_date')
        return dict(kind='date', value=(y, mo, d), raw=raw, style=style)
    return None


def parse_quantity(raw):
    text = compact(raw)
    qualifier = ''
    m = re.match(r'^(In\s+\d{4}\s*,\s*|at least\s+|at most\s+|about\s+|approximately\s+)', text, re.I)
    if m: qualifier, text = text[:m.end()], text[m.end():]
    m = re.fullmatch(r'([$£€]?)(' + _NUM + r')(.*)', text)
    if m:
        currency, literal, suffix = m.groups()
        try: value = Decimal(literal.replace(',', ''))
        except InvalidOperation: return None
        number_start = len(qualifier) + len(currency)
    else:
        currency = ''; value = None
        for candidate in sorted(set(SMALL) | set(TENS) |
                                {t + sep + w for t in TENS for w, n in SMALL.items() if 0 < n < 10 for sep in ('-', ' ')},
                                key=len, reverse=True):
            if text.casefold() == candidate or text.casefold().startswith(candidate + ' '):
                value = Decimal(word_number(candidate)); literal = text[:len(candidate)]
                suffix = text[len(candidate):]; break
        if value is None: return None
        number_start = len(qualifier)
    unit = compact(suffix.casefold().replace('-', ' '))
    scale = Decimal(1)
    if unit in ('billion', 'million', 'thousand'):
        scale = {'billion': Decimal(10**9), 'million': Decimal(10**6), 'thousand': Decimal(1000)}[unit]
        dimension = currency or 'scalar'
    elif unit in ('', 'player', 'players', 'trophies', 'units', 'votes'): dimension = currency or 'count'
    elif unit in ('%', 'percent'): dimension = 'percent'
    elif unit in ('year old', 'years old', 'year', 'years'): dimension = 'months'; scale = Decimal(12)
    elif unit in ('month', 'months', 'months old'): dimension = 'months'
    else: return None
    return dict(kind='quantity', value=value * scale, amount=value, dimension=dimension,
        currency=currency, unit=unit, scale=scale, raw=raw, number_start=number_start,
        number_end=number_start + len(literal), literal=literal)


def parse_alias(raw, question):
    text = compact(raw).strip(',;.!? \t\n\"`')
    if not text: return dict(kind='invalid', reason='empty_raw_alias')
    parsed = parse_date(text)
    if parsed: return parsed
    if re.fullmatch(r'[12]\d{3}', text) and re.search(r'\b(?:when|year|date|season)\b', question, re.I):
        return dict(kind='year', value=int(text), raw=raw)
    if re.search(r'\b\d{3,4}s\b|\d\s*[-–—]\s*\d|\d[/-]\d', text):
        return dict(kind='invalid', reason='unsupported_temporal_or_numeric_range')
    number = parse_quantity(text)
    if number: return number
    key = entity_key(text)
    if key in COUNTRIES or key in COUNTRY_NAMES:
        # Distinct historical polity names are not guessed to be synonyms.
        return dict(kind='country', value=key, raw=raw)
    if key in {x.casefold() for x in POOLS['continent']}:
        return dict(kind='continent', value=key, raw=raw)
    m = re.fullmatch(r'(native to\s+)(.+)', key)
    if m and m[2] in {x.casefold() for x in POOLS['continent']}:
        return dict(kind='continent', value=m[2], raw=raw, prefix=compact(raw)[:len('native to ')])
    m = re.fullmatch(r'((?:northern|southern|eastern|western|central)\s+)?(.+)', key)
    if m and m[2] in STATES:
        return dict(kind='us_region', value=key, raw=raw, prefix=m[1] or '')
    position = re.fullmatch(r'(beneath|under)\s+(?:the\s+)?(.+)', key)
    if position and position[2] in POOLS['organ']:
        return dict(kind='relative_organ', value=('under', position[2]), raw=raw)
    for kind in ('color', 'organ', 'meningeal_layer', 'body_region', 'tissue', 'brain_structure', 'gland', 'lobe'):
        if key in POOLS[kind]: return dict(kind=kind, value=key, raw=raw)
    if key == 'torso': return dict(kind='body_region', value='trunk', raw=raw)
    # Names require an explicit person-role question and a multiword proper name.
    # WHO alone is insufficient (countries, teams, organizations also answer WHO).
    if re.search(r'\b(author|wrote|writer|sang|singer|actor|actress|played|president|person|scientist|invented)\b', question, re.I):
        prefix_match = re.match(r'^(?:written by|by)\s+', text, re.I)
        prefix = text[:prefix_match.end()] if prefix_match else ''
        name = text[len(prefix):]
        parts = name.split()
        if 2 <= len(parts) <= 5 and all(re.fullmatch(r"[A-Z][A-Za-z'’-]*\.?", p) for p in parts):
            if any(w.casefold() in ('empire', 'kingdom', 'university', 'astros', 'dodgers', 'states', 'city', 'team') for w in parts):
                return dict(kind='invalid', reason='person_hint_conflicts_with_entity_shape')
            middle = parts[1:-1]
            if middle and any(len(p.rstrip('.')) != 1 for p in middle):
                return dict(kind='invalid', reason='unresolved_middle_name_identity')
            return dict(kind='person', value=(parts[0].casefold(), parts[-1].casefold()), raw=raw,
                        first=parts[0], last=parts[-1], prefix=prefix)
    return dict(kind='invalid', reason='unsupported_or_uncertain_answer_type')


def equivalence_forms(text):
    """Finite whole-answer variants; never substring or an unrestricted judge."""
    text = compact(text)
    result = {text}
    q = parse_quantity(text)
    if q:
        # Numeric words/digits are equivalent only at the same declared unit.
        unit = text[q['number_end']:]
        plain = format(q['amount'], 'f').rstrip('0').rstrip('.') if '.' in format(q['amount'], 'f') else str(q['amount'])
        result.add(text[:q['number_start']] + plain + unit)
        word = number_words(q['amount'])
        if word: result.add(text[:q['number_start']] + word + unit)
        # These are literal numeric/unit equalities, not a semantic judge.
        if q['dimension']=='percent':
            for amount in (plain, word):
                if amount:result.update((amount+'%',amount+' percent'))
        elif q['dimension']=='months':
            total=q['value']
            result.add(format(total,'f')+' months')
            if total%12==0:
                years=total/12; ys=format(years,'f')
                result.update((ys+' years',ys+' years old',ys+'-year-old'))
                w=number_words(years)
                if w:result.update((w+' years',w+' years old',w+'-year-old'))
            # A provided bare numeric age gold licenses a bare answer. The
            # current phrase itself explicitly states the unit, so its value
            # is preserved; count classifiers never guess a hidden age unit.
        elif q['dimension']=='count':
            result.add(plain)
            if word:result.add(word)

    m = re.fullmatch(r'(beneath|under)\s+(?:the\s+)?(.+)', text, re.I)
    if m:
        for relation in ('beneath', 'under'):
            for article in ('', 'the '): result.add(relation + ' ' + article + m[2])
    if text.casefold().startswith('native to '):result.add(text[len('native to '):])
    if entity_key(text) in ('trunk','torso'):result.update(('trunk','torso'))
    d = parse_date(text)
    if d and d['kind'] == 'date':
        y, mo, day = d['value']
        result.update((f'{day} {calendar.month_name[mo]} {y}', f'{calendar.month_name[mo]} {day}, {y}', f'{y:04d}-{mo:02d}-{day:02d}'))
    if d and d['kind'] == 'day_range':
        mo, start, end = d['value']
        result.update(f'{start}{dash}{end} {calendar.month_name[mo]}' for dash in ('-', '–', '—'))
    return {compact(x) for x in result if x.strip()}


def invalid(reason):
    return dict(status='INVALID_INTERVENTION', invalid_reason=reason)


def make_plan(golds, seed_key, question):
    if not golds or not all(isinstance(g, str) and g.strip() for g in golds): return invalid('empty_raw_gold_aliases')
    raw = sorted(set(compact(g).strip(',;.!? \t\n\"`') for g in golds), key=lambda s: (-len(s), s))
    parsed = [parse_alias(g, question) for g in raw]
    if any(p['kind'] == 'invalid' for p in parsed):
        return invalid(next(p['reason'] for p in parsed if p['kind'] == 'invalid'))
    kinds = {p['kind'] for p in parsed}
    if kinds <= {'date', 'year'}:
        years = {p['value'][0] if p['kind'] == 'date' else p['value'] for p in parsed}
        full = {p['value'] for p in parsed if p['kind'] == 'date'}
        if len(years) != 1 or len(full) > 1: return invalid('conflicting_date_aliases_or_events')
        kind = 'date_family'
    elif len(kinds) != 1: return invalid('mixed_answer_types_or_slots')
    else:
        kind = parsed[0]['kind']
        if kind == 'quantity':
            values = {(p['value'], p['dimension']) for p in parsed}
            # A bare count can accompany an age measured in years, but only for
            # an explicitly age question and a provided year-old gold alias.
            if len(values) > 1 and re.search(r'\b(how old|what age|at what age)\b', question, re.I):
                for p in parsed:
                    if p['dimension'] == 'count': p['value'] *= 12; p['dimension'] = 'months'; p['scale'] *= 12
                values = {(p['value'], p['dimension']) for p in parsed}
        else: values = {str(p['value']) for p in parsed}
        if len(values) != 1: return invalid('multiple_distinct_answer_values_or_entities')
    seed = int(hashlib.sha256(json.dumps(seed_key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16], 16)
    mapping = {}
    if kind == 'date_family':
        delta = 4 * (1 + seed % 20)
        for g, p in zip(raw, parsed):
            if p['kind'] == 'year': changed = str(p['value'] + delta)
            else:
                y, mo, d = p['value']; y += delta
                try: date(y, mo, d)
                except ValueError: return invalid('shifted_date_invalid_leap_day')
                changed = f'{d} {calendar.month_name[mo]} {y}'
            mapping[g] = changed
    elif kind == 'day_range':
        mo, start, end = parsed[0]['value']; room = calendar.monthrange(2000, mo)[1] - end
        if room < 1: return invalid('date_interval_has_no_safe_forward_shift')
        delta = 1 + seed % room
        for g in raw: mapping[g] = f'{start + delta}–{end + delta} {calendar.month_name[mo]}'
    elif kind == 'quantity':
        base = parsed[0]; offset = Decimal(1 + seed % 9) * base['scale']
        target = base['value'] + offset
        if base['dimension'] == 'percent' and not 0 <= target <= 100: return invalid('percentage_outside_valid_domain')
        for g, p in zip(raw, parsed):
            amount = target / p['scale']
            precision = len(p['literal'].split('.', 1)[1]) if '.' in p['literal'] else 0
            if amount != amount.quantize(Decimal(1).scaleb(-precision)): return invalid('quantity_alias_precision_conflict')
            rendered = format(amount, f'.{precision}f')
            mapping[g] = g[:p['number_start']] + rendered + g[p['number_end']:]
    else:
        pool = POOLS['organ'] if kind == 'relative_organ' else POOLS[kind]
        original_forms = {entity_key(x) for g in raw for x in equivalence_forms(g)}
        choices = [x for x in pool if entity_key(x) not in original_forms]
        if kind == 'relative_organ': choices = [x for x in choices if x != parsed[0]['value'][1]]
        if kind == 'us_region': choices = [x for x in choices if entity_key(x) not in {entity_key(g)[len(p.get('prefix', '')):] for g, p in zip(raw, parsed)}]
        if not choices: return invalid('no_non_equivalent_typed_candidate')
        target = choices[seed % len(choices)]
        for g, p in zip(raw, parsed):
            if kind == 'relative_organ': mapping[g] = 'beneath the ' + target
            elif p.get('prefix'): mapping[g] = p['prefix'] + target
            else: mapping[g] = target
    originals = sorted({x for g in raw for x in equivalence_forms(g)})
    if any(entity_key(v) in {entity_key(x) for x in originals} for v in mapping.values()): return invalid('equivalent_typed_candidate')
    return dict(status='VALID_PLAN', replacement_kind=kind, alias_to_value=mapping,
                original_semantic_aliases=originals,
                person_first=parsed[0].get('first'), person_last=parsed[0].get('last'))
