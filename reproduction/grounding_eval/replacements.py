"""Transparent deterministic coarse replacement rules, not an author-exact recipe.

Recognized quantities retain their form/unit. Question hints select entity pools;
unknown types are explicitly marked fallback. Every accepted CF alias is an
actual injected value (no substring acceptance or unobserved short aliases).
"""
import calendar
import hashlib
import json
import re
import string
from decimal import Decimal

_MONTH = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
_MONTH_RE = re.compile(r'(?<![A-Za-z])'+_MONTH+r'(?![A-Za-z])', re.I)
_YEAR = re.compile(r'(?<!\d)([12]\d{3})(?!\d)')
_NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_QUANTITY = re.compile(r'^('+_NUMBER+r')(\s*)([^\d%]*)$')
_PERCENT = re.compile(r'^('+_NUMBER+r')(\s*)%$')
_SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty').split(), range(21)))
_POOLS = dict(person=['Ada Lovelace','Charles Darwin','Marie Curie','Rosalind Franklin','Isaac Newton','Katherine Johnson'],
    location=['Helsinki','Lisbon','Kyoto','Nairobi','Vienna','Melbourne'],
    country=['Brazil','Norway','Kenya','Portugal','Canada','Japan'],
    anatomy_lobe=['frontal lobe','parietal lobe','occipital lobe','temporal lobe'],
    anatomy_gland=['pituitary gland','thyroid gland','adrenal gland','pineal gland'],
    anatomy_brain=['hypothalamus','hippocampus','cerebellum','brainstem'],
    anatomy_body=['liver','pancreas','spleen','kidney'],
    fallback_entity=['Aster Research Institute','Silverwood Foundation','Harbor College','Northbridge Society'])


def _norm(x):
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in x.lower() if c not in string.punctuation)).split())


def _seed(key):
    return int(hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()[:16],16)


def _roman(n):
    if not 0<n<4000:raise ValueError('Roman value outside supported range')
    out=''
    for value,token in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while n>=value:out+=token;n-=value
    return out


def _roman_int(text):
    if not re.fullmatch('[IVXLCDM]+',text):return None
    values=dict(I=1,V=5,X=10,L=50,C=100,D=500,M=1000)
    n=sum(-values[c]if i+1<len(text)and values[c]<values[text[i+1]]else values[c]for i,c in enumerate(text))
    return n if 0<n<4000 and _roman(n)==text else None


def _format_numeric(original, value):
    precision=len(original.split('.',1)[1])if'.'in original else 0
    text=format(value,'.%df'%precision)if precision else str(int(value))
    if ',' in original:
        whole, dot, fraction=text.partition('.')
        text=format(int(whole),',')+(dot+fraction if dot else '')
    if original.startswith('+') and not text.startswith('-'):text='+'+text
    return text


def _date_candidate(alias):
    return bool((_MONTH_RE.search(alias)and re.search(r'\d',alias))or
        re.fullmatch(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',alias)or
        re.fullmatch(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',alias))


def _pack(mapping,kind,golds):
    forbidden={_norm(g)for g in golds}
    if not mapping or any(not _norm(v)or _norm(v)in forbidden for v in mapping.values()):
        return None
    aliases=sorted(set(mapping.values()))
    # Pick the longest actually-injected form as the display representative.
    representative=max(aliases,key=lambda x:(len(x),x))
    return dict(alias_to_value=mapping,cf_aliases=aliases,injected_value=representative,replacement_kind=kind)


def make_replacements(golds, seed_key, question):
    if not isinstance(golds,(list,tuple))or not golds or not all(isinstance(x,str)for x in golds):
        raise ValueError('Nonempty string gold aliases required')
    golds=sorted(set(x.strip()for x in golds if x.strip()and _norm(x)))
    if not golds:raise ValueError('No usable gold aliases')
    n=_seed(seed_key);q=question.casefold()
    # Repeated outer loop only changes a collision-avoiding deterministic offset.
    roman=[]
    for alias in golds:
        match=re.search(r'(?:^|\s)([IVXLCDM]+)$',alias)
        value=_roman_int(match.group(1))if match else None
        if value is not None:roman.append((alias,match,value))
    year_aliases=[a for a in golds if re.fullmatch(r'[12]\d{3}',a)]
    if roman and (year_aliases or re.search(r'super\s*bowl|roman|numeral|pope|monarch|king|queen',q)):
        for trial in range(32):
            delta=1+(n+trial)%15;mapping={}
            for alias,match,value in roman:
                if value+delta>=4000:continue
                mapping[alias]=alias[:match.start(1)]+_roman(value+delta)
            for alias in year_aliases:mapping[alias]=str(int(alias)+delta)
            if mapping:
                fallback=max(mapping.values(),key=len)
                mapping.update({a:fallback for a in golds if a not in mapping})
                result=_pack(mapping,'roman_and_season_year_offset'if year_aliases else'roman_ordinal_offset',golds)
                if result:return result
    dates=[a for a in golds if _date_candidate(a)]
    if dates or (year_aliases and len(year_aliases)==len(golds)and re.search(r'year|when|date|season',q)):
        for trial in range(32):
            delta=4*(1+(n+trial)%20);mapping={}
            leap_day = any(re.search(r'(?:Feb(?:ruary)?[^0-9]*29|29[^A-Za-z]*Feb(?:ruary)?|[-/]02[-/]29)',a,re.I) for a in dates)
            if leap_day and any(not calendar.isleap(int(m.group(1))+delta)for a in dates for m in _YEAR.finditer(a)):
                continue
            years={m.group(1)for a in dates+year_aliases for m in _YEAR.finditer(a)}
            if years:
                for alias in dates+year_aliases:
                    changed=_YEAR.sub(lambda m:str(int(m.group(1))+delta),alias)
                    # Keep leap-day substitutions valid across century boundaries.
                    iso=re.fullmatch(r'(\d{4})([-/])(02)([-/])(29)',changed)
                    if iso and not calendar.isleap(int(iso.group(1))):changed=changed[:-2]+'28'
                    if changed!=alias:mapping[alias]=changed
            else:
                # Month/day aliases without a year: retain the month and change
                # day consistently to 1..28 (valid for every month).
                day_match=re.search(r'(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?!\d)',dates[0])
                if day_match:
                    old=day_match.group(1);new=str(1+(int(old)+1+(n+trial)%20)%28)
                    if new==str(int(old)):continue
                    for alias in dates:
                        mapping[alias]=re.sub(r'(?<!\d)'+re.escape(old)+r'(?!\d)',new,alias)
                    for alias in golds:
                        if alias==old:mapping[alias]=new
            if mapping:
                fallback=max(mapping.values(),key=len);mapping.update({a:fallback for a in golds if a not in mapping})
                result=_pack(mapping,'date_preserved_year_offset'if years else'date_preserved_day_offset',golds)
                if result:return result
    percentages={a:_PERCENT.fullmatch(a)for a in golds}
    if any(percentages.values()):
        first=next(m for m in percentages.values()if m);base=Decimal(first.group(1).replace(',',''))
        for trial in range(32):
            value=Decimal((int(base)+1+(n+trial)%79)%101)
            mapping={}
            for alias,match in percentages.items():
                if match:mapping[alias]=_format_numeric(match.group(1),value)+match.group(2)+'%'
                elif re.fullmatch(_NUMBER,alias):mapping[alias]=_format_numeric(alias,value)
                else:mapping[alias]=_format_numeric(first.group(1),value)+'%'
            result=_pack(mapping,'percentage_preserved',golds)
            if result:return result
    quantities={a:_QUANTITY.fullmatch(a)for a in golds}
    numeric=[m for m in quantities.values()if m]
    if numeric or all(a.casefold()in _SMALL for a in golds):
        first=numeric[0]if numeric else None
        base=Decimal(first.group(1).replace(',',''))if first else Decimal(_SMALL[golds[0].casefold()])
        for trial in range(32):
            value=base+Decimal(1+(n+trial)%97);mapping={}
            # Unknown spelled-out aliases inherit a full actual numeric+unit form.
            unit_example=next((m for m in numeric if m.group(3).strip()),first)
            fallback=(_format_numeric(unit_example.group(1),value)+unit_example.group(2)+unit_example.group(3))if unit_example else str(int(value))
            for alias,match in quantities.items():
                mapping[alias]=(_format_numeric(match.group(1),value)+match.group(2)+match.group(3))if match else fallback
            result=_pack(mapping,'numeric_quantity_unit_preserved'if any(m.group(3).strip()for m in numeric)else'numeric_quantity',golds)
            if result:return result
    # Names of TV programmes (Grey's Anatomy) and institutional contexts may
    # contain anatomy/country hints. An explicit WHO asks for a person first.
    if re.search(r'\bwho\b|\bwhose\b|\bwhich person\b',q):
        pool=_POOLS['person'];start=n%len(pool)
        for i in range(len(pool)):
            value=pool[(start+i)%len(pool)]
            result=_pack({a:value for a in golds},'question_hint_person',golds)
            if result:return result
    anatomical=' '.join(golds).casefold()
    position=re.compile(r'^(beneath|under|above|behind|below|near|inside|within|next to)\s+(the\s+)?(.+)$',re.I)
    if any(position.match(a) for a in golds) and re.search(r'body|gallbladder|organ|liver|stomach|kidney',q+' '+anatomical):
        pool=_POOLS['anatomy_body']
        for i in range(len(pool)):
            organ=pool[(n+i)%len(pool)];mapping={}
            for a in golds:
                m=position.match(a)
                mapping[a]=(m.group(1)+' '+(m.group(2)or'')+organ)if m else organ
            result=_pack(mapping,'anatomical_relative_position_preserved',golds)
            if result:return result
    if re.search(r'\blobes?\b|\bcortex\b',q+' '+anatomical):kind='anatomy_lobe'
    elif re.search(r'\bgland\b|pituitary|adrenal|thyroid|endocrine|\bhormone\b',q+' '+anatomical):kind='anatomy_gland'
    elif re.search(r'\bbrain\b|cerebell|hypothalam|hippocamp',q):kind='anatomy_brain'
    elif re.search(r'\bbody\b|anatom|\borgan\b|\bkidney\b|\bliver\b|\bpancreas\b',q):kind='anatomy_body'
    elif re.search(r'\bcountry\b|\bnation\b',q):kind='country'
    elif re.search(r'\bwho\b|\bperson\b|\bwhose\b',q):kind='person'
    elif re.search(r'\bwhere\b|\bcity\b|\btown\b|\blocation\b',q):kind='location'
    else:kind='fallback_entity'
    pool=_POOLS[kind];start=n%len(pool)
    for i in range(len(pool)):
        value=pool[(start+i)%len(pool)]
        result=_pack({a:value for a in golds},'question_hint_'+kind if kind!='fallback_entity'else'fallback_entity_unknown_type',golds)
        if result:return result
    raise ValueError('No non-equivalent deterministic replacement available for '+kind)
