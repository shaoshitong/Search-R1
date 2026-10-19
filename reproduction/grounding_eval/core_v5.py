"""Independent conservative strict-v5 preparation, classification and aggregation.

Operational choices (not claimed as exact author settings): deterministic typed
fake values, boundary-aware aliases, one user chat with the actual seen documents,
greedy BF16, 512 new tokens, no retrieval during the counterfactual decode.
"""
import hashlib
import json
import re
import string
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from search_r1.qa_protocol import extract_final_answer

PROTOCOL = 'strict-v5-conservative-typed'
CF_CONFIG = dict(protocol_version=PROTOCOL, replacement_version='raw-typed-conservative-v5', comparison_version='finite-whole-answer-equivalence-v5',
    seed=1729, prompt_version='single-user-question-docs-v1',
    parser_version='terminal-nonnested-qa-v1', normalization='qa-em-ascii-punct-articles-v1',
    max_new_tokens=512, max_model_len=8192, temperature=0., top_p=1., top_k=-1,
    dtype='bfloat16', new_retrieval=False, ban_search_tokens=False,
    max_attempts=3, tensor_parallel_size=1, answer_stop='</answer>',
    include_stop_str_in_output=True, stop_version='normal-terminal-answer-action-v1')

def utc(): return datetime.now(timezone.utc).isoformat()
def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic(path, obj, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace: raise FileExistsError(path)
    temp = path.with_name(path.name+'.tmp')
    import os
    with open(temp, 'wb') as f:
        f.write(canonical(obj)+b'\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def read(path): return json.loads(Path(path).read_text())
def read_jsonl(path):
    import gzip
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def occurrence(row): return [str(row['data_source']), int(row['evaluation_ordinal']), str(row['qid'])]

def semantic_keys(text):
    from .replacements_v5 import equivalence_forms
    # Numeric/date strings retain punctuation and articles during equivalence.
    # QA EM normalization is reserved for original correctness, not type parsing.
    return {re.sub(r'\s+', ' ', re.sub(r'\b(?:a|an|the)\b',' ',x.casefold())).strip().strip(',;.!?') for x in equivalence_forms(text)}

def _alias_pattern(alias):
    # Safe spelling variants only: whitespace, dashes, and date commas. Never
    # strip arbitrary punctuation (which would corrupt money, decimals, initials).
    chunks=re.split(r'(\s+|[-–—]|,)', alias)
    return ''.join(r'\s+' if x.isspace() else r'[-–—]' if x in '-–—' else r',?' if x==',' else re.escape(x) for x in chunks if x)

def numeric_span_valid(doc,start,end,alias):
    from .replacements_v5 import parse_quantity, word_number
    quantity=parse_quantity(alias)
    if quantity is None and not re.fullmatch(r'[12]\d{3}',alias):return True
    left,right=doc[:start],doc[end:]
    # A naked answer 7 is not a component of 3.7, 7.5, -7, $7,
    # 7,000 or twenty-seven. These guards also govern residual detection.
    if re.search(r'[\d][.,]\s*$',left) or re.match(r'^[.,]\s*\d',right):return False
    if re.search(r'[-+−]\s*$',left):return False
    if quantity is not None and not quantity.get('currency') and re.search(r'[$£€]\s*$',left):return False
    if re.search(r'[A-Za-z][ -]$',left):
        before=re.search(r'([A-Za-z]+)[ -]$',left)
        if before and (word_number(before[1]) is not None or before[1].lower() in ('hundred','thousand','million','billion','and','point')):
            return False
    if re.match(r'^[- ](?:hundred|thousand|million|billion|[a-z]+)',right,re.I):
        after=re.match(r'^[- ]([a-z]+)',right,re.I)
        if after and (word_number(after[1]) is not None or after[1].lower() in ('hundred','thousand','million','billion','point')):
            # Explicit scale aliases such as $2.187 billion already consume it.
            return False
    return True

def intervene(docs, golds, key, question=''):
    from .replacements_v5 import make_plan, equivalence_forms, parse_date
    plan=make_plan(golds,key,question)
    if plan['status']!='VALID_PLAN': return plan
    mapping={}
    for alias,value in plan['alias_to_value'].items():
        for form in equivalence_forms(alias):
            if form in mapping and mapping[form]!=value:
                common=equivalence_forms(mapping[form]) & equivalence_forms(value)
                if not common:return dict(status='INVALID_INTERVENTION',invalid_reason='ambiguous_alias_mapping')
                mapping[form]=min(common,key=lambda x:(len(x),x))
            else:mapping[form]=value
    aliases=sorted(mapping,key=lambda x:(-len(x),x))
    compiled=[(re.compile(r'(?<!\w)'+_alias_pattern(a)+r'(?!\w)',re.I),a) for a in aliases]
    modified=[]; replacements=[]; observed_values=set(); grounded_dates=set()
    for index,doc in enumerate(docs):
        matches=[]
        for pattern,alias in compiled:
            for match in pattern.finditer(doc):
                if numeric_span_valid(doc,match.start(),match.end(),alias):matches.append((match.start(),match.end(),alias,match.group()))
        matches.sort(key=lambda x:(x[0],-(x[1]-x[0])))
        chosen=[]; end=-1
        for item in matches:
            if item[0]>=end:chosen.append(item);end=item[1]
        parts=[];end=0
        for start,stop,alias,actual in chosen:
            value=mapping[alias];parts.extend((doc[end:start],value));end=stop
            replacements.append(dict(doc_index=index,start=start,end=stop,original=actual,injected=value))
            observed_values.add(value)
            # If the actual evidence supplies a full date around a year alias,
            # admit that exact altered date (not an unobserved date candidate).
            if re.fullmatch(r'[12]\d{3}',actual) and re.fullmatch(r'[12]\d{3}',value):
                context=doc[max(0,start-32):stop+2]
                from .replacements_v5 import MONTH_PATTERN
                pattern=re.compile(r'(?:\d{1,2}\s+'+MONTH_PATTERN+r'\s*,?\s*|'+MONTH_PATTERN+r'\s+\d{1,2}\s*,?\s*)'+re.escape(actual),re.I)
                for date_match in pattern.finditer(context):
                    if date_match.end()==len(context.rstrip(' .,;')):
                        candidate=date_match.group()[:-len(actual)]+value
                        parsed_date=parse_date(candidate)
                        if parsed_date and parsed_date['kind']=='date':grounded_dates.add(candidate)
        parts.append(doc[end:]);modified.append(''.join(parts))
    if not replacements:return dict(status='UNPERTURBABLE',invalid_reason='no_boundary_matched_raw_or_explicit_equivalent_gold')
    # Check every declared original equivalent after the one-pass operation.
    # A nested short alias left elsewhere is not quietly accepted.
    for doc in modified:
        if any(numeric_span_valid(doc,m.start(),m.end(),alias) for pattern,alias in compiled for m in pattern.finditer(doc)):
            return dict(status='INVALID_INTERVENTION',invalid_reason='original_alias_residual')
    if plan.get('person_last'):
        # A bare surname or an unrecognized middle-name variant is ambiguous;
        # do not assert that it denotes another entity or silently leave it.
        first,last=plan['person_first'],plan['person_last']
        for doc in modified:
            if re.search(r'(?<!\w)'+re.escape(last)+r'(?!\w)',doc,re.I):
                return dict(status='INVALID_INTERVENTION',invalid_reason='unresolved_person_surname_residual')
    original_keys={k for x in plan['original_semantic_aliases'] for k in semantic_keys(x)}
    cf_aliases={x for value in observed_values|grounded_dates for x in equivalence_forms(value)}
    if re.search(r'\b(how old|what age|at what age)\b',question,re.I) and any(re.fullmatch(r'\d+',g.strip()) for g in golds):
        from .replacements_v5 import parse_quantity,number_words
        for value in observed_values:
            quantity=parse_quantity(value)
            if quantity and quantity['dimension']=='months' and quantity['value']%12==0:
                years=quantity['value']/12; cf_aliases.add(format(years,'f'))
                words=number_words(years)
                if words:cf_aliases.add(words)
    cf_aliases=sorted(cf_aliases)
    if any(semantic_keys(v)&original_keys for v in cf_aliases):
        return dict(status='INVALID_INTERVENTION',invalid_reason='original_and_injected_alias_collision')
    return dict(status='READY',modified_docs=modified,injected_value=sorted(observed_values)[0],
        cf_aliases=cf_aliases, original_semantic_aliases=plan['original_semantic_aliases'],
        replacement_kind=plan['replacement_kind'],replacement_mapping=replacements,
        replacement_count=len(replacements),modified_docs_sha256=digest(modified),
        comparison_version=CF_CONFIG['comparison_version'])

def classify(original, cf_answer, cf_aliases, golds):
    if not cf_answer:return 'Amb'
    new=semantic_keys(cf_answer)
    old={k for g in golds for k in semantic_keys(g)}
    injected={k for g in cf_aliases for k in semantic_keys(g)}
    if new&old:return 'Hall'
    if new&injected:return 'True'
    return 'Amb'

def prepare_row(row, manifest):
    key = occurrence(row)
    base = dict(row, occurrence_key=key, protocol_version=PROTOCOL,
        checkpoint_hash=manifest['checkpoint_hash'], cf_config_sha256=manifest['cf_config_sha256'],
        source_record_hash=digest(row), prepared_at=utc())
    base['request_id'] = digest([manifest['method'], manifest['scale'], manifest['seed'],
        manifest['checkpoint_hash'], key, PROTOCOL, manifest['cf_config_sha256'], base['source_record_hash']])
    required = ('question', 'gold', 'model_spans', 'seen_docs', 'actual_retrieval_calls', 'source_shard_sha256')
    if any(k not in row for k in required):
        return dict(base, status='SOURCE_INCOMPLETE', invalid_reason='missing_required_source_fields')
    if (not isinstance(row['gold'], list) or not row['gold'] or
        not all(isinstance(x,str) for x in row['gold']+row['model_spans']+row['seen_docs'])):
        return dict(base, status='SOURCE_INCOMPLETE', invalid_reason='invalid_source_schema')
    answer = extract_final_answer(row['model_spans'])
    correct = bool(answer and normalize(answer) in {normalize(g) for g in row['gold']})
    markers = sum(len(re.findall(r'<search>.*?</search>', span, re.S)) for span in row['model_spans'])
    base.update(original_answer=answer, normalized_original_answer=normalize(answer or ''), original_em=int(correct),
                model_search_markers=markers, seen_docs_sha256=digest(row['seen_docs']))
    if not correct: return dict(base, status='NOT_CORRECT')
    if not markers and not row['actual_retrieval_calls'] and not row['seen_docs']:
        return dict(base, status='DIRECT_CORRECT', classification='Direct')
    if not markers or not row['actual_retrieval_calls'] or not row['seen_docs']:
        return dict(base, status='SOURCE_INCOMPLETE', invalid_reason='search_provenance_inconsistent')
    # Historical methods sometimes serialize bare qid differently. The immutable
    # common dataset ordinal/question/golds make the same intervention assignment
    # across methods; the full original qid still belongs to the resume identity.
    seed_key=[CF_CONFIG['seed']]+key[:2]+[digest([row['question'],sorted(row['gold'])])]
    return dict(base, replacement_seed_key=seed_key, **intervene(row['seen_docs'], row['gold'], seed_key,row['question']))

def messages(row):
    docs = '\n\n'.join(row['modified_docs'])
    return [dict(role='user', content=(
        'Answer the question using the search results below. You may reason inside '
        '<think> and </think>. Put only the final short answer inside <answer> and </answer>.\n'
        'Question: '+row['question']+'\n<information>'+docs+'</information>'))]

def aggregate(prepared, results):
    by_id = {}
    for r in results:
        key=r['request_id']
        if key in by_id:
            if digest(by_id[key]) != digest(r): raise ValueError('Conflicting duplicate result: '+key)
        by_id[key]=r
    keys=set(); counts=Counter(); invalid=Counter(); pending=0; completed=0; perturbable=0
    for p in prepared:
        key=p['request_id']
        if key in keys: raise ValueError('Duplicate prepared occurrence')
        keys.add(key); status=p['status']
        counts['normal_questions'] += 1; counts['normal_correct'] += p.get('original_em',0)
        if status=='DIRECT_CORRECT': counts['Direct']+=1; continue
        if status=='NOT_CORRECT': continue
        if p.get('original_em'): counts['correct_searched']+=1
        if status!='READY': invalid[status+':'+p.get('invalid_reason','')]+=1; continue
        perturbable+=1; result=by_id.get(key)
        if result is None or result['status']!='CLASSIFIED':
            pending+=1; invalid[(result or {}).get('status','PENDING')]+=1; continue
        if any(result.get(k)!=p.get(k) for k in ('checkpoint_hash','cf_config_sha256','source_record_hash')):
            raise ValueError('Result provenance mismatch')
        category=classify(p['original_answer'],result['cf_answer'],p['cf_aliases'],p.get('original_semantic_aliases',p['gold']))
        if category!=result['classification']: raise ValueError('Stored classification differs from recomputation')
        counts[category]+=1; completed+=1
    if set(by_id)-keys: raise ValueError('Foreign results')
    direct, true, amb, hall = (counts[k] for k in ('Direct','True','Amb','Hall'))
    searched=true+amb+hall; total=direct+searched
    normal_tokens={}
    for field in ('initial_prompt_tokens','model_generated_tokens','raw_generated_tokens','observation_tokens',
                  'trajectory_total_tokens','cumulative_model_input_tokens'):
        values=[p[field] for p in prepared if isinstance(p.get(field),(int,float))]
        ordered=sorted(values)
        normal_tokens[field]=dict(total=sum(values) if values else None, measured_questions=len(values),
            mean=sum(values)/len(values) if values else None,
            p50=ordered[len(ordered)//2] if ordered else None,
            p95=ordered[min(len(ordered)-1,int(len(ordered)*.95))] if ordered else None,
            max=max(values) if values else None)
    cf_results=[r for key,r in by_id.items() if key in keys and r['status']=='CLASSIFIED']
    cf_tokens={}
    for field in ('model_input_tokens','generation_tokens','trajectory_total_tokens'):
        values=[r[field] for r in cf_results if isinstance(r.get(field),(int,float))]
        cf_tokens[field]=dict(total=sum(values) if values else None, measured_questions=len(values),
            mean=sum(values)/len(values) if values else None,max=max(values) if values else None)
    return dict(protocol_version=PROTOCOL, status='provisional' if pending or any(x.startswith('SOURCE_INCOMPLETE') for x in invalid) else 'complete',
        counts=dict(counts), coverage=dict(correct_searched=counts['correct_searched'], perturbable=perturbable,
        inference_completed=completed, classified=searched, pending_or_error=pending, invalid_reasons=dict(invalid)),
        TC=(direct+true+.6*amb+.3*hall)/total if total else None,
        D_Inv=hall/searched if searched else None, TC_denominator=total, D_Inv_denominator=searched,
        TC_null_reason=None if total else 'no_correct_direct_or_classified_searched',
        D_Inv_null_reason=None if searched else 'no_classified_correct_searched', seed_sample_sd=None,
        token_usage=dict(normal=normal_tokens,counterfactual=cf_tokens,
            note='Normal cumulative input counts repeated history at every actual generation request; observations are not model output. CF excludes retry failures from unique-success totals; attempts remain separately auditable.'),utc=utc())
