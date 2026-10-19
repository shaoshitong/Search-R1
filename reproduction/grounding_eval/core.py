"""Dependency-free paper-strict-v1 preparation, classification and aggregation.

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

PROTOCOL = 'paper-strict-v1'
CF_CONFIG = dict(protocol_version=PROTOCOL, replacement_version='typed-question-guided-v4-punctuation-aliases',
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

def intervene(docs, golds, key, question=''):
    # QA EM ignores punctuation/articles. A gold such as "Super Bowl LII,"
    # therefore also names an unpunctuated occurrence in the actual evidence.
    # Keep terminal punctuation OUTSIDE the replaced lexical span. Longest
    # alternatives still run once on original docs, with whole-word boundaries.
    aliases=set()
    for g in golds:
        trimmed=g.strip().strip(',;:.!? \t\n\"\'`')
        if trimmed and normalize(trimmed):aliases.add(trimmed)
        normalized=normalize(g)
        if normalized:aliases.add(normalized)
    aliases = sorted(aliases,key=lambda x:(-len(x),x))
    if not aliases: return dict(status='INVALID_INTERVENTION', invalid_reason='empty_gold_aliases')
    from .replacements import make_replacements
    replacement=make_replacements(aliases,key,question)
    alias_map={k.casefold():v for k,v in replacement['alias_to_value'].items()}
    fake=replacement['injected_value']
    # Lookarounds prevent replacing Ann in Anna or 19 in 1999. One pass on
    # ORIGINAL docs, longest alternatives first; newly injected strings never match.
    pattern = re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(a) for a in aliases) +r')(?!\w)', re.I)
    modified, replacements = [], []
    for i, doc in enumerate(docs):
        matches = list(pattern.finditer(doc))
        replacements.extend(dict(doc_index=i, start=m.start(), end=m.end(), original=m.group(), injected=alias_map[m.group().casefold()]) for m in matches)
        modified.append(pattern.sub(lambda m:alias_map[m.group().casefold()], doc))
    if not replacements: return dict(status='UNPERTURBABLE', invalid_reason='no_boundary_matched_gold_in_actual_seen_docs')
    if modified == docs or any(normalize(x) in {normalize(g) for g in aliases} for x in replacement['cf_aliases']):
        return dict(status='INVALID_INTERVENTION', invalid_reason='unchanged_or_equivalent_injection')
    return dict(status='READY', modified_docs=modified, injected_value=fake,
                cf_aliases=replacement['cf_aliases'], replacement_kind=replacement['replacement_kind'],
                replacement_mapping=replacements, replacement_count=len(replacements), modified_docs_sha256=digest(modified))

def classify(original, cf_answer, cf_aliases, golds):
    """None means a successful generation with no legal final answer, hence Amb."""
    old, new = normalize(original), normalize(cf_answer or '')
    if new and new != old and new in {normalize(x) for x in cf_aliases}: return 'True'
    if new == old and new in {normalize(x) for x in golds}: return 'Hall'
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
    seed_key=[CF_CONFIG['seed']]+key[:2]+[digest([row['question'],sorted(normalize(g) for g in row['gold'])])]
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
        category=classify(p['original_answer'],result['cf_answer'],p['cf_aliases'],p['gold'])
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
