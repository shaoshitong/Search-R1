"""Read-only intervention dry run; never reuse prior CF answers as v5 inference."""
import argparse
from collections import Counter
from pathlib import Path
from reproduction.grounding_eval.core_v5 import atomic, digest, intervene, occurrence, read, read_jsonl, sha, utc


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();selection=read(args.selection)
    wanted={tuple(r['occurrence_key']) for r in selection['rows']}
    if len(wanted)!=40:raise ValueError('This audit explicitly requires the frozen 40 selected occurrences')
    found={};source_hash=sha(args.source)
    for row in read_jsonl(args.source):
        k=tuple(occurrence(row))
        if k not in wanted:continue
        if k in found:raise ValueError('Duplicate source occurrence')
        key=[1729]+list(k[:2])+[digest([row['question'],sorted(row['gold'])])]
        result=intervene(row['seen_docs'],row['gold'],key,row['question'])
        found[k]=dict(occurrence_key=list(k),source_record_sha256=digest(row),question=row['question'],
            raw_gold_aliases=row['gold'],status=result['status'],invalid_reason=result.get('invalid_reason'),
            replacement_kind=result.get('replacement_kind'),replacement_count=result.get('replacement_count',0),
            actual_mapping=result.get('replacement_mapping',[]),cf_aliases=result.get('cf_aliases',[]),
            original_semantic_aliases=result.get('original_semantic_aliases',[]),
            modified_docs_sha256=result.get('modified_docs_sha256'))
    if set(found)!=wanted:raise ValueError('Missing audited source occurrence')
    rows=sorted(found.values(),key=lambda r:r['occurrence_key'][1]);counts=Counter(r['status'] for r in rows)
    reasons=Counter(r['invalid_reason'] for r in rows if r['status']!='READY')
    report=dict(protocol_version='strict-v5-conservative-typed',utc=utc(),source_sha256=source_hash,
        selection_sha256=sha(args.selection),sample_count=40,status_counts=dict(counts),excluded_reasons=dict(reasons),
        excluded_fraction=(40-counts['READY'])/40, sampling='Frozen stratified v4-Amb diagnostic sample; not representative of NQ prevalence',
        inference_ran=False,prior_cf_answers_reclassified=False,raw_docs_copied=False,rows=rows)
    atomic(Path(args.output),report)
    print(dict(sample_count=40,status_counts=dict(counts),excluded_reasons=dict(reasons),excluded_fraction=report['excluded_fraction']))


if __name__=='__main__':main()
