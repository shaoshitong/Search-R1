import pathlib,json,re,hashlib,time
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020');p=root/'normal/normal.jsonl'
rows=[json.loads(x) for x in p.read_text().splitlines()];markers=[sum(len(re.findall(r'<search>.*?</search>',s,re.S)) for s in r['model_spans']) for r in rows];calls=[r['actual_retrieval_calls'] for r in rows]
assert len(rows)==3610 and sum(r['original_native_em'] for r in rows)==1706
report=dict(NQ_questions=3610,NQ_correct=1706,NQ_EM=1706/3610,Search_Q=sum(markers)/3610,parsed_response_search_markers=sum(markers),actual_retrieval_calls=sum(calls),actual_retrieval_calls_mean=sum(calls)/3610,all_native_counts_equal_parsed=markers==calls,normal_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),checked_unix=time.time())
f=root/'normal/scalar-summary.json'
if not f.exists():f.write_text(json.dumps(report))
print(json.dumps(report))
