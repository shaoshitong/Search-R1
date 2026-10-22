"""Verify every canonical own-G10 ID from accepted small text metadata only."""
import hashlib,json,time
from pathlib import Path
start=time.time();A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');F=A/'online-packed-full169028-M3to10-P20-20261022-v1/fixture';d=json.loads((F/'full-training-descriptor.json').read_bytes());ref=d['groups'];p=Path(ref['path']);raw=p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];x=json.loads(raw);del raw;rows=x['train']if isinstance(x,dict)and'train'in x else x
if isinstance(rows,dict):rows=list(rows.values())
qids=[g['qid']for g in rows];ids=set();bad=0
for g in rows:
 bad+=len(g['trajectory_ids'])!=10 or len(g['labels'])!=10 or len(set(g['trajectory_ids']))!=10 or any(type(y)is not int or y not in(0,1)for y in g['labels']);ids.update(g['trajectory_ids'])
assert len(qids)==len(set(qids))==169028 and len(ids)==1690280 and not bad
ordered=hashlib.sha256(json.dumps(qids,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();assert ordered==d['canonical_ordered_qid_sha256']
print(json.dumps(dict(status='ALL_CANONICAL169028_OWN_G10_IDENTITIES_FRESH_TEXT_SHA_VERIFIED',time=time.time(),seconds=time.time()-start,groups=ref,questions=169028,unique_qids=169028,unique_own_trajectory_ids=1690280,all_questions_have10_distinct_own_trajectories=True,all_labels_real_integer_binary=True,canonical_ordered_qid_sha256=ordered,full_payloads_opened=False,fullP80_rows_not_yet_extracted_or_accepted=True,final4096_read=False,checkpoint_or_hidden_tensors_read=False)))
