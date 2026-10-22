import gzip,hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2');A=B.parent
IDS={'70188b8e50fd4b45bc420ce58014f46e-outer000001-c107521-g3':1,'70188b8e50fd4b45bc420ce58014f46e-outer000001-c107529-g3':1,'70188b8e50fd4b45bc420ce58014f46e-outer000001-c107543-g2':0}
def read(p):return json.loads(Path(p).read_bytes())
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
seal=A/'hpc3-tail-newseedG5-20261021-v1/sampling/shard-01/batches/batch-000420/complete.json';s=read(seal);ref=s['full']['full'];assert proof(ref['path'])=={k:ref[k]for k in('path','bytes','sha256')};raw={}
with gzip.open(ref['path'],'rt')as f:
 for line in f:
  r=json.loads(line);t=r['trajectory_id']
  if t in IDS:
   p=[]
   for turn in r['behavior_turns']:p.extend(turn['raw_generated_token_ids'][:turn['kept_token_count']])
   raw[t]=dict(qid=r['qid'],trajectory_id=t,prompt=r['initial_prompt_token_ids'],policy_token_ids=p[:80],role='train');assert r['strict_em']==IDS[t]
found={}
for receipt in sorted((B/'chunk-receipts').glob('chunk-*.json')):
 for ref2 in read(receipt)['token_shards']:
  with gzip.open(ref2['path'],'rt')as f:
   for line in f:
    r=json.loads(line);t=r['trajectory_id']
    if t in IDS:
     assert t not in found;found[t]=dict(qid=r['qid'],trajectory_id=t,actual_compact_SHA256=sha(r),chosen_RAW_compact_SHA256=sha(raw[t]),prompt_and_all_first80_exact=r==raw[t],expected_true_EM=IDS[t],chunk_receipt=proof(receipt),token_shard=ref2)
 if len(found)==3:break
assert len(found)==3 and all(r['prompt_and_all_first80_exact']for r in found.values())
print(json.dumps(dict(time=time.time(),status='ACTUAL_V2_THREE_FAILED_CASES_USE_CHOSEN_PHYSICAL_P80_EXACTLY',cases=list(found.values()),chosen_physical_seal=proof(seal),chosen_full=ref,execution_run_id=s['execution_run_id'],full_responses_or_token_ids_returned=False,canonical_labels_changed=False,final4096_read=False)))
