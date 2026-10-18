"""CPU comparison of existing native features for equal-token Mixed G5 groups."""
import json
import subprocess
import time

from .control import ROOT, route


REMOTE=r'''
import collections,hashlib,json,pathlib,sys,time,numpy as np
h=json.load(sys.stdin)['handle'];b=pathlib.Path(h['base'])
def read(p):return json.loads(pathlib.Path(p).read_text())
latest=read(b/'probe/latest.json');n=latest['cycle'];bundles=[]
for name,meta in [('fixed_probe512',b/'fixed-test/complete.json'),('first_train512',b/'cycles/cycle-0001.json'),('latest_train512',b/'cycles'/('cycle-%04d.json'%n))]:
 bundles.append((name,read(meta)))
results=[]
for name,bundle in bundles:
 scan=pathlib.Path(bundle['scan_path']);manifest=read(scan/'scan-complete.json')
 assert hashlib.sha256((scan/'scan-complete.json').read_bytes()).hexdigest()==bundle['scan_sha256']
 raw=(scan/'index.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['files']['index.jsonl']['sha256']
 rows=[json.loads(line) for line in raw.splitlines()];groups=collections.defaultdict(list)
 for row in rows:groups[str(row['qid'])].append(row)
 raw=pathlib.Path(bundle['own_em']).read_bytes();assert hashlib.sha256(raw).hexdigest()==bundle['own_em_sha256']
 labels={x['trajectory_id']:x['strict_em'] for x in map(json.loads,raw.splitlines())}
 selected=[]
 for siblings in groups.values():
  if len({r['policy_prefix_sha256'] for r in siblings})==1 and 0<sum(labels[r['trajectory_id']] for r in siblings)<5:
   selected.append(siblings)
  if len(selected)==16:break
 assert len(selected)==16
 arrays={rank:np.load(item['path'],mmap_mode='r',allow_pickle=False) for rank,item in enumerate(manifest['actor_hidden_shards'])}
 equal=0;unique=collections.Counter();maximum_abs=0.;maximum_relative_l2=0.;hashes=[];bytes_read=0
 for siblings in selected:
  features=[]
  for row in siblings:
   assert row['prefix_length']==5
   original=arrays[row['feature_rank']][row['feature_row']]
   qlen=row['initial_context']['request_length']
   active=np.concatenate((np.asarray(original[:qlen]),np.asarray(original[273:278]))).copy()
   assert active.dtype==np.uint16
   features.append(active);bytes_read+=active.nbytes
  ds=[hashlib.sha256(x.tobytes()).hexdigest() for x in features]
  count=len(set(ds));unique[count]+=1;equal+=count==1
  ref=(features[0].astype(np.uint32)<<16).view(np.float32)
  assert np.isfinite(ref).all()
  for item in features[1:]:
   arr=(item.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(arr).all()
   diff=arr-ref;maximum_abs=max(maximum_abs,float(np.max(np.abs(diff))))
   maximum_relative_l2=max(maximum_relative_l2,float(np.linalg.norm(diff.astype(np.float64))/max(np.linalg.norm(ref.astype(np.float64)),1e-30)))
  hashes.append(hashlib.sha256(''.join(ds).encode()).hexdigest())
 results.append(dict(scope=name,cycle=bundle['cycle'],actor_version=bundle['actor_version'],selection='first16 completeG5 trueMixed groups whose retained policy5 hashes match exactly',
  groups_compared=16,all5_valid_native_features_bit_identical=equal,unique_feature_rows_per_group=dict(unique),
  max_abs_BF16_feature_difference=maximum_abs,max_relative_L2_feature_difference=maximum_relative_l2,
  relevant_feature_bytes_read=bytes_read,feature_digest_aggregate=hashlib.sha256(''.join(hashes).encode()).hexdigest(),
  features_already_present=True,no_new_model_forward=True,raw_tensor_returned=False))
 arrays.clear()
print(json.dumps(dict(role='existing_native_prefix_feature_identity_CPU_diagnostic',checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],latest_complete_epochs=latest['completed_epochs'],rows=results,no_GPU_queries=True,no_generation=True)))
'''


def main():
    h=json.loads((ROOT/'preparation/actor50-probe100-resume-deployment-20261018.json').read_text())
    result=subprocess.run(route(REMOTE),input=json.dumps(dict(handle=h)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-probe100-prefix-feature-identity-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n')
    if result.returncode:raise RuntimeError('Inspect feature audit receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
