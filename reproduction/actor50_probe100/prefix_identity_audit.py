"""Inspect existing Prefix5 identity metadata and strict labels; no generation."""
import json
import subprocess
import time

from .control import ROOT, route


REMOTE = r'''
import collections,hashlib,json,pathlib,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);limit=160*1024**2;used=0
def raw(path):
 global used
 path=pathlib.Path(path);assert path.resolve()==path and not path.is_symlink()
 size=path.stat().st_size;used+=size;assert used<=limit
 return path.read_bytes()
def read(path):return json.loads(raw(path))
def sha(value):return hashlib.sha256(value).hexdigest()
def signature(row):
 return (row['question_hash'],row['policy_prefix_sha256'],row['position_ids_sha256'],json.dumps(row['segment_bounds'],sort_keys=True),
         tuple((s['context_sha256'],s['context_length'],s['request_length']) for s in row['segments']))
def audit(name,scan_path,scan_sha,em_path=None,em_sha=None,scores=None):
 scan=pathlib.Path(scan_path);manifest_raw=raw(scan/'scan-complete.json');assert sha(manifest_raw)==scan_sha
 manifest=json.loads(manifest_raw);index_raw=raw(scan/'index.jsonl');assert sha(index_raw)==manifest['files']['index.jsonl']['sha256']
 rows=[json.loads(x) for x in index_raw.splitlines()];index={x['trajectory_id']:x for x in rows};assert len(index)==len(rows)==manifest['count']
 groups=collections.defaultdict(list)
 for row in rows:groups[str(row['qid'])].append(row)
 labels={};emhash=None
 if em_path:
  data=raw(em_path);emhash=sha(data);assert emhash==em_sha
  for line in data.splitlines():
   row=json.loads(line);tid=row['trajectory_id'];assert tid in index and tid not in labels and row['actor_version']==manifest['actor_version']
   assert type(row['strict_em']) is int and row['strict_em'] in (0,1) and row['prefix_sha256']==index[tid]['prefix_sha256']
   assert row.get('prefix_scan_sha256')==scan_sha
   labels[tid]=row['strict_em']
 dist=collections.Counter();classes={x:collections.Counter() for x in ('all_correct','all_wrong','mixed')}
 pair_equal=0;input_equal=0;short=0;labelled=0;score_count=0;pj_equal=0;scorehash=None
 score_by_qid={}
 if scores:
  data=raw(scores);scorehash=sha(data)
  for line in data.splitlines():
   row=json.loads(line);score_by_qid[str(row['qid'])]=row
 for qid,siblings in groups.items():
  assert len(siblings)==5 and {x['response_index'] for x in siblings}==set(range(5)) and len({x['question_hash'] for x in siblings})==1
  hashes=[x['policy_prefix_sha256'] for x in siblings];unique=len(set(hashes));dist[unique]+=1
  pair_equal+=sum(hashes[i]==hashes[j] for i in range(5) for j in range(i+1,5))
  same_input=len({signature(x) for x in siblings})==1;input_equal+=same_input
  short+=any(x['prefix_length']<5 for x in siblings)
  if all(x['trajectory_id'] in labels for x in siblings):
   labelled+=1;k=sum(labels[x['trajectory_id']] for x in siblings);cls='all_wrong' if k==0 else 'all_correct' if k==5 else 'mixed'
   classes[cls]['groups']+=1;classes[cls]['all5_prefix_equal']+=unique==1;classes[cls]['all5_input_metadata_equal']+=same_input
  if qid in score_by_qid:
   item=score_by_qid[qid];assert set(item['trajectory_ids'])=={x['trajectory_id'] for x in siblings}
   values=item['trajectory_probabilities'];assert len(values)==5
   score_count+=1;pj_equal+=max(values)-min(values)<1e-10
 return dict(scope=name,actor_version=manifest['actor_version'],questions=len(groups),trajectories=len(rows),
  unique_prefix_count_distribution=dict(dist),all5_prefix_equal=dist[1],all5_prefix_equal_fraction=dist[1]/len(groups),
  all5_input_metadata_equal=input_equal,pair_equal=pair_equal,pair_total=10*len(groups),groups_with_short_prefix=short,
  labelled_groups=labelled,by_true_class={k:dict(v) for k,v in classes.items()},
  saved_prediction_groups=score_count,saved_pj_all5_equal_at1e10=pj_equal,
  scan_manifest_sha256=scan_sha,index_sha256=sha(index_raw),own_EM_sha256=emhash,saved_score_sha256=scorehash,
  no_new_generation=True,no_model_forward=True,no_tensor_read=True)
latest=read(b/'probe/latest.json');n=latest['cycle'];output=[]
for cycle in range(1,n+1):
 bundle=read(b/'cycles'/('cycle-%04d.json'%cycle))
 output.append(audit('warmup_train_cycle%d_epochs%d-%d'%(cycle,2*cycle-1,2*cycle),bundle['scan_path'],bundle['scan_sha256'],bundle['own_em'],bundle['own_em_sha256']))
fixed=read(b/'fixed-test/complete.json')
output.append(audit('warmup_fixed_probe_test512',fixed['scan_path'],fixed['scan_sha256'],fixed['own_em'],fixed['own_em_sha256']))
formal=pathlib.Path(h['joint50_path']).parents[2]
for outer in (50,51):
 root=formal/'outers'/('outer-%06d'%outer);prefix=root/'prefix/complete.json'
 if not prefix.exists():continue
 meta=read(prefix);fullpath=next((root/sub/'complete.json' for sub in ('full-pool','full') if (root/sub/'complete.json').exists()),None)
 full=read(fullpath) if fullpath else None;score=root/'probe-score/candidate-scores.jsonl'
 output.append(audit('former_formal_outer%d_pre_actor_update'%outer,meta['scan_path'],meta['scan_sha256'],
  full['own_em']['path'] if full else None,full['own_em']['sha256'] if full else None,score if score.exists() else None))
value=dict(role='existing_prefix5_identity_diagnostic',checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],
 runtime_source_commit=h['source_commit'],latest_complete_epochs=latest['completed_epochs'],latest_EMA_updates=latest['ema_updates'],
 live_status=read(b/'logs/status.json'),current_failure=read(b/'logs/failure.json') if (b/'logs/failure.json').exists() else None,
 source50_checkpoint_sha256=h['joint50_sha256'],comparison_scope='within each question own five trajectories; not across questions',
 exact_token_identity='producer SHA256 of actual retained policy token IDs, verified index manifest SHA',
 input_metadata_signature='question_hash, policy_tokens_hash, positions_hash, segment bounds and causal context hashes',
 remote_metadata_bytes_read=used,raw_data_returned=False,rows=output)
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    hpath=ROOT/'preparation/actor50-probe100-resume-deployment-20261018.json'
    h=json.loads(hpath.read_text())
    result=subprocess.run(route(REMOTE),input=json.dumps(dict(handle=h)).encode(),
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-probe100-prefix-identity-'+str(time.time_ns())+'.json')
    receipt=dict(checked_unix=time.time(),exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:raise RuntimeError('Inspect identity audit receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
