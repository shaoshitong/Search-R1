import sys,pathlib,json,datetime,subprocess,hashlib
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016');sys.path.insert(0,str(repo))
from reproduction.prefix_only_correctness_ab.control import a800_route
work=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work');m=json.loads((work/'prefix10-20-a800-reference-source-metadata.json').read_text());teacher=str(pathlib.Path(m['teacher']['identity.json']['path']).parent);test=str(pathlib.Path(m['test']['teacher-cache/labels.json']['path']).parent);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
code=r'''
import json,pathlib,collections,datetime
import torch
teacher=pathlib.Path('__TEACHER__');test=pathlib.Path('__TEST__')
complete=json.loads((teacher/'complete.json').read_text());descs=[]
for e in complete['chunks']:
 c=json.loads((teacher/e['relative_path']).read_text());descs.append(('train',teacher/c['raw_rollout']['relative_path']))
descs.append(('test',test/'rollout.pt'))
out=dict(role='first_native_turn_raw_vs_actual_kept_boundary_audit',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_inference=True,no_raw_tokens_exported=True,no_repeated_full_sha=True,splits={})
for split,p in descs:
 raw=torch.load(p,map_location='cpu',weights_only=False);turns=raw['output'].non_tensor_batch['behavior_turns'];tids=raw['output'].non_tensor_batch['behavior_trajectory_id']
 s=out['splits'].setdefault(split,dict(trajectories=0,raw_gt_kept=0,kept_zero=0,raw_gt_kept_eos_only=0,kept_length_histogram={},raw_length_histogram={},limits={str(L):dict(raw_prefix_exceeds_actual_kept=0,discarded_tokens_in_raw_prefix=0,only_trailing_eos_exceeds=0,kept_short=0) for L in (10,20,40,80)}))
 for tid,ts in zip(tids,turns):
  t=ts[0];assert t['trajectory_id']==tid and t['turn_index']==0
  r=t['raw_generated_token_ids'];k=t['kept_token_count'];assert 0<=k<=len(r) and t['prefix_cut']['kept_token_ids']==r[:k]
  n=len(r);s['trajectories']+=1;s['raw_gt_kept']+=n>k;s['kept_zero']+=k==0
  eosonly=n==k+1 and r[-1]==t['eos_token_id'];s['raw_gt_kept_eos_only']+=eosonly
  for key,length in [('kept_length_histogram',k),('raw_length_histogram',n)]:s[key][str(length)]=s[key].get(str(length),0)+1
  for L in (10,20,40,80):
   v=s['limits'][str(L)];nr=min(L,n);nk=min(L,k);v['raw_prefix_exceeds_actual_kept']+=nr>nk;v['discarded_tokens_in_raw_prefix']+=nr-nk;v['only_trailing_eos_exceeds']+=nr>nk and eosonly;v['kept_short']+=k<L
 del raw
for s in out['splits'].values():
 for v in s['limits'].values():v['exceed_fraction']=v['raw_prefix_exceeds_actual_kept']/s['trajectories']
print(json.dumps(out))
'''.replace('__TEACHER__',teacher).replace('__TEST__',test)
wrapper="import subprocess,os,sys\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='/data/shaoshitong/Search-G1-a800-fresh200-16034e59031e4f21a323ebb2556bef7f/repo')\nr=subprocess.run(['/data/search-g1/env/bin/python','-'],input="+repr(code.encode())+",capture_output=True,env=env,timeout=210)\nsys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)\n"
r=subprocess.run(a800_route(wrapper),capture_output=True,text=True,timeout=240)
e=work/('prefix-only-kept-boundary-transport-'+tag+'.json');e.write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2))
if r.returncode:raise RuntimeError(r.stderr)
s=json.loads(r.stdout);p=work/('prefix-only-kept-boundary-summary-'+tag+'.json');p.write_text(json.dumps(s,indent=2));print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),summary=s)))
