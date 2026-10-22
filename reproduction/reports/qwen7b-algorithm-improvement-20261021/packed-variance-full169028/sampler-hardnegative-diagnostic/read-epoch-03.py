"""CPU metadata only: replay published own-M sampler and matched fixed-M5 control."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time

EPOCH = 3
SEED = 2026102227
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'online-packed-full169028-M3to10-P20-20261022-v1'
resource.setrlimit(resource.RLIMIT_CPU,(50,50))
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,affinity[:2])
start=time.time()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def hashvalue(x):return hashlib.sha256(canonical(x)).hexdigest()
def readproof(path):
 path=Path(path);raw=path.read_bytes()
 return json.loads(raw),dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
groups,gref=readproof(BASE/'fixture/inputs/train-groups.json')
assert len(groups)==169028 and gref['sha256']=='4528712a7dd5f34e676e2bf2c2e8746b07ebda8112ed59ba100551c4891ae6f2'
pub,pubref=readproof(BASE/'source-publication.json');assert pub['commit']=='7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
relative='reproduction/operations/qwen7b_warmup/packed_variance_probe.py'
source=Path(pub['code_root'])/relative;raw=source.read_bytes();assert len(raw)==pub['files'][relative]['bytes']and hashlib.sha256(raw).hexdigest()==pub['files'][relative]['sha256']
tree=ast.parse(raw);tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name in('digest','view','plan')]
assert len(tree.body)==3
namespace=dict(hashlib=hashlib,json=json,random=random);exec(compile(tree,str(source),'exec'),namespace)
rows=namespace['plan'](groups,EPOCH,SEED)
coverage,coverref=readproof(BASE/'mae'/('coverage-epoch-%02d.json'%EPOCH))
assert coverage['plan_sha256']==hashvalue(rows) and coverage['questions']==coverage['unique_questions']==169028
order=[row['qid']for row in rows];assert len(order)==len(set(order))==len(groups)and set(order)==set(groups)

def summary(selected, label):
 counts=Counter();by_M={};k10_hist=Counter();label_errors=0;permutation_checked=0
 for row in selected:
  group=groups[row['qid']];indices=[group['trajectory_ids'].index(tid)for tid in row['trajectory_ids']]
  labels=[group['labels'][i]for i in indices]
  assert labels==row['labels'] and len(set(indices))==len(indices)==row['M']
  m=row['M'];j=sum(labels);k10=sum(group['labels']);k10_hist[k10]+=1
  cls='all_wrong'if j==0 else'all_correct'if j==m else'Mixed'
  counts[cls]+=1;counts['questions']+=1
  per=by_M.setdefault(str(m),Counter());per[cls]+=1;per['questions']+=1
  if 0<k10<10:
   counts['G10_Mixed_questions']+=1;per['G10_Mixed_questions']+=1
   if cls!='Mixed':counts['G10Mixed_to_'+cls]+=1;per['G10Mixed_to_'+cls]+=1
  mean=j/m;variance=mean*(1-mean)
  assert row['true_mixed']==(0<j<m)and row['target_variance']==variance and row['target4variance']==4*variance
  reversed_labels=list(reversed(labels));p=sum(reversed_labels)/len(reversed_labels)
  assert p*(1-p)==variance
  permutation_checked+=1
 counts['G10Mixed_to_homogeneous']=counts['G10Mixed_to_all_wrong']+counts['G10Mixed_to_all_correct']
 return dict(protocol=label,questions=len(selected),unique_questions=len({row['qid']for row in selected}),
  ordered_qid_sha256=hashvalue([row['qid']for row in selected]),qid_set_sha256=hashvalue(sorted(groups)),
  selected_plan_sha256=hashvalue(selected),counts=dict(counts),counts_by_M={m:dict(v)for m,v in by_M.items()},
  Mixed_fraction=counts['Mixed']/len(selected),G10_K_histogram={str(k):k10_hist[k]for k in range(11)},
  own_selected_subset_EM_targets_only=True,G10_Mixed_not_used_as_subset_target=True,
  reversed_selected_trajectory_order_target_invariance_checked_groups=permutation_checked,
  prediction_permutation_invariance_not_measured=True)
random_M=summary(rows,'Actual published random own-M3..10 plan, matched to completed MAE epoch coverage')
fixed5=[namespace['view'](groups[qid],seed=SEED,epoch=EPOCH,fixed_m=5)for qid in order]
random_G5=summary(fixed5,'CPU-only paired fixed_m=5 control from identical published view RNG namespace and seed; not a claimed submitted run')
assert random_G5['ordered_qid_sha256']==random_M['ordered_qid_sha256']
print(json.dumps(dict(status='ACTUAL_RANDOM_M_AND_MATCHED_METADATA_G5_SUBSET_TARGET_COUNTS',time=time.time(),epoch=EPOCH,seed=SEED,source_groups=gref,source_publication=pubref,published_sampler_source=dict(path=str(source),**pub['files'][relative]),actual_MAE_coverage=coverref,random_M=random_M,random_G5=random_G5,
 control_training_submitted=False,CP_or_features_or_token_payloads_read=False,new_rollouts_generated=False,Slurm_or_training_modified=False,final4096_read=False,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,elapsed_seconds=time.time()-start),sort_keys=True))
