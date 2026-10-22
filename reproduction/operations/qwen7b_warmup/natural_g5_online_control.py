"""Paired natural-random own-G5 control over the unchanged8192 online pilot.

The existing online prefill, native block/head, optimizer, DDP and evaluation
math are reused verbatim. Only metadata plans differ: label-blind random5of10,
then authentic Mixed targets and whole-epoch inverse-frequency BCE weights.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
SOURCE=A/'online-mixed-layer-prefix-20261022-v1'
SUITE=A/'online-natural-g5-8192-20261022-v1'
FIXTURE=SUITE/'fixture'
SEED=2026102207
FIXED_SHA='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def proof(path):
 path=Path(path);raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 actual=proof(ref['path']);assert all(actual[k]==ref[k]for k in('bytes','sha256')),ref['path'];return Path(ref['path'])
def immutable(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);raw=canonical(value)+b'\n'
 if path.exists():assert path.read_bytes()==raw,path
 else:path.write_bytes(raw)


def natural_plan(groups,order,epoch):
 """RNG selection sees only qids and trajectory indices; labels are read later."""
 derived=int(digest(['paired-natural-own-G5',SEED,epoch]),16)
 rng=random.Random(derived)
 choices=[sorted(rng.sample(range(10),5))for _ in order]
 result=[];hard_wrong=hard_correct=0;g10_mixed=0
 for qid,indices in zip(order,choices):
  group=groups[qid];assert len(group['trajectory_ids'])==len(group['labels'])==10
  labels=[group['labels'][j]for j in indices]
  assert all(type(y)is int and y in(0,1)for y in group['labels'])
  k5=sum(labels);k10=sum(group['labels']);mixed=0<k5<5
  if 0<k10<10:
   g10_mixed+=1;hard_wrong+=k5==0;hard_correct+=k5==5
  result.append(dict(qid=qid,indices=indices,trajectory_ids=[group['trajectory_ids'][j]for j in indices],labels=labels,true_mixed=mixed))
 n=len(result);positive=sum(row['true_mixed']for row in result);negative=n-positive
 assert n==8192 and positive and negative
 stats=dict(epoch=epoch,seed=SEED,derived_seed_sha256=hashlib.sha256(str(derived).encode()).hexdigest(),
  questions=n,unique_questions=len(set(order)),Mixed=positive,nonMixed=negative,
  label_blind_G5_selection=True,own_trajectory_indices_only=True,hard_replay=False,
  G10_Mixed_questions=g10_mixed,G10Mixed_to_G5_homogeneous=hard_wrong+hard_correct,
  G10Mixed_to_G5_all_wrong=hard_wrong,G10Mixed_to_G5_all_correct=hard_correct,
  epoch_BCE_class_weights=dict(Mixed=n/(2*positive),nonMixed=n/(2*negative)),
  plan_sha256=digest(result),ordered_qids_sha256=digest(order))
 return result,stats


def prepare():
 FIXTURE.mkdir(parents=True,exist_ok=True)
 with(SUITE/'fixture-prepare.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  done=FIXTURE/'fixture-complete.json'
  if done.exists():return preflight()
  source=read(SOURCE/'fixture-complete.json')
  assert source['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'
  assert source['training_questions']==8192 and source['validation_questions']==2048
  assert source['fixed_G5_sha256']==FIXED_SHA and len(source['token_shards'])==4
  groups=read(SOURCE/'inputs/train-groups.json');assert len(groups)==8192
  inputs=FIXTURE/'inputs';inputs.mkdir(exist_ok=True);shared={}
  for name in('train-groups.json','validation-groups.json','fixed-train2048.json'):
   target=SOURCE/'inputs'/name;shared[name]=proof(target);dest=inputs/name
   if dest.is_symlink():assert dest.resolve()==target
   else:assert not dest.exists();dest.symlink_to(target)
  plans=[];previous=None
  for epoch in(1,2,3):
   original=SOURCE/'inputs'/('epoch-%02d-plan.json'%epoch);old=read(original);order=[row['qid']for row in old]
   assert len(order)==len(set(order))==8192 and set(order)==set(groups)
   plan,stats=natural_plan(groups,order,epoch)
   assert[row['qid']for row in plan]==order
   old_by={row['qid']:row for row in old}
   stats['same_G5_as_original_plan']=sum(set(row['trajectory_ids'])==set(old_by[row['qid']]['trajectory_ids'])for row in plan)
   stats['same_G5_as_previous_new_epoch']=sum(set(row['trajectory_ids'])==set(previous[row['qid']])for row in plan)if previous else None
   previous={row['qid']:row['trajectory_ids']for row in plan}
   path=inputs/('epoch-%02d-plan.json'%epoch);immutable(path,plan)
   stats.update(source_original_plan=proof(original),question_order_exactly_original=True,plan=proof(path))
   plans.append(stats)
  fixture=dict(status='NATURAL_RANDOM_OWN_G5_PAIRED8192_FIXTURE_READY',source_fixture=proof(SOURCE/'fixture-complete.json'),
   training_questions=8192,training_trajectories=81920,validation_questions=2048,validation_trajectories=10240,
   fixed_G5_sha256=source['fixed_G5_sha256'],fixed_train_plan_sha256=source['fixed_train_plan_sha256'],
   token_shards=source['token_shards'],shared_small_inputs=shared,plans=plans,seed=SEED,
   online_math_source='unchanged online_mixed_prefix_explore.run',sampling_intervention='label-blind uniform own5of10 then real EM target',
   original_8192_qids_and_three_epoch_orders_preserved=True,all_layers_same_new_plans=True,
   no_rollout_regeneration=True,no_new_token_shards=True,no_persisted_hidden_features=True,final4096_read=False,
   total_optimizer_updates=192,epochs=3,global_batch_questions=128,world_size=2,micro_questions=8,accumulation=8,
   fresh_official_block_and_head=True,learning_rate=5e-6,EMA_decay=.99)
  immutable(done,fixture)
  receipt=dict(status='NATURAL_RANDOM_G5_PAIRED_CONTROL_PREPARED_NOT_SUBMITTED',fixture=proof(done),plans=plans,
   training_questions=8192,validation_questions=2048,all_layers_share_identical_plans=True,
   new_token_or_hidden_or_rollout_artifacts=False,final4096_read=False,Slurm_submitted=False,time=time.time())
  immutable(SUITE/'prepared.json',receipt)
  return receipt


def preflight():
 receipt=read(SUITE/'prepared.json');checked(receipt['fixture']);fixture=read(receipt['fixture']['path'])
 assert fixture['status']=='NATURAL_RANDOM_OWN_G5_PAIRED8192_FIXTURE_READY'and fixture['seed']==SEED
 checked(fixture['source_fixture'])
 for ref in fixture['shared_small_inputs'].values():checked(ref)
 for item in fixture['plans']:checked(item['plan']);checked(item['source_original_plan'])
 return receipt


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train'),required=True)
 p.add_argument('--layer',type=int,choices=(21,27),default=21);p.add_argument('--prefix',type=int,choices=(20,),default=20)
 p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=5e-6);args=p.parse_args()
 if args.action=='prepare':print(json.dumps(prepare()),flush=True);return
 assert args.epochs==3 and args.lr==5e-6
 preflight()
 from . import online_mixed_prefix_explore as implementation
 implementation.SUITE=SUITE;implementation.FIXTURE_SUITE=FIXTURE
 base=SUITE/('layer%d-prefix%d'%(args.layer,args.prefix));base.mkdir(parents=True,exist_ok=True)
 try:
  implementation.run(args)
  if os.environ.get('RANK')=='0':
   result=read(base/'complete.json');assert result['cursor']==192 and result['epochs']==3
 except BaseException as error:
  import traceback
  implementation.atomic(base/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time()))
  raise


if __name__=='__main__':main()
