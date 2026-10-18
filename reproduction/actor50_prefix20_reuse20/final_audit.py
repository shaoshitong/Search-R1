"""Terminal matched20 state/feature/ledger validation; no fresh evaluation."""
import json
import subprocess
import time

from reproduction.actor50_probe100.control import ROOT,route


REMOTE=r'''
import hashlib,json,pathlib,sys,time,torch
h=json.load(sys.stdin)['handle'];b=pathlib.Path(h['base'])
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):
 out=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):out.update(block)
 return out.hexdigest()
assert read(b/'preparation/deployment.json')==h
c=read(b/'preparation/config.json');end=read(b/'complete.json');last=read(b/'probe/latest.json')
assert end['completed_epochs']==20 and end['complete_cycles']==10 and end['actor_updates']==end['new_rollouts']==0
assert end['latest_probe']==last and last['completed_epochs']==20 and last['ema_updates']==276
assert read(b/'logs/exit.json')['success'] and not (b/'logs/failure.json').exists()
cleanup=read(b/'logs/cleanup-status.json');assert cleanup['cleaned'] and cleanup['owned_births_all_gone'] and cleanup['remaining_owned_birth_count']==0
assert sha(b/'preparation/config.json')==read(b/'preparation/source-receipt.json')['config_sha256']
data=read(b/'preparation/data-prepared.json');assert data['total_trajectories']==28160 and data['training_test_disjoint'] and data['new_rollouts']==0
feature_count=0;rep={};feature_digest=[]
for n in range(11):
 name='test' if n==0 else 'cycle-%04d'%n;root=b/'data'/name;meta=read(root/'prepared.json')
 assert sha(root/'index.jsonl')==meta['index_sha256'] and meta['original_strict_EM_reused']
 shards=read(root/'features-complete.json')['shards'];assert len(shards)==4
 for rank,item in enumerate(shards):
  assert item==read(root/('rank-%d-complete.json'%rank))
  assert item['rank']==rank and item['rows']==640 and item['width']==293 and item['dim']==2048
  assert item['preserved_question_and_first5_bitwise'] and item['new_sampled_tokens']==item['new_rollouts']==0
  assert not item['gold_or_final_outcomes_passed_to_actor']
  assert pathlib.Path(item['path']).stat().st_size==item['bytes'];feature_digest.append(item['sha256']);feature_count+=1
 rep[name]=meta['repetition']
curve=[json.loads(line) for line in (b/'metrics.jsonl').read_text().splitlines() if line];assert len(curve)==10
source=pathlib.Path(c['source_prefix5_base']);baseline=[];previous=None;clock=0;points=[]
for n,m in enumerate(curve,1):
 complete=read(b/'training'/('cycle-%04d'%n)/'complete.json')
 old=read(source/'training'/('cycle-%04d'%n)/'complete.json');baseline.append(old['metrics'])
 assert complete['metrics']==m and m['completed_epochs']==2*n and m['actor_updates']==0
 assert m['epoch_sampling']==old['metrics']['epoch_sampling']
 count=sum(row['optimizer_updates'] for row in m['epoch_sampling']);clock+=count
 assert complete['actual_adam_steps_per_rank']==[clock]*4 and m['ema_updates']==m['new_adam_updates']==clock
 updates=[json.loads(line) for line in (b/'training'/('cycle-%04d'%n)/'updates.jsonl').read_text().splitlines() if line]
 assert len(updates)==count
 for step,row in enumerate(updates,clock-count+1):assert row['new_adam_updates']==row['ema_updates']==step and row['actor_updates']==0 and row['learning_rate']==5e-6
 if previous is not None:assert previous==m['before']['test']
 previous=m['after']['test']
 for moment in ('before','after'):
  assert m[moment]['test']['count']==512 and m[moment]['test']['mixed_count']==114
  assert m[moment]['test']['nonmixed_count']==398 and not m[moment]['test']['resampled_evaluation']
 points.append(dict(epoch=2*n,Adam_EMA=clock,prefix20_before=m['before'],prefix20_after=m['after'],prefix5_before=old['metrics']['before'],prefix5_after=old['metrics']['after'],checkpoint_sha256=complete['sha256']))
assert clock==276 and sha(last['checkpoint'])==last['sha256']
sys.path.insert(0,str(b/'repo'))
from reproduction.actor50_prefix20_reuse20.model import validate_model_state
from reproduction.prefix_selection200.prefix5_model import validate_native_buffers
state=torch.load(last['checkpoint'],map_location='cpu',weights_only=False)
assert state['completed_epochs']==20 and state['prefix_policy_tokens']==20 and state['new_adam_updates']==state['ema_updates']==276
assert state['actor_updates']==0 and state['source_probe50_sha256']==c['probe50_sha256'] and state['metrics']==curve[-1]
validate_model_state(state['raw_model']);validate_model_state(state['ema_model'])
validate_native_buffers(state['raw_buffers']);validate_native_buffers(state['ema_buffers'])
assert len(state['optimizer']['state'])==14 and len(state['rank_rng'])==4 and [x['rank'] for x in state['rank_rng']]==list(range(4))
for item in state['optimizer']['state'].values():
 assert int(item['step'])==276
 for value in item.values():
  if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
assert all(x['lr']==5e-6 and x['weight_decay']==0 for x in state['optimizer']['param_groups'])
raw=json.dumps(dict(role='actual_prefix20_matched20_final_acceptance',checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
 complete_epochs=20,complete_cycles=10,actual_Adam_per_rank=[276]*4,EMA_updates=276,actor_updates=0,new_rollouts=0,
 all10_sampling_plans_equal_prefix5=True,all_real_update_ledgers_contiguous=True,all_fixed_test_pre_post_metrics_continuous=True,
 exact_preserved_question_first5_feature_shards=feature_count,feature_sha_aggregate=hashlib.sha256(''.join(feature_digest).encode()).hexdigest(),
 final_raw_EMA_Adam_native_buffers_finite=True,all_four_rank_RNG_present=True,final_checkpoint=last['checkpoint'],final_checkpoint_sha256=last['sha256'],
 prefix5_initial_test=baseline[0]['before']['test'],prefix20_initial_test=curve[0]['before']['test'],prefix5_final_test=baseline[-1]['after']['test'],prefix20_final_test=curve[-1]['after']['test'],
 curve=points,repetition=rep,cleanup=cleanup,exit=read(b/'logs/exit.json'),no_new_evaluations=True),allow_nan=False)
assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    h=json.loads((ROOT/'preparation/actor50-prefix20-reuse20-deployment-20261018.json').read_text())
    r=subprocess.run(route(REMOTE),input=json.dumps(dict(handle=h)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-prefix20-final-audit-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect audit receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
