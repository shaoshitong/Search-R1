"""One terminal CPU/state/ledger audit. Does not generate or evaluate new data."""
import argparse
import hashlib
import json
import subprocess
import time

from .control import ROOT, route


REMOTE = r'''
import hashlib,json,os,pathlib,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base'])
assert b.resolve()==b and b.name.endswith('-resume-'+h['execution_id'])
def read(path):
 path=pathlib.Path(path)
 assert path.resolve()==path and not path.is_symlink() and path.stat().st_size<=64*1024**2
 return json.loads(path.read_text())
def sha(path):
 out=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):out.update(block)
 return out.hexdigest()
def who(pid):
 try:
  q=pathlib.Path('/proc')/str(pid);raw=(q/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  return dict(pid=pid,sid=int(v[3]),uid=q.stat().st_uid,start_ticks=int(v[19]))
 except (FileNotFoundError,ProcessLookupError):return None
assert read(b/'preparation/deployment.json')==h
c=read(b/'preparation/launch-config.json');end=read(b/'complete.json');latest=read(b/'probe/latest.json')
assert c['actor_training_enabled'] is False and c['target_probe_epochs']==100 and c['cycles']==50
assert c['actor_version']=='54ac33dcebe64e62a11fe4177f41f116-actor0050'
assert end['run_id']==h['run_id'] and end['completed_epochs']==100 and end['complete_cycles']==50 and end['actor_updates']==0
assert latest==end['latest_probe'] and latest['completed_epochs']==100 and latest['cycle']==50
exit=read(b/'logs/exit.json');cleanup=read(b/'logs/cleanup-status.json')
assert exit['success'] is True and exit['cycle']==50
assert cleanup['cleaned'] and cleanup['owned_births_all_gone'] and cleanup['remaining_owned_birth_count']==0
assert not (b/'logs/failure.json').exists()
leader=read(b/'preparation/submit-receipt.json')['leader'];actual=who(leader['pid'])
assert actual is None or any(actual[k]!=leader[k] for k in ('pid','sid','uid','start_ticks'))
assert sha(b/'preparation/launch-config.json')==read(b/'preparation/source-receipt.json')['config_sha256']
assert sha(c['joint50_path'])==c['joint50_sha256']
fixed=read(b/'fixed-test/complete.json')
assert fixed['fixed_test_identity_sha256']=='1ebb2d88013ef335275ca60d86e423fd75e33a7149136525a92f1f2308a2dbbd'
assert fixed['actor_version']==c['actor_version'] and fixed['questions']==512 and fixed['trajectories']==2560 and fixed['strict_EM_binary']
curve_path=b/'metrics.jsonl';assert curve_path.stat().st_size<4*1024**2
curve=[json.loads(line) for line in curve_path.read_text().splitlines() if line]
assert len(curve)==50
prior=0;previous_test=None;points=[];checkpoints=[]
metric_keys=('accuracy','balanced_accuracy','mixed_recall','nonmixed_recall','mean_q','total_loss','trajectory_accuracy','tp','fp','tn','fn')
for n,m in enumerate(curve,1):
 complete=read(b/'training'/('cycle-%04d'%n)/'complete.json')
 bundle=read(b/'cycles'/('cycle-%04d.json'%n))
 assert m==complete['metrics'] and m['cycle']==n and m['completed_epochs']==2*n
 assert complete['run_id']==h['run_id'] and m['actor_version']==c['actor_version'] and m['actor_updates']==0
 assert bundle['actor_version']==c['actor_version'] and bundle['actor_updates']==0 and bundle['strict_EM_binary']
 assert bundle['questions']==512 and bundle['trajectories']==2560 and bundle['training_questions_used']
 assert m['test_events_fixed_across_all_cycles'] and m['train_eval_on_original_unsampled512']
 assert m['train_questions']==m['test_questions']==512 and m['weight_kind']=='EMA0.99'
 plans=m['epoch_sampling'];assert [x['epoch'] for x in plans]==[2*n-1,2*n]
 for plan in plans:
  original=plan['original_class_counts'];balanced=plan['balanced_class_counts'];maximum=max(original.values())
  assert sum(original.values())==512 and min(original.values())>0
  assert balanced==dict(all_correct=maximum,all_wrong=maximum,mixed=maximum)
  assert plan['balanced_groups']==3*maximum and plan['balanced_trajectories']==15*maximum
 steps=prior+sum(x['optimizer_updates'] for x in plans)
 assert m['new_adam_updates']==m['ema_updates']==complete['ema_updates']==steps
 assert complete['actual_adam_steps_per_rank']==[steps]*4
 cp=pathlib.Path(complete['checkpoint']);assert cp.resolve()==cp and cp.is_file() and not cp.is_symlink() and cp.stat().st_size==complete['bytes']
 updatepath=cp.parents[1]/'training'/('cycle-%04d'%n)/'updates.jsonl'
 assert updatepath.stat().st_size<4*1024**2
 updates=[json.loads(line) for line in updatepath.read_text().splitlines() if line]
 assert len(updates)==steps-prior
 for clock,row in enumerate(updates,prior+1):
  assert row['cycle']==n and row['new_adam_updates']==row['ema_updates']==clock
  assert row['learning_rate']==5e-6 and row['actor_updates']==0
  assert row['global_trajectories']==5*row['global_groups']
 if previous_test is not None:assert m['before']['test']==previous_test
 for moment in ('before','after'):
  for split in ('train','test'):
   item=m[moment][split];assert item['count']==512 and not item['resampled_evaluation']
   assert item['tp']+item['fn']==item['mixed_count'] and item['tn']+item['fp']==item['nonmixed_count']
   assert item['weight_kind']=='EMA0.99' and item['threshold']==.5
   assert abs(item['balanced_accuracy']-.5*(item['mixed_recall']+item['nonmixed_recall']))<1e-12
  assert m[moment]['test']['mixed_count']==114 and m[moment]['test']['nonmixed_count']==398
 previous_test=m['after']['test'];prior=steps
 compact={moment:{split:{k:m[moment][split][k] for k in metric_keys} for split in ('train','test')} for moment in ('before','after')}
 points.append(dict(cycle=n,epoch=2*n,Adam=steps,EMA=steps,**compact,original_class_counts=plans[0]['original_class_counts'],balanced_each_class=max(plans[0]['original_class_counts'].values()),steps_per_epoch=[x['optimizer_updates'] for x in plans],checkpoint_sha256=complete['sha256']))
 checkpoints.append(dict(cycle=n,path=str(cp),bytes=cp.stat().st_size,sha256=complete['sha256']))
assert latest['ema_updates']==prior
cp=pathlib.Path(latest['checkpoint']);assert sha(cp)==latest['sha256']
import torch
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.prefix5_model import validate_model_state,validate_native_buffers
state=torch.load(cp,map_location='cpu',weights_only=False)
assert state['run_id']==h['run_id'] and state['cycle']==50 and state['completed_epochs']==100 and state['actor_updates']==0
assert state['source_probe50_sha256']==c['probe50_sha256'] and state['new_adam_updates']==state['ema_updates']==prior
validate_model_state(state['raw_model']);validate_model_state(state['ema_model'])
validate_native_buffers(state['raw_buffers']);validate_native_buffers(state['ema_buffers'])
assert len(state['optimizer']['state'])==14 and len(state['rank_rng'])==4
assert [x['rank'] for x in state['rank_rng']]==list(range(4))
for item in state['optimizer']['state'].values():
 assert int(item['step'])==prior
 for value in item.values():
  if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
assert all(x['lr']==5e-6 and x['weight_decay']==0 for x in state['optimizer']['param_groups'])
for row in state['rank_rng']:
 assert row['torch_cpu'].dtype==torch.uint8 and row['torch_cpu'].numel()>0
 assert row['torch_cuda'].dtype==torch.uint8 and row['torch_cuda'].numel()>0
 assert row['python'] and row['numpy']
assert state['metrics']==curve[-1]
value=dict(role='actual_terminal_frozen_actor50_probe100_acceptance',checked_unix=time.time(),run_id=h['run_id'],execution_id=h['execution_id'],runtime_source_commit=h['source_commit'],
 complete_cycles=50,complete_epochs=100,actor_updates=0,actor_version=c['actor_version'],actual_Adam_per_rank=[prior]*4,EMA_updates=prior,
 all_cycles_balance_and_real_update_ledgers_verified=True,all_pre_post_fixed_test_metrics_continuous=True,
 final_raw_EMA_Adam_native_buffers_finite=True,all_four_rank_RNG_present=True,final_checkpoint=str(cp),final_checkpoint_sha256=latest['sha256'],
 source50_joint_sha256=c['joint50_sha256'],source50_probe_sha256=c['probe50_sha256'],source50_HF_initial_full_SHA_qualification=end['source50_qualification'],
 fixed_probe512_identity_sha256=fixed['fixed_test_identity_sha256'],fixed_test_bundle_sha256=sha(b/'fixed-test/complete.json'),fixed_test_events_reused=True,
 metadata_curve_sha256=sha(curve_path),initial_test=curve[0]['before']['test'],final_test=curve[-1]['after']['test'],curve=points,checkpoint_inventory=checkpoints,
 cleanup=cleanup,exit=exit,owned_leader_gone=True,new_evaluations_run=False)
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024
print(raw)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--handle', required=True)
    a = p.parse_args()
    from pathlib import Path
    handle = json.loads(Path(a.handle).read_text())
    payload = json.dumps(dict(handle=handle)).encode()
    result = subprocess.run(route(REMOTE), input=payload, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=65)
    receipt = dict(checked_unix=time.time(),exit_code=result.returncode,
                   stdout=result.stdout.decode(),stderr=result.stderr.decode(),
                   run_id=handle['run_id'],execution_id=handle['execution_id'])
    out = ROOT/'preparation'/('actor50-probe100-final-audit-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode or len(result.stdout)>128*1024:
        raise RuntimeError('Inspect final audit receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__ == '__main__':
    main()
