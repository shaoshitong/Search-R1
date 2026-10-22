"""Independent CPU checkpoint audit for the paired8192 online natural-G5 runs.

Run on HPC3 after the corresponding owned job exits. No GPU, actor model,
rollout, feature or final-confirmation data is loaded. Reports actual raw/EMA,
Adam and rank-specific RNG contents; does not claim a GPU next-update replay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
SUITE=A/'online-natural-g5-8192-20261022-v1'
JOBS={21:'749459',27:'749460'}


def read(path):return json.loads(Path(path).read_text())
def proof(path):
 path=Path(path);h=hashlib.sha256()
 with path.open('rb')as stream:
  for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def checked(ref):
 actual=proof(ref['path']);assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256'],ref['path'];return Path(ref['path'])


def tensor_scan(tensor,torch):
 assert tensor.device.type=='cpu'
 h=hashlib.sha256();flat=tensor.detach().reshape(-1)
 for begin in range(0,flat.numel(),1024*1024):
  piece=flat[begin:begin+1024*1024].contiguous()
  if piece.is_floating_point()or piece.is_complex():assert bool(torch.isfinite(piece).all())
  h.update(piece.view(torch.uint8).numpy().tobytes())
 return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),numel=tensor.numel(),sha256=h.hexdigest())


def state_scan(state,torch):
 assert isinstance(state,dict)and state
 fields={name:tensor_scan(value,torch)for name,value in state.items()}
 return dict(parameter_or_buffer_tensors=len(fields),elements=sum(x['numel']for x in fields.values()),
  component_sha256=digest(fields),fields=fields)


def audit(layer,epoch):
 import numpy as np
 import torch
 base=SUITE/('layer%d-prefix20'%layer);checkpoint=base/('checkpoint-epoch-%02d.pt'%epoch)
 if not checkpoint.exists():return dict(status='WAITING_FOR_REAL_CHECKPOINT',layer=layer,epoch=epoch)
 assert not str(checkpoint).startswith('/Users/'),'Checkpoint payload must remain remote'
 cursor=64*epoch;publication=read(SUITE/'source-publication.json');preparation=read(SUITE/'prepared.json');fixture=read(checked(preparation['fixture']))
 job=(base/'job.id').read_text().strip();assert job==JOBS[layer]
 actual=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip().splitlines()
 allocation=[line.split('|')for line in actual if line.split('|')[0]==job]
 assert len(allocation)==1 and allocation[0][1]=='COMPLETED'and allocation[0][2]=='0:0','Wait for the exact original job to finish cleanly'
 if epoch==3:
  complete=read(base/'complete.json');assert complete['cursor']==192 and complete['epochs']==3 and complete['layer']==layer and complete['prefix']==20
 ref=proof(checkpoint)
 if epoch==3:assert read(base/'latest.json')==ref,'Actual latest file does not reference this final checkpoint'
 blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
 assert set(blob)=={'raw','EMA','Adam','RNG_by_rank','cursor','epoch','plan_sha256','layer','prefix','learning_rate','raw_updates_EMA','actor_updates'}
 assert blob['cursor']==cursor and blob['epoch']==epoch and blob['layer']==layer and blob['prefix']==20
 assert blob['learning_rate']==5e-6 and blob['raw_updates_EMA']is True and blob['actor_updates']==0
 plan_ref=fixture['plans'][epoch-1]['plan'];plan=read(checked(plan_ref));assert len(plan)==len({x['qid']for x in plan})==8192
 assert blob['plan_sha256']==digest(plan)
 raw=state_scan(blob['raw'],torch);ema=state_scan(blob['EMA'],torch)
 assert raw['fields'].keys()==ema['fields'].keys()
 for name,field in raw['fields'].items():assert field['shape']==ema['fields'][name]['shape']and field['dtype']==ema['fields'][name]['dtype'],name
 names=list(blob['raw']);assert any(n.startswith('block.')for n in names)and any(n.startswith('group_head.')for n in names)
 different=[name for name in names if raw['fields'][name]['sha256']!=ema['fields'][name]['sha256']]
 assert any(name.startswith('block.')for name in different)and any(name.startswith('group_head.')for name in different),'Raw and EMA cannot alias after64+ real updates'
 optim=blob['Adam'];assert set(optim)=={'state','param_groups'}and len(optim['param_groups'])==1
 pg=optim['param_groups'][0];assert pg['lr']==5e-6 and pg['betas']==(0.9,0.999)
 ids=pg['params'];assert len(ids)==len(set(ids))==len(blob['raw'])and set(optim['state'])==set(ids)
 moments={};clocks=set()
 for parameter_id,name in zip(ids,names):
  state=optim['state'][parameter_id];assert set(state)=={'step','exp_avg','exp_avg_sq'}
  step=state['step'];step=int(step.item()if torch.is_tensor(step)else step);clocks.add(step)
  assert step==cursor
  assert state['exp_avg'].shape==state['exp_avg_sq'].shape==blob['raw'][name].shape
  moments[name]=dict(step=step,exp_avg=tensor_scan(state['exp_avg'],torch),exp_avg_sq=tensor_scan(state['exp_avg_sq'],torch))
 assert clocks=={cursor}
 assert isinstance(blob['RNG_by_rank'],list)and len(blob['RNG_by_rank'])==2
 rngs=[]
 for rank,rng in enumerate(blob['RNG_by_rank']):
  assert set(rng)=={'python','numpy','torch','cuda'}
  generator=random.Random();generator.setstate(rng['python']);first_python=generator.random()
  generator_np=np.random.RandomState();generator_np.set_state(rng['numpy']);first_numpy=float(generator_np.random_sample())
  assert rng['torch'].dtype==rng['cuda'].dtype==torch.uint8 and rng['torch'].ndim==rng['cuda'].ndim==1
  generator_torch=torch.Generator(device='cpu');generator_torch.set_state(rng['torch']);first_torch=float(torch.rand((),generator=generator_torch))
  rngs.append(dict(rank=rank,python_state_loadable=True,numpy_state_loadable=True,CPU_torch_state_loadable=True,
   isolated_next_random=dict(python=first_python,numpy=first_numpy,torch=first_torch),
   CPU_torch_state=tensor_scan(rng['torch'],torch),CUDA_state=tensor_scan(rng['cuda'],torch),
   CUDA_state_bytes_present=True,CUDA_runtime_restore_tested=False))
 for model in('raw','EMA'):
  assert all(torch.is_tensor(v)for v in blob[model].values())
 coverage=read(base/('coverage-epoch-%02d.json'%epoch));assert coverage['questions']==coverage['unique_questions']==8192
 assert coverage['G5_trajectory_visits']==40960 and coverage['plan_sha256']==blob['plan_sha256']
 assert coverage['Mixed']==fixture['plans'][epoch-1]['Mixed']and coverage['nonMixed']==fixture['plans'][epoch-1]['nonMixed']
 evaluations=read(base/('evaluation-epoch-%02d.json'%epoch));assert evaluations['cursor']==cursor and set(evaluations['metrics'])=={'fixed_train2048','fixed_validation2048'}
 result=dict(status='NATURAL_G5_ONLINE_CPU_RAW_EMA_ADAM_RNG_CHECKPOINT_ACCEPTED',layer=layer,epoch=epoch,cursor=cursor,
  checkpoint=ref,exact_original_job_id=job,sacct=actual,source_publication=proof(SUITE/'source-publication.json'),
  source_commit=publication['commit'],fixture=preparation['fixture'],epoch_plan=plan_ref,
  raw=dict(elements=raw['elements'],tensors=raw['parameter_or_buffer_tensors'],sha256=raw['component_sha256']),
  EMA=dict(elements=ema['elements'],tensors=ema['parameter_or_buffer_tensors'],sha256=ema['component_sha256']),
  raw_EMA_separate_and_different=True,raw_EMA_different_tensors=len(different),
  Adam=dict(parameter_tensors=len(ids),all_actual_steps=cursor,all_native_and_head_moments_finite=True,
   component_sha256=digest(moments),learning_rate=5e-6),RNG_by_rank=rngs,
  epoch_coverage=proof(base/('coverage-epoch-%02d.json'%epoch)),evaluation=proof(base/('evaluation-epoch-%02d.json'%epoch)),
  full_CPU_deserialization=True,GPU_next_update_replay_tested=False,
  EMA_update_recursion_source_verified_by_publication_not_replayed_here=True,
  final4096_read=False,actor_model_or_features_loaded=False,time=time.time(),auditor=proof(__file__))
 output=base/('CPU-checkpoint-audit-epoch-%02d.json'%epoch);temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(result,sort_keys=True)+'\n');temporary.replace(output)
 return dict(status=result['status'],checkpoint=ref,audit=proof(output),cursor=cursor,Adam_steps=cursor,raw_EMA_different_tensors=len(different),RNG_ranks=2,GPU_resume_not_claimed=True)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--layer',type=int,choices=(21,27),required=True);parser.add_argument('--epoch',type=int,choices=(1,2,3),default=3);args=parser.parse_args()
 print(json.dumps(audit(args.layer,args.epoch)),flush=True)


if __name__=='__main__':main()
