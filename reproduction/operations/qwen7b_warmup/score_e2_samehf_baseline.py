"""Supplement sealed E2 on the identical qualified new-HF fixed-G5 pool.
Read-only fair baseline; never added to the preregistered E4/peer calibration grid.
"""
import argparse,fcntl,json,os,platform,shlex,time,traceback
from pathlib import Path
if __package__:
 from . import score_peer_g10_calibration as s
else:
 import score_peer_g10_calibration as s
TARGET=s.A/'E2-same-HF-baseline-20261021'
VALIDATION=s.A/'validation-fixedG5-hf-replay-20261021/capture/complete.json'
SHA='6a970815ff3aec26eedf4d6a65589bbebac3c4ccbdf8eefdf859a83e2a11bd2d'
STATUS='SEALED_E2_SAME_QUALIFIED_NEW_HF_FIXED_G5_BASELINE_COMPLETE'
JOB='q7b-E2-same-HF'

def render(code,configsha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=@@JOB@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=@@BASE@@/logs/scoring-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@RUNTIME@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u @@ENTRY@@ --action run --config-sha256 @@SHA@@
'''.replace('@@JOB@@',JOB).replace('@@BASE@@',str(TARGET)).replace('@@RUNTIME@@',shlex.quote(str(s.TRAIN/('code-'+s.PIN[:12])))).replace('@@ENTRY@@',shlex.quote(str(Path(code)/Path(__file__).name))).replace('@@SHA@@',configsha)

def prepare():
 s.runtime(s.read(s.TRAIN/'source-publication.json'))
 from reproduction.operations.qwen7b_warmup import peer_newseed_fit_operation as op
 from reproduction.operations.qwen7b_warmup.probe_g10_train import frozen_validation
 old=s.B/'frozen-validation-ema3-20261021/inputs/config.json';original=s.read(old)
 d=next(r for r in original['checkpoints'] if r['name']=='E2');receipt=s.read(s.checked(d['source_receipt']));sc=s.read(s.checked(d['source_config']))
 if d['sha256']!=SHA or d['steps']!=11782 or d['epoch']!=2 or receipt['sha256']!=SHA or receipt['completed_epoch'] is not True or receipt['independent_cpu_reload'] is not True:raise ValueError('Actual sealed E2 source identity differs')
 if Path(d['path']).stat().st_size!=d['bytes']:raise ValueError('E2 checkpoint bytes differ')
 v,groups,proof,e=op.validation_inputs(VALIDATION);fixed=frozen_validation(groups,s.read(s.checked(e['fixed_g5'])))
 s.immutable(TARGET/'groups-fixedG5.json',fixed);pub=s.read(TARGET/'source-publication.json');root=Path(pub['code_root'])
 for name,item in pub['files'].items():s.checked(dict(path=str(root/name),**item))
 c=dict(role='E2_FAIR_SAME_HF_VIEWED_VALIDATION_ONLY',checkpoint=d,original_sealed_config=s.proof(old),model_dir=d['model_dir'],model_config=s.proof(Path(d['model_dir'])/'config.json'),validation_complete=s.proof(VALIDATION),groups=s.proof(TARGET/'groups-fixedG5.json'),features=v['features'],fixed_G5_sha256=op.FIXED_G5_SHA,source_publication=s.proof(TARGET/'source-publication.json'),pinned_runtime_publication=s.proof(s.TRAIN/'source-publication.json'),actor_updates=0,probe_updates=0,optimizer_constructed=False,calibration_grid_member=False,final_pool_read=False)
 s.immutable(TARGET/'config.json',c);script=render(root,s.proof(TARGET/'config.json')['sha256']);p=TARGET/'scoring.sbatch'
 if p.exists() and p.read_text()!=script:raise ValueError('Immutable E2 sbatch differs')
 if not p.exists():p.write_text(script)
 r=dict(status='SEALED_E2_SAME_HF_PREPARED',config=s.proof(TARGET/'config.json'),sbatch=s.proof(p),checkpoint=d,questions=2048,trajectories=10240,checkpoint_payload_read=False,feature_payload_read=False);s.immutable(TARGET/'preparation.json',r);return r

def run(configsha):
 if platform.system()!='Linux' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Only actual remote owned GPU may read tensors')
 configref=s.proof(TARGET/'config.json')
 if configref['sha256']!=configsha:raise ValueError('E2 configSHA differs')
 c=s.read(configref['path']);s.runtime(s.read(s.checked(c['pinned_runtime_publication'])));pub=s.read(s.checked(c['source_publication']))
 for name,item in pub['files'].items():s.checked(dict(path=str(Path(pub['code_root'])/name),**item))
 from reproduction.operations.qwen7b_warmup import probe_ablation_train as u,evaluate_frozen_validation as e
 from reproduction.operations.qwen7b_warmup.model import Probe
 from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
 import torch
 if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0) or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('One deterministic H100 required')
 groups=s.read(s.checked(c['groups']));manifests=s.read(s.checked(c['features']));s.checked(c['model_config']);s.checked(c['validation_complete']);d=c['checkpoint']
 state=torch.load(s.checked(d),map_location='cpu',mmap=True,weights_only=False);e.validate_checkpoint_metadata(d,s.read(s.checked(d['source_receipt'])),s.read(s.checked(d['source_config'])),state)
 os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20';torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;device=torch.device('cuda',0);torch.cuda.set_device(device)
 model=Probe(c['model_dir'],273,20).to(device).eval().requires_grad_(False);model.load_state_dict(state['ema'],strict=True);native=state['ema_buffers'];u.restore_buffers(model,native);ema=u.model_hash(model)
 if any(r['ema_sha256']!=ema for r in state['rank_states']):raise ValueError('E2 actual four-rank EMA hash differs')
 del state;reader=ReplayFeatureReader(manifests,26)
 s.atomic(TARGET/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],config=configref,ema_sha256=ema,time=time.time()))
 try:metrics=e.evaluate(model,reader,groups,device,TARGET/'predictions.jsonl.gz',TARGET/'progress.json','E2-same-HF',micro_groups=8)
 finally:reader.close()
 if u.model_hash(model)!=ema or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('E2 frozen inference modified weights or created gradients')
 u.restore_buffers(model,native)
 if u.tensor_hash(u.buffers(model))!=u.tensor_hash(native):raise ValueError('E2 native buffers failed restoration')
 if s.proof(d['path'])!={k:d[k] for k in ('path','bytes','sha256')}:raise ValueError('E2 sourcecheckpoint changed')
 r=dict(status=STATUS,config=configref,checkpoint=d,groups=c['groups'],predictions=s.proof(TARGET/'predictions.jsonl.gz'),metrics=metrics,ema_sha256=ema,fixed_G5_sha256=c['fixed_G5_sha256'],actor_updates=0,probe_updates=0,optimizer_constructed=False,parameters_unchanged=True,calibration_grid_member=False,final_pool_read=False,normal_exit=True);s.immutable(TARGET/'complete.json',r);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','run'),required=True);p.add_argument('--config-sha256');a=p.parse_args();TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
 with (TARGET/'scoring.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:r=prepare() if a.action=='prepare' else run(a.config_sha256);print(json.dumps(r if a.action=='prepare' else {k:r[k] for k in ('status','predictions','metrics')}),flush=True)
  except BaseException as error:s.atomic(TARGET/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
