"""Stage one fresh second-seed Z replica against accepted read-only full fixture.

Never submits Slurm, starts training, regenerates data or reads checkpoints.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];ROOT=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-Z-secondseed-20261023'
ORIGINAL='7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1'
FIXTURE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1/fixture'
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-full-Z-s19137
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=@@BASE@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_variance_seeded --action train --base @@BASE@@ --fixture-base @@FIXTURE@@ --training-questions 169028 --epochs 10 --lr 0.000005 --loss z_mse_anchor --ema-decay 0.9995 --anchor-weight 1 --normalization-eps 0.000001 --init-seed 19137 --train-plan-seed 2026102302 --start-mode fresh
'''
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);fixture=pathlib.Path(v['fixture']);code=b/('code-'+v['commit'][:12])
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Existing immutable seeded binding differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def mount_and_source():
 assert b.parent.is_dir()and(fixture/'fixture-complete.json').is_file(),'Real shared mount/accepted fixture required; do not create a fake path'
 pub=read(fixture.parent/'source-publication.json');assert pub['commit']==v['original']
 old=read(fixture/'fixture-complete.json');assert old['status']=='PACKED_VARIANCE_OWN_G10_POLICY20_FIXTURE_READY'and old['training_questions']==169028 and old['validation_questions']==2048
 assert old['train_groups']['sha256']=='4528712a7dd5f34e676e2bf2c2e8746b07ebda8112ed59ba100551c4891ae6f2'
 assert pathlib.Path(old['train_groups']['path']).stat().st_size==old['train_groups']['bytes']
 for name in('packed_variance_probe.py','runtime.sh'):
  rel='reproduction/operations/qwen7b_warmup/'+name;assert pub['files'][rel]==v['files'][rel]
 return old,pub
'''
PART='''mount_and_source();b.mkdir(exist_ok=True);code.mkdir(exist_ok=True);p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['chunk_sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''old,pub=mount_and_source();b.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
with(b/'stage.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(b/'latest.json').exists()and not(b/'controller.json').exists(),'Fresh replica root already contains training state'
 for name,ref in v['files'].items():
  p=code/name;parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
  if name.endswith('.py'):compile(raw,name,'exec')
  for part in parts:part.unlink()
 publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),original_runtime_commit=v['original'],files=v['files'],file_origins=v['origins']);once(b/'source-publication.json',publication)
 entry=b/'training.sbatch';text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@FIXTURE@@',str(fixture))
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 fixture_proof=proof(fixture/'fixture-complete.json');once(b/'fixture-reuse.json',dict(status='EXISTING_ACCEPTED_FULL169028_G10_P20_FIXTURE_READ_ONLY_REUSED',fixture=fixture_proof,train_groups=old['train_groups'],validation_groups=old['validation_groups'],fixed_train_G5=old['fixed_train_G5'],fixed_validation_G5=old['fixed_validation_G5'],new_rollouts_or_tokens_or_hidden_features=False))
 ready=dict(status='FRESH_SECOND_TRAINING_SEED_Z_REPLICA_STAGED_NOT_SUBMITTED',source_publication=proof(b/'source-publication.json'),entry=proof(entry),fixture=fixture_proof,base=str(b),code_root=str(code),source_commit=v['commit'],original_runtime_commit=v['original'],training=dict(init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228,start_mode='fresh',resume_checkpoint=None,official_actor_revision='a09a35458c702b33eeacc393d103063234e8bc28',layer=21,capture_layer=20,prefix=20,loss='z_mse_anchor',anchor_weight=1.0,normalization_epsilon=1e-6,normalization_scope='actual_global_optimizer_batch',questions=169028,epochs=10,updates_per_epoch=1321,updates=13210,global_batch=128,world_size=2,local_batch=64,micro=8,accumulation=8,lr=5e-6,EMA_decay=.9995,raw_updates_EMA=True,fixed_dev_and_diagnostic_seed_unchanged=True,head_initialization_SHA_recorded_in_controller_and_CP=True),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='08:00:00'),Slurm_submitted=False,training_started=False,final4096_read=False);once(b/'ready.json',ready)
 print(json.dumps(ready))
'''
STATUS='''out=dict(time=time.time(),base=str(b),Slurm_changed=False)
for name in('ready.json','source-publication.json','fixture-reuse.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 if(b/name).is_file():out[name]=read(b/name)
print(json.dumps(out))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 payload={};origins={};files={}
 for name,origin in [('online_packed_variance_seeded.py',a.commit),('packed_variance_probe.py',ORIGINAL),('runtime.sh',ORIGINAL)]:
  rel=P+name;raw=subprocess.check_output(['git','show',origin+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());origins[rel]=origin
 spec=importlib.util.spec_from_file_location('seeded_full_private',ROOT/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  values=dict(base=BASE,fixture=FIXTURE,commit=a.commit,original=ORIGINAL,branch=BRANCH,files=files,origins=origins,entry=ENTRY,**extra);script='v='+repr(values)+'\n'+COMMON+body;packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;res=ctl.run(command,account='mou176',source_address=ip,timeout=45)
  if res.returncode:raise RuntimeError('Stage/control did not complete: '+res.stdout[-1500:]+res.stderr[-1500:])
  return json.loads(res.stdout)
 if a.action=='stage':
  for name,raw in payload.items():
   for index,offset in enumerate(range(0,len(raw),4096)):
    chunk=raw[offset:offset+4096];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call(STATUS)
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/seeded-full169028-Z';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))


for body in(PART,SEAL,STATUS):compile(COMMON+body,'seeded-full-stage-static','exec')
if __name__=='__main__':main()
