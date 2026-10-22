"""Publish/stage/prepare/status control for packed variance pilot; never submits."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-packed-variance-full169028-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1'
FILES=['packed_variance_probe.py','online_packed_variance_scaling.py','runtime.sh']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-full-var-@@LOSS@@
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=@@BASE@@/@@LOSS@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/@@LOSS@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "${QWEN7B_CODE_ROOT}/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_variance_scaling --action train --base @@BASE@@/@@LOSS@@ --fixture-base @@BASE@@/fixture --training-questions 169028 --epochs 10 --lr 0.000005 --loss @@LOSS@@ --ema-decay 0.9995
'''
COMMON='''import pathlib,json,hashlib,base64,zlib,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+'\\n').encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable packed variance binding differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(exist_ok=True,parents=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for x in parts:x.unlink()
publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']);once(b/'source-publication.json',publication)
A=b.parent;initial=[dict(path=str(A/'joint169028-native-training-bank-v1/complete.json'),bytes=9418,sha256='18aec908cfaeb5cd5a0a1bb486fe8b09a76d674c1f32b2643e12c319101398a8'),dict(path=str(A/'hpc3-tail-full169028-qualified-raw-union-v1/complete.json'),bytes=7943,sha256='3dabf439073b1c49b4b0be7e9885100a214f101443e3b7a983ddca1fbbf40b1b')];todo=list(initial);seen=set();prefixes={};metadata=[]
def walk(x,key=''):
 if isinstance(x,dict):
  if {'path','bytes','sha256'}<=set(x):
   p=pathlib.Path(x['path'])
   if any(w in str(p)for w in('final4096','confirmation-materialization')):return
   if p.name in('source-events.json','source-refs.json')or(p.name=='complete.json'and key in('complete','raw_complete','full_raw_G10_acceptance')):
    if str(p)not in seen:todo.append({k:x[k]for k in('path','bytes','sha256')})
  for k,y in x.items():walk(y,k)
 elif isinstance(x,list):
  for y in x:walk(y,key)
def only(x,key=''):
 if isinstance(x,dict):
  if key=='prefix'and {'path','bytes','sha256'}<=set(x):
   old=prefixes.get(x['path']);ref={k:x[k]for k in('path','bytes','sha256')};assert old is None or old==ref;prefixes[x['path']]=ref
  for k,y in x.items():only(y,k)
 elif isinstance(x,list):
  for y in x:only(y,key)
while todo:
 ref=todo.pop();p=pathlib.Path(ref['path'])
 if str(p)in seen:continue
 seen.add(str(p));assert len(seen)<100 and ref['bytes']<3000000;assert proof(p)==ref;raw=json.loads(p.read_bytes());metadata.append(ref)
 if p.name=='source-events.json':only(raw)
 else:walk(raw)
bank=json.loads(pathlib.Path(initial[0]['path']).read_bytes());gref=bank['groups'];assert proof(gref['path'])==gref;raw=json.loads(pathlib.Path(gref['path']).read_bytes());groups=raw['train']if'train'in raw else raw
if isinstance(groups,dict):groups=list(groups.values())
assert len(groups)==169028 and len({g['qid']for g in groups})==169028 and all(len(g['trajectory_ids'])==len(g['labels'])==10 and len(set(g['trajectory_ids']))==10 and all(type(x)is int and x in(0,1)for x in g['labels'])for g in groups);tids=[t for g in groups for t in g['trajectory_ids']];assert len(set(tids))==1690280
ordered=hashlib.sha256(json.dumps([g['qid']for g in groups],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();assert ordered==bank['ordered_qid_sha256'];del tids,raw
fixture=b/'fixture';fixture.mkdir(exist_ok=True);descriptor=dict(status='FULL169028_G10_ACCEPTED_GROUPS_AND_PREFIX_SOURCES_BOUND',groups=gref,prefix_sources=sorted(prefixes.values(),key=lambda x:x['path']),source_metadata=metadata,questions=169028,trajectories=1690280,own_EM_preserved=True,canonical_ordered_qid_sha256=ordered,label_based_identity_selection=False,final4096_read=False,prefix_payloads_rehashed_by_prepare=True);once(fixture/'full-training-descriptor.json',descriptor)
entries=[]
for loss in('mse','mae'):
 dest=b/loss;dest.mkdir(exist_ok=True);entry=dest/'training.sbatch';text=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@LOSS@@',loss)
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)]);ready=dict(status='FULL169028_PACKED_VARIANCE_READY_FIXTURE_REQUIRED_NOT_SUBMITTED',loss=loss,entry=proof(entry),source_publication=proof(b/'source-publication.json'),runtime_commit=v['commit'],descriptor=proof(fixture/'full-training-descriptor.json'),fixture_base=str(fixture),root_submit_controller=v['submit_controller'],training=dict(questions=169028,trajectories_G10=1690280,epochs=10,updates_per_epoch=1321,updates=13210,tail_global_batch=68,global_batch=128,local_batch=64,micro=8,accumulation=8,layer=21,prefix=20,M_range=[3,10],seed=2026102227,loss=loss,loss_scale='4*population_variance',EMA_decay=.9995,lr=5e-6,fresh_official_block_head_Adam=True,pilot_checkpoint_resumed=False),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='08:00:00'),selection_score='predicted_population_EM_variance',shared_fixture_read_only=True,final4096_read=False,Slurm_submitted=False);once(dest/'ready.json',ready);entries.append(dict(loss=loss,ready=proof(dest/'ready.json'),entry=proof(entry)))
summary=dict(status='FULL169028_TWO_LOSS_ENTRIES_STAGED_CPU_FIXTURE_REQUIRED',publication=proof(b/'source-publication.json'),descriptor=proof(fixture/'full-training-descriptor.json'),prefix_source_files=len(prefixes),prefix_compressed_bytes=sum(r['bytes']for r in prefixes.values()),metadata_files=len(metadata),questions=169028,trajectories=1690280,entries=entries,Slurm_submitted=False);print(json.dumps(summary))
'''
PREPARE='''pub=json.loads((b/'source-publication.json').read_bytes());assert pub['commit']==v['commit']
fixture=b/'fixture'
if(fixture/'fixture-complete.json').exists():print(json.dumps(dict(status='EXISTING_PACKED_FIXTURE_ACCEPTED',fixture=proof(fixture/'fixture-complete.json'))))
else:
 lock=(fixture/'prepare.lock').open('a+');active=False
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:active=True
 if active:print(json.dumps(dict(status='EXISTING_CPU_TOKEN_PREPARATION_ACTIVE_NO_DUPLICATE')))
 else:
  lock.close();receipt=b/'prepare-process.json'
  if receipt.exists():
   prior=json.loads(receipt.read_bytes());p=pathlib.Path('/proc/%d/cmdline'%prior['pid'])
   if p.exists()and b'online_packed_variance_scaling'in p.read_bytes():raise RuntimeError('Existing preparation identity is alive; do not duplicate')
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'prepare.log').open('ab');cmd=['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.online_packed_variance_scaling','--action','prepare','--base',str(fixture),'--training-questions','169028','--full-training-descriptor',str(fixture/'full-training-descriptor.json')];p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);result=dict(pid=p.pid,stage='CPU_TOKEN_ONLY_PREPARATION_STARTED_NOT_VERIFIED',runtime_commit=v['commit'],time=time.time());receipt.write_text(json.dumps(result)+'\\n');print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time())
for n in('source-publication.json','prepare-process.json','fixture/prepare-progress.json','fixture/fixture-complete.json','fixture/failure-rank-cpu.json','mse/controller.json','mse/progress-rank-0.json','mse/progress-rank-1.json','mse/latest.json','mse/complete.json','mae/controller.json','mae/progress-rank-0.json','mae/progress-rank-1.json','mae/latest.json','mae/complete.json'):
 if(b/n).exists():out[n]=json.loads((b/n).read_bytes())
for p in b.glob('failure-rank-*.json'):out[p.name]=json.loads(p.read_bytes())
if(b/'prepare.log').exists():
 with(b/'prepare.log').open('rb')as f:f.seek(max(0,(b/'prepare.log').stat().st_size-1500));out['prepare_log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','prepare','status'],required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 payload={};files={}
 for name in FILES:
  rel=P+name;raw=subprocess.check_output(['git','show',a.commit+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 submit='reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028/submit-both-once.py';raw=subprocess.check_output(['git','show',a.commit+':'+submit],cwd=R);submitref=dict(commit=a.commit,path=submit,sha256=hashlib.sha256(raw).hexdigest())
 spec=importlib.util.spec_from_file_location('packed_variance_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(source,**more):
  v=dict(base=BASE,branch=BRANCH,commit=a.commit,files=files,sbatch=SBATCH,submit_controller=submitref,**more);script='v='+repr(v)+'\n'+COMMON+source;packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-3000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for i,start in enumerate(range(0,len(raw),3072)):
    chunk=raw[start:start+3072];call(PART,name=name,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call(PREPARE if a.action=='prepare'else STATUS)
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance-full169028';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(COMMON+SEAL,'packed_stage_static','exec');compile(COMMON+PREPARE,'packed_prepare_static','exec')
if __name__=='__main__':main()
