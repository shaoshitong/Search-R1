"""Publish/stage/prepare/status control for packed variance pilot; never submits."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-packed-variable-M-variance-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1'
FILES=['packed_variance_probe.py','online_packed_variance.py','runtime.sh']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-packed-M3to10-var
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "${QWEN7B_CODE_ROOT}/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_variance --action train --base @@BASE@@ --training-questions 8192 --epochs 3 --lr 0.000005
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
entry=b/'training.sbatch';text=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
ready=dict(status='PACKED_VARIANCE_IMPLEMENTATION_STAGED_FIXTURE_REQUIRED_NOT_SUBMITTED',entry=proof(entry),source_publication=proof(b/'source-publication.json'),runtime_commit=v['commit'],root_submit_controller=v['submit_controller'],training=dict(questions=8192,pilot_not_full169028=True,epochs=3,updates=192,global_batch=128,local_batch=64,micro=8,accumulation=8,layer=21,prefix=20,M_range=[3,10],M_and_subset_drawn_before_EM=True,question_order_each_epoch_random=True,trajectory_order_each_epoch_random=True,question_prompt_count=1,actor_prefill_rows_per_group=1,loss='ordinary_MSE(sigmoid(logit),4*selected_M_population_EM_variance)',BCE_losses=[],trajectory_losses=[],ranking_losses=[],lr=5e-6,EMA_decay=.99,fresh_official_block_raw_and_EMA_and_Adam=True),selection=dict(score='predicted_population_EM_variance',order='descending',same_M_all_candidates=True,fixed_independent_selection_M=5,strict_EM_used_for_ranking=False),evaluation=dict(fixed_independent_original_G5_questions=2048,diagnostics_M=[3,5,10],diagnostic_M_seed=2026102228,variance_G5_classification_threshold=.08,classification_threshold_tuned=False,final4096_read=False),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='00:30:00'),interpretation='Packed context, randomM and objective all change; not an isolated MSE causal comparison',existing_tasks_modified=False,Slurm_submitted=False)
once(b/'ready.json',ready);print(json.dumps(dict(status=ready['status'],ready=proof(b/'ready.json'),entry=proof(entry),publication=proof(b/'source-publication.json'))))
'''
PREPARE='''pub=json.loads((b/'source-publication.json').read_bytes());assert pub['commit']==v['commit']
if(b/'fixture-complete.json').exists():print(json.dumps(dict(status='EXISTING_PACKED_FIXTURE_ACCEPTED',fixture=proof(b/'fixture-complete.json'))))
else:
 lock=(b/'prepare.lock').open('a+');active=False
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:active=True
 if active:print(json.dumps(dict(status='EXISTING_CPU_TOKEN_PREPARATION_ACTIVE_NO_DUPLICATE')))
 else:
  lock.close();receipt=b/'prepare-process.json'
  if receipt.exists():
   prior=json.loads(receipt.read_bytes());p=pathlib.Path('/proc/%d/cmdline'%prior['pid'])
   if p.exists()and b'online_packed_variance'in p.read_bytes():raise RuntimeError('Existing preparation identity is alive; do not duplicate')
  env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'prepare.log').open('ab');cmd=['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.online_packed_variance','--action','prepare','--base',str(b),'--training-questions','8192'];p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);result=dict(pid=p.pid,stage='CPU_TOKEN_ONLY_PREPARATION_STARTED_NOT_VERIFIED',runtime_commit=v['commit'],time=time.time());receipt.write_text(json.dumps(result)+'\\n');print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time())
for n in('source-publication.json','ready.json','prepare-process.json','prepare-progress.json','fixture-complete.json','root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
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
 submit='reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance/submit-once.py';raw=subprocess.check_output(['git','show',a.commit+':'+submit],cwd=R);submitref=dict(commit=a.commit,path=submit,sha256=hashlib.sha256(raw).hexdigest())
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
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-variance';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(COMMON+SEAL,'packed_stage_static','exec');compile(COMMON+PREPARE,'packed_prepare_static','exec')
if __name__=='__main__':main()
