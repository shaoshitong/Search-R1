"""Stage five immutable matched pilot entries and authentic CPU token fixture.

Never submits, cancels or modifies a Slurm job. Mutations use one explicit direct
control route; status is read-only. Root owns every future job submission.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-matched-packed-objective-pilot-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-L15P80-8192-20261023-v1'
FILES=['packed_variance_probe.py','online_packed_variance_scaling.py','group_topk_losses.py','online_packed_matched_objective.py','runtime.sh']
CELLS=[('mse-Mrandom','mse',0),('mse-M5','mse',5),('mixed-bce-Mrandom','mixed_bce',0),('mixed-bce-M5','mixed_bce',5),('topk-ramp-M5','topk_ramp',5)]
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-L15P80-@@CELL@@
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=@@BASE@@/@@CELL@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/@@CELL@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "${QWEN7B_CODE_ROOT}/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_matched_objective --action train --base @@BASE@@/@@CELL@@ --fixture-base @@BASE@@/fixture --objective @@OBJECTIVE@@ --fixed-m @@M@@ --layer 15 --prefix 80 --epochs 10 --lr 0.000005 --ema-decay 0.9995 --margin 1.0
'''
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl,gzip
b=pathlib.Path(v['base']);assert b.parent.is_dir()and(b.parent/'online-mixed-layer-prefix-20261022-v1/fixture-complete.json').is_file(),'Actual shared task mount and accepted real Prefix80 fixture required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 p=pathlib.Path(ref['path']);assert proof(p)=={k:ref[k]for k in('path','bytes','sha256')};return p
def once(p,x):
 raw=canonical(x)+bytes([10])
 if p.exists():assert p.read_bytes()==raw,'Immutable matched pilot publication/binding differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def identity():
 p=b/'prepare-process.json'
 if not p.exists():return False
 old=read(p);proc=pathlib.Path('/proc/%d'%old['pid'])
 return proc.exists()and b'online_packed_matched_objective'in(proc/'cmdline').read_bytes()and(proc/'stat').read_text().rsplit(')',1)[1].split()[19]==old['start_ticks']
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for part in parts:part.unlink()
once(b/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']))
entries=[];(b/'fixture').mkdir(exist_ok=True)
for name,objective,m in v['cells']:
 dest=b/name;dest.mkdir(exist_ok=True);text=v['sbatch']
 for key,val in dict(BASE=str(b),CODE=str(code),CELL=name,OBJECTIVE=objective,M=str(m)).items():text=text.replace('@@'+key+'@@',val)
 entry=dest/'training.sbatch'
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 ready=dict(status='MATCHED_L15_P80_PILOT_ENTRY_STAGED_FIXTURE_REQUIRED_NOT_SUBMITTED',cell=name,objective=objective,fixed_m=m,entry=proof(entry),source_publication=proof(b/'source-publication.json'),runtime_commit=v['commit'],fixture_base=str(b/'fixture'),training=dict(questions=8192,source_trajectories=81920,epochs=10,global_batch_attempt_budget=640,global_batch=128,local_batch=64,micro=8,accumulation=8,layer=15,prefix=80,head_max_positions=2048,seed=2026102227,LR=5e-6,EMA_decay=.9995,sampling_independent_of_EM=True,raw_updates_EMA=True,fresh_official_block_head_Adam=True,topk_margin=1.0 if objective=='topk_ramp'else None,topk_scope='actual_global_batch_KceilNdiv4'if objective=='topk_ramp'else None,zero_ramp_skips_Adam_and_EMA=objective=='topk_ramp'),evaluation=dict(train2048_ownG5=True,viewed_validation2048_ownG5=True,primary='Top512_real_Mixed',BA_secondary=True,classification_threshold=.32 if objective=='mse'else.5,threshold_fitted=False,dev_G10_required=False,final4096_read=False),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='02:00:00'),root_submits_only=True,Slurm_submitted=False)
 once(dest/'ready.json',ready);entries.append(dict(cell=name,objective=objective,fixed_m=m,ready=proof(dest/'ready.json'),entry=proof(entry)))
result=dict(status='FIVE_MATCHED_L15_P80_ENTRIES_STAGED_NOT_SUBMITTED',publication=proof(b/'source-publication.json'),entries=entries,source_files_fresh_SHA_verified=True,CPU_fixture_required=True,Slurm_submitted=False,final4096_read=False);once(b/'staged.json',result);print(json.dumps(result))
'''
PREPARE='''guard=(b/'prepare-launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);pub=read(b/'source-publication.json');assert pub['commit']==v['commit'];fixture=b/'fixture'
if(fixture/'fixture-complete.json').exists():print(json.dumps(dict(status='EXISTING_CPU_FIXTURE_COMPLETE_NEEDS_ACCEPTANCE',fixture=proof(fixture/'fixture-complete.json'))))
elif identity():print(json.dumps(dict(status='EXISTING_ACTUAL_CPU_PREPARATION_REUSED',process=read(b/'prepare-process.json'))))
else:
 lock=(fixture/'prepare.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close();env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'prepare.log').open('ab');cmd=['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.online_packed_matched_objective','--action','prepare','--base',str(fixture)];p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);proc=pathlib.Path('/proc/%d'%p.pid);receipt=dict(pid=p.pid,start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],status='CPU_REAL_POLICY80_FIXTURE_STARTED_NOT_ACCEPTED',runtime_commit=v['commit'],time=time.time());q=b/'prepare-process.json.tmp';q.write_bytes(canonical(receipt)+bytes([10]));q.replace(b/'prepare-process.json');print(json.dumps(receipt))
'''
ACCEPT='''guard=(b/'fixture-acceptance.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not identity(),'CPU fixture preparation must actually exit before acceptance';fixture=b/'fixture';f=read(fixture/'fixture-complete.json');assert f['status']=='MATCHED_PACKED_REAL_POLICY80_G10_FIXED2048_READY'and f['training_questions']==8192 and f['validation_questions']==2048 and f['max_real_policy_prefix']==80;source=read(checked(f['source_fixture']));assert source['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'and f['source_token_shards']==source['token_shards'];train=read(checked(f['train_groups']));validation=read(checked(f['validation_groups']));fixed=read(checked(f['fixed_train_G5']));assert len(train)==8192 and len(validation)==len(fixed)==2048
wanted={t:(role,q,y)for role,gs in(('train',train),('validation',validation))for q,g in gs.items()for t,y in zip(g['trajectory_ids'],g['labels'])};assert len(wanted)==92160 and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in train.values())and all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in validation.values());assert not(set(train)&set(validation))and not({g['question_hash']for g in train.values()}&{g['question_hash']for g in validation.values()});expected={};started=time.time()
for ref in f['source_token_shards']:
 with gzip.open(checked(ref),'rt')as src:
  for line in src:
   r=json.loads(line);tid=r['trajectory_id'];role,q,y=wanted[tid];assert r['qid']==q and r['role']==role and r['label']==y;policy=[]
   for s in r['segments']:
    n=min(80-len(policy),s['policy_length']);policy+=s['context_tokens'][s['request_length']:s['request_length']+n]
    if len(policy)==80:break
   assert len(policy)==min(80,r['policy_tokens'])and policy;expected[tid]=digest(dict(qid=q,trajectory_id=tid,role=role,prompt=r['prompt'],policy_token_ids=policy))
assert len(expected)==92160;seen=set();short=0
for ref in f['token_shards']:
 with gzip.open(checked(ref),'rt')as src:
  for line in src:
   r=json.loads(line);tid=r['trajectory_id'];assert tid not in seen and expected[tid]==digest(r);seen.add(tid);short+=len(r['policy_token_ids'])<80
assert seen==set(wanted)
for g in fixed:
 original=train[g['qid']];by=dict(zip(original['trajectory_ids'],original['labels']));assert len(g['trajectory_ids'])==5 and g['labels']==[by[t]for t in g['trajectory_ids']]
result=dict(status='MATCHED_PACKED_REAL_PREFIX80_FIXTURE_INDEPENDENT_CPU_ACCEPTED',time=time.time(),seconds=time.time()-started,fixture=proof(fixture/'fixture-complete.json'),training_questions=8192,training_trajectories=81920,validation_questions=2048,validation_trajectories=10240,total_real_own_rows=92160,genuine_shorter_than80_trajectories=short,source_and_compact_fresh_SHA_verified=True,source_policy80_exact_match=True,all_own_ids_and_EM_verified=True,fixed_validation_G5_sha256=f['fixed_G5_sha256'],fixed_train_G5=proof(checked(f['fixed_train_G5'])),no_GPU=True,Slurm_submitted=False,final4096_read=False);path=fixture/'accepted.json';tmp=path.with_suffix('.json.tmp');tmp.write_bytes(canonical(result)+bytes([10]));tmp.replace(path);print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time(),actual_CPU_preparation_live=identity(),Slurm_mutation=False,final4096_read=False)
for name in('source-publication.json','staged.json','prepare-process.json','fixture/fixture-complete.json','fixture/accepted.json','fixture/failure-rank-cpu.json'):
 if(b/name).exists():out[name]=read(b/name)
if(b/'prepare.log').exists():
 with(b/'prepare.log').open('rb')as f:f.seek(max(0,(b/'prepare.log').stat().st_size-1800));out['prepare_log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','accept','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 payload={};files={}
 for name in FILES:
  rel=P+name;raw=subprocess.check_output(['git','show',a.commit+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 spec=importlib.util.spec_from_file_location('matched_packed_direct_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**more):
  v=dict(base=BASE,branch=BRANCH,commit=a.commit,files=files,cells=CELLS,sbatch=SBATCH,**more);script='v='+repr(v)+'\n'+COMMON+body;compile(script,'matched_packed_deployment','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2500:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for i,start in enumerate(range(0,len(raw),3072)):
    chunk=raw[start:start+3072];call(PART,name=name,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call({'prepare':PREPARE,'accept':ACCEPT,'status':STATUS}[a.action])
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/matched-packed-objective';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,SEAL,PREPARE,ACCEPT,STATUS):compile(COMMON+body,'matched_deployment_static','exec')
if __name__=='__main__':main()
