"""Stage only a fresh explicit second-seed matched MSE-M5 replica."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-matched-MSE-M5-secondseed-20261023'
ORIGINAL='0436c2c1e3c1b38a2abcbc252bf3524eff0e6deb'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1'
FIXTURE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-L15P80-8192-20261023-v1/fixture'
FILES=['packed_variance_probe.py','online_packed_variance_scaling.py','group_topk_losses.py','online_packed_matched_objective.py','online_packed_matched_objective_seeded.py','test_online_packed_matched_objective_seeded.py','runtime.sh']
CELLS=[('mse-M5',0)]
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-L15P80-mse5-s19137
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
srun --kill-on-bad-exit=1 bash "${QWEN7B_CODE_ROOT}/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_matched_objective_seeded --action train --base @@BASE@@/@@CELL@@ --fixture-base @@FIXTURE@@ --objective mse --fixed-m 5 --layer 15 --prefix 80 --epochs 10 --lr 0.000005 --ema-decay 0.9995 --margin 1.0 --init-seed 19137 --train-plan-seed 2026102302 --start-mode fresh
'''
CPU_SEED_CHECK="import json,tempfile\nfrom pathlib import Path\nfrom reproduction.operations.qwen7b_warmup.online_packed_matched_objective_seeded import epoch_plan,seed_model_rng,head_state_sha256,enforce_start_mode,read,checked,digest\nfrom reproduction.operations.qwen7b_warmup.online_packed_matched_objective import epoch_plan as original_plan\nfrom reproduction.operations.qwen7b_warmup.packed_variance_probe import PackedVarianceHead\nf=read(Path(FIXTURE_BASE)/'fixture-complete.json');groups=read(checked(f['train_groups']));old=epoch_plan(groups,1,5,train_plan_seed=2026102227);new=epoch_plan(groups,1,5,train_plan_seed=2026102302);again=epoch_plan(groups,2,5,train_plan_seed=2026102302);assert old==original_plan(groups,1,5)\na={r['qid']:r for r in old};b={r['qid']:r for r in new};assert len(a)==len(b)==8192 and set(a)==set(b)==set(groups);changed=sum(a[q]['trajectory_ids']!=b[q]['trajectory_ids']for q in a);assert changed>8100 and [r['qid']for r in old]!=[r['qid']for r in new]and [r['qid']for r in new]!=[r['qid']for r in again]\nseed_model_rng(9137);h1=head_state_sha256(PackedVarianceHead(max_positions=2048));seed_model_rng(19137);h2=head_state_sha256(PackedVarianceHead(max_positions=2048));seed_model_rng(19137);h3=head_state_sha256(PackedVarianceHead(max_positions=2048));assert h1!=h2 and h2==h3\nwith tempfile.TemporaryDirectory(prefix='matched-mse-seed-CPU-')as tmp:\n enforce_start_mode(tmp,'fresh');(Path(tmp)/'latest.json').write_text('{}')\n try:enforce_start_mode(tmp,'fresh')\n except ValueError:pass\n else:raise AssertionError('Fresh start silently resumed')\nprint(json.dumps(dict(status='ACTUAL8192_MATCHED_MSE_M5_SEED_PLUMBING_CPU_PASSED',original_plan_sha256=digest(old),new_plan_sha256=digest(new),new_epoch2_plan_sha256=digest(again),new_question_order_sha256=digest([r['qid']for r in new]),same_unique_question_coverage=8192,own_G5_views_changed=changed,initial_head_component_sha256={'9137':h1,'19137':h2},head_initialization_repeatable=True,head_test_scope='CPU fresh PackedVarianceHead; no actor loaded; actual runtime records post-actor head SHA',fresh_cannot_implicitly_resume=True,init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228,fixture_train_groups=f['train_groups'],validation_groups=f['validation_groups'],fixed_train_G5=f['fixed_train_G5'],final4096_read=False,model_or_checkpoint_tensors_read=False,Slurm_submitted=False)))\n"
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);fbase=pathlib.Path(v['fixture']);assert b.parent.is_dir()and(fbase/'accepted.json').is_file(),'Actual shared task mount and independently accepted Prefix80 fixture required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def once(p,x):
 raw=canonical(x)+bytes([10])
 if p.exists():assert p.read_bytes()==raw,'Immutable global64 window entry/publication differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not(b/'mse-M5/latest.json').exists()and not(b/'mse-M5/controller.json').exists(),'Fresh seed root already contains training state'
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for part in parts:part.unlink()
once(b/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],file_origins=v['origins'],original_runtime_commit=v['original']))
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
check_source='FIXTURE_BASE='+repr(str(fbase))+chr(10)+v['seed_check'];check=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',check_source],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert check.returncode==0,check.stderr[-2000:];seed_result=json.loads(check.stdout.splitlines()[-1]);once(b/'actual-seed-component-checks.json',seed_result)
fixture=read(fbase/'fixture-complete.json');accepted=read(fbase/'accepted.json');assert accepted['status']=='MATCHED_PACKED_REAL_PREFIX80_FIXTURE_INDEPENDENT_CPU_ACCEPTED'and accepted['fixture']==proof(fbase/'fixture-complete.json')and accepted['source_policy80_exact_match']and accepted['total_real_own_rows']==92160
for ref in fixture['token_shards']+[fixture['train_groups'],fixture['validation_groups'],fixture['fixed_train_G5']]:assert proof(ref['path'])==ref
reuse=dict(status='EXISTING_AUTHENTIC_POLICY80_FIXTURE_FRESH_SHA_REUSED',fixture=proof(fbase/'fixture-complete.json'),independent_acceptance=proof(fbase/'accepted.json'),training_questions=8192,training_trajectories=81920,validation_questions=2048,validation_trajectories=10240,real_own_rows=92160,policy_prefix_capacity=80,fixed_validation_G5_sha256=accepted['fixed_validation_G5_sha256'],fixed_train_G5=fixture['fixed_train_G5'],no_reextraction=True,no_GPU=True,final4096_read=False);once(b/'fixture-reuse.json',reuse);entries=[]
for name,capacity in v['cells']:
 dest=b/name;dest.mkdir(exist_ok=True);text=v['sbatch']
 for key,val in dict(BASE=str(b),CODE=str(code),CELL=name,FIXTURE=str(fbase),WINDOW=str(capacity)).items():text=text.replace('@@'+key+'@@',val)
 entry=dest/'training.sbatch'
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 ready=dict(status='FRESH_SECOND_SEED_MATCHED_MSE_M5_READY_NOT_SUBMITTED',cell=name,objective='mse',fixed_m=5,entry=proof(entry),source_publication=proof(b/'source-publication.json'),fixture_reuse=proof(b/'fixture-reuse.json'),actual_seed_checks=proof(b/'actual-seed-component-checks.json'),runtime_commit=v['commit'],training=dict(questions=8192,epochs=10,global_batch_attempt_budget=640,global_batch=128,local_batch=64,micro=8,accumulation=8,layer=15,prefix=80,head_max_positions=2048,seed=2026102302,init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228,start_mode='fresh',resume_checkpoint=None,seed_source='explicit_CLI',LR=5e-6,EMA_decay=.9995,fixed_own_G5=True,objective='population_variance_MSE',target='4*population_EM_variance_of_selected_ownG5',raw_updates_EMA=True,fresh_official_block_head_Adam=True),evaluation=dict(primary='viewed_independent2048_ownG5_Top512_real_Mixed',fixed_train2048_ownG5=True,secondary_BA_threshold=.32,threshold_fitted=False,rank_score='4*predicted_population_EM_variance',final4096_read=False),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='02:00:00'),root_submits_only=True,Slurm_submitted=False);once(dest/'ready.json',ready);entries.append(dict(cell=name,objective='mse',fixed_m=5,entry=proof(entry),ready=proof(dest/'ready.json')))
result=dict(status='FRESH_MATCHED_MSE_M5_SECOND_SEED_STAGED_ACCEPTED_FIXTURE_NOT_SUBMITTED',publication=proof(b/'source-publication.json'),fixture_reuse=proof(b/'fixture-reuse.json'),entries=entries,existing_128batch_entries_modified=False,Slurm_submitted=False,final4096_read=False);once(b/'staged.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time(),Slurm_mutation=False,final4096_read=False)
for name in('source-publication.json','staged.json','fixture-reuse.json','actual-seed-component-checks.json'):
 if(b/name).exists():out[name]=read(b/name)
for cell,capacity in v['cells']:
 for name in('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
  p=b/cell/name
  if p.exists():out[cell+'/'+name]=read(p)
 for p in(b/cell).glob('failure-rank-*.json'):out[cell+'/'+p.name]=read(p)
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);payload={};files={};origins={}
 for name in FILES:
  rel=P+name;origin=a.commit if name in ('online_packed_matched_objective_seeded.py','test_online_packed_matched_objective_seeded.py')else ORIGINAL;raw=subprocess.check_output(['git','show',origin+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());origins[rel]=origin
 spec=importlib.util.spec_from_file_location('window_topk_direct_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**more):
  v=dict(base=BASE,fixture=FIXTURE,branch=BRANCH,commit=a.commit,files=files,origins=origins,original=ORIGINAL,cells=CELLS,sbatch=SBATCH,seed_check=CPU_SEED_CHECK,**more);script='v='+repr(v)+'\n'+COMMON+body;compile(script,'window_topk_deploy','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:
   decode=lambda x:x.decode(errors='replace')if isinstance(x,bytes)else x
   raise RuntimeError('Direct control/stage failed: '+decode(r.stdout)[-2000:]+decode(r.stderr)[-2500:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for i,start in enumerate(range(0,len(raw),6144)):
    chunk=raw[start:start+6144];call(PART,name=name,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call(STATUS)
 dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/seeded-matched-MSE-M5';dest.mkdir(exist_ok=True,parents=True);(dest/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,SEAL,STATUS):compile(COMMON+body,'window_topk_deploy_static','exec')
if __name__=='__main__':main()
