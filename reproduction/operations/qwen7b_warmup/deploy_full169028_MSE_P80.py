"""Publish/source-stage one full169028 L15/P80 ownG5 MSE run; never submits."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-L15P80-MSE-ownG5-20261023';ORIGINAL='92ec87b0c1a02a068517c8ed84aa4b8f5727599c'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3'
BASE_SECOND=BASE.replace('seed9137-plan2026102227','seed19137-plan2026102302')
FIXTURE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2/fixture'
FILES=['online_packed_full169028_MSE_P80.py','test_full169028_MSE_P80_cpu.py','packed_variance_probe.py','online_packed_variance_scaling.py','group_topk_losses.py','runtime.sh']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-fullP80-MSE5-@@SEED_LABEL@@
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=@@BASE@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_packed_full169028_MSE_P80 --action train --base @@BASE@@ --fixture-base @@FIXTURE@@ --objective mse --fixed-m 5 --layer 15 --prefix 80 --epochs 10 --lr 0.000005 --ema-decay 0.9995 --init-seed @@INIT_SEED@@ --train-plan-seed @@TRAIN_PLAN_SEED@@ --start-mode fresh
'''
CLI_CHECK="import argparse,inspect,shlex,os\nfrom pathlib import Path\nimport reproduction.operations.qwen7b_warmup.online_packed_full169028_MSE_P80 as m\ntext=Path(os.environ['FULL_ENTRY_PATH']).read_text()\nline=next(z for z in text.splitlines()if' --action 'in z)\nargv=['--action']+shlex.split(line.split(' --action ',1)[1])\nconstruction=inspect.getsource(m.main).splitlines()[1].split(';a=p.parse_args();',1)[0].strip()\nns={'argparse':argparse,'Path':Path}\nexec(construction,ns)\na=ns['p'].parse_args(argv)\nassert a.action=='train'and a.fixture_base==Path(os.environ['FULL_FIXTURE_PATH'])and a.init_seed==int(os.environ['FULL_INIT_SEED'])and a.train_plan_seed==int(os.environ['FULL_PLAN_SEED'])and a.layer==15 and a.prefix==80 and a.objective=='mse'and a.fixed_m==5 and a.epochs==10\nassert m.TRAINING_QUESTIONS==169028 and m.actual_local_batch_layout(34)['actual_global_batch']==68\nprint('FULL_P80_SOURCE_CPU_IMPORT_AND_ACTUAL_ENTRY_CLI_PASS')\n"
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);fbase=pathlib.Path(v['fixture']);assert b.parent.is_dir(),'Existing actual shared experiment mount required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(8388608),b''):h.update(z)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable full run source/entry changed'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
STAGE='''guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(b/'controller.json').exists()and not(b/'latest.json').exists(),'Do not source-stage over an active full run'
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(exist_ok=True,parents=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(z.read_bytes()for z in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for z in parts:z.unlink()
once(b/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],file_origins=v['origins'],expanded_from_runtime_commit=v['original']))
text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@FIXTURE@@',str(fbase)).replace('@@INIT_SEED@@',str(v['init_seed'])).replace('@@TRAIN_PLAN_SEED@@',str(v['train_plan_seed'])).replace('@@SEED_LABEL@@','s'+str(v['init_seed']));entry=b/'training.sbatch'
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',FULL_ENTRY_PATH=str(entry),FULL_FIXTURE_PATH=str(fbase),FULL_INIT_SEED=str(v['init_seed']),FULL_PLAN_SEED=str(v['train_plan_seed']))
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',v['cli_check']],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0 and'FULL_P80_SOURCE_CPU_IMPORT_AND_ACTUAL_ENTRY_CLI_PASS'in r.stdout,r.stderr[-1200:]
result=dict(status='FULL169028_L15P80_MSE_OWNG5_SOURCE_STAGED_NOT_TRAINING_READY',publication=proof(b/'source-publication.json'),entry=proof(entry),fixture_expected=str(fbase),fixture_expected_status='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY',training=dict(questions=169028,epochs=10,updates_per_epoch=1321,total_updates=13210,global_batch=128,local_batch=64,micro=8,accumulation=8,tail_global_batch=68,tail_local_batch=34,tail_micro_counts=[8,8,8,8,2],fixed_m=5,layer=15,prefix=80,head_max_positions=2048,LR=5e-6,EMA_decay=.9995,init_seed=v['init_seed'],train_plan_seed=v['train_plan_seed'],diagnostic_seed=2026102228,fresh_official_block=True,pilot_resume=False,raw_to_EMA_continuous=True),resources=dict(partition='acd_u',qos='normal',GPUs=2,CPUs=12,memory_GiB=128,time='12:00:00'),replica=v['replica'],evaluation_protocol=dict(primary='viewed_independent2048_ownG5_Top512_true_Mixed',epoch_and_raw_or_EMA_selected_using_seed1_dev_only=True,same_selected_epoch_state_applied_to_seed2=True,seed2_not_used_for_selection=True,final4096_read=False),actual_SBATCH_full_CLI_parser_passed=True,CPU_import_and_entry_syntax_passed=True,fixture_CPU_acceptance_required_before_root_submission=True,P20_alignment_required_before_root_submission=True,actual_physical_execution_and_seal_source_required_before_root_submission=True,Slurm_submitted=False,final4096_read=False);once(b/'source-staged.json',result);print(json.dumps(result))
'''
QUALIFY='''if not(fbase/'fixture-complete.json').exists()or not(fbase/'accepted.json').exists():print(json.dumps(dict(status='WAITING_INDEPENDENT_FULL_FIXTURE_ACCEPTANCE_NO_TRAINING',fixture=str(fbase),Slurm_submitted=False)));raise SystemExit(0)
f=read(fbase/'fixture-complete.json');a=read(fbase/'accepted.json');assert f['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY'and f['training_questions']==169028 and f['max_real_policy_prefix']==80 and str(a['status']).endswith('ACCEPTED')and a['fixture']==proof(fbase/'fixture-complete.json')
if a.get('actual_physical_execution_and_seal_source_exact')is not True:print(json.dumps(dict(status='WAITING_ACTUAL_PHYSICAL_EXECUTION_SEAL_OWN_EM_RESOLUTION_NO_TRAINING',fixture=str(fbase),Slurm_submitted=False)));raise SystemExit(0)
physical_ref=a['physical_source_contract'];assert proof(physical_ref['path'])==physical_ref;physical=read(physical_ref['path']);assert physical['status']=='ALL169028_OWN_G10_PHYSICAL_SEAL_EXECUTION_AND_OWN_EM_RESOLVED'and physical['all_selected_own_EM_matches_canonical']is True and physical['whole_G5_not_mixed']is True and physical['selection_used_EM']is False
if a.get('fresh_full_P20_prefix_prompt_and_own_EM_exact')is not True:print(json.dumps(dict(status='WAITING_FRESH_FULL_P20_ALIGNMENT_NO_TRAINING',fixture=str(fbase),Slurm_submitted=False)));raise SystemExit(0)
assert a.get('fresh_full_P20_prefix_prompt_and_own_EM_exact')is True,'Full originalP20 prompt/prefix/ownEM alignment required';aligned=a['P20_alignment'];assert proof(aligned['path'])==aligned
assert f['validation_groups']['sha256']=='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'and f['fixed_train_G5']['sha256']=='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b'
for ref in(f['train_groups'],f['validation_groups'],f['fixed_train_G5']):assert proof(ref['path'])==ref
train=read(f['train_groups']['path']);dev=read(f['validation_groups']['path']);assert len(train)==169028 and len(dev)==2048 and not(set(train)&set(dev))and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in train.values())and all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in dev.values())
ready=dict(status='FULL169028_L15P80_MSE_OWNG5_ACCEPTED_FIXTURE_READY_NOT_SUBMITTED',source_staged=proof(b/'source-staged.json'),publication=proof(b/'source-publication.json'),entry=proof(b/'training.sbatch'),fixture=proof(fbase/'fixture-complete.json'),independent_acceptance=proof(fbase/'accepted.json'),fresh_full_P20_prefix_prompt_and_own_EM_exact=True,P20_alignment=aligned,actual_physical_execution_and_seal_source_exact=True,physical_source_contract=physical_ref,training_questions=169028,own_G10_trajectories=1690280,fixed_dev_G5=f['validation_groups'],fixed_train_G5=f['fixed_train_G5'],Slurm_submitted=False,final4096_read=False);once(b/'ready.json',ready);print(json.dumps(ready))
'''
STATUS='''out=dict(time=time.time(),Slurm_mutation=False,final4096_read=False,fixture_complete_exists=(fbase/'fixture-complete.json').exists(),fixture_accepted_exists=(fbase/'accepted.json').exists())
for name in('source-publication.json','source-staged.json','ready.json','root-submission.json','controller.json','latest.json','complete.json'):
 if(b/name).exists():out[name]=read(b/name)
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','qualify','status'),required=True);p.add_argument('--commit',required=True);p.add_argument('--replica',choices=('seed1','seed2'),default='seed1');a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);payload={};files={};origins={}
 for n in FILES:
  rel=P+n;origin=a.commit if n in('online_packed_full169028_MSE_P80.py','test_full169028_MSE_P80_cpu.py')else ORIGINAL;raw=subprocess.check_output(['git','show',origin+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());origins[rel]=origin
 spec=importlib.util.spec_from_file_location('full_p80_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  v=dict(base=BASE if a.replica=='seed1'else BASE_SECOND,replica=a.replica,init_seed=9137 if a.replica=='seed1'else 19137,train_plan_seed=2026102227 if a.replica=='seed1'else 2026102302,fixture=FIXTURE,branch=BRANCH,commit=a.commit,original=ORIGINAL,files=files,origins=origins,entry=SBATCH,cli_check=CLI_CHECK,**extra);source='v='+repr(v)+'\n'+COMMON+body;compile(source,'full_source_stage','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(str(r.stdout)[-1600:]+str(r.stderr)[-1600:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for i,start in enumerate(range(0,len(raw),6144)):
    chunk=raw[start:start+6144];call(PART,name=name,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(STAGE)
 else:result=call(QUALIFY if a.action=='qualify'else STATUS)
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-L15P80-MSE-ownG5';out=out/a.replica;out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,STAGE,QUALIFY,STATUS):compile(COMMON+body,'full_p80_source_stage_static','exec')
if __name__=='__main__':main()
