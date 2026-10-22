"""Source-stage future selected fullP80 order audit, never CP/GPU/final/Slurm."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-L15P80-permutation-audit-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1'
TRAINING='7a8ff6ae8a33324846d85f77ffc28e9ab699c028';PROTOCOL='4a7e2e80ca98ac3fe6cce75e312cdf95defb4cd3'
FILES=['packed_full169028_trajectory_permutation_audit.py','test_packed_full169028_trajectory_permutation_audit.py','deploy_packed_full169028_permutation_audit.py','packed_trajectory_permutation_audit.py','online_packed_full169028_MSE_P80.py','packed_variance_probe.py','runtime.sh']
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-fullP80-order-audit
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=@@BASE@@/audit-%j.log
set -euo pipefail
[[ "$#" == 2 ]] || { echo 'Usage: sbatch audit.sbatch ROOT_BOUND_VIEWED_AUDIT_JSON SHA256'; exit 2; }
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/audit-job.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.packed_full169028_trajectory_permutation_audit --config "$1" --config-sha "$2"
'''
COMMON='''import pathlib,json,hashlib,base64,os,subprocess,time,fcntl
b=pathlib.Path(v['base']);assert b.parent.is_dir(),'Actual shared mount required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable audit stage changed'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
STAGE='''guard=(b/'source-stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(b/'controller.json').exists()and not(b/'complete.json').exists(),'No source-stage over active audit'
for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(exist_ok=True,parents=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(q.read_bytes()for q in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for q in parts:q.unlink()
once(b/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],file_origins=v['origins'],training_source=v['training'],mature_permutation_protocol=v['protocol']))
entry=b/'audit.sbatch';text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.test_packed_full169028_trajectory_permutation_audit'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0 and'Ran 5 tests'in r.stderr,r.stderr[-1200:]
script='import json;from reproduction.operations.qwen7b_warmup.packed_full169028_trajectory_permutation_audit import future_template;print(json.dumps(future_template('+repr(proof(b/'source-publication.json'))+')))'
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert r.returncode==0,r.stderr[-1200:];template=json.loads(r.stdout.splitlines()[-1]);assert template['status']=='FUTURE_ONLY_UNSELECTED_NOT_AUTHORIZED'and template['epoch']is None and template['state']is None;once(b/'future-unbound-audit-template.json',template)
result=dict(status='FULL169028_L15P80_MSE_PERMUTATION_AUDIT_SOURCE_STAGED_ONLY_NOT_AUTHORIZED_NOT_SUBMITTED',source_publication=proof(b/'source-publication.json'),entry=proof(entry),invalid_future_template=proof(b/'future-unbound-audit-template.json'),CPU_tests_passed=5,source_import_and_entry_syntax_passed=True,training_geometry=dict(layer=15,prefix=80,head_max_positions=2048,training_questions=169028,objective='mse'),protocol=dict(pool='viewed_NQ_fixed2048_ownG5',same_selected_epoch_state_both_seeds=True,orders=['original','reverse','cyclic1'],no_order_selection=True,original_saved_score_alignment_required=True,fixed_absolute_score_tolerance=.001),resources=dict(GPUs=2,CPUs=12,memory_GiB=128,partition='acd_u',qos='normal',time='01:00:00'),actual_work_after_root_authorization=dict(questions=2048,replicas=2,orders=3,actor_prefill_rows=12288,probe_states_per_replica=1),checkpoint_files_statted_hashed_or_opened=False,real_prediction_or_EM_aggregate_read=False,GPU_started=False,Slurm_submitted=False,final4096_read=False,existing_jobs_changed=False);once(b/'staged.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(time=time.time(),base=str(b),Slurm_submitted=False,final4096_read=False,checkpoint_files_statted_hashed_or_opened=False)
for name in('source-publication.json','staged.json'):
 if(b/name).exists():out[name]=dict(proof=proof(b/name),value=json.loads((b/name).read_bytes()))
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','status'),default='stage');a=p.parse_args();tip=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,tip],cwd=R);payload={};files={};origins={}
 for name in FILES:
  origin=TRAINING if name in('online_packed_full169028_MSE_P80.py','packed_variance_probe.py','runtime.sh')else PROTOCOL if name=='packed_trajectory_permutation_audit.py'else a.commit;rel=P+name;raw=subprocess.check_output(['git','show',origin+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());origins[rel]=origin
 spec=importlib.util.spec_from_file_location('full_permutation_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  value=dict(base=BASE,commit=a.commit,branch=BRANCH,training=TRAINING,protocol=PROTOCOL,files=files,origins=origins,entry=ENTRY,**extra);script='v='+repr(value)+'\n'+COMMON+body;compile(script,'stage_full_permutation','exec');data=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(data)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-1600:]+r.stderr[-1600:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for index,start in enumerate(range(0,len(raw),6144)):
    chunk=raw[start:start+6144];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(STAGE)
 else:result=call(STATUS)
 dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-L15P80-permutation-audit';dest.mkdir(exist_ok=True,parents=True);(dest/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
for body in(PART,STAGE,STATUS):compile(COMMON+body,'full_permutation_stage_static','exec')
if __name__=='__main__':main()
