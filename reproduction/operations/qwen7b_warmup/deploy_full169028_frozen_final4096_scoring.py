"""Stage the future-only full169028 MSE admission scorer; never freeze, score or submit.

Only immutable source, an intentionally invalid descriptor and a CPU import
receipt are produced. Checkpoint tensors and sealed final journals stay closed.
"""
import argparse,base64,hashlib,importlib.util,json,os,shlex,signal,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];ROOT=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-final4096-source-stage-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-full169028-L15P80-MSE-20261023-v2'
SCORER_SOURCE='82c352abfbea8e6411b2a52727ef1d13555a264f'
MATURE_SOURCE='b680a97bc5965910402092f7957ccb4155f934cf'
FILES=['score_frozen_architectures_final4096.py','score_frozen_final4096.py','online_mixed_prefix_explore.py','peer_long_prefix_diagnostic.py','native_mixed_group_classifier.py','packed_variance_probe.py','online_packed_matched_objective.py','online_packed_variance_scaling.py','runtime.sh','deploy_frozen_architectures_final4096.py','score_frozen_full169028_MSE_final4096.py','test_frozen_full169028_MSE_final4096.py','deploy_full169028_frozen_final4096_scoring.py']
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-final4096-fullMSE
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=@@BASE@@/score-%j.log
set -euo pipefail
[[ "$#" == 2 ]] || { echo 'Usage: sbatch scoring.sbatch REMOTE_FREEZE_JSON SHA256'; exit 2; }
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>@@BASE@@/scoring-job.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.score_frozen_full169028_MSE_final4096 --freeze-json "$1" --freeze-sha "$2"
'''
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);code=b/('code-'+v['commit'][:12])
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Different immutable prepared scorer artifact'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def existing_mount():
 assert b.parent.is_dir()and(b.parent/'online-packed-full169028-M3to10-P20-20261022-v1/source-publication.json').is_file(),'Real existing experiment mount required; never create a fake shared path'
'''
PART='''existing_mount();b.mkdir(exist_ok=True);code.mkdir(exist_ok=True);p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['chunk_sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''existing_mount();b.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
with(b/'stage.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(b/'final-use-started.json').exists()and not(b/'complete.json').exists(),'Scorer already has actual run state'
 for name,ref in v['files'].items():
  p=code/name;parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
  if name.endswith('.py'):compile(raw,name,'exec')
  for part in parts:part.unlink()
 publication=dict(commit=v['commit'],branch=v['branch'],scorer_source_commit=v['scorer_source'],mature_architecture_source_commit=v['mature_source'],code_root=str(code),files=v['files']);once(b/'source-publication.json',publication)
 entry=b/'scoring.sbatch';text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)])
 # Import plus descriptor validation touches no checkpoint or sealed pool.
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 script='''+'"""'+'''import json
from reproduction.operations.qwen7b_warmup import score_frozen_full169028_MSE_final4096 as s
import unittest
from reproduction.operations.qwen7b_warmup import test_frozen_full169028_MSE_final4096 as test
checks=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(test))
assert checks.wasSuccessful() and checks.testsRun==5
t=s.future_descriptor()
assert t['status']=='FUTURE_ONLY_NOT_FROZEN' and t['configuration']['primary_state'] is None and t['configuration']['secondary_states']==[]
try:s.freeze_contract(t)
except (AssertionError,ValueError,TypeError,KeyError):pass
else:raise AssertionError('Future-only template unexpectedly accepted as a freeze')
print(json.dumps(dict(template=t,metadata_tests_run=checks.testsRun,metadata_tests_passed=checks.wasSuccessful()),sort_keys=True))
'''+ '"""'+'''
 result=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35)
 assert result.returncode==0,'CPU import/template rejection failed: '+result.stderr[-1800:]
 cpu=json.loads(result.stdout.strip().splitlines()[-1]);schema=cpu['template'];schema['scorer_publication']=proof(b/'source-publication.json');once(b/'candidate-freeze-TEMPLATE.json',schema)
 receipt=dict(status='FULL169028_MSE_ADMISSION_FINAL4096_SCORER_STAGED_FUTURE_ONLY',publication=proof(b/'source-publication.json'),entry=proof(entry),template=proof(b/'candidate-freeze-TEMPLATE.json'),CPU_import_passed=True,metadata_CPU_tests_run=cpu['metadata_tests_run'],metadata_CPU_tests_passed=cpu['metadata_tests_passed'],invalid_template_rejected=True,full169028_clock_and_saved_epoch_phase_required=True,both_seeds_same_physical_fixture_and_P20_alignment_required=True,CPU_acceptance_accepted_input_recursive_refs_required=True,bash_syntax_passed=True,final_journals_or_labels_read=False,model_or_checkpoint_tensors_read=False,Slurm_submitted=False,candidate_frozen=False,expected_GPUs=2,root_submission_requires=['actual_two_training_seed_acceptance_proofs','same_frozen_epoch_and_state_both_seeds','exact_freeze_JSON_and_SHA256','same_name_and_lock_check'])
 once(b/'staged.json',receipt);print(json.dumps(receipt))
'''
STATUS='''existing_mount();out=dict(time=time.time(),base=str(b),final4096_read=False,Slurm_changed=False)
for name in('staged.json','source-publication.json'):
 if(b/name).is_file():out[name]=read(b/name)
for name in('scoring.sbatch','candidate-freeze-TEMPLATE.json'):
 if(b/name).is_file():out[name]=proof(b/name)
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','status'),required=True);p.add_argument('--commit',required=True);p.add_argument('--route',choices=('direct','campus'),default='direct');a=p.parse_args()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);subprocess.check_call(['git','merge-base','--is-ancestor',SCORER_SOURCE,a.commit],cwd=R)
 payload={P+n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};files={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for name,raw in payload.items()}
 assert files[P+'score_frozen_full169028_MSE_final4096.py']['sha256']=='65d853fefd922b85ec723c9194b8e1d823703652925059a8d74e898e2349d791'
 spec=importlib.util.spec_from_file_location('architecture_final_private',ROOT/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  values=dict(base=BASE,commit=a.commit,scorer_source=SCORER_SOURCE,mature_source=MATURE_SOURCE,branch=BRANCH,files=files,entry=ENTRY,**extra);script='v='+repr(values)+'\n'+COMMON+body;packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000
  if a.route=='direct':result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  else:
   child=subprocess.Popen(['python3',str(ROOT/'work/hpc3_private_control.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
   try:stdout,stderr=child.communicate(command.encode(),timeout=60)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.communicate();raise RuntimeError('Explicit campus stage route timed out; inspect actual outcome before retry')
   result=subprocess.CompletedProcess(child.args,child.returncode,stdout,stderr)
  if result.returncode:
   decode=lambda value:value.decode(errors='replace')if isinstance(value,bytes)else value
   raise RuntimeError('Stage/control failed: '+decode(result.stdout)[-1800:]+decode(result.stderr)[-1800:])
  return json.loads(result.stdout.strip().splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for index,offset in enumerate(range(0,len(raw),6144)):
    chunk=raw[offset:offset+6144];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call(STATUS)
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-full169028-MSE-final4096-scoring';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))

for body in(PART,SEAL,STATUS):compile(COMMON+body,'architecture-final-stage-static','exec')
if __name__=='__main__':main()
