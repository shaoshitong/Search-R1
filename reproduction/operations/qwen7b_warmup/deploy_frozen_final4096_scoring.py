"""Publish and stage a final scoring entry; never submits or opens final data."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]; CONTROL=R.parents[1]; P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-frozen-final4096-scoring-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-scoring-20261022-v1'
FILES=['score_frozen_final4096.py','deploy_frozen_final4096_scoring.py','online_mixed_prefix_explore.py','peer_long_prefix_diagnostic.py','native_mixed_group_classifier.py','runtime.sh']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-final4096-paired
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
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.score_frozen_final4096 --freeze-json "$1" --freeze-sha "$2"
'''
COMMON='''import pathlib,json,hashlib,base64,subprocess,os
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,x):
 raw=(json.dumps(x,sort_keys=True)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Different immutable prepared scorer artifact'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''for name,item in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for x in parts:x.unlink()
pub=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']);once(b/'source-publication.json',pub)
entry=b/'scoring.sbatch';text=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
schema=dict(status='TEMPLATE_ONLY_NOT_CANDIDATE_FREEZE',required_status='CANDIDATE_FROZEN_BEFORE_FINAL_SCORES',freeze_epoch='ROOT_ACTUAL_FREEZE_TIME',final_labels_or_scores_used_for_selection=False,further_candidate_selection_on_final_allowed=False,primary_validation_questions=2048,baseline=v['baseline'],candidate=dict(checkpoint=dict(path='REMOTE_ACCEPTED_CHECKPOINT',bytes='EXACT_BYTES',sha256='EXACT_SHA256'),state_key='EMA',layer=27,prefix=80),viewed_validation_evaluation=dict(path='REMOTE_VIEWED_VALIDATION_EVALUATION',bytes='EXACT_BYTES',sha256='EXACT_SHA256'),scorer_publication=proof(b/'source-publication.json'))
once(b/'candidate-freeze-TEMPLATE.json',schema)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');script='import reproduction.operations.qwen7b_warmup.score_frozen_final4096 as s; assert s.E1 == '+repr(v['baseline']['checkpoint'])+'; print("FINAL_SCORER_CPU_IMPORT_WITHOUT_FINAL_READ_PASS")';r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',script],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0 and'FINAL_SCORER_CPU_IMPORT_WITHOUT_FINAL_READ_PASS'in r.stdout,r.stderr[-1000:]
receipt=dict(status='FINAL4096_SCORER_STAGED_NOT_SUBMITTED_NOT_SCORED',publication=proof(b/'source-publication.json'),entry=proof(entry),template=proof(b/'candidate-freeze-TEMPLATE.json'),CPU_import_passed=True,bash_syntax_passed=True,final_journals_or_labels_read=False,model_or_checkpoint_tensors_read=False,Slurm_submitted=False,candidate_frozen=False,expected_GPUs=2,root_submission_requires=['exact_candidate_freeze_JSON','freeze_SHA256','same_name_and_lock_check'])
once(b/'staged.json',receipt);print(json.dumps(receipt))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0]==a.commit
 payload={P+n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in payload.items()}
 spec=importlib.util.spec_from_file_location('final_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 baseline=dict(checkpoint=dict(path='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1/checkpoints/cursor-002577.pt',bytes=3781073882,sha256='c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917'),state_key='ema',layer=27,prefix=20)
 def call(source,**more):
  v=dict(base=BASE,commit=a.commit,branch=BRANCH,files=files,sbatch=SBATCH,baseline=baseline,**more);raw='v='+repr(v)+'\n'+COMMON+source;packed=base64.b64encode(zlib.compress(raw.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-1000:])
  return json.loads(r.stdout.splitlines()[-1])
 for name,raw in payload.items():
  for index,start in enumerate(range(0,len(raw),3072)):
   chunk=raw[start:start+3072];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
 result=call(SEAL);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/frozen-final4096-scoring';out.mkdir(parents=True,exist_ok=True);(out/'actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


for script in (COMMON,PART,SEAL):compile(script,'final_scoring_stage_static','exec')
if __name__=='__main__':main()
