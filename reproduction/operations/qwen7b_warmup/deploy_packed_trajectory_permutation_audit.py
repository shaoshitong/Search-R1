"""Stage an independent E8 viewed-dev permutation audit. Never submit Slurm."""
import argparse, base64, hashlib, importlib.util, json, shlex, subprocess, zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]; CONTROL=R.parents[1]; P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-packed-E8-trajectory-permutation-audit-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit-20261023-v1'
FILES=['packed_trajectory_permutation_audit.py','test_packed_trajectory_permutation_audit.py','deploy_packed_trajectory_permutation_audit.py','packed_variance_probe.py','runtime.sh']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-ZE8-order-audit
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=00:25:00
#SBATCH --output=@@BASE@@/audit-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>@@BASE@@/audit-job.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.packed_trajectory_permutation_audit --config @@CONFIG@@ --config-sha @@CONFIGSHA@@
'''
COMMON='''import pathlib,json,hashlib,base64,os,subprocess
b=pathlib.Path(v['base']);a=b.parent;full=a/'online-packed-full169028-M3to10-P20-20261022-v1';assert full.is_dir()and(full/'source-publication.json').is_file(),'Actual shared experiment mount required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable audit stage differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''for name,ref in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for part in parts:part.unlink()
training_pub=json.loads((full/'source-publication.json').read_bytes());key='reproduction/operations/qwen7b_warmup/packed_variance_probe.py';assert v['files'][key]['sha256']==training_pub['files'][key]['sha256'],'Exact E8 packed architecture source required'
pub=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']);once(b/'source-publication.json',pub)
source=full/'z_mse_anchor';cp=source/'checkpoint-epoch-08.pt';assert cp.stat().st_size==3773206036
ev=source/'evaluation-epoch-08.json';actual=json.loads(ev.read_bytes());assert actual['epoch']==8 and actual['cursor']==10568
fixture=a/'online-mixed-layer-prefix-20261022-v1/fixture-complete.json';fixed=a/'online-mixed-layer-prefix-20261022-v1/inputs/validation-groups.json';assert proof(fixture)['sha256']=='2aa8f1de7dbb64412bb03088803b1c69d4e61fa63476e8c10536d2bd20911d5d'and proof(fixed)['sha256']=='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'
official=full.parent.parent/'assets/qwen25-7b-instruct-a09a35458c70/complete.json';oi=json.loads(official.read_bytes());assert oi['revision']=='a09a35458c702b33eeacc393d103063234e8bc28'and oi['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'
cfg=dict(status='VIEWED_DEV_E8_PACKED_PERMUTATION_AUDIT_BOUND',base=str(b),checkpoint=dict(path=str(cp),bytes=3773206036,sha256='2d81f15cd4af00fe192963cba6f6db86aba22bf47523bd3e10982b86f7178483'),source_evaluation=proof(ev),source_saved_predictions=proof(source/'epoch-08-fixed_validation2048_G5-predictions.jsonl.gz'),fixture=proof(fixture),fixed_validation=proof(fixed),official_inventory=proof(official),source_training_publication=proof(full/'source-publication.json'),source_publication=proof(b/'source-publication.json'),orders=['original','reverse','cyclic1'],states=['raw','EMA'],questions=2048,layer=21,prefix=20,M=5,selection_score='predicted_population_EM_variance',variance_threshold=.08,final4096_read=False)
once(b/'audit-config.json',cfg);config=proof(b/'audit-config.json');entry=b/'audit.sbatch';text=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@CONFIG@@',config['path']).replace('@@CONFIGSHA@@',config['sha256'])
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)])
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.test_packed_trajectory_permutation_audit'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=35);assert r.returncode==0,r.stderr[-1200:]
ready=dict(status='PACKED_Z_E8_DEV2048_TRAJECTORY_ORDER_AUDIT_STAGED_NOT_SUBMITTED',entry=proof(entry),config=config,source_publication=proof(b/'source-publication.json'),checkpoint=cfg['checkpoint'],CPU_packing_tests_passed=True,CPU_tests=3,checkpoint_contents_opened=False,checkpoint_fresh_SHA_deferred_to_owned_GPU_job=True,final4096_read=False,Slurm_submitted=False,resources=dict(GPUs=2,CPUs=12,memoryGiB=128,partition='acd_u',qos='normal',time='00:25:00'),expected_work=dict(questions=2048,orders=3,actor_prefill_rows=6144,raw_EMA_share_actor_features=True),existing_jobs_changed=False)
once(b/'staged.json',ready);print(json.dumps(ready))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);args=p.parse_args()
    tip=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',args.commit,tip],cwd=R)
    payload={P+n:subprocess.check_output(['git','show',args.commit+':'+P+n],cwd=R)for n in FILES};files={n:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in payload.items()}
    spec=importlib.util.spec_from_file_location('audit_existing_direct',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(source,**more):
        v=dict(base=BASE,commit=args.commit,branch=BRANCH,files=files,sbatch=SBATCH,**more);data=base64.b64encode(zlib.compress(('v='+repr(v)+'\n'+COMMON+source).encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(data)+')))');assert len(cmd.encode())<16000
        r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
        if r.returncode:raise RuntimeError(r.stdout[-1200:]+r.stderr[-1200:])
        return json.loads(r.stdout.splitlines()[-1])
    for name,raw in payload.items():
        for index,start in enumerate(range(0,len(raw),3072)):
            part=raw[start:start+3072];call(PART,name=name,index=index,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest())
    ready=call(SEAL);dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/packed-Z-E8-trajectory-permutation-audit';dest.mkdir(parents=True,exist_ok=True);(dest/'actual-stage.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(ready))

for source in(COMMON,PART,SEAL):compile(source,'packed_permutation_stage','exec')
if __name__=='__main__':main()
