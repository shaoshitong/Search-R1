"""One isolated normal H100 diagnostic; no main-state/process mutation."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
CONTROL=R.parents[1]
P='reproduction/operations/qwen7b_warmup/'
MAIN='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1'
BASE=MAIN.rsplit('/',1)[0]+'/mixed-direct-performance-20261022-v1'
REMOTE=r'''import pathlib,json,hashlib,base64,zlib,os,subprocess,fcntl,time,re
v=__V__;b=pathlib.Path(v['base']);main=pathlib.Path(v['main']);b.mkdir(exist_ok=True);code=b/'code';code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def checked(p,ref):
 raw=pathlib.Path(p).read_bytes();assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'];return raw
def immutable(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Existing diagnostic entry differs'
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert r.returncode==0,r.stderr[-1200:];return r.stdout.strip()
if v['action']=='stage':
 publication=read(main/'source-publication.json');original=pathlib.Path(publication['code_root']);origins={}
 for name,ref in publication['files'].items():
  raw=checked(original/name,ref);immutable(code/name,raw);origins[name]=dict(origin=publication['file_origins'][name],sha256=ref['sha256'],bytes=len(raw))
 for name,ref in v['files'].items():
  raw=zlib.decompress(base64.b64decode(ref['data']));assert hashlib.sha256(raw).hexdigest()==ref['sha256'];immutable(code/name,raw);origins[name]={k:ref[k]for k in('origin','sha256','bytes')}
 immutable(b/'source-publication.json',(json.dumps(dict(files=origins,code_root=str(code),main_config_sha256='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393'),sort_keys=True)+'\n').encode())
 entry=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@MAIN@@',str(main));immutable(b/'profile.sbatch',entry.encode());(b/'logs').mkdir(exist_ok=True);subprocess.check_call(['bash','-n',str(b/'profile.sbatch')]);print(json.dumps(dict(status='PROFILE_STAGED_NOT_SUBMITTED',base=str(b),source_files=len(origins))))
elif v['action']=='submit':
 lock=(b/'submit.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(b/'submit-intent.json').exists(),'Existing/uncertain submit: inspect instead of duplicate'
 assert not command(['squeue','-h','-u','mou176','-n','q7b-mixed-perf','-o','%i|%T']), 'Existing profile job'
 (b/'submit-intent.json').write_text(json.dumps(dict(time=time.time(),source=read(b/'source-publication.json'))))
 job=command(['sbatch','--parsable',str(b/'profile.sbatch')]);assert re.fullmatch(r'[0-9]+',job)
 (b/'submission.json').write_text(json.dumps(dict(job_id=job,time=time.time())))
 print(json.dumps(dict(status='PROFILE_SUBMITTED_NOT_COMPLETED',job_id=job,identity=command(['scontrol','show','job','-o',job]))))
elif v['action']=='status':
 out=dict(epoch=time.time(),base=str(b))
 for name in('submission.json','reader-timing.json','compute-timing.json','complete.json'):
  if(b/name).exists():out[name]=read(b/name)
 if(b/'submission.json').exists():
  job=read(b/'submission.json')['job_id'];out['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed']);log=b/'logs'/('profile-'+job+'.log')
  if log.exists():
   with log.open('rb')as f:f.seek(0,2);size=f.tell();f.seek(max(0,size-2500));out['log_tail']=f.read().decode(errors='replace')
 print(json.dumps(out))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-mixed-perf
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/logs/profile-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.profile_mixed_direct_io --source-config @@MAIN@@/config.json --output @@BASE@@/reader-timing.json
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.native_mixed_compute_benchmark --source-config @@MAIN@@/config.json --output @@BASE@@/compute-timing.json
python3 -c 'import json,time,pathlib;pathlib.Path("@@BASE@@/complete.json").write_text(json.dumps(dict(status="ISOLATED_READER_AND_GPU_PROFILE_FINISHED",epoch=time.time())))'
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','submit','status'),required=True);p.add_argument('--source-commit');p.add_argument('--benchmark-commit');a=p.parse_args()
 files={}
 if a.action=='stage':
  if not a.source_commit or not a.benchmark_commit:raise ValueError('Both exact published origins required')
  for name,origin in ((P+'replay_feature_reader_fast.py',a.source_commit),(P+'profile_mixed_direct_io.py',a.source_commit),(P+'native_mixed_compute_benchmark.py',a.benchmark_commit)):
   raw=subprocess.check_output(['git','show',origin+':'+name],cwd=R);files[name]=dict(origin=origin,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(zlib.compress(raw)).decode())
 values=dict(action=a.action,base=BASE,main=MAIN,files=files,entry=ENTRY)
 source=REMOTE.replace('__V__',repr(values),1);packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
 spec=importlib.util.spec_from_file_location('profile_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=m.run(command,account='mou176',source_address=ip,timeout=55);print(r.stdout.decode());
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1200:])
if __name__=='__main__':main()
