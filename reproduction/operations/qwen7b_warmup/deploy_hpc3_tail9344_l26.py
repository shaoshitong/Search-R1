"""Code-only stage / CPU projection / guarded one-H100 fixed9344 capture."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail9344-independent-L26-prepared-v2'
QUALIFIED='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-joint124194-audited243-L26-v1/code-db5b5ac04173'
RAW='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail9344-old5-qualified-g10-v1/complete.json'
RAW_SHA='0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd'
FILES=('prepare_hpc3_tail9344_l26.py','capture_hpc3_tail9344_l26.py','test_prepare_hpc3_tail9344_l26.py')
OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail9344-L26-capture'
COMMON='''import pathlib,json,hashlib,subprocess,time,os,fcntl,base64,tarfile,io,gzip
b=pathlib.Path(v['base']);code=b/('code-'+v['commit'][:12]);qualified=pathlib.Path(v['qualified']);b.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for raw in iter(lambda:f.read(8*1024**2),b''):h.update(raw)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def once(p,value):
 raw=(json.dumps(value,sort_keys=True,indent=2)+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable control metadata conflict'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr[-1200:];return r.stdout.strip()
def birth(pid):
 try:return pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def active():return command(['squeue','-h','-u','mou176','-n','q7b-tail9344-L26','-o','%i|%T|%R'])
def pub():
 p=read(b/'source-publication.json');assert p['commit']==v['commit']and p['code_root']==str(code)
 for name,item in p['files'].items():assert fp(code/name)==dict(path=str(code/name),**item)
 return p
'''
STAGE='''code.mkdir(exist_ok=True);raw=base64.b64decode(v['chunk']);assert hashlib.sha256(raw).hexdigest()==v['chunk_sha'];p=code/('upload-%04d.part'%v['part'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(part=v['part'],accepted=True)))
'''
EXTRACT='''raw=b''.join((code/('upload-%04d.part'%i)).read_bytes()for i in range(v['parts']));assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
 for item in tar:
  assert item.name in v['files']and item.isfile()and'/'not in item.name;data=tar.extractfile(item).read();p=code/item.name;assert len(data)==v['files'][item.name]['bytes']and hashlib.sha256(data).hexdigest()==v['files'][item.name]['sha256']
  if p.exists():assert p.read_bytes()==data
  else:p.write_bytes(data);p.chmod(0o444)
once(b/'source-publication.json',dict(commit=v['commit'],code_root=str(code),files=v['files'],external_qualified_runtime=str(qualified),external_qualified_publication_SHA='c8547a3dae899547fd4620afde9350c8a034c0dfee5bebd1dd9b5323a4dbdff3',raw_complete_SHA=v['raw_sha']))
pub();print(json.dumps(dict(status='SOURCE_RESOLVED_TAIL9344_L26_IMMUTABLE_CODE_STAGED',publication=fp(b/'source-publication.json'))))
'''
PREPARE='''pub();raw=pathlib.Path(v['raw'])
if not raw.exists():print(json.dumps(dict(status='WAITING_FOR_ACTUAL_FULL9344_RAW_G10_COMPLETE',GPU_submitted=False)));raise SystemExit(0)
assert fp(raw)['sha256']==v['raw_sha'];guard=(b/'prepare-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'preparation.json').exists():print(json.dumps(dict(status='EXISTING_SOURCE_RESOLVED_PREPARATION_PRESERVED',preparation=fp(b/'preparation.json'))))
elif(b/'cpu-prepare-launch.json').exists():
 old=read(b/'cpu-prepare-launch.json');print(json.dumps(dict(status='PRIOR_CPU_PREPARE_NOT_RELAUNCHED',launch=old,same_process_birth=birth(old['pid'])==old['birth_ticks'])))
else:
 assert not(b/'cpu-prepare-intent.json').exists();assert not active()
 once(b/'cpu-prepare-intent.json',dict(raw_complete=fp(raw),publication=fp(b/'source-publication.json')))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(qualified),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 args=['bash',str(qualified/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u',str(code/'prepare_hpc3_tail9344_l26.py'),'--prepare-only','--raw-sha256',v['raw_sha']]
 with(b/'cpu-prepare.log').open('ab')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(status='ACTUAL_FIXED9344_CPU_PREFIX_PROJECTION_STARTED_NOT_YET_READY',pid=p.pid,birth_ticks=birth(p.pid),raw_complete=fp(raw),epoch=time.time());once(b/'cpu-prepare-launch.json',result);print(json.dumps(result))
'''
FINALIZE='''pub();prep=read(b/'preparation.json');assert prep['status']=='FIXED9344_FULL_QUALIFIED_G10_SOURCE_RESOLVED_L26_INPUTS_PREPARED_NO_GPU'and prep['questions']==9344 and prep['trajectories']==93440 and prep['source_raw_complete']['sha256']==v['raw_sha']
for item in prep['selections']:assert fp(item['path'])==item
text=v['entry'].replace('@@BASE@@',str(b)).replace('@@NEW@@',str(code)).replace('@@QUALIFIED@@',str(qualified)).replace('@@SHA@@',fp(b/'preparation.json')['sha256']);p=b/'capture.sbatch'
if p.exists():assert p.read_text()==text
else:p.write_text(text)
subprocess.check_call(['bash','-n',str(p)]);(b/'logs').mkdir(exist_ok=True)
result=dict(status='FIXED9344_ONE_NORMAL_H100_SEQUENTIAL_FOUR_SHARDS_READY_NOT_SUBMITTED',preparation=fp(b/'preparation.json'),entry=fp(p),publication=fp(b/'source-publication.json'),gpus=1,cpus=8,memory='128G',time='02:00:00',QOS='normal',partition='acd_u',dependency=None);once(b/'submission-plan.json',result);print(json.dumps(result))
'''
SUBMIT='''pub();guard=(b/'submit-control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(b/'submission-plan.json')
for key in('preparation','entry','publication'):assert fp(plan[key]['path'])==plan[key]
assert not active(),'Same-name job exists; never repeat'
assert not(b/'submission-intent.json').exists(),'Unknown prior submission; reconcile before retry'
assert not(b/'capture-complete.json').exists()
for i in range(4):assert not(b/('shard-%02d'%i)/'capture').exists(),'Prior capture preserved; inspect exact progress first'
once(b/'submission-intent.json',dict(epoch=time.time(),plan=plan))
r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'capture.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not r.stdout.strip().split(';')[0].isdigit():raise RuntimeError('Uncertain/rejected submit, inspect same-name/intent before retry: '+r.stdout[-200:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];raw=command(['scontrol','show','job','-o',job])
for item in('JobName=q7b-tail9344-L26','Account=mou176','QOS=normal','Partition=acd_u','NumCPUs=8','MinMemoryNode=128G','TimeLimit=02:00:00','Dependency=(null)','Command='+str(b/'capture.sbatch')):assert item in raw,item
assert'gres/gpu=1'in raw or'TresPerNode=gres/gpu:1'in raw
once(b/'submission-identity.json',dict(job_id=job,job=raw,plan=plan,epoch=time.time()));command(['scontrol','release',job]);result=dict(status='FIXED9344_ONE_H100_CAPTURE_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,identity=fp(b/'submission-identity.json'));once(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),same_name_jobs=active(),base=str(b))
for name in('cpu-prepare-launch.json','progress.json','preparation.json','submission-plan.json','submission-result.json'):
 p=b/name
 if p.exists():
  x=read(p);out[name]=dict(ref=fp(p),**{k:x[k]for k in('status','pid','birth_ticks','questions','trajectories','counts','job_id','elapsed_seconds')if k in x})
for i in range(4):
 cap=b/('shard-%02d'%i)/'capture';own=dict(chunks=len(list(cap.glob('chunk-*/complete.json'))))
 for name in('progress.json','failure.json','complete.json'):
  p=cap/name
  if p.exists():
   x=read(p);own[name]=dict(ref=fp(p),**{k:x[k]for k in('status','trajectories','expected_trajectories','completed_chunks','job_id','elapsed_seconds','error','error_type')if k in x})
 out[str(i)]=own
for name in('cpu-prepare.log',):
 p=b/name
 if p.exists():out[name]=p.read_text(errors='replace')[-1800:]
for name in('bank-acceptance-progress.json','bank/complete.json'):
 p=b/name
 if p.exists():out[name]=dict(ref=fp(p),value=read(p))
if(b/'submission-result.json').exists():
 job=read(b/'submission-result.json')['job_id'];out['main_sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed']);out['log_tail']={p.name:p.read_text(errors='replace')[-1400:]for p in(b/'logs').glob('*.log')}
print(json.dumps(out))
'''
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-tail9344-L26
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=@@BASE@@/logs/capture-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@QUALIFIED@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>@@BASE@@/gpu-job.lock
flock -n 9
bash "@@QUALIFIED@@/reproduction/operations/qwen7b_warmup/runtime.sh" -u "@@NEW@@/capture_hpc3_tail9344_l26.py" --action capture --preparation-sha256 @@SHA@@
bash "@@QUALIFIED@@/reproduction/operations/qwen7b_warmup/runtime.sh" -u "@@NEW@@/capture_hpc3_tail9344_l26.py" --action merge
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','finalize','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Explicit submit authorization flag required')
 own=Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);assert own.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)
 spec=importlib.util.spec_from_file_location('fixed9344_l26_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(block,extra=None):
  args=dict(base=BASE,qualified=QUALIFIED,raw=RAW,raw_sha=RAW_SHA,commit=commit);args.update(extra or {});source='v='+repr(args)+'\n'+COMMON+block;compile(source,'fixed9344-control','exec');packed=base64.b64encode(zlib.compress(source.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-1500:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  data=io.BytesIO();files={}
  with gzip.GzipFile(fileobj=data,mode='wb',mtime=0)as gz:
   with tarfile.open(fileobj=gz,mode='w')as archive:
    for name in FILES:
     raw=subprocess.check_output(['git','show',commit+':'+P+name],cwd=R);assert raw==(R/P/name).read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o444;archive.addfile(item,io.BytesIO(raw))
  payload=data.getvalue();parts=[payload[i:i+4096]for i in range(0,len(payload),4096)]
  for i,chunk in enumerate(parts):call(STAGE,dict(part=i,chunk=base64.b64encode(chunk).decode(),chunk_sha=hashlib.sha256(chunk).hexdigest()))
  result=call(EXTRACT,dict(files=files,parts=len(parts),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
 else:result=call({'prepare':PREPARE,'finalize':FINALIZE,'submit':SUBMIT,'status':STATUS}[a.action],dict(entry=ENTRY)if a.action=='finalize'else None)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
