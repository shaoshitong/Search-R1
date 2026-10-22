"""Publish/stage/start/status the single owned CPU Prefix80 preparation."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-full169028-policy80-inputs-20261023'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-ownG10-20261023-v1'
FILES=['prepare_full169028_policy80.py','test_prepare_full169028_policy80.py','deploy_full169028_policy80.py']
ENTRY='''#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec python3 -u @@CODE@@/reproduction/operations/qwen7b_warmup/prepare_full169028_policy80.py --base @@BASE@@ --chunk-sources 64
'''
COMMON='''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);a=b.parent;assert a.is_dir()and(a/'online-packed-full169028-M3to10-P20-20261022-v1/fixture/full-training-descriptor.json').is_file()and(a/'online-packed-matched-L15P80-8192-20261023-v1/fixture/accepted.json').is_file(),'Actual shared accepted source mount required';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
def once(p,x):
 raw=(json.dumps(x,sort_keys=True,separators=(',',':'))+chr(10)).encode()
 if p.exists():assert p.read_bytes()==raw,'Immutable CPU publication or entry differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def live():
 p=b/'prepare-process.json'
 if not p.exists():return False
 r=read(p);proc=pathlib.Path('/proc/%d'%r['pid'])
 if not proc.exists():return False
 command=(proc/'cmdline').read_bytes();return(proc/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']and(str(b).encode()in command or str(b/'prepare.sh').encode()in command)and(b'prepare_full169028_policy80.py'in command or b'prepare.sh'in command)
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''guard=(b/'stage.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,ref in v['files'].items():
 p=code/name;parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for part in parts:part.unlink()
once(b/'source-publication.json',dict(commit=v['commit'],branch=v['branch'],code_root=str(code),producer_relative='reproduction/operations/qwen7b_warmup/prepare_full169028_policy80.py',files=v['files']))
entry=b/'prepare.sh';text=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code))
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text)
subprocess.check_call(['bash','-n',str(entry)]);out=dict(status='FULL169028_TRUE_POLICY80_CPU_PIPELINE_STAGED_NOT_STARTED',base=str(b),fixture=str(b/'fixture'),publication=proof(b/'source-publication.json'),entry=proof(entry),new_rollouts_generated=0,GPU_used=False,Slurm_submitted=False,final4096_read=False);once(b/'staged.json',out);print(json.dumps(out))
'''
START='''guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
if(b/'complete.json').exists():print(json.dumps(dict(status='EXISTING_COMPLETE_CPU_PIPELINE_REUSED',complete=read(b/'complete.json'))))
elif live():print(json.dumps(dict(status='EXISTING_OWNED_CPU_PROCESS_REUSED',process=read(b/'prepare-process.json'))))
else:
 check=(b/'prepare.lock').open('a+');fcntl.flock(check,fcntl.LOCK_EX|fcntl.LOCK_NB);check.close();log=(b/'prepare.log').open('ab');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');p=subprocess.Popen(['bash',str(b/'prepare.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);proc=pathlib.Path('/proc/%d'%p.pid);receipt=dict(status='PERSISTENT_CPU_POLICY80_PIPELINE_STARTED_NOT_ACCEPTED',pid=p.pid,start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],commit=v['commit'],entry=proof(b/'prepare.sh'),time=time.time(),new_rollouts_generated=0,GPU_used=False,Slurm_submitted=False,final4096_read=False);tmp=b/'prepare-process.json.tmp';tmp.write_text(json.dumps(receipt,sort_keys=True)+chr(10));tmp.replace(b/'prepare-process.json');print(json.dumps(receipt))
'''
STATUS='''out=dict(time=time.time(),base=str(b),actual_owned_process_live=live(),final4096_read=False,Slurm_changed=False)
for name in('prepare-process.json','controller.json','progress.json','failure.json','complete.json','fixture/accepted.json'):
 if(b/name).is_file():out[name]=read(b/name)
if(b/'prepare.log').is_file():
 with(b/'prepare.log').open('rb')as f:f.seek(max(0,(b/'prepare.log').stat().st_size-2200));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','start','status'),required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,universal_newlines=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 payload={P+n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};files={k:dict(bytes=len(x),sha256=hashlib.sha256(x).hexdigest())for k,x in payload.items()};spec=importlib.util.spec_from_file_location('full_P80_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],universal_newlines=True).strip()
 def call(body,**more):
  v=dict(base=BASE,branch=BRANCH,commit=a.commit,files=files,entry=ENTRY,**more);script='v='+repr(v)+'\n'+COMMON+body;packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=45)
  if r.returncode:
   decode=lambda x:x.decode(errors='replace')if isinstance(x,bytes)else x
   raise RuntimeError('CPU pipeline control did not complete: '+decode(r.stdout)[-1200:]+decode(r.stderr)[-1800:])
  return json.loads(r.stdout.strip().splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for index,offset in enumerate(range(0,len(raw),6144)):
    part=raw[offset:offset+6144];call(PART,name=name,index=index,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest())
  out=call(SEAL)
 else:out=call(START if a.action=='start'else STATUS)
 dest=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-P80-preparation';dest.mkdir(exist_ok=True,parents=True);(dest/('actual-'+a.action+'.json')).write_text(json.dumps(out,sort_keys=True)+'\n');print(json.dumps(out))
for body in(PART,SEAL,START,STATUS):compile(COMMON+body,'full_policy80_cpu_deployment','exec')
if __name__=='__main__':main()
