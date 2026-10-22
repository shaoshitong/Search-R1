"""Only stage/start/status the owned durable CPU physical-source auditor."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[4];CONTROL=REPO.parents[1];BRANCH='codex/q7b-viewed-label-ownEM-audit-20261023';REL='reproduction/reports/qwen7b-algorithm-improvement-20261021/viewed-label-ownEM-audit/audit-pilot8192-physical.py';BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed-pilot8192-physical-source-audit-20261023-v1'
COMMON='''import pathlib,os,json,time,hashlib,base64,subprocess,fcntl
b=pathlib.Path(v['base']);assert b.parent.is_dir();b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);runtime=code/'audit-pilot8192-physical.py'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def live():
 p=b/'started.json'
 if not p.exists():return False
 r=read(p);proc=pathlib.Path('/proc/%d'%r['pid'])
 return proc.exists()and(proc/'stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']and str(runtime).encode()in(proc/'cmdline').read_bytes()and str(b).encode()in(proc/'cmdline').read_bytes()
'''
STAGE='''assert not live();raw=base64.b64decode(v['payload']);assert len(raw)==v['runtime']['bytes']and hashlib.sha256(raw).hexdigest()==v['runtime']['sha256'];compile(raw,str(runtime),'exec')
if runtime.exists():assert runtime.read_bytes()==raw
else:runtime.write_bytes(raw);runtime.chmod(0o444)
pub=dict(commit=v['commit'],branch=v['branch'],runtime=v['runtime'],code_root=str(code),CPU_only=True,Slurm_mutated=False,final4096_read=False);p=b/'source-publication.json';raw=(json.dumps(pub,sort_keys=True,separators=(',',':'))+chr(10)).encode()
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(status='OWNED_CPU_PHYSICAL_AUDITOR_STAGED_NOT_STARTED',runtime=proof(runtime),publication=proof(p),Slurm_mutated=False,final4096_read=False)))
'''
START='''guard=(b/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);pub=read(b/'source-publication.json');assert pub['commit']==v['commit']and proof(runtime)==dict(pub['runtime'],path=str(runtime))
if(b/'complete.json').exists():print(json.dumps(dict(status='EXISTING_COMPLETE_AUDIT_REUSED',complete=read(b/'complete.json'))))
elif live():print(json.dumps(dict(status='EXISTING_AUDIT_PROCESS_REUSED',started=read(b/'started.json'))))
else:
 assert not(b/'failure.json').exists(),'Inspect actual prior failure before any retry';g=(b/'audit.lock').open('a+');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);g.close();log=(b/'audit.log').open('ab');p=subprocess.Popen(['python3','-u',str(runtime),'--base',str(b)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'));proc=pathlib.Path('/proc/%d'%p.pid);receipt=dict(time=time.time(),pid=p.pid,start_ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19],runtime=proof(runtime),commit=v['commit'],CPU_only=True,Slurm_mutated=False,final4096_read=False);(b/'started.json').write_text(json.dumps(receipt)+'\n');print(json.dumps(dict(status='OWNED_CPU_PHYSICAL_AUDITOR_STARTED_NOT_ACCEPTED',started=receipt)))
'''
STATUS='''out=dict(time=time.time(),actual_owned_process_live=live(),CPU_only=True,Slurm_mutated=False,final4096_read=False)
for name in('source-publication.json','started.json','controller.json','progress.json','failure.json','complete.json'):
 if(b/name).exists():out[name]=read(b/name)
if(b/'audit.log').exists():
 with(b/'audit.log').open('rb')as f:f.seek(max(0,(b/'audit.log').stat().st_size-1800));out['log_tail']=f.read().decode(errors='replace')
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',required=True,choices=['stage','start','status']);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=REPO);raw=subprocess.check_output(['git','show',a.commit+':'+REL],cwd=REPO);v=dict(base=BASE,commit=a.commit,branch=BRANCH,runtime=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),payload=base64.b64encode(raw).decode());src='v='+repr(v)+'\n'+COMMON+{'stage':STAGE,'start':START,'status':STATUS}[a.action];packed=base64.b64encode(zlib.compress(src.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');s=importlib.util.spec_from_file_location('pilot_own_audit_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=m.run(command,account='mou176',source_address=ip,timeout=30)
 if r.returncode:raise RuntimeError(str(r.stdout)[-1600:]+str(r.stderr)[-1600:])
 out=json.loads(r.stdout.splitlines()[-1]);dest=Path(__file__).resolve().parent/('actual-pilot-physical-'+a.action+'.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
for body in(STAGE,START,STATUS):compile(COMMON+body,'own_audit_control','exec')
if __name__=='__main__':main()
