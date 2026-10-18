"""Authorized CP64 recovery on idle H1008, using retained18 NFS in place."""
import argparse,base64,gzip,hashlib,json,subprocess,time
from pathlib import Path
from .control import REPO,OUT,remote
NAME='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp64-h1008-20261019-r1'
P='/mnt/data/shaoshitong/'+NAME
C='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+NAME
SOURCE='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp50-eight-20261018-r2'

def call(script,phase,node='h100-10'):
 f=OUT/('resume64-'+phase+'-'+str(time.time_ns())+'-transport.json')
 v=json.loads(remote(script,f,target=node,established_controller=True));dest=f.with_name(f.name.replace('-transport','-actual'))
 dest.write_text(json.dumps(v,indent=2));print(json.dumps(dict(evidence=str(dest),actual=v),ensure_ascii=False),flush=True);return v

def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['stage','poll','submit','observe']);args=a.parse_args()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 pre='import pathlib,json,subprocess,time,os\np=pathlib.Path('+repr(P)+');cbase='+repr(C)+'\n'
 if args.phase=='stage':
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
  assert subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/b-diverse10to5-cp50-eight-20261018'],cwd=REPO,text=True).split()[0]==commit
  changed=subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMRT','ff04ea81e4f3d7783d9c84c558faa515e5eca79f',commit],cwd=REPO,text=True).splitlines()
  archive=base64.b64encode(gzip.compress(subprocess.check_output(['git','archive',commit,'--',*changed],cwd=REPO))).decode()
  code=r'''
import sys,uuid,pathlib,json,time
from reproduction.prefix_selection200.diverse_native_resume import qualify,prepare,PROTECTED
from reproduction.prefix_selection200.diverse_resume import read,sha,once
b=pathlib.Path(BASE);source=pathlib.Path(SOURCE);old=read(source/'preparation/launch-config.json')
q=qualify(source,b/'preparation/native-cp64-full-sha-cpu.json');assert q['outer']==64
def rebase(v):
 if isinstance(v,dict):return{k:rebase(x)for k,x in v.items()}
 if isinstance(v,list):return[rebase(x)for x in v]
 if isinstance(v,str)and(v==str(source)or v.startswith(str(source)+'/')):return str(b)+v[len(str(source)):]
 return v
c=rebase(old);uuids=UUIDS
c.update(base=str(b),execution_id=uuid.uuid4().hex,source_commit=COMMIT,expected_execution_node='h100-8',physical_gpu_uuids=uuids,auxiliary_gpu_uuids=uuids)
c['resume_joint']=dict(adapter='b_diverse_4to8_cp50_v1',recovery_mode='diverse_same_eight_latest_joint_v1',source_base=str(source),completed_outer=64)
for key in ('resume_cached_sampling','runtime_compatibility','startup_sampling_resume'):c.pop(key,None)
protected={name:sha(source/'repo/reproduction/prefix_selection200'/name)for name in PROTECTED}
assert all(sha(b/'repo/reproduction/prefix_selection200'/name)==value for name,value in protected.items())
proof=dict(status='DIVERSE_SAME_EIGHT_NATIVE_RECOVERY',runtime_commit=COMMIT,binding_commit=old['actor_checkpoint_binding_commit'],source_commit=old['source_commit'],source_base=str(source),completed_outer=64,protected_sha256=protected)
compat=b/'preparation/runtime-compatibility.json';once(compat,proof);c['runtime_compatibility']=dict(path=str(compat),sha256=sha(compat))
once(b/'preparation/launch-config.json',c)
r=prepare(source,b,c,b/'preparation/native-cp64-full-sha-cpu.json')
once(b/'preparation/native-recovery-complete.json',dict(config=c,imported=r,ready_unix=time.time()))
print(json.dumps(dict(status='CP64_NATIVE_READY',base=str(b),actor=r['actor_adam'],probe=r['probe_adam'])))
'''
  resources=sorted(OUT.glob('resources-h100-8-*-actual.json'))[-1];inventory=json.loads(resources.read_text())
  uuids=[line.split(',')[1].strip()for line in inventory['gpu'].splitlines()]
  assert len(uuids)==8
  code='BASE='+repr(C)+'\nSOURCE='+repr(SOURCE)+'\nCOMMIT='+repr(commit)+'\nUUIDS='+repr(uuids)+'\n'+code
  s=pre+'ARCHIVE='+repr(archive)+'\nWORKER='+repr(code)+'\n'+r'''
import base64,gzip,tarfile,io,hashlib,shutil
assert not p.exists(),'Existing recovery must be reconciled, never overwritten'
source=p.parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp50-eight-20261018-r2'
shutil.copytree(source/'repo',p/'repo',ignore=shutil.ignore_patterns('__pycache__','.git'));(p/'preparation').mkdir();(p/'logs').mkdir()
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(ARCHIVE))))as tf:
 for m in tf.getmembers():assert not m.name.startswith('/')and'..'not in pathlib.Path(m.name).parts and(m.isfile()or m.isdir())
 tf.extractall(p/'repo')
(p/'preparation/cpu-recovery.py').write_text(WORKER)
rows=[dict(path=str(f.relative_to(p/'repo')),sha256=hashlib.sha256(f.read_bytes()).hexdigest())for f in(p/'repo').rglob('*')if f.is_file()]
(p/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,files=rows)))
wrapper=p/'repo/reproduction/operations/b_replica_retained18/private_paths.sh'
command=['unshare','--mount','--propagation','private','bash',str(wrapper),'env','SEARCH_G1_PHYSICAL_WORLD_SIZE=8','SEARCH_G1_B_PREFIX_TOKENS=20','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+cbase+'/repo','/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u',cbase+'/preparation/cpu-recovery.py']
with(p/'logs/cpu-recovery.log').open('ab',buffering=0)as out:
 child=subprocess.Popen(command,stdout=out,stderr=out,start_new_session=True)
stat=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text();receipt=dict(pid=child.pid,start_ticks=int(stat.rsplit(')',1)[1].split()[19]),created_unix=time.time())
(p/'preparation/cpu-recovery-receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''
  call('COMMIT='+repr(commit)+'\n'+s,'stage')
 elif args.phase=='poll':
  call(pre+r'''
ready=p/'preparation/native-recovery-complete.json';receipt=p/'preparation/cpu-recovery-fixed-receipt.json'
if not receipt.exists():receipt=p/'preparation/cpu-recovery-receipt.json'
r=json.loads(receipt.read_text());proc=pathlib.Path('/proc')/str(r['pid']);log=p/'logs/cpu-recovery-fixed-import.log'
if not log.exists():log=p/'logs/cpu-recovery.log'
print(json.dumps(dict(ready=json.loads(ready.read_text())if ready.exists()else None,process_stat=(proc/'stat').read_text()if proc.exists()else None,log=log.read_text()[-8000:])))
''','poll')
 elif args.phase=='submit':
  s='import pathlib,json,subprocess,time,os\nb=pathlib.Path('+repr(C)+')\n'+r'''
assert(b/'preparation/native-recovery-complete.json').exists()
intent=b/'preparation/slurm-submit-intent.json'
if intent.exists():
 print(json.dumps(dict(status='EXISTING_INTENT_RECONCILE',intent=json.loads(intent.read_text()))));raise SystemExit
runner=b/'preparation/slurm-run.sh'
body="""#!/bin/bash
set -euo pipefail
exec unshare --mount --propagation private bash "BASE/repo/reproduction/operations/b_replica_retained18/private_shared_paths.sh" env SEARCH_G1_PHYSICAL_WORLD_SIZE=8 SEARCH_G1_B_PREFIX_TOKENS=20 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="BASE/repo" /dataset/shaoshitong/Search-G1-common8/env/bin/python -u -m reproduction.operations.b_replica_retained18.slurm_native_wait --base "BASE"
""".replace('BASE',str(b));runner.write_text(body)
with intent.open('x')as f:json.dump(dict(created_unix=time.time(),eligible_nodes=['h100-8','h100-10'],world=8),f)
value=subprocess.run(['sbatch','--parsable','--nodes=1','--ntasks=1','--exclude=h100-5,h100-11','--gres=gpu:8','--cpus-per-task=32','--mem=256G','--time=1-12:00:00','--job-name=sst-b-diverse-cp64-eight','--output='+str(b/'logs/slurm-%j.log'),str(runner)],capture_output=True,text=True,timeout=40)
r=dict(returncode=value.returncode,stdout=value.stdout,stderr=value.stderr,observed_unix=time.time());(b/'preparation/slurm-submit-result.json').write_text(json.dumps(r));print(json.dumps(r))
'''
  call(s,'submit','h100-8')
 else:
  query=call('import subprocess,json,time\nprint(json.dumps(dict(observed_unix=time.time(),queue=subprocess.check_output(["squeue","-h","-n","sst-b-diverse-cp64-eight","-o","%i %T %N %R"],text=True))))','queue')
  fields=query['queue'].split()
  if fields and fields[1]=='PENDING':
   (OUT/'RESUME64_LATEST.json').write_text(json.dumps(query,indent=2));return
  node=fields[2]if fields and fields[1]=='RUNNING'else'h100-10'
  s='import pathlib,json,subprocess,time\nb=pathlib.Path('+repr(C)+')\n'+r'''
def read(p):return json.loads(p.read_text())if p.exists()else None
stage=read(b/'logs/pipeline-status.json');n=(stage or{}).get('outer',65);root=b/'outers'/('outer-%06d'%n)
progress={str(f.relative_to(b)):read(f)for f in root.rglob('*progress*.json')if f.stat().st_size<15000}
receipt=read(b/'preparation/submit-receipt.json');who=(receipt or{}).get('leader',{});proc=pathlib.Path('/proc')/str(who.get('pid'));leader=(proc/'stat').read_text()if proc.exists()else None
facts=[]
for i in range(65,201):
 j=read(b/'outers'/('outer-%06d'%i)/'complete.json')
 if j:facts.append(dict(outer=i,actor=j['actor']['actual_adam_steps_per_rank'],probe=j['probe']['actual_adam_steps_per_rank'],selection_metrics=j['selection_metrics'],phase_timings=j['phase_timings'],ranks=j['actor']['rank_receipts']))
logs=[]
for f in sorted((b/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:4]:
 with f.open('rb')as q:q.seek(max(0,f.stat().st_size-3000));tail=q.read().decode(errors='replace')
 logs.append(dict(path=str(f.relative_to(b)),tail=tail,bytes=f.stat().st_size,mtime=f.stat().st_mtime))
print(json.dumps(dict(observed_unix=time.time(),stage=stage,joint=read(b/'latest-joint.json'),progress=progress,receipt=receipt,leader=leader,cleanup=read(b/'logs/cleanup-status.json'),slurm=read(b/'preparation/slurm-submit-result.json'),facts=facts,logs=logs,queue=subprocess.run(['squeue','-h','-n','sst-b-diverse-cp64-eight','-o','%i %T %N %R'],capture_output=True,text=True).stdout)))
'''
  v=call(s,'observe',node);(OUT/'RESUME64_LATEST.json').write_text(json.dumps(v,indent=2))

if __name__=='__main__':main()
