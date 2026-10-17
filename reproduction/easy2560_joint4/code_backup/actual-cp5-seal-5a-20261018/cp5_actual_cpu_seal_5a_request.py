"""Execute published5a CPU-only CP5 seal once, with durable small observations."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

WS=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
OUT=Path(__file__).parent
PYTHON='/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017/env/bin/python'
COMMIT='5a16192f8c8de43b6c1f122b77ae49c7a4144603'
REMOTE=r'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
new=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018')
old=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018')
commit='5a16192f8c8de43b6c1f122b77ae49c7a4144603';source=new/'source'/commit
root=new/'preparation/cp5-cpu-seal-5a16192';target=old/'preparation/checkpoint-evidence/global_step_5'
assert os.getuid()==14090 and source.resolve()==source and new.resolve()==new and old.resolve()==old
binding=json.loads((new/'preparation/run-source.json').read_text());assert binding['source_commit']=='d91575dbca221f66c467dccafa8508d9fb703688' and binding['runtime_repo']==str(new/'repo')
run=json.loads((old/'logs/run-manifest.json').read_text());assert run['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da' and run['source_commit']=='8a5bac94b8230a99b81977f5290589e3d3708dfe'
assert (old/'job.id').read_text().strip()=='730226'
mode=MODE
GUARD="""import builtins,os,runpy,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','transformers','vllm','ray','faiss'}:raise RuntimeError('CPU_ONLY_IMPORT_GUARD_'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
runpy.run_module('reproduction.easy2560_joint4.checkpoint_evidence',run_name='__main__')
assert not any(x in sys.modules for x in ('torch','transformers','vllm','ray','faiss'))
"""
commands={'seal':[sys.executable,'-B','-c',GUARD,'seal','--base',str(old),'--training-commit','8a5bac94b8230a99b81977f5290589e3d3708dfe','--run-id',run['run_id'],'--source-job','730226','--step','5'],
 'verify':[sys.executable,'-B','-c',GUARD,'verify','--directory',str(target)]}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def birth(pid):
 path=Path('/proc')/str(pid);v=(path/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(v[19]),uid=path.stat().st_uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid_namespace=os.readlink(path/'ns/pid'))
def put(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if mode=='inspect':
 out={'role':'published5a_CPU_CP5_prior_claim_and_actual_CLI_help','checked_unix':time.time(),'source':str(source),'runtime_commit':binding['source_commit'],'runtime_binding_sha256':digest(new/'preparation/run-source.json'),'target':str(target),'target_exists':target.exists(),'target_marker_exists':(target/'complete.json').exists(),'auditroot_exists':root.exists(),'existing_claim':json.loads((root/'claim.json').read_text()) if (root/'claim.json').is_file() else None,'source_operator_sha256':digest(source/'reproduction/easy2560_joint4/checkpoint_evidence.py'),'source_publication_sha256':digest(source/'source-publication.json')}
 parent=target.parent
 out['existing_evidence_entries']=[p.name for p in parent.iterdir()] if parent.exists() else []
 out['CLI_help']={}
 for args in (['--help'],['seal','--help'],['verify','--help']):
  r=subprocess.run([sys.executable,'-B','-c',GUARD]+args,cwd=source,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
  out['CLI_help'][' '.join(args)]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 print(json.dumps(out,sort_keys=True))
elif mode=='start':
 assert not root.exists() and not target.exists(),'EXISTING_CLAIM_OR_TARGET_NO_RELAUNCH'
 assert not any(p.name.startswith('.partial-') for p in target.parent.iterdir()) if target.parent.exists() else True
 root.mkdir(mode=0o700);plan={'source':str(source),'target':str(target),'runtime_commit':binding['source_commit'],'runtime_binding_sha256':digest(new/'preparation/run-source.json'),'commands':commands,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(source),'started_unix':time.time(),'new_training_updates':0,'model_reloads':0,'gpu_allocations':0};put(root/'plan.json',plan)
 supervisor="""import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);plan=json.loads((root/'plan.json').read_text());source=Path(plan['source'])
def birth(pid):
 p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(pid=pid,start_ticks=int(v[19]),uid=p.stat().st_uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid_namespace=os.readlink(p/'ns/pid'))
def put(name,value):
 with (root/name).open('x') as f:json.dump(value,f,sort_keys=True);f.write('\\n');f.flush();os.fsync(f.fileno())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
for phase in ('seal','verify'):
 with (root/(phase+'.stdout')).open('xb') as out,(root/(phase+'.stderr')).open('xb') as err:
  p=subprocess.Popen(plan['commands'][phase],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
  put(phase+'-claim.json',dict(phase=phase,process_identity=birth(p.pid),started_unix=time.time()))
  rc=p.wait()
 put(phase+'-exit.json',dict(phase=phase,returncode=rc,finished_unix=time.time()))
 if rc:put('result.json',dict(status='FAILED_STOPPED_NO_AUTORETRY',phase=phase,returncode=rc,finished_unix=time.time()));raise SystemExit(rc)
put('result.json',dict(status='ACTUAL_CP5_CPU_SEAL_AND_VERIFY_COMPLETED',finished_unix=time.time(),new_training_updates=0,model_reloads=0,gpu_allocations=0))
""".replace('ROOT',repr(str(root)))
 with (root/'supervisor.log').open('xb') as log:
  p=subprocess.Popen([sys.executable,'-B','-c',supervisor],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  claim={'supervisor':birth(p.pid),'started_unix':time.time(),'source_commit':commit,'training_runtime_commit':binding['source_commit']};put(root/'claim.json',claim)
 print(json.dumps({'status':'STARTED_ONCE','root':str(root),'claim':claim},sort_keys=True))
elif mode=='observe':
 assert root.is_dir() and root.stat().st_uid==os.getuid()
 out={'checked_unix':time.time(),'root':str(root),'runtime_commit':binding['source_commit'],'runtime_binding_sha256':digest(new/'preparation/run-source.json'),'target_exists':target.exists(),'target_marker_exists':(target/'complete.json').is_file()}
 for name in ('claim.json','seal-claim.json','seal-exit.json','verify-claim.json','verify-exit.json','result.json'):
  p=root/name;out[name]=json.loads(p.read_text()) if p.exists() else None
 for name in ('supervisor.log','seal.stdout','seal.stderr','verify.stdout','verify.stderr'):
  p=root/name
  out[name]={'bytes':p.stat().st_size,'text':p.read_text()[-100000:]} if p.exists() else None
 for phase in ('seal','verify'):
  claim=out[phase+'-claim.json']
  if claim:
   path=Path('/proc')/str(claim['process_identity']['pid']);live=path.exists()
   row={'exists':live}
   if live:
    actual=birth(claim['process_identity']['pid']);row.update(actual_birth=actual,same_birth=actual==claim['process_identity'])
    if row['same_birth']:
     v=(path/'stat').read_text().rsplit(')',1)[1].split();row['state']=v[0];row['cpu_ticks']={'utime':int(v[11]),'stime':int(v[12])}
     row['io']=(path/'io').read_text() if (path/'io').exists() else None
   out[phase+'_live']=row
 if (target/'complete.json').is_file():
  out['target_complete']=json.loads((target/'complete.json').read_text());out['target_complete_sha256']=digest(target/'complete.json');out['target_supplement_sha256']=digest(target/'supplement.json')
 print(json.dumps(out,sort_keys=True))
else:raise SystemExit('inspect/start/observe')
'''


def main():
    mode=sys.argv[1]
    code=REMOTE.replace('MODE',repr(mode));ast.parse(code)
    p=WS/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
    s=importlib.util.spec_from_file_location('bridge',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    r=m.run('jxu957',shlex.join([PYTHON,'-B','-c',code]),'',str(WS/'work'),login_host='10.120.48.31',expected_host_identity=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]'),observation_timeout=55)
    value=dict(mode=mode,checked_unix=time.time(),source_commit=COMMIT,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    path=OUT/('actual-cp5-cpu-seal-5a-'+mode+'-'+str(int(time.time()))+'.json');path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(str(path));print(json.dumps(value,sort_keys=True) if r.returncode else json.dumps(json.loads(r.stdout),sort_keys=True))

if __name__=='__main__':main()
