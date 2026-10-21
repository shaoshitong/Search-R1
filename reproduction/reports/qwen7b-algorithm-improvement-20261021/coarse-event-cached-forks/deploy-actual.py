"""Typed deployment of qualified coarse forks. Submit only under explicit root GO."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];O=pathlib.Path(__file__).parent
QUALIFIED='c888a2281859e1a930218d490aae5bbeb5adcc22'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
PACKET=dict(path=A+'/frozenE1-head-objective-family-v1/coarse-fork-inputs.json',bytes=9776,sha256='d40bddcc05f7d5319e1f77040264b8e5f10f799b7d6013ce4f44f4b9dfdc9f10')
BODY='''import pathlib,json,hashlib,subprocess,fcntl,os,time,re
A=pathlib.Path(v['A']);b=A/('coarse-event-frozenE1-'+v['mode']+'-fit30016-hold5120-v1');b.mkdir(exist_ok=True);guard=(b/'deployment.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def check(ref):
 assert proof(ref['path'])==ref,'Fresh small descriptor SHA differs';return read(ref['path'])
def immutable(p,x):
 if p.exists():assert read(p)==x,'Immutable metadata differs'
 else:
  with p.open('x')as f:f.write(json.dumps(x,sort_keys=True)+'\\n');f.flush();os.fsync(f.fileno())
def unlocked():
 handles=[]
 for name in ('training.lock','prepare.lock','audit-first.lock'):
  f=(b/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
 return handles
def active():
 r=subprocess.run(['squeue','-h','-u','mou176','-n','q7b-coarse-'+v['mode'],'-o','%i|%j|%T|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);assert r.returncode==0,r.stderr[-1000:];return r.stdout.strip().splitlines()
code=A/('coarse-event-cached-prepare-code-v1/code-'+v['qualified'][:12]);qref=dict(path=str(code/'cpu-qualification.json'),bytes=3574,sha256='a58dd76a24cd6cbe3b0605a8f249e81ce2b8cb56bc4dd4560262422beb838368');q=check(qref);assert q['tests']==3 and not q['GPU_used'];packet=check(v['packet'])
assert packet['role']=='FROZEN_E1_SHARED_CACHE_CP0_COARSE_FORK_INPUTS';assert packet['baseline_CP0']['sha256']=='acd1c4f4bc277a8cab83847f966cfbf2462b6391b6db1d9c52c9434447f562df';assert packet['cache_complete']['sha256']=='b39168d130626151c5793887783c6c6db96474a74f90fb80216880a7c8bda350'
if v['action']=='prepare':
 handles=unlocked();assert not active(),'Existing own job must be reconciled, never duplicated';assert not(b/'submission-intent.json').exists(),'Unknown submission intent must be reconciled'
 original=check(q['baseline_publication']);files={}
 for name,ref in original['files'].items():
  p=code/name;actual=proof(p)
  if p.name in q['files']:assert actual==q['files'][p.name]
  else:assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256']
  files[name]=actual
 for name,ref in q['files'].items():files['reproduction/operations/qwen7b_warmup/'+name]=proof(ref['path']);assert files['reproduction/operations/qwen7b_warmup/'+name]==ref
 pub=dict(commit=v['qualified'],remote_sha=v['qualified'],code_root=str(code),files=files,baseline_runtime_commit=original['commit'],controller_commit=v['controller'],controller_source=v['controller_source'],cpu_recovery_qualification=qref,input_packet=v['packet'],shared_immutable_code_reused=True)
 immutable(b/'source-publication.json',pub)
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with(b/'cpu-prepare.log').open('ab')as log:r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.coarse_event_cached_train','--action','prepare','--mode',v['mode'],'--packet',v['packet']['path']],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=48)
 assert r.returncode==0,(b/'cpu-prepare.log').read_text()[-3500:]
 result=dict(status='ACTUAL_COARSE_FORK_SINGLE_SUBMIT_READY',mode=v['mode'],publication=proof(b/'source-publication.json'),preparation=proof(b/'preparation.json'),config=proof(b/'config.json'),entry=proof(b/'training.sbatch'),shared_packet=v['packet'],cpu_prepare_tensors_loaded=False,source_CP0_actual_restore_pending_on_allocation=True,GPU_submitted=False,owner_lock_free=True,active_same_name_jobs=active())
 subprocess.check_call(['bash','-n',str(b/'training.sbatch')]);assert not result['active_same_name_jobs'];immutable(b/'submit-ready.json',result)
elif v['action']=='submit':
 handles=unlocked();ready=read(b/'submit-ready.json')
 for key in ('publication','preparation','config','entry'):assert proof(ready[key]['path'])==ready[key]
 assert read(b/'source-publication.json')['controller_commit']==v['controller'];assert not active(),'Same-name job already exists'
 if(b/'submission-intent.json').exists():raise RuntimeError('Existing submission intent; inspect/reconcile before any retry')
 immutable(b/'submission-intent.json',dict(mode=v['mode'],epoch=time.time(),controller=v['controller'],entry=ready['entry'],config=ready['config'],qualified=v['qualified'],held_before_release=True))
 r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Submission uncertain; retained intent. Inspect same-name job before retry: '+r.stdout[-500:]+r.stderr[-1000:])
 job=r.stdout.strip().split(';')[0];raw=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True);assert 'JobName=q7b-coarse-'+v['mode']in raw and 'QOS=normal'in raw and 'Partition=acd_u'in raw and 'TimeLimit=02:00:00'in raw and 'Account=mou176'in raw and 'MinMemoryNode=128G'in raw and 'NumNodes=1'in raw and ('Command='+str(b/'training.sbatch'))in raw and ('WorkDir='+str(b))in raw and 'Dependency=(null)'in raw;assert re.search(r'NumCPUs=8(?: |$)',raw)and ('gres/gpu=1'in raw or 'gres:gpu=1'in raw)
 identity=dict(status='ACTUAL_COARSE_FORK_HELD_IDENTITY_VERIFIED',mode=v['mode'],job_id=job,job=raw,entry=ready['entry'],config=ready['config'],epoch=time.time());immutable(b/'submission-identity.json',identity)
 subprocess.check_call(['scontrol','release',job]);result=dict(status='ACTUAL_COARSE_FORK_SUBMITTED_RELEASED',mode=v['mode'],job_id=job,identity=proof(b/'submission-identity.json'),entry=ready['entry'],config=ready['config']);immutable(b/'submission-result.json',result)
else:
 result=dict(mode=v['mode'],epoch=time.time(),active_same_name_jobs=active())
 for name in ('submit-ready.json','submission-intent.json','submission-identity.json','submission-result.json','controller.json','progress.json','first-update-replay.json','first-cp-independent-audit.json','latest.json','failure.json','complete.json'):
  if(b/name).exists():
   try:result[name[:-5]]=read(b/name)
   except FileNotFoundError:pass
print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','submit','status'),required=True);p.add_argument('--mode',choices=('binary','three_event','both'),required=True);a=p.parse_args()
 own=pathlib.Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);assert subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)==own.read_bytes();source=dict(path=str(own.relative_to(R)),bytes=own.stat().st_size,sha256=hashlib.sha256(own.read_bytes()).hexdigest())
 spec=importlib.util.spec_from_file_location('coarse_actual_private_control',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 for mode in ('binary','three_event')if a.mode=='both'else(a.mode,):
  script='v='+repr(dict(action=a.action,mode=mode,A=A,qualified=QUALIFIED,controller=commit,controller_source=source,packet=PACKET))+'\n'+BODY;compile(script,'coarse-actual-remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=65)
  if r.returncode:raise RuntimeError(r.stdout[-3500:]+r.stderr[-1000:])
  result=json.loads(r.stdout.splitlines()[-1]);(O/('actual-'+mode+'-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
