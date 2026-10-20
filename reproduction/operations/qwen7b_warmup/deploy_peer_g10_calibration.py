"""One normal-GPU immutable frozen seen-training calibration scoring operator.
Only explicit actions touch remote state; unknown sbatch outcome is never retried.
The qualified d8 runtime is referenced read-only rather than copied/hot-edited.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile,tempfile
from pathlib import Path
from . import score_peer_g10_calibration as s
FILES=('score_peer_g10_calibration.py','deploy_peer_g10_calibration.py','test_peer_g10_calibration.py')
BRANCH='codex/peer8192-frozen-calibration-scoring-20261021'
COMMON=r'''
import pathlib,json,hashlib,base64,os,fcntl,time,subprocess,tarfile,io,sys
sys.dont_write_bytecode=True
b=pathlib.Path(inputs['base']);assert str(b)==inputs['target'] and b.resolve()==b
b.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix('.tmp')
 with t.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def verify(r):
 p=pathlib.Path(r['path']);assert p.resolve()==p and p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def lock():
 h=(b/'deployment.lock').open('a+');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
def command(args,timeout=25):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 if p.returncode and args[0]=='squeue' and 'Invalid job id specified' in p.stderr:return ''
 assert p.returncode==0,(args,p.stdout[-500:],p.stderr[-1500:]);return p.stdout.strip()
def pub():
 v=read(b/'source-publication.json');root=pathlib.Path(v['code_root']);assert root.parent==b and root.name=='code-'+v['commit'][:12]
 for name,r in v['files'].items():verify(dict(path=str(root/name),**r))
 return v,root
'''
INIT=r'''
h=lock();v={k:inputs[k] for k in ('commit','bundle_sha256','bundle_bytes','chunks','files','remote_branch')}
if (b/'upload-controller.json').exists():assert read(b/'upload-controller.json')==v
else:atomic(b/'upload-controller.json',v)
(b/('upload-'+v['bundle_sha256'][:16])).mkdir(exist_ok=True)
print(json.dumps(dict(status='FROZEN_CALIBRATION_UPLOAD_BOUND',chunks=v['chunks'])))
'''
CHUNK=r'''
h=lock();v=read(b/'upload-controller.json');assert v['commit']==inputs['commit'];data=base64.b64decode(inputs['data']);i=inputs['index'];assert 0<=i<v['chunks'] and len(data)<=4096 and hashlib.sha256(data).hexdigest()==inputs['sha256'];p=b/('upload-'+v['bundle_sha256'][:16])/('part-%04d'%i)
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data)
print(json.dumps(dict(status='CODE_CHUNK_SHA_PASSED',index=i,bytes=len(data))))
'''
EXTRACT=r'''
h=lock();v=read(b/'upload-controller.json');parts=sorted((b/('upload-'+v['bundle_sha256'][:16])).glob('part-*'));assert len(parts)==v['chunks'];data=b''.join(p.read_bytes() for p in parts);assert len(data)==v['bundle_bytes'] and hashlib.sha256(data).hexdigest()==v['bundle_sha256'];root=b/('code-'+v['commit'][:12]);root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 members=archive.getmembers();assert {m.name for m in members}==set(v['files'])
 for m in members:
  assert m.isfile() and pathlib.Path(m.name).name==m.name;p=root/m.name;payload=archive.extractfile(m).read();assert hashlib.sha256(payload).hexdigest()==v['files'][m.name]['sha256']
  if p.exists():assert p.read_bytes()==payload
  else:p.write_bytes(payload);p.chmod(0o444)
publication=dict(commit=v['commit'],remote_sha=v['commit'],remote_branch=v['remote_branch'],code_root=str(root),files=v['files'],bundle_sha256=v['bundle_sha256'],pinned_runtime_commit=inputs['pin'])
if (b/'source-publication.json').exists():assert read(b/'source-publication.json')==publication
else:atomic(b/'source-publication.json',publication)
pub();print(json.dumps(dict(status='IMMUTABLE_CALIBRATION_RUNTIME_READY',code_root=str(root),commit=v['commit'])))
'''
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():
 p=read(b/'preparation.json');verify(p['config']);verify(p['sbatch']);print(json.dumps(p))
else:
 controller=b/'cpu-prepare-controller.json'
 if controller.exists():
  old=read(controller);pid=old['pid'];active=False
  try:active=str(root/'score_peer_g10_calibration.py') in pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().decode(errors='replace')
  except OSError:pass
  if active:print(json.dumps(dict(status='CPU_PREPARATION_ALREADY_ACTIVE',pid=pid)));sys.exit(0)
  raise RuntimeError('Prior CPU preparation exited without acceptance; inspect failure before retry')
 pinned=pathlib.Path(inputs['train'])/('code-'+inputs['pin'][:12]);env=dict(os.environ,QWEN7B_CODE_ROOT=str(pinned),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');log=(b/'cpu-prepare.log').open('ab',0)
 args=['bash',str(pinned/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u',str(root/'score_peer_g10_calibration.py'),'--action','prepare'];p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env);atomic(controller,dict(pid=p.pid,args=args,source_commit=v['commit'],time=time.time()));print(json.dumps(dict(status='CPU_PREPARATION_STARTED',pid=p.pid)))
'''
SUBMIT=r'''
h=lock();v,root=pub();p=read(b/'preparation.json');verify(p['config']);script=verify(p['sbatch']);assert not (b/'failure.json').exists(),'Actual failure must be resolved before submission'
job=b/'job.id'
if job.exists():print(json.dumps(dict(status='EXISTING_CALIBRATION_JOB_NO_DUPLICATE',job_id=job.read_text().strip())));sys.exit(0)
name='q7b-peer8192-frozen-cal';q=command(['squeue','-u','mou176','-h','-o','%i|%j|%T']);same=[x for x in q.splitlines() if '|'+name+'|' in x];assert not same,('Same-name job exists',same)
assert not (b/'submit-intent.json').exists(),'Unknown previous submission outcome: do not repeat'
atomic(b/'submit-intent.json',dict(job_name=name,source_commit=v['commit'],sbatch=p['sbatch'],time=time.time(),held=True))
out=command(['sbatch','--hold','--parsable',str(script)]);jid=out.split(';')[0];assert jid.isdigit(),out
job.write_text(jid+'\n');info=command(['scontrol','show','job','-o',jid]);assert 'JobName='+name in info and 'WorkDir='+str(b) in info and 'Account=mou176' in info
atomic(b/'submission.json',dict(job_id=jid,source_commit=v['commit'],config=p['config'],sbatch=p['sbatch'],identity=info,dependency=None,qos='normal',partition='acd_u',time=time.time()))
command(['scontrol','release',jid]);print(json.dumps(dict(status='ONE_FROZEN_CALIBRATION_GPU_JOB_SUBMITTED',job_id=jid,source_commit=v['commit'],qos='normal',partition='acd_u',dependency=None)))
'''
STATUS=r'''
v=dict(status='FROZEN_CALIBRATION_CURRENT_ACTUAL_STATE',observed_epoch=time.time())
for name in ('source-publication.json','cpu-prepare-controller.json','preparation.json','controller.json','progress.json','failure.json','submission.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root','pinned_runtime_commit')}
  v[name[:-5]]=x
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList']);v['start_estimate']=command(['squeue','--start','-j',jid,'-h','-o','%i|%T|%S|%R'])
for name in ('E4','peer256'):
 p=b/name/'complete.json'
 if p.exists():
  x=read(p);v[name]={k:x[k] for k in ('status','predictions','ema_sha256','timing','verification')}
p=b/'complete.json'
if p.exists():
 x=read(p);v['complete']={k:x[k] for k in ('status','groups','identities','normal_exit','actor_updates','probe_updates')}
 v['complete_ref']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
if (b/'cpu-prepare.log').exists() and not (b/'preparation.json').exists():v['cpu_log_tail']=(b/'cpu-prepare.log').read_text()[-2500:]
if (b/'logs').exists():
 logs=sorted((b/'logs').glob('scoring-*.log'))
 if logs:v['gpu_log_tail']=logs[-1].read_text()[-2500:]
print(json.dumps(v))
'''

def remote_source(values,action):return 'inputs='+repr(values)+'\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]

def bundle(repo,commit):
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if remote!=commit:raise ValueError('Dedicated source commit must be pushed and remoteSHAverified first')
 data=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=data,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name in FILES:
    rel='reproduction/operations/qwen7b_warmup/'+name;payload=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo)
    if payload!=(repo/rel).read_bytes():raise ValueError('Unpublished local scorer source '+name)
    files[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest());m=tarfile.TarInfo(name);m.size=len(payload);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(payload))
 return data.getvalue(),files

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit' and not a.submit_authorized:p.error('--submit-authorized required')
 spec=importlib.util.spec_from_file_location('calibration_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 values=dict(base=str(s.TARGET),target=str(s.TARGET),train=str(s.TRAIN),pin=s.PIN)
 def call(action,extra=None):
  own=dict(values);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action))
  if len(command.encode())>=16000:raise ValueError('Control payload exceeds16KB')
  r=control.run(command,account='mou176',source_address=address,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('Remote outcome unconfirmed; inspect status before retry: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-3000:])
  return rows[-1]
 if a.action=='stage':
  data,files=bundle(Path(__file__).resolve().parents[3],a.commit);chunks=[data[i:i+4096] for i in range(0,len(data),4096)];v=dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files);call('init',v)
  for i,chunk in enumerate(chunks):call('chunk',dict(commit=a.commit,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
  result=call('extract',v)
 else:result=call(a.action)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer8192-calibration-score-'+a.action+'-latest.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
