"""Immutable CP256+682 operator; explicit prepare/submit, never blind retry."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import peer_append_train as r,deploy_peer_within1_matched as prior
BRANCH='codex/peer-expanded-append682-20261021'
REFERENCE='ebc452595fddfd5eb095b45091a642c1e02ea060'
OWN=('peer_append_train.py','peer_append_only_plan.py','deploy_peer_append_train.py','test_peer_append_train.py')
COMMON,INIT,CHUNK,EXTRACT=prior.COMMON,prior.INIT,prior.CHUNK,prior.EXTRACT
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():
 p=read(b/'preparation.json');verify(p['config']);verify(p['sbatch']);print(json.dumps(p));sys.exit(0)
controller=b/'cpu-prepare-controller.json'
if controller.exists():
 old=read(controller);active=False
 try:active='peer_append_train' in pathlib.Path('/proc/%d/cmdline'%old['pid']).read_bytes().decode(errors='replace')
 except OSError:pass
 if active:print(json.dumps(dict(status='APPEND_CPU_PREPARATION_ACTIVE',pid=old['pid'])));sys.exit(0)
 raise RuntimeError('Prior CPU preparation exited; inspect actual failure, do not blindly restart')
code="import unittest,sys,runpy; s=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_peer_append_train'); v=unittest.TextTestRunner(verbosity=2).run(s); assert v.wasSuccessful(); sys.argv=['peer_append_train','--action','prepare']; runpy.run_module('reproduction.operations.qwen7b_warmup.peer_append_train',run_name='__main__')"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'cpu-prepare.log').open('ab',0)
args=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',code];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(controller,dict(pid=p.pid,args=args,source_commit=v['commit'],time=time.time()));print(json.dumps(dict(status='APPEND_FOCUSED_TESTS_AND_CPU_PREPARATION_STARTED',pid=p.pid)))
'''
SUBMIT="training_guard=(b/'append-only.lock').open('a+');fcntl.flock(training_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"+prior.SUBMIT.replace('q7b-peer8192-within1','q7b-peer-append682').replace('ONE_FROZEN_CALIBRATION_GPU_JOB_SUBMITTED','ONE_APPEND_ONLY682_GPU_JOB_SUBMITTED')
STATUS=r'''
v=dict(status='APPEND_ONLY682_ACTUAL_CURRENT_STATE',observed_epoch=time.time())
for name in ('source-publication.json','cpu-prepare-controller.json','preparation.json','submission.json','controller.json','progress.json','failure.json','first-cp-independent-audit.json','first-cp-audit-failure.json','first-cp-audit-controller.json','next-update-replay.json','latest.json','interrupted.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root','base_runtime_commit','callback_interface_commit')}
  v[name[:-5]]=x
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList']);v['start_estimate']=command(['squeue','--start','-j',jid,'-h','-o','%i|%T|%S|%R'])
for cursor in (512,768,938):
 p=b/'evaluations'/('cursor-%06d'%cursor)/'complete.json'
 if p.exists():
  x=read(p);v['evaluation_'+str(cursor)]={k:x[k] for k in ('status','cursor','steps','metrics','training_six_components_unchanged')};v['evaluation_'+str(cursor)]['ref']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
if (b/'complete.json').exists():
 x=read(b/'complete.json');v['complete']={k:x[k] for k in ('status','cursor','steps','additional_updates','checkpoint','metrics','config_sha256','actor_updates')};v['complete_ref']=dict(path=str(b/'complete.json'),bytes=(b/'complete.json').stat().st_size,sha256=sha(b/'complete.json'))
if (b/'cpu-prepare.log').exists() and not (b/'preparation.json').exists():v['cpu_log_tail']=(b/'cpu-prepare.log').read_text()[-4000:]
logs=sorted((b/'logs').glob('train-*.log')) if (b/'logs').exists() else []
if logs:v['gpu_log_tail']=logs[-1].read_text()[-2200:]
print(json.dumps(v))
'''

def remote_source(values,action):return 'inputs='+repr(values)+'\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]

def bundle(repo,commit,reference):
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if remote!=commit or reference['commit']!=REFERENCE:raise ValueError('Published append source and exact accepted reference closure required')
 origins=reference['file_origins'];payloads={};where={}
 for name,item in reference['files'].items():
  origin=origins['overrides'].get(name,origins['default_commit']);data=subprocess.check_output(['git','show',origin+':'+name],cwd=repo)
  if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Accepted runtime closure differs: '+name)
  payloads[name]=data;where[name]=origin
 for short in OWN:
  name=r.PKG+short;data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
  if data!=(repo/name).read_bytes():raise ValueError('Unpublished append source '+name)
  payloads[name]=data;where[name]=commit
 output=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name,data in sorted(payloads.items()):
    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(data))
 return output.getvalue(),files,dict(default_commit=origins['default_commit'],overrides={n:c for n,c in where.items() if c!=origins['default_commit']})

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit' and not a.submit_authorized:p.error('Actual parent GO and explicit --submit-authorized required')
 spec=importlib.util.spec_from_file_location('append_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();values=dict(base=str(r.TARGET),target=str(r.TARGET),train=str(r.CONTROL),pin=r.D8,callback_commit=r.CALLBACK)
 def call(action,extra=None):
  v=dict(values);v.update(extra or {});cmd='python3 -c '+shlex.quote(remote_source(v,action))
  if len(cmd.encode())>=16000:raise ValueError('Control command exceeds16KB')
  result=ctl.run(cmd,account='mou176',source_address=address,timeout=55)
  out=result.stdout.decode(errors='replace') if isinstance(result.stdout,bytes) else result.stdout;err=result.stderr.decode(errors='replace') if isinstance(result.stderr,bytes) else result.stderr
  rows=[json.loads(line) for line in out.splitlines() if line.startswith('{')]
  if result.returncode or not rows:raise RuntimeError('Outcome unconfirmed; inspect identity before retry: '+err[-1600:]+' '+out[-2600:])
  return rows[-1]
 if a.action=='stage':
  repo=Path(__file__).resolve().parents[3];reference=json.loads((repo/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/reference-source-packet.json').read_text())['within_publication']
  data,files,origins=bundle(repo,a.commit,reference);chunks=[data[i:i+4096] for i in range(0,len(data),4096)]
  call('init',dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,file_origins=origins))
  for i,part in enumerate(chunks):call('chunk',dict(commit=a.commit,index=i,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))
  result=call('extract',dict(commit=a.commit))
 else:result=call(a.action)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer-append-'+a.action+'-latest.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
