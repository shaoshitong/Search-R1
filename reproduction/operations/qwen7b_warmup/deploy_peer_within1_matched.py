"""Immutable matched within1 CPU-prepare / normal single-GPU operator.
Stage SHA-exact D8 tree plus published callback-only interface48eb; all state
changes are explicit. Submission is held until identity+jobID are durable.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import peer_within1_matched as w,deploy_peer_g10_calibration as tr
BRANCH='codex/peer8192-within1-matched256-20261021'
OWN=('peer_within1_matched.py','deploy_peer_within1_matched.py','test_peer_within1_matched.py','deploy_peer_g10_calibration.py','score_peer_g10_calibration.py')
LEARN='3fc8f70908761d7a8ceff33a987fe0720d6aa64c'
COMMON=tr.COMMON
INIT=tr.INIT.replace("'remote_branch')","'remote_branch','file_origins')")
CHUNK=tr.CHUNK
EXTRACT=tr.EXTRACT.replace("pinned_runtime_commit=inputs['pin']", "base_runtime_commit=inputs['pin'],callback_interface_commit=inputs['callback_commit'],file_origins=v['file_origins']")
EXTRACT=EXTRACT.replace("assert m.isfile() and pathlib.Path(m.name).name==m.name;p=root/m.name;payload=archive.extractfile(m).read();","rel=pathlib.Path(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts;p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);payload=archive.extractfile(m).read();")
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():
 p=read(b/'preparation.json');verify(p['config']);verify(p['sbatch']);print(json.dumps(p));sys.exit(0)
controller=b/'cpu-prepare-controller.json'
if controller.exists():
 old=read(controller);pid=old['pid'];active=False
 try:active='peer_within1_matched' in pathlib.Path('/proc/%d/cmdline'%pid).read_bytes().decode(errors='replace')
 except OSError:pass
 if active:print(json.dumps(dict(status='WITHIN1_CPU_PREPARATION_ACTIVE',pid=pid)));sys.exit(0)
 raise RuntimeError('Prior within1 CPU prepare exited; inspect real failure, no blind restart')
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'cpu-prepare.log').open('ab',0)
args=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.peer_within1_matched','--action','prepare'];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(controller,dict(pid=p.pid,args=args,source_commit=v['commit'],time=time.time()));print(json.dumps(dict(status='WITHIN1_CPU_SIGNATURE_PREPARATION_STARTED',pid=p.pid)))
'''
SUBMIT=tr.SUBMIT.replace('q7b-peer8192-frozen-cal','q7b-peer8192-within1').replace("b/'submit-intent.json'","b/'submit-intent.json'")
STATUS=r'''
v=dict(status='MATCHED_WITHIN1_ACTUAL_CURRENT_STATE',observed_epoch=time.time())
for name in ('source-publication.json','cpu-prepare-controller.json','cpu-signature-progress.json','preparation.json','matched-control-audit.json','controller.json','progress.json','failure.json','first-cp-independent-audit.json','first-cp-audit-failure.json','first-cp-audit-controller.json','next-update-replay.json','latest.json','interrupted.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root','base_runtime_commit','callback_interface_commit')}
  if name=='controller.json':x={k:x[k] for k in ('pid','job_id','status','source_steps','global_original_questions')}
  v[name[:-5]]=x
if (b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();v['job_id']=jid;v['squeue']=command(['squeue','-j',jid,'-h','-o','%i|%j|%T|%M|%R|%N']);v['sacct']=command(['sacct','-j',jid,'-n','-P','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,NodeList']);v['start_estimate']=command(['squeue','--start','-j',jid,'-h','-o','%i|%T|%S|%R'])
if (b/'complete.json').exists():
 x=read(b/'complete.json');v['complete']={k:x[k] for k in ('status','cursor','steps','checkpoint','metrics','config_sha256','actor_updates')};v['complete_ref']=dict(path=str(b/'complete.json'),bytes=(b/'complete.json').stat().st_size,sha256=sha(b/'complete.json'))
if (b/'cpu-prepare.log').exists() and not (b/'preparation.json').exists():v['cpu_log_tail']=(b/'cpu-prepare.log').read_text()[-2500:]
logs=sorted((b/'logs').glob('train-*.log')) if (b/'logs').exists() else []
if logs:v['gpu_log_tail']=logs[-1].read_text()[-2200:]
print(json.dumps(v))
'''
def remote_source(values,action):return 'inputs='+repr(values)+'\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,submit=SUBMIT,status=STATUS)[action]

def bundle(repo,commit,oldpub):
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if remote!=commit:raise ValueError('Publish exact within1 source before staging')
 if oldpub['commit']!=w.D8:raise ValueError('Actual control D8 source publication required')
 def blob(ref,name):return subprocess.check_output(['git','show',ref+':'+name],cwd=repo)
 payloads={};origins={}
 for name,item in oldpub['files'].items():
  data=blob(w.D8,name)
  if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Local D8 differs from actual remote control: '+name)
  payloads[name]=data;origins[name]=w.D8
 name=w.PKG+'probe_g10_train.py';payloads[name]=blob(w.CALLBACK,name);origins[name]=w.CALLBACK
 name=w.PKG+'peer_g10_learnability.py';payloads[name]=blob(LEARN,name);origins[name]=LEARN
 for short in OWN:
  name=w.PKG+short;data=blob(commit,name)
  if data!=(repo/name).read_bytes():raise ValueError('Unpublished within1 dependency '+name)
  payloads[name]=data;origins[name]=commit
 data=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=data,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name,payload in sorted(payloads.items()):
    files[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest());m=tarfile.TarInfo(name);m.size=len(payload);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(payload))
 return data.getvalue(),files,dict(default_commit=w.D8,overrides={n:c for n,c in origins.items() if c!=w.D8})

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit' and not a.submit_authorized:p.error('--submit-authorized required')
 spec=importlib.util.spec_from_file_location('within1_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();values=dict(base=str(w.TARGET),target=str(w.TARGET),train=str(w.CONTROL),pin=w.D8,callback_commit=w.CALLBACK)
 def call(action,extra=None):
  own=dict(values);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action))
  if len(command.encode())>=16000:raise ValueError('Control payload exceeds16KB')
  r=control.run(command,account='mou176',source_address=address,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('Within1 action unconfirmed; inspect actual state: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-2500:])
  return rows[-1]
 if a.action=='stage':
  oldpub=json.loads((a.control_root/'outputs/qwen7b-warmup-20261019/reports/peer-control-D8-runtime-source-publication.json').read_text());data,files,origins=bundle(Path(__file__).resolve().parents[3],a.commit,oldpub);chunks=[data[i:i+4096] for i in range(0,len(data),4096)];v=dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,file_origins=origins);call('init',v)
  for i,chunk in enumerate(chunks):call('chunk',dict(commit=a.commit,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
  result=call('extract',dict(commit=a.commit))
 else:result=call(a.action)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer-within1-'+a.action+'-latest.json');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
