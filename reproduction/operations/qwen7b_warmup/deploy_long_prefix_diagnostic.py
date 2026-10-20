"""Publish/stage/start a bounded CPU-only causal journal diagnostic; no GPU APIs."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
from . import peer_long_prefix_diagnostic as d
BRANCH='codex/peer8192-long-prefix-diagnostic-20261021'
FILES=('peer_long_prefix_diagnostic.py','test_peer_long_prefix_diagnostic.py','deploy_long_prefix_diagnostic.py')
COMMON=r'''
import base64,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
b=Path(v['base']);b.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def atomic(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp.%d'%os.getpid());q.write_text(json.dumps(x,sort_keys=True));q.replace(p)
def immutable(p,x):
 if p.exists():
  if read(p)!=x:raise ValueError('Immutable diagnostic binding differs')
 else:atomic(p,x)
def alive(pid,code):
 try:
  raw=Path('/proc/%s/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode();birth=Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
  return dict(live=str(code/'peer_long_prefix_diagnostic.py') in raw,pid=pid,birth=birth,cmd=raw[:350])
 except FileNotFoundError:return dict(live=False,pid=pid)
'''
STAGE=r'''
c=b/('code-'+v['commit'][:12]);c.mkdir(exist_ok=True)
for name,record in v['files'].items():
 if '/' in name:raise ValueError('Basename only')
 p=c/name
 if p.exists() and ref(p)['sha256']!=record['sha256']:raise ValueError('Existing code differs')
immutable(b/'stage-manifest.json',dict(commit=v['commit'],files=v['files'],code_root=str(c)))
print(json.dumps(dict(status='CPU_ONLY_CODE_STAGE_INITIALIZED',code_root=str(c))))
'''
CHUNK=r'''
m=read(b/'stage-manifest.json');c=Path(m['code_root']);name=v['name'];assert name in m['files'];p=c/(name+'.part-%04d'%v['index']);x=base64.b64decode(v['data']);assert hashlib.sha256(x).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x)
print(json.dumps(dict(status='CODE_CHUNK_STORED',name=name,index=v['index'])))
'''
SEAL=r'''
m=read(b/'stage-manifest.json');c=Path(m['code_root'])
for name,want in m['files'].items():
 p=c/name
 if not p.exists():
  parts=sorted(c.glob(name+'.part-*'));assert parts;p.write_bytes(b''.join(x.read_bytes() for x in parts))
 if ref(p)['sha256']!=want['sha256'] or p.stat().st_size!=want['bytes']:raise ValueError('Published source bytes/SHA differ')
 p.chmod(0o444)
 for x in c.glob(name+'.part-*'):x.unlink()
subprocess.check_call([sys.executable,'-m','py_compile',str(c/'peer_long_prefix_diagnostic.py')])
immutable(b/'source-publication.json',dict(commit=m['commit'],code_root=str(c),files=m['files'],remote_branch=v['branch'],CPU_only=True))
print(json.dumps(dict(status='PUBLISHED_CPU_SOURCE_ALL_FILES_SHA_VERIFIED',commit=m['commit'],code_root=str(c))))
'''
PREPARE=r'''
m=read(b/'source-publication.json');c=Path(m['code_root'])
for name,x in m['files'].items():assert ref(c/name)['sha256']==x['sha256']
subprocess.check_call([sys.executable,str(c/'peer_long_prefix_diagnostic.py'),'--action','prepare','--source-commit',m['commit']])
x=read(b/'config.json');immutable(b/'preparation.json',dict(config=ref(b/'config.json'),source_commit=m['commit'],CPU=2,memory_limit=3221225472,wall_seconds=1200,questions=8192,trajectories=81920))
print(json.dumps(dict(status='BOUNDED_CPU_DIAGNOSTIC_PREPARED',preparation=read(b/'preparation.json'))))
'''
START=r'''
m=read(b/'source-publication.json');c=Path(m['code_root']);p=read(b/'preparation.json');assert ref(b/'config.json')==p['config']
if (b/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE',complete=ref(b/'complete.json'))))
else:
 with (b/'operator.lock').open('a') as g:
  fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (b/'process.json').exists():
   previous=read(b/'process.json');state=alive(previous['pid'],c)
   if state['live']:raise ValueError('Existing diagnostic CPU writer is active')
   if not v['resume']:raise ValueError('Previous attempt exists; explicit resume required')
  elif (b/'start-intent.json').exists():raise ValueError('Unknown prior launch outcome; inspect before retry')
  for name,x in m['files'].items():assert ref(c/name)['sha256']==x['sha256']
  with (b/'analysis.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  intent=dict(config=p['config'],code_root=str(c),attempt=time.time(),resume=v['resume']);atomic(b/'start-intent.json',intent)
  log=(b/'analysis.log').open('ab');proc=subprocess.Popen([sys.executable,str(c/'peer_long_prefix_diagnostic.py'),'--action','analyze','--config-sha256',p['config']['sha256']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);log.close();record=dict(pid=proc.pid,config=p['config'],code_root=str(c),started=time.time());atomic(b/'process.json',record)
  print(json.dumps(dict(status='CPU_ONLY_ANALYSIS_STARTED',process=record)))
'''
STATUS=r'''
x=dict(status='ACTUAL_CPU_ONLY_LONG_PREFIX_STATE',observed_epoch=time.time())
for name in ('preparation.json','controller.json','progress.json','partial.json','failure.json'):
 p=b/name
 if p.exists():x[name[:-5]]=read(p)
if (b/'process.json').exists():
 own=read(b/'process.json');x['process']=dict(own,actual=alive(own['pid'],Path(own['code_root'])))
 try:x['process']['rss']=Path('/proc/%d/status'%own['pid']).read_text().split('VmRSS:',1)[1].splitlines()[0].strip()
 except (FileNotFoundError,IndexError):pass
if (b/'complete.json').exists():
 z=read(b/'complete.json');x['complete']=ref(b/'complete.json');x['summary']={k:z[k] for k in ('status','questions','trajectories','P20_tokens_positions_contexts_exact_matches','P20_previous_collision_counts_exact','elapsed_seconds','peak_RSS_bytes','statistics')}
if (b/'analysis.log').exists():x['log_tail']=(b/'analysis.log').read_text()[-1800:]
print(json.dumps(x))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','start','resume','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 spec=importlib.util.spec_from_file_location('prefix_diag_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(source,extra=None):
  v=dict(base=str(d.TARGET));v.update(extra or {});cmd='python3 -c '+shlex.quote('v='+repr(v)+'\n'+COMMON+source)
  if len(cmd.encode())>=16000:raise ValueError('CPU control command must remain below16KB')
  r=control.run(cmd,account='mou176',source_address=address,timeout=55);rows=[json.loads(line) for line in r.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('CPU outcome unconfirmed; inspect actual status before retry: '+r.stderr.decode(errors='replace')[-1800:]+' '+r.stdout.decode(errors='replace')[-1500:])
  return rows[-1]
 if a.action=='stage':
  remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
  if remote!=a.commit:raise ValueError('Published branch commit required')
  payloads={name:subprocess.check_output(['git','show',a.commit+':reproduction/operations/qwen7b_warmup/'+name],cwd=repo) for name in FILES}
  if any(x!=(repo/'reproduction/operations/qwen7b_warmup'/name).read_bytes() for name,x in payloads.items()):raise ValueError('Unpublished local code')
  files={name:dict(bytes=len(x),sha256=hashlib.sha256(x).hexdigest()) for name,x in payloads.items()};call(STAGE,dict(commit=a.commit,files=files))
  for name,x in payloads.items():
   for i,start in enumerate(range(0,len(x),4096)):
    data=x[start:start+4096];call(CHUNK,dict(name=name,index=i,data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest()))
  result=call(SEAL,dict(branch=BRANCH))
 else:result=call(dict(prepare=PREPARE,start=START,resume=START,status=STATUS)[a.action],dict(resume=a.action=='resume'))
 print(json.dumps(result,sort_keys=True));path=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('long-prefix-'+a.action+'-latest.json');path.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
