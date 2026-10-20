"""Stage/prepare/monitor immutable CPU-only native-prefix confidence extraction."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
from . import deploy_long_prefix_diagnostic as t,prefix_confidence_data as d
BRANCH='codex/prefix-confidence-incremental-20261021'
FILES=('prefix_confidence_data.py','audit_prefix_logprob_availability.py','test_prefix_confidence_data.py','deploy_prefix_confidence_data.py')
COMMON=t.COMMON.replace('peer_long_prefix_diagnostic.py','prefix_confidence_data.py')
SEAL=t.SEAL.replace('peer_long_prefix_diagnostic.py','prefix_confidence_data.py')
PREPARE=r'''
m=read(b/'source-publication.json');c=Path(m['code_root'])
for name,x in m['files'].items():assert ref(c/name)['sha256']==x['sha256']
subprocess.check_call([sys.executable,str(c/'prefix_confidence_data.py'),'--action','prepare','--code-root',str(c),'--commit',m['commit']])
immutable(b/'data-preparation.json',dict(config=ref(b/'config.json'),split=ref(b/'split.json'),source_commit=m['commit'],CPU=2,memory_limit=3221225472,wall_seconds=1560,questions=8192,trajectories=81920))
print(json.dumps(dict(status='FROZEN_PREFIX_CONFIDENCE_DATA_PREPARED',preparation=read(b/'data-preparation.json'))))
'''
START=r'''
m=read(b/'source-publication.json');c=Path(m['code_root']);p=read(b/'data-preparation.json');assert ref(b/'config.json')==p['config'];data=b/'data'
if (data/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE',complete=ref(data/'complete.json'))))
else:
 with (b/'data-operator.lock').open('a') as g:
  fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (b/'data-process.json').exists():
   previous=read(b/'data-process.json');state=alive(previous['pid'],c)
   if state['live']:raise ValueError('Existing native LP extraction writer is active')
   if not v['resume']:raise ValueError('Previous extraction attempt exists; explicit resume required')
  elif (b/'data-start-intent.json').exists():raise ValueError('Unknown prior data launch outcome; inspect before retry')
  for name,x in m['files'].items():assert ref(c/name)['sha256']==x['sha256']
  with (data/'extract.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  atomic(b/'data-start-intent.json',dict(config=p['config'],code_root=str(c),attempt=time.time(),resume=v['resume']))
  log=(b/'data-extract.log').open('ab');proc=subprocess.Popen([sys.executable,str(c/'prefix_confidence_data.py'),'--action','extract','--config-sha256',p['config']['sha256']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL);log.close();record=dict(pid=proc.pid,config=p['config'],code_root=str(c),started=time.time());atomic(b/'data-process.json',record);print(json.dumps(dict(status='BOUNDED_CPU_ONLY_CONFIDENCE_EXTRACTION_STARTED',process=record)))
'''
STATUS=r'''
x=dict(status='ACTUAL_PREFIX_CONFIDENCE_CPU_EXTRACTION',observed_epoch=time.time());data=b/'data'
for name in ('controller.json','progress.json','failure.json'):
 p=data/name
 if p.exists():x[name[:-5]]=read(p)
if (b/'data-process.json').exists():
 p=read(b/'data-process.json');x['process']=dict(p,actual=alive(p['pid'],Path(p['code_root'])))
 try:x['process']['rss']=Path('/proc/%d/status'%p['pid']).read_text().split('VmRSS:',1)[1].splitlines()[0].strip()
 except (FileNotFoundError,IndexError):pass
if (data/'complete.json').exists():
 z=read(data/'complete.json');x['complete']=ref(data/'complete.json');x['summary']={k:z[k] for k in ('status','features','groups','split','questions','trajectories','feature_names','quality_by_seed','same_causal_LP_numeric_differences','elapsed_seconds','peak_RSS_bytes','all_own_ID_EM_order_native_LP_and_P20_exact','external_payload_opened')}
if (b/'data-extract.log').exists():x['log_tail']=(b/'data-extract.log').read_text()[-1800:]
print(json.dumps(x))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','start','resume','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('prefix_confidence_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(source,extra=None):
  v=dict(base=str(d.BASE));v.update(extra or {});cmd='python3 -c '+shlex.quote('v='+repr(v)+'\n'+COMMON+source)
  if len(cmd.encode())>=16000:raise ValueError('CPU control command must remain below16KB')
  r=control.run(cmd,account='mou176',source_address=address,timeout=55);rows=[json.loads(s) for s in r.stdout.decode(errors='replace').splitlines() if s.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('Actual CPU data state unconfirmed; inspect before retry: '+r.stderr.decode(errors='replace')[-1700:]+' '+r.stdout.decode(errors='replace')[-1700:])
  return rows[-1]
 if a.action=='stage':
  assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]==a.commit
  payloads={name:subprocess.check_output(['git','show',a.commit+':reproduction/operations/qwen7b_warmup/'+name],cwd=repo) for name in FILES}
  for name,x in payloads.items():assert x==(repo/'reproduction/operations/qwen7b_warmup'/name).read_bytes()
  protocol='reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence-PROTOCOL.md';payloads['prefix-confidence-PROTOCOL.md']=subprocess.check_output(['git','show','4e5d2b803954e959cb470d2bf07fa35420d3ed10:'+protocol],cwd=repo);files={name:dict(bytes=len(x),sha256=hashlib.sha256(x).hexdigest()) for name,x in payloads.items()};call(t.STAGE,dict(commit=a.commit,files=files))
  for name,x in payloads.items():
   for i,start in enumerate(range(0,len(x),4096)):
    chunk=x[start:start+4096];call(t.CHUNK,dict(name=name,index=i,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest()))
  result=call(SEAL,dict(branch=BRANCH))
 else:result=call(dict(prepare=PREPARE,start=START,resume=START,status=STATUS)[a.action],dict(resume=a.action=='resume'))
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('prefix-confidence-data-'+a.action+'-latest.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
