"""Explicit immutable inter1/bothrank1 operators; independent names and locks."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import peer_rank_matched as r,deploy_peer_within1_matched as prior
BRANCH='codex/peer8192-rank-matched256-20261021'
OWN=('peer_rank_matched.py','deploy_peer_rank_matched.py','test_peer_rank_matched.py')
COMMON,INIT,CHUNK=prior.COMMON,prior.INIT,prior.CHUNK
EXTRACT=prior.EXTRACT.replace("file_origins=v['file_origins'])","file_origins=v['file_origins'],matched_rank_reference_commit=inputs['reference_commit'])")
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():
 p=read(b/'preparation.json');verify(p['config']);verify(p['sbatch']);print(json.dumps(p));sys.exit(0)
controller=b/'cpu-prepare-controller.json'
if controller.exists():
 old=read(controller);active=False
 try:active='peer_rank_matched' in pathlib.Path('/proc/%d/cmdline'%old['pid']).read_bytes().decode(errors='replace')
 except OSError:pass
 if active:print(json.dumps(dict(status='RANK_CPU_PREPARATION_ACTIVE',pid=old['pid'])));sys.exit(0)
 raise RuntimeError('Prior CPU prepare exited; inspect failure instead of blind restart')
code="import unittest,sys,runpy; s=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_peer_rank_matched'); v=unittest.TextTestRunner(verbosity=2).run(s); assert v.wasSuccessful(); sys.argv=['peer_rank_matched','--recipe',"+repr(inputs['recipe'])+",'--action','prepare']; runpy.run_module('reproduction.operations.qwen7b_warmup.peer_rank_matched',run_name='__main__')"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'cpu-prepare.log').open('ab',0)
args=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',code];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(controller,dict(pid=p.pid,args=args,source_commit=v['commit'],time=time.time()));print(json.dumps(dict(status='MATCHED_RANK_TESTS_AND_CPU_PREPARATION_STARTED',pid=p.pid,recipe=inputs['recipe'])))
'''
STATUS=prior.STATUS.replace('MATCHED_WITHIN1','MATCHED_RANK')

def remote_source(values,action):
 recipe=values['recipe'];r.target(recipe)
 submit=prior.SUBMIT.replace('q7b-peer8192-within1','q7b-peer8192-'+recipe)
 submit="training_guard=(b/'rank-matched.lock').open('a+');fcntl.flock(training_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"+submit
 return 'inputs='+repr(values)+'\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,submit=submit,status=STATUS)[action]

def bundle(repo,commit,reference):
 published=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if published!=commit or reference['commit']!=r.REFERENCE:raise ValueError('Publish exact new source and use exact accepted within1 closure')
 origins=reference['file_origins'];payloads={};source_origins={}
 for name,item in reference['files'].items():
  origin=origins['overrides'].get(name,origins['default_commit']);data=subprocess.check_output(['git','show',origin+':'+name],cwd=repo)
  if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Accepted immutable reference closure differs: '+name)
  payloads[name]=data;source_origins[name]=origin
 for short in OWN:
  name=r.PKG+short;data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
  if data!=(repo/name).read_bytes():raise ValueError('Unpublished matched rank source: '+name)
  payloads[name]=data;source_origins[name]=commit
 out=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=out,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name,data in sorted(payloads.items()):
    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(data))
 return out.getvalue(),files,dict(default_commit=reference['file_origins']['default_commit'],overrides={n:c for n,c in source_origins.items() if c!=reference['file_origins']['default_commit']})

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--recipe',choices=tuple(r.RECIPES),required=True);p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit' and not a.submit_authorized:p.error('Explicit --submit-authorized required')
 spec=importlib.util.spec_from_file_location('peer_rank_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();base=r.target(a.recipe)
 values=dict(base=str(base),target=str(base),train=str(r.CONTROL),pin=r.w.D8,callback_commit=r.w.CALLBACK,reference_commit=r.REFERENCE,recipe=a.recipe)
 def call(action,extra=None):
  v=dict(values);v.update(extra or {});cmd='python3 -c '+shlex.quote(remote_source(v,action))
  if len(cmd.encode())>=16000:raise ValueError('Control command exceeds16KB')
  result=ctl.run(cmd,account='mou176',source_address=address,timeout=55)
  stdout=result.stdout.decode(errors='replace') if isinstance(result.stdout,bytes) else result.stdout
  stderr=result.stderr.decode(errors='replace') if isinstance(result.stderr,bytes) else result.stderr
  rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
  if result.returncode or not rows:raise RuntimeError('Outcome unconfirmed; inspect actual identity before any retry: '+stderr[-1500:]+' '+stdout[-2500:])
  return rows[-1]
 if a.action=='stage':
  packet=json.loads((Path(__file__).resolve().parents[3]/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/reference-source-packet.json').read_text())
  data,files,origins=bundle(Path(__file__).resolve().parents[3],a.commit,packet['within_publication']);chunks=[data[i:i+4096] for i in range(0,len(data),4096)]
  v=dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,file_origins=origins);call('init',v)
  for index,part in enumerate(chunks):call('chunk',dict(commit=a.commit,index=index,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))
  result=call('extract',dict(commit=a.commit))
 else:result=call(a.action)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer-rank-'+a.recipe+'-'+a.action+'-latest.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
