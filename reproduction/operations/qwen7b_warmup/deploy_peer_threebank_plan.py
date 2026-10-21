"""Publish-bound finite CPU metadata operator; never submits any GPU job."""
import argparse,base64,gzip,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
from .peer_threebank_plan import TARGET
P='reproduction/operations/qwen7b_warmup/'
FILES=[P+n for n in ['peer_threebank_plan.py','test_peer_threebank_plan.py','deploy_peer_threebank_plan.py','train_g10_domain_statistics.py']]
BRANCH='codex/peer-threebank58432-plan-20261021'
SOURCE='111342b816ca8674a6468a889c4f9cc50ad8c91b'
SOURCE_BASE=TARGET.parent/'peer-audited100-after366-L26-replay-v1'
COMMON=r'''
import base64,fcntl,gzip,hashlib,json,os,pathlib,shutil,subprocess,sys,time
b=pathlib.Path(inputs['base']);b.mkdir(parents=True,exist_ok=True);code=b/('cpu-code-'+inputs['commit'][:12]);runtime=code/'reproduction/operations/qwen7b_warmup/runtime.sh'
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def lock():
 f=(b/'operator.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def publication():
 v=read(code/'cpu-source-publication.json');assert v['commit']==inputs['commit']
 for name,item in v['files'].items():assert fp(code/name)==dict(item,path=str(code/name))
 assert fp(b/'config.json')['sha256']==v['files']['cohort-config.json']['sha256'];return v
'''
CHUNK=r'''
guard=lock();p=b/'source-bundle.gz.upload';data=base64.b64decode(inputs['chunk_data']);assert hashlib.sha256(data).hexdigest()==inputs['chunk_sha256'];size=p.stat().st_size if p.exists() else 0
if size==inputs['offset']:
 with p.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
else:
 assert size>=inputs['offset']+len(data)
 with p.open('rb') as f:f.seek(inputs['offset']);assert f.read(len(data))==data
print(json.dumps(dict(status='OWN_CPU_METADATA_SOURCE_CHUNK_ACCEPTED',offset=inputs['offset'])))
'''
STAGE=r'''
guard=lock();data=(b/'source-bundle.gz.upload').read_bytes();assert hashlib.sha256(data).hexdigest()==inputs['bundle_sha256'];values=json.loads(gzip.decompress(data));assert set(values)==set(inputs['files'])
if not code.exists():
 parent=pathlib.Path(inputs['source_base']);pub=read(parent/'source-publication.json');assert pub['commit']==inputs['source_commit'];original=pathlib.Path(pub['code_root']);stage=code.with_name(code.name+'.staging');assert not stage.exists(),'Unknown source partial preserved'
 for name,item in pub['files'].items():assert fp(original/name)==dict(item,path=str(original/name))
 shutil.copytree(str(original),str(stage))
 for name,encoded in values.items():
  raw=base64.b64decode(encoded);item=inputs['files'][name];assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];p=stage/name;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
 atomic(stage/'cpu-source-publication.json',dict(commit=inputs['commit'],files=inputs['files'],base_capture_source=fp(parent/'source-publication.json'),tensor_payload_copied=False));stage.rename(code)
raw=(code/'cohort-config.json').read_bytes();p=b/'config.json'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw);p.chmod(0o444)
publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
p=subprocess.run(['bash',str(runtime),'-m','unittest','reproduction.operations.qwen7b_warmup.test_peer_threebank_plan'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=40);(b/'cpu-tests.log').write_text(p.stdout);assert p.returncode==0,p.stdout[-1500:]
print(json.dumps(dict(status='THREEBANK_CPU_SOURCE_AND_FOUR_TESTS_ACCEPTED',code=str(code),publication=fp(code/'cpu-source-publication.json'),config=fp(b/'config.json'),tests=fp(b/'cpu-tests.log'))))
'''
START=r'''
guard=lock();publication();receipt=b/'cpu-launch.json'
if (b/'complete.json').exists():print(json.dumps(dict(status='EXISTING_METADATA_PLAN_COMPLETION_PRESERVED',complete=fp(b/'complete.json'))))
elif receipt.exists():
 v=read(receipt);print(json.dumps(dict(status='EXISTING_CPU_OPERATOR_PRESERVED',launch=v,same_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-launch-intent.json').exists(),'Unknown prior launch outcome; no duplicate'
 args=['bash',str(runtime),'-u','-m','reproduction.operations.qwen7b_warmup.peer_threebank_plan','--config',str(b/'config.json')];atomic(b/'cpu-launch-intent.json',dict(config=fp(b/'config.json'),publication=fp(code/'cpu-source-publication.json'),args=args,epoch=time.time()))
 supervisor='import json,pathlib,subprocess,sys,time,os; start=time.time(); p=subprocess.run(json.loads(sys.argv[1])); value=dict(exit_code=p.returncode,elapsed_seconds=time.time()-start,epoch=time.time()); q=pathlib.Path(sys.argv[2]); tmp=q.with_suffix(".tmp");tmp.write_text(json.dumps(value));tmp.replace(q);sys.exit(p.returncode)'
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');allowed=sorted(os.sched_getaffinity(0))[:2]
 with (b/'cpu.log').open('ab') as log:p=subprocess.Popen([sys.executable,'-u','-c',supervisor,json.dumps(args),str(b/'cpu-exit.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env,preexec_fn=lambda:os.sched_setaffinity(0,allowed))
 v=dict(status='CPU_METADATA_STARTED_NOT_PROGRESS_VERIFIED',pid=p.pid,start_ticks=ticks(p.pid),cpu_affinity=allowed,config=fp(b/'config.json'),epoch=time.time());atomic(receipt,v);print(json.dumps(v))
'''
STATUS=r'''
out=dict(status='ACTUAL_THREEBANK_METADATA_PLAN_OBSERVATION',epoch=time.time(),accepted_source_bank_receipts=len(list(b.glob('source-*-accepted.json'))))
for name in ['cpu-launch.json','cpu-exit.json']:
 if (b/name).exists():out[name]=read(b/name)
if (b/'cpu-launch.json').exists():v=read(b/'cpu-launch.json');out['same_process_birth']=ticks(v['pid'])==v['start_ticks']
if (b/'complete.json').exists():out['complete']=read(b/'complete.json');out['complete_ref']=fp(b/'complete.json')
if (b/'cpu.log').exists():out['log_tail']=(b/'cpu.log').read_text(errors='replace')[-2500:]
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--action',choices=('stage','start','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 config_name=str(a.config.resolve().relative_to(repo));names=FILES+[config_name];payload={};refs={}
 for name in names:
  raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes();dest='cohort-config.json' if name==config_name else name;payload[dest]=base64.b64encode(raw).decode();refs[dest]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 v=dict(base=str(TARGET),source_base=str(SOURCE_BASE),source_commit=SOURCE,commit=a.commit,files=refs)
 scripts=dict(chunk=CHUNK,stage=STAGE,start=START,status=STATUS)
 def source(action,extra=None):return 'inputs='+repr(dict(v,**(extra or {})))+'\n'+COMMON+'\n'+scripts[action]
 spec=importlib.util.spec_from_file_location('threebank_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  script=source(action,extra);compile(script,action,'exec');cmd='python3 -c '+shlex.quote(script);assert len(cmd.encode())<16000;out=ctl.run(cmd,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x) for x in out.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  if out.returncode or not rows:raise RuntimeError(out.stderr.decode(errors='replace')[-1500:]+' '+out.stdout.decode(errors='replace')[-2000:])
  return rows[-1]
 if a.action=='stage':
  actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0];assert actual==a.commit,'Publish exact CPU metadata source first'
  blob=gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0)
  for i in range(0,len(blob),5000):
   part=blob[i:i+5000];call('chunk',dict(offset=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
  answer=call('stage',dict(bundle_sha256=hashlib.sha256(blob).hexdigest()))
 else:answer=call(a.action)
 path=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('peer-threebank-plan-'+a.action+'.json');path.write_text(json.dumps(answer,indent=2,sort_keys=True)+'\n');print(json.dumps(answer))
if __name__=='__main__':main()
