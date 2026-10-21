"""Code-only isolated CPU acceptance after four actual capture completions."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
P='reproduction/operations/qwen7b_warmup/'
FILES=[P+'peer_joint_subset_l26_merge.py']
BRANCH='codex/peer-audited122-L26-audits-20261021'
SOURCE='57bd9e337d6d2a75d2603b684d28752bfe3bf520'
COMMON=r'''
import base64,fcntl,gzip,hashlib,io,json,os,pathlib,subprocess,shutil,tarfile,time
b=pathlib.Path(inputs['base']);assert b.parent==pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')and b.name.startswith('peer-joint124194-')
code=b/('cpu-bank-acceptance-'+inputs['commit'][:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
lock=(b/'cpu-bank-operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pub=read(b/'source-publication.json')
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
'''
STAGE=r'''
if not code.exists():
 stage=code.with_name(code.name+'.staging');assert not stage.exists(),'Unknown partial code stage preserved'
 shutil.copytree(pub['code_root'],str(stage))
 blob=base64.b64decode(inputs['blob']);assert hashlib.sha256(blob).hexdigest()==inputs['blob_sha256']
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
  members=tar.getmembers();assert {v.name for v in members}==set(inputs['files'])
  for m in members:
   assert m.isfile() and not pathlib.Path(m.name).is_absolute() and '..' not in pathlib.Path(m.name).parts
   x=tar.extractfile(m).read();assert hashlib.sha256(x).hexdigest()==inputs['files'][m.name]['sha256'];p=stage/m.name;assert not p.exists();p.write_bytes(x);p.chmod(0o444)
 atomic(stage/'cpu-source-publication.json',dict(commit=inputs['commit'],source_producer=fp(b/'source-publication.json'),files=inputs['files']));stage.rename(code)
for name,item in inputs['files'].items():assert fp(code/name)==dict(item,path=str(code/name))
print(json.dumps(dict(status='ISOLATED_CPU_SUBSET_BANK_CODE_PUBLISHED',code_root=str(code),source=fp(code/'cpu-source-publication.json'))))
'''
START=r'''
assert code.exists();receipt=b/'cpu-bank-launch.json'
if (b/'bank/complete.json').exists():print(json.dumps(dict(status='EXISTING_BANK_COMPLETION_PRESERVED',complete=fp(b/'bank/complete.json'))))
elif receipt.exists():
 v=read(receipt);print(json.dumps(dict(status='EXISTING_CPU_BANK_LAUNCH_PRESERVED',launch=v,same_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-bank-launch-intent.json').exists(),'Unknown prior launch outcome; never repeat'
 assert all((b/('shard-%02d'%i)/'capture/complete.json').exists() for i in range(4)),'Four actual finished feature shards required'
 atomic(b/'cpu-bank-launch-intent.json',dict(source=fp(code/'cpu-source-publication.json'),preparation=fp(b/'preparation.json'),epoch=time.time()))
 allowed=sorted(os.sched_getaffinity(0))[:2]
 with (b/'cpu-bank-acceptance.log').open('ab') as log:p=subprocess.Popen(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.peer_joint_subset_l26_merge','--base',str(b)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,allowed))
 v=dict(status='CPU_BANK_STARTED_NOT_YET_ACCEPTED',pid=p.pid,start_ticks=ticks(p.pid),code_root=str(code),cpu_affinity=allowed,epoch=time.time());atomic(receipt,v);print(json.dumps(v))
'''
STATUS=r'''
out=dict(status='ACTUAL_CPU_BANK_ACCEPTANCE_OBSERVATION',epoch=time.time(),four_capture_completes=sum((b/('shard-%02d'%i)/'capture/complete.json').exists() for i in range(4)))
for name in ['cpu-bank-launch.json','bank-acceptance-progress.json']:
 if (b/name).exists():out[name]=read(b/name)
if (b/'cpu-bank-launch.json').exists():v=read(b/'cpu-bank-launch.json');out['same_process_birth']=ticks(v['pid'])==v['start_ticks']
if (b/'bank/complete.json').exists():out['complete']=read(b/'bank/complete.json');out['complete_ref']=fp(b/'bank/complete.json')
if (b/'cpu-bank-acceptance.log').exists():out['log_tail']=(b/'cpu-bank-acceptance.log').read_text(errors='replace')[-1800:]
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--branch',required=True);p.add_argument('--base',required=True);p.add_argument('--action',choices=('stage','start','status'),default='status');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 v=dict(base=a.base,commit=a.commit)
 if a.action=='stage':
  actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+a.branch],cwd=repo,text=True).split()[0]
  if actual!=a.commit:raise ValueError('Exact latest published CPU source required')
  data=io.BytesIO();files={}
  with tarfile.open(fileobj=data,mode='w:gz') as tar:
   for name in FILES:
    raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
  blob=data.getvalue();v.update(blob=base64.b64encode(blob).decode(),blob_sha256=hashlib.sha256(blob).hexdigest(),files=files)
 source='inputs='+repr(v)+'\n'+COMMON+'\n'+dict(stage=STAGE,start=START,status=STATUS)[a.action];command='python3 -c '+shlex.quote(source);assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('delta_cpu_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();result=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 rows=[json.loads(x) for x in result.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
 if result.returncode or not rows:raise RuntimeError(result.stderr.decode(errors='replace')[-1500:]+' '+result.stdout.decode(errors='replace')[-1500:])
 out=rows[-1];path=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('joint-subset-bank-'+Path(a.base).name+'-'+a.action+'.json');path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
