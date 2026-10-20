"""Operator for published independent NAS10 CPU snapshot; ctl.route direct only."""
import base64,hashlib,importlib.util,json,shlex,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
CONTROL=REPO.parent.parent
COMMIT='2043880a13df12566d010a8d83edbbfec679d727'
NAMES=['peer_newseed_export.py','peer_newseed_audit_reference.py','test_peer_newseed_export.py','peer_newseed_hf_stream.py']
ROOT='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021'
SNAPSHOT='newseed-20261021-185747-445fc691'
CODE=ROOT+'/_code-'+COMMIT[:12]
spec=importlib.util.spec_from_file_location('nas10_published_export_control','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)


def call(script,timeout=55):
 if len(script.encode())>=16000:raise ValueError('Operator command payload must stay below16KB')
 r=subprocess.run(ctl.route('h100-10','python3 -'),input=script,text=True,capture_output=True,timeout=timeout)
 rows=[json.loads(x) for x in r.stdout.splitlines() if x.startswith('{')]
 if r.returncode or not rows:raise RuntimeError('Action outcome unconfirmed; reconcile before repeating: '+r.stderr[-1600:]+' '+r.stdout[-1800:])
 print(json.dumps(rows[-1]),flush=True);return rows[-1]


def main():
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO,text=True).split()[0]
 if head!=COMMIT or remote!=COMMIT:raise RuntimeError('Exact approved source must be committed and pushed first')
 files={}
 for name in NAMES:
  relative='reproduction/operations/qwen7b_warmup/'+name;raw=(REPO/relative).read_bytes()
  if raw!=subprocess.check_output(['git','show',COMMIT+':'+relative],cwd=REPO):raise RuntimeError('Unpublished source '+name)
  files[relative]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  for offset in range(0,len(raw),4096):
   part=raw[offset:offset+4096]
   script='values='+repr(dict(root=CODE,relative=relative,offset=offset,encoded=base64.b64encode(part).decode(),chunk_sha=hashlib.sha256(part).hexdigest(),total=len(raw),file_sha=files[relative]['sha256']))+'\n'+r'''
import pathlib,json,base64,hashlib,os,socket
assert socket.gethostname().split('.')[0]=='h100-10'
r=pathlib.Path(values['root']);r.mkdir(parents=True,exist_ok=True);p=r/values['relative'];p.parent.mkdir(parents=True,exist_ok=True)
raw=base64.b64decode(values['encoded']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==values['chunk_sha']
tmp=p.with_name(p.name+'.upload');n=tmp.stat().st_size if tmp.exists() else 0
if p.exists():assert p.stat().st_size==values['total'] and hashlib.sha256(p.read_bytes()).hexdigest()==values['file_sha']
else:
 if n==values['offset']:
  with tmp.open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 else:
  assert n>=values['offset']+len(raw)
  with tmp.open('rb') as f:f.seek(values['offset']);assert f.read(len(raw))==raw
 if tmp.stat().st_size==values['total']:
  assert hashlib.sha256(tmp.read_bytes()).hexdigest()==values['file_sha'];os.link(str(tmp),str(p));p.chmod(0o444);tmp.unlink()
print(json.dumps(dict(status='PUBLISHED_EXPORT_CODE_CHUNK_SHA_PASSED',file=values['relative'],offset=values['offset'],bytes=len(raw))))
'''
   call(script)
 publication=dict(commit=COMMIT,origin_sha=COMMIT,code_root=CODE,files=files,exporter_module='reproduction.operations.qwen7b_warmup.peer_newseed_export',no_peer_source_mutation=True)
 script='values='+repr(dict(publication=publication,snapshot=SNAPSHOT,root=ROOT,code=CODE))+'\n'+r'''
import pathlib,json,hashlib,os,subprocess,sys,time,fcntl,socket
assert socket.gethostname().split('.')[0]=='h100-10'
code=pathlib.Path(values['code']);pub=values['publication']
for relative,item in pub['files'].items():
 p=code/relative;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
p=code/'source-publication.json'
if p.exists():assert json.loads(p.read_text())==pub
else:
 with p.open('x') as f:json.dump(pub,f,sort_keys=True);f.flush();os.fsync(f.fileno())
operator=pathlib.Path(values['root'])/'_operators'/values['snapshot'];operator.mkdir(parents=True,exist_ok=True)
lock=(operator/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ,PYTHONPATH=str(code),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run([sys.executable,'-m','unittest','reproduction.operations.qwen7b_warmup.test_peer_newseed_export'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=80)
(operator/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:]
entry=code/'reproduction/operations/qwen7b_warmup/peer_newseed_export.py'
r=subprocess.run([sys.executable,str(entry),'--discover'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
assert r.returncode==0,r.stderr[-1000:];discovery=json.loads(r.stdout.splitlines()[-1])
p=(operator/'discovery.json');p.write_text(json.dumps(discovery,sort_keys=True))
launch=operator/'launch.json'
if launch.exists():
 receipt=json.loads(launch.read_text());pid=receipt['pid'];proc=pathlib.Path('/proc')/str(pid)
 alive=proc.exists() and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==receipt['start_ticks']
 print(json.dumps(dict(status='EXISTING_EXPORT_PROCESS_PRESERVED',identity_alive=alive,receipt=receipt,discovery=discovery)))
else:
 assert not (operator/'launch-intent.json').exists(),'Unknown previous start must be reconciled before retry'
 args=[sys.executable,'-u',str(entry),'--export','--snapshot-id',values['snapshot']]
 intent=dict(status='SOURCE_CPU_EXPORT_LAUNCH_INTENT',command=args,commit=pub['commit'],snapshot_path=str(pathlib.Path(values['root'])/values['snapshot']),source_run_id='70188b8e50fd4b45bc420ce58014f46e',epoch=time.time())
 with (operator/'launch-intent.json').open('x') as f:json.dump(intent,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 log=(operator/'export.log').open('xb');child=subprocess.Popen(args,env=env,cwd=str(code),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
 receipt=dict(intent,status='SOURCE_CPU_EXPORT_STARTED_NOT_YET_ACCEPTED',pid=child.pid,start_ticks=ticks,log=str(operator/'export.log'),source_publication=str(code/'source-publication.json'),actual_cpu_tests_passed=True)
 with launch.open('x') as f:json.dump(receipt,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(status=receipt['status'],receipt=receipt,discovery=discovery)))
'''
 result=call(script,110);result['operator_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 p=CONTROL/'outputs/qwen7b-warmup-20261019/peer-newseed-export-launch.json';p.write_text(json.dumps(result,indent=2)+'\n')
 (Path(__file__).parent/'peer-newseed-export-launch.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
