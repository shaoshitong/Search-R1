"""Code-only stage plus bounded CPU qualification/preparation; no submit API."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import direct_hidden_count_prepare as r,deploy_peer_append_train as prior
BRANCH='codex/direct-hidden-count-prepare-20261021'
REFERENCE='b8c3f8a76fa195efc7c2e95cfa1a7d36669576ec'
OWN=('direct_hidden_count.py','direct_hidden_count_update.py','test_direct_hidden_count.py','direct_hidden_count_prepare.py','deploy_direct_hidden_count_prepare.py')
COMMON,INIT,CHUNK=prior.COMMON,prior.INIT,prior.CHUNK
EXTRACT=prior.EXTRACT.replace("file_origins=v['file_origins'])","file_origins=v['file_origins'],reference_append_runtime_commit=inputs['reference'])")
PREPARE=r'''
h=lock();v,root=pub()
if (b/'preparation.json').exists():print(json.dumps(read(b/'preparation.json')));sys.exit(0)
if (b/'cpu-controller.json').exists():print(json.dumps(dict(status='EXISTING_CPU_PREPARATION_NO_DUPLICATE',controller=read(b/'cpu-controller.json'))));sys.exit(0)
code="import sys,runpy; sys.argv=['direct_hidden_count_prepare','--action','qualify']; runpy.run_module('reproduction.operations.qwen7b_warmup.direct_hidden_count_prepare',run_name='__main__'); sys.argv=['direct_hidden_count_prepare','--action','prepare']; runpy.run_module('reproduction.operations.qwen7b_warmup.direct_hidden_count_prepare',run_name='__main__')"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=(b/'cpu-prepare.log').open('ab',0)
args=['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',code];p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env);atomic(b/'cpu-controller.json',dict(pid=p.pid,args=args,commit=v['commit'],CPU_only=True,time=time.time()));print(json.dumps(dict(status='DIRECT_HIDDEN_COUNT_CPU_QUALIFICATION_AND_PREPARATION_STARTED',pid=p.pid)))
'''
STATUS=r'''
v=dict(status='DIRECT_HIDDEN_COUNT_PREPARE_ONLY_ACTUAL_STATE',time=time.time())
for name in ('source-publication.json','cpu-controller.json','cpu-qualification.json','preparation.json'):
 p=b/name
 if p.exists():
  x=read(p)
  if name=='source-publication.json':x={k:x[k] for k in ('commit','code_root','base_runtime_commit','callback_interface_commit','reference_append_runtime_commit')}
  v[name[:-5]]=x
if (b/'cpu-prepare.log').exists():v['log_tail']=(b/'cpu-prepare.log').read_text()[-3500:]
v['GPU_submission_available']=False;print(json.dumps(v))
'''
def remote_source(values,action):return 'inputs='+repr(values)+'\n'+COMMON+dict(init=INIT,chunk=CHUNK,extract=EXTRACT,prepare=PREPARE,status=STATUS)[action]
def bundle(repo,commit):
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]
 if remote!=commit:raise ValueError('Published exact preparation source required')
 reference=json.loads((repo/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-ranking-matched256/reference-source-packet.json').read_text())['within_publication']
 old,files,origins=prior.bundle(repo,REFERENCE,reference);payloads={}
 with tarfile.open(fileobj=io.BytesIO(old),mode='r:gz') as archive:
  for member in archive.getmembers():payloads[member.name]=archive.extractfile(member).read()
 for short in OWN:
  name=r.a.PKG+short;data=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
  if data!=(repo/name).read_bytes():raise ValueError('Unpublished count preparation source '+name)
  payloads[name]=data;origins['overrides'][name]=commit
 output=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name,data in sorted(payloads.items()):
    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o444;m.mtime=0;archive.addfile(m,io.BytesIO(data))
 return output.getvalue(),files,origins
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','prepare','status'),required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('direct_count_prepare_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();values=dict(base=str(r.TARGET),target=str(r.TARGET),pin=r.a.D8,callback_commit=r.a.CALLBACK,reference=REFERENCE)
 def call(action,extra=None):
  v=dict(values);v.update(extra or {});cmd='python3 -c '+shlex.quote(remote_source(v,action))
  if len(cmd.encode())>=16000:raise ValueError('Control command exceeds16KB')
  out=ctl.run(cmd,account='mou176',source_address=address,timeout=55)
  if out.returncode:raise RuntimeError(out.stderr[-1800:])
  return json.loads(out.stdout.splitlines()[-1])
 if a.action=='stage':
  data,files,origins=bundle(Path(__file__).resolve().parents[3],a.commit);chunks=[data[i:i+4096] for i in range(0,len(data),4096)]
  call('init',dict(commit=a.commit,remote_branch=BRANCH,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,file_origins=origins))
  for i,part in enumerate(chunks):call('chunk',dict(commit=a.commit,index=i,data=base64.b64encode(part).decode(),sha256=hashlib.sha256(part).hexdigest()))
  result=call('extract',dict(commit=a.commit))
 else:result=call(a.action)
 print(json.dumps(result));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('direct-hidden-count-'+a.action+'-latest.json');out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
