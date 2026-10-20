"""Code-only immutable stage and guarded bounded CPU diagnostic operator."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO=CONTROL/'work/qwen7b-probe-warmup-20261019'
COMMIT='262f549'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-paired-old-new-20261021'
FILE='reproduction/operations/qwen7b_warmup/peer_oldnew_diagnostic.py'
COMMON='''import pathlib,json,hashlib,base64,fcntl,os,subprocess,time
b=pathlib.Path(inputs['base']);b.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def atomic(p,v):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\\n');t.replace(p)
def lock():
 f=(b/'operator.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
'''
BLOCKS=dict(stage='''
f=lock();root=b/('code-'+inputs['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%05d'%inputs['index']);raw=base64.b64decode(inputs['payload']);assert len(raw)<=4096 and hashlib.sha256(raw).hexdigest()==inputs['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(status='CODE_CHUNK_SHA_ACCEPTED',index=inputs['index'])))
''',finish='''
f=lock();root=b/('code-'+inputs['commit'][:12]);raw=b''.join((root/('part-%05d'%i)).read_bytes() for i in range(inputs['chunks']));assert hashlib.sha256(raw).hexdigest()==inputs['file_sha256'];p=root/'peer_oldnew_diagnostic.py'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw);p.chmod(0o444)
compile(raw,str(p),'exec');v=dict(commit=inputs['commit'],remote_sha=inputs['remote_sha'],code_root=str(root),source=dict(path=str(p),bytes=len(raw),sha256=inputs['file_sha256']))
if (b/'source-publication.json').exists():assert read(b/'source-publication.json')==v
else:atomic(b/'source-publication.json',v)
print(json.dumps(dict(status='CPU_DIAGNOSTIC_IMMUTABLE_SOURCE_ACCEPTED',source=v)))
''',start='''
f=lock();pub=read(b/'source-publication.json');p=pathlib.Path(pub['source']['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==pub['source']['sha256']
launch=b/('cpu-launch-'+inputs['attempt']+'.json');intent=b/('cpu-intent-'+inputs['attempt']+'.json')
if (b/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE_PRESERVED',complete=read(b/'complete.json')['status'])))
elif launch.exists():
 v=read(launch);print(json.dumps(dict(status='EXISTING_CPU_LAUNCH_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not intent.exists(),'Unknown existing launch outcome; do not repeat';v=dict(attempt=inputs['attempt'],source=pub['source'],epoch=time.time(),action='CPU_ONLY_PAIRED_TRAINING_DIAGNOSTIC');atomic(intent,v)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/('cpu-'+inputs['attempt']+'.log')).open('ab') as log:child=subprocess.Popen(['python3','-u',str(p),'--model','both'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 v.update(pid=child.pid,start_ticks=ticks(child.pid),status='STARTED_NOT_PROGRESS_VERIFIED');atomic(launch,v);print(json.dumps(v))
''',status='''
out=dict(epoch=time.time(),base=str(b))
for name in ('controller.json','progress.json','failure.json'):
 p=b/name
 if p.exists():out[name]=read(p)
for name in ('latest.json','complete.json'):
 p=b/name
 if p.exists():
  v=read(p);out[name]=dict(status=v['status'],models=list(v['results']),epoch=v['epoch'])
out['models']={}
for name in ('E4','peer256'):
 p=b/name/'complete.json'
 if p.exists():
  v=read(p);out['models'][name]=dict(status=v['status'],epoch=v['epoch'],elapsed_seconds=v['elapsed_seconds'],source_predictions=v['source_predictions'],old5_group=v['old5']['group'],new5_group=v['new5']['group'],paired=v['paired'])
print(json.dumps(out))
''')
def main():
 a=argparse.ArgumentParser();a.add_argument('--action',choices=('stage','start','status'),required=True);a.add_argument('--attempt',default='v1');args=a.parse_args()
 spec=importlib.util.spec_from_file_location('oldnew_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  v=dict(base=TARGET,attempt=args.attempt);v.update(extra or {});code='inputs='+repr(v)+'\n'+COMMON+'\n'+BLOCKS[action];command='python3 -c '+shlex.quote(code);assert len(command.encode())<16000
  r=ctl.run(command,account='mou176',source_address=address,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout.decode()[-2000:]+r.stderr.decode()[-1000:])
  return json.loads(r.stdout.decode().splitlines()[-1])
 if args.action=='stage':
  commit=subprocess.check_output(['git','rev-parse',COMMIT],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO,text=True).split()[0]
  subprocess.run(['git','merge-base','--is-ancestor',commit,remote],cwd=REPO,check=True);data=subprocess.check_output(['git','show',commit+':'+FILE],cwd=REPO);h=hashlib.sha256(data).hexdigest()
  for i,start in enumerate(range(0,len(data),4096)):
   raw=data[start:start+4096];call('stage',dict(commit=commit,index=i,payload=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
  result=call('finish',dict(commit=commit,remote_sha=remote,chunks=(len(data)+4095)//4096,file_sha256=h))
 else:result=call(args.action)
 out=CONTROL/'outputs/qwen7b-warmup-20261019'/('peer-oldnew-'+args.action+'-'+args.attempt+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
