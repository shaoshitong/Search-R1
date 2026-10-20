"""Code-only control of the H100 -> management RAM -> private HF uploader."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PARENT='/home/lichen/qwen7b-newseed-to-hpc3-20261021'
REMOTE=r'''
import pathlib,json,hashlib,os,time,fcntl,subprocess,sys,importlib.util
b=pathlib.Path(inputs['base']);source=pathlib.Path(inputs['source']);name=inputs['snapshot']
assert b.parent==pathlib.Path('/home/lichen/qwen7b-newseed-to-hpc3-20261021') and b.name==name
assert source.parent==pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021') and source.name==name
def read(p):return json.loads(pathlib.Path(p).read_text())
def atomic(p,v):
 p=pathlib.Path(p);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 q.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def identity(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
path=b/('peer-newseed-hf-'+inputs['code_sha256']+'.py')
'''


def render(inputs,action):
    code='inputs='+repr(inputs)+'\n'+REMOTE
    if action=='chunk':
        code+=r'''
b.mkdir(parents=True,exist_ok=True);folder=b/'code-chunks';folder.mkdir(exist_ok=True)
import base64
raw=base64.b64decode(inputs['data']);assert hashlib.sha256(raw).hexdigest()==inputs['chunk_sha256']
p=folder/('%s-%04d'%(inputs['code_sha256'],inputs['index']))
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='CODE_CHUNK_SHA_ACCEPTED',index=inputs['index'])))
'''
    elif action=='stage':
        code+=r'''
raw=b''.join((b/'code-chunks'/('%s-%04d'%(inputs['code_sha256'],i))).read_bytes() for i in range(inputs['chunks']))
assert hashlib.sha256(raw).hexdigest()==inputs['code_sha256'];compile(raw.decode(),str(path),'exec')
if path.exists():assert sha(path)==inputs['code_sha256']
else:
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 path.chmod(0o444)
pub=dict(status='PUBLISHED_HF_RAM_STREAM_CODE_STAGED_ONLY',source_commit=inputs['commit'],source=str(source),path=str(path),sha256=inputs['code_sha256'],bytes=len(raw),epoch=time.time())
atomic(b/'source-publication.json',pub);print(json.dumps(pub))
'''
    elif action in ('start','resume-failed'):
        if action=='resume-failed':
            inputs=dict(inputs,resume_failed=True)
            code='inputs='+repr(inputs)+'\n'+REMOTE
        code+=r'''
assert sha(path)==inputs['code_sha256']
secret=json.loads(sys.stdin.readline());assert secret['hf_token'] and set(secret)=={'hf_token'}
with (b/'launch.lock').open('a') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=b/'launch.json'
 if inputs.get('resume_failed'):
  assert receipt.exists() and (b/'failure.json').exists() and not (b/'complete.json').exists()
  previous=read(receipt);assert identity(previous['pid']) is None,'Prior uploader PID still exists'
  with (b/'uploader.lock').open('a') as writer:
   fcntl.flock(writer,fcntl.LOCK_EX|fcntl.LOCK_NB)
  history=b/'failed-attempts'/('pid-%d-%s'%(previous['pid'],previous['start_ticks']))
  history.mkdir(parents=True,exist_ok=False)
  atomic(history/'retirement.json',dict(previous_launch=previous,verified_pid_absent=True,verified_writer_lock_free=True,epoch=time.time()))
  for filename in ('launch.json','launch-intent.json','failure.json','controller.json','retry.json'):
   old=b/filename
   if old.exists():old.replace(history/filename)
 if receipt.exists():
  v=read(receipt);print(json.dumps(dict(status='EXISTING_LAUNCH_NOT_REPLAYED',launch=v,actual_start_ticks=identity(v['pid']))))
 else:
  assert not (b/'launch-intent.json').exists(),'Unknown prior launch, reconcile before replay'
  spec=importlib.util.spec_from_file_location('peer_hf_preflight',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  plan=m.remote_snapshot(source)
  assert plan['snapshot_id']==name
  atomic(b/'launch-intent.json',dict(source=str(source),code_sha256=inputs['code_sha256'],source_questions=plan['questions'],source_trajectories=plan['trajectories'],epoch=time.time()))
  with (b/'uploader.log').open('ab') as log:
   p=subprocess.Popen(['python3','-u',str(path),'--base',str(b),'--source',str(source)],stdin=subprocess.PIPE,
    stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,HF_HUB_DISABLE_XET='1',HF_HUB_ENABLE_HF_TRANSFER='0',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_VERBOSITY='error',CUDA_VISIBLE_DEVICES=''))
  p.stdin.write(json.dumps(secret).encode()+bytes([10]));p.stdin.close()
  v=dict(status='RAM_STREAM_UPLOADER_STARTED_NOT_MIGRATION_COMPLETE',pid=p.pid,start_ticks=identity(p.pid),source=str(source),code_sha256=inputs['code_sha256'],token_saved=False,Mac_data_transit=False,epoch=time.time())
  atomic(receipt,v);print(json.dumps(v))
'''
    elif action=='status':
        code+=r'''
out=dict(observed_epoch=time.time(),base=str(b),exists=b.exists())
for name in ('launch.json','progress.json','failure.json','retry.json','controller.json'):
 p=b/name
 if p.exists():out[name]=read(p)
if (b/'launch.json').exists():
 v=read(b/'launch.json');out['actual_process_same_birth']=identity(v['pid'])==v['start_ticks']
 if out['actual_process_same_birth']:
  stat=pathlib.Path('/proc/%d/io'%v['pid'])
  if stat.exists():out['actual_io_counters']={k:int(v) for k,v in (line.split(':',1) for line in stat.read_text().splitlines())}
if (b/'complete.json').exists():
 v=read(b/'complete.json');out['complete']=dict(status=v['status'],snapshot_id=v['snapshot_id'],groups=len(v['groups']),catalog_path=v['catalog_path'],catalog_commit=v['catalog_commit'],source_questions=v['source_snapshot']['questions'],source_trajectories=v['source_snapshot']['trajectories'])
print(json.dumps(out))
'''
    else:raise ValueError(action)
    return code


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--action',choices=['stage','start','resume-failed','status'],required=True);p.add_argument('--control-root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[3];raw=Path(__file__).with_name('peer_newseed_hf_stream.py').read_bytes()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if a.action=='stage':
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
        if commit!=remote:raise ValueError('Publish current source before staging')
        for name in ('peer_newseed_hf_stream.py','deploy_peer_newseed_hf.py'):
            file=Path(__file__).with_name(name);tracked=subprocess.check_output(['git','show',commit+':'+str(file.relative_to(repo))],cwd=repo)
            if file.read_bytes()!=tracked:raise ValueError('Unpublished transfer code')
    snapshot=a.source.name
    inputs=dict(base=PARENT+'/'+snapshot,source=str(a.source),snapshot=snapshot,
        code_sha256=hashlib.sha256(raw).hexdigest(),commit=commit,chunks=(len(raw)+4095)//4096)
    def call(action,extra=None,secret=None):
        values=dict(inputs);values.update(extra or {})
        cmd='python3 -c '+shlex.quote(render(values,action))
        if len(cmd.encode())>=16000:raise ValueError('Management control command exceeds16KB')
        argv=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=8','mgmt-host',cmd]
        payload=json.dumps(secret).encode()+bytes([10]) if secret is not None else None
        r=subprocess.run(argv,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
        rows=[json.loads(s) for s in r.stdout.decode(errors='replace').splitlines() if s.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Remote action unconfirmed; inspect same intent before retry')
        return rows[-1]
    if a.action=='stage':
        for i in range(inputs['chunks']):
            data=raw[i*4096:(i+1)*4096];call('chunk',dict(index=i,data=base64.b64encode(data).decode(),chunk_sha256=hashlib.sha256(data).hexdigest()))
        result=call('stage')
    elif a.action in ('start','resume-failed'):
        token=(Path.home()/'.cache/huggingface/token').read_text().strip()
        result=call(a.action,secret=dict(hf_token=token));token=None
    else:result=call('status')
    out=a.control_root/'outputs/qwen7b-warmup-20261019'/('peer-newseed-hf-'+a.action+'.json')
    out.write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result))


if __name__=='__main__':main()
