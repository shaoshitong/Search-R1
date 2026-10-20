"""Explicit real12-shard gate, standalone published CPU merge stage/start/status.

No scheduler calls/GPU requests. Original collector and source banks stay intact.
"""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
REPO=pathlib.Path(__file__).resolve().parents[4]
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
B='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
A=B+'/algorithm-improvement-20261021'
TARGET=A+'/training-G20-matched6528-20261021/materialized'
RECEIVER=B+'/peer-newseed-h100-20261021/newseed-20261021-185747-445fc691'
COMMON=r'''
import pathlib,json,hashlib,os,time,fcntl,subprocess,base64,importlib.util,sys
A=pathlib.Path(inputs['A']);b=pathlib.Path(inputs['target'])
phases={o:A/('training-extraG5-matched6528'+('' if o==5 else '-offset%d'%o)+'-20261021/materialized') for o in (5,10,15)}
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def gate():
 result={}
 for offset,root in phases.items():
  own={}
  for shard in range(4):
   d=root/'sampling'/('shard-%02d'%shard)
   own[str(shard)]={n:(fp(d/n) if (d/n).exists() else None) for n in ['complete.json','extra-g5-complete.json','extra-g5-adapter-controller.json']}
  result[str(offset)]=own
 return result
'''
STATUS=r'''
gates=gate();out=dict(status='REAL_G20_COMPLETION_GATE_OBSERVATION',epoch=time.time(),gates=gates,all12ready=all(x for p in gates.values() for s in p.values() for x in s.values()),target=str(b))
for name in ['g20-cpu-launch.json','merge-progress.json','merge-failure.json','bank/complete.json']:
 p=b/name
 if p.exists():
  v=read(p)
  if name=='g20-cpu-launch.json':v=dict(v,same_process_birth=ticks(v['pid'])==v['start_ticks'])
  out[name]=v
p=b/'g20-cpu.log'
if p.exists():out['log_tail']=p.read_text(errors='replace')[-2500:]
print(json.dumps(out))
'''
CHUNK=r'''
root=pathlib.Path(inputs['code_root']);root.mkdir(exist_ok=True,parents=True)
p=root/('%05d.part'%inputs['chunk_index']);data=base64.b64decode(inputs['chunk_data']);assert len(data)<=4096
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data)
print(json.dumps(dict(status='STANDALONE_G20_CODE_CHUNK_ACCEPTED',chunk=inputs['chunk_index'])))
'''
START=r'''
gates=gate();assert all(x for p in gates.values() for s in p.values() for x in s.values()),'All12 real shards must be actually sealed before CPU merge'
assert not pathlib.Path(inputs['receiver']).joinpath('complete.json').exists(),'Peer actual receiver complete takes priority; defer this CPU merge'
root=pathlib.Path(inputs['code_root']);data=b''.join(p.read_bytes() for p in sorted(root.glob('*.part')));assert hashlib.sha256(data).hexdigest()==inputs['source_sha256']
p=root/'g20_real_merge.py'
if p.exists():assert p.read_bytes()==data
else:p.write_bytes(data);p.chmod(0o444)
source=read(phases[5]/'config.json');old=pathlib.Path(source['code_root']);assert old.name=='code-2b293d6e6a0b'
for name,digest in source['code_files_sha256'].items():assert sha(old/name)==digest,name
pub=dict(source_commit=inputs['commit'],source=fp(p),old_frozen_collector_root=str(old),old_code_files_sha256=source['code_files_sha256'],gates=gates,actor_updates=0,probe_updates=0)
manifest=root/'source-publication.json'
if manifest.exists():assert read(manifest)==pub
else:atomic(manifest,pub)
# Import only the new standalone merger; all relative verifier imports resolve
# against the old immutable2b293 runtime and retain its exact accepted hashes.
body="import sys,importlib.util,json; sys.dont_write_bytecode=True; sys.path.insert(0,%r); spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.g20_real_merge',%r); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(json.dumps(m.merge(%r)),flush=True)"%(str(old),str(p),str(b))
b.mkdir(parents=True,exist_ok=True);lock=(b/'operator-launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (b/'bank/complete.json').exists():print(json.dumps(dict(status='EXISTING_REAL_G20_COMPLETE_PRESERVED',complete=fp(b/'bank/complete.json'))))
elif (b/'g20-cpu-launch.json').exists():
 v=read(b/'g20-cpu-launch.json');print(json.dumps(dict(status='EXISTING_G20_CPU_LAUNCH_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'g20-cpu-launch-intent.json').exists(),'Unknown prior CPU launch result; do not repeat'
 intent=dict(status='G20_CPU_LAUNCH_INTENT',publication=fp(manifest),source=fp(p),epoch=time.time());atomic(b/'g20-cpu-launch-intent.json',intent)
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(old),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/'g20-cpu.log').open('ab') as log:
  child=subprocess.Popen(['bash',str(old/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',body],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(intent,status='G20_CPU_STARTED_NOT_PROGRESS_VERIFIED',pid=child.pid,start_ticks=ticks(child.pid));atomic(b/'g20-cpu-launch.json',value);print(json.dumps(value))
'''


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=('status','start'),default='status');args=parser.parse_args()
 source=REPO/'reproduction/operations/qwen7b_warmup/g20_real_merge.py';payload=source.read_bytes();digest=hashlib.sha256(payload).hexdigest()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 assert payload==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(REPO))],cwd=REPO)
 inputs=dict(A=A,target=TARGET,receiver=RECEIVER,code_root=A+'/g20-real-merge-code-'+digest[:12],source_sha256=digest,commit=commit)
 spec=importlib.util.spec_from_file_location('g20_merge_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  own=dict(inputs);own.update(extra or {});code='inputs='+repr(own)+'\n'+COMMON+dict(status=STATUS,chunk=CHUNK,start=START)[action]
  command='python3 -c '+shlex.quote(code);assert len(command.encode())<16000
  r=control.run(command,account='mou176',source_address=address,timeout=55)
  values=[json.loads(s) for s in r.stdout.decode(errors='replace').splitlines() if s.startswith('{')]
  if r.returncode or not values:raise RuntimeError('Unconfirmed G20 action; no automatic retry: '+r.stdout.decode(errors='replace')[-2000:]+r.stderr.decode(errors='replace')[-500:])
  return values[-1]
 state=call('status')
 if args.action=='start':
  if not state['all12ready']:print(json.dumps(dict(status='NOT_READY_ALL12_REAL_SHARDS',state=state)));return
  for i,start in enumerate(range(0,len(payload),4096)):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(payload[start:start+4096]).decode()))
  state=call('start')
 output=CONTROL/'outputs/qwen7b-warmup-20261019'/('g20-real-merge-'+args.action+'.json');output.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,sort_keys=True))
if __name__=='__main__':main()
