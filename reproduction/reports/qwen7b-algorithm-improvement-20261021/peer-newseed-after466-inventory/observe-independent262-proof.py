"""One finite read-only actual proof observation; never restarts the worker."""
import importlib.util,json,pathlib,shlex,subprocess
HERE=pathlib.Path(__file__).resolve().parent;x=json.loads((HERE/'actual-independent262-launch.json').read_bytes())
worker='''import json,sys,pathlib,time,hashlib
x=json.loads(sys.stdin.buffer.read());r=pathlib.Path(x['root']);p=pathlib.Path('/proc/%d/stat'%x['pid']);out=dict(epoch=time.time(),root=str(r),pid=x['pid'],alive=p.exists())
if p.exists():
 s=p.read_text().split(') ')[1].split();out['owner_ticks_match']=int(s[19])==x['start_ticks'];out['process_state']=s[0]
for name in('controller.json','progress.json','complete.json','failure.json'):
 p=r/name
 if p.exists():
  b=p.read_bytes();v=json.loads(b);out[name]=dict(ref=dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),summary={k:z for k,z in v.items()if k not in('proofs','source_reference','code')})
log=r/'controller.log'
if log.exists()and log.stat().st_size:out['log_tail']=log.read_text(errors='replace')[-1400:]
print(json.dumps(out))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(x).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode()[-1000:])
v=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-independent262-latest.json').write_text(json.dumps(v,indent=2)+'\n')
with(HERE/'actual-independent262-observations.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v))
