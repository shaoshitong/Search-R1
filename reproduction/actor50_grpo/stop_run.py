"""Stop exactly the superseded execution on a fresh, explicit human intent."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from reproduction.actor50_probe100.control import route, ROOT

REMOTE = r'''
import json,os,pathlib,signal,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base'])
def read(f):
 f=pathlib.Path(f);assert f.resolve()==f and not f.is_symlink() and f.stat().st_size<8*1024**2
 return json.loads(f.read_text())
assert read(b/'preparation/deployment.json')==h
assert p['intent']['stop_execution_id']==h['execution_id']
identity=read(b/'preparation/submit-receipt.json')['leader']
def actual():
 try:
  f=pathlib.Path('/proc')/str(identity['pid']);s=(f/'stat').read_text();v=s[s.rfind(')')+2:].split()
  now=dict(pid=identity['pid'],start_ticks=int(v[19]),sid=int(v[3]),uid=f.stat().st_uid,state=v[0])
  return now if all(now[k]==identity[k] for k in ('pid','start_ticks','sid','uid')) else None
 except (FileNotFoundError,ProcessLookupError):return None
def small(f):return read(f) if f.is_file() else None
sent=False
if p['action']=='stop':
 target=b/'preparation/human-restart-mixed30.json'
 if not target.exists():
  with target.open('x') as f:json.dump(dict(intent=p['intent'],intent_sha256=p['intent_sha256'],
   operator_sha256=p['operator_sha256'],checked_unix=time.time(),pipeline_identity=identity,
   latest_joint=small(b/'latest-joint.json'),status=small(b/'logs/pipeline-status.json'),
   role='USER_STOP_SUPERSEDED_EXECUTION_RESTART_FROM50_WITH_MIXED30_FLOOR'),f,sort_keys=True)
 if actual() and not small(b/'logs/cleanup-status.json'):
  fd=os.pidfd_open(identity['pid'])
  try:
   assert actual() and actual()['uid']==os.getuid()
   signal.pidfd_send_signal(fd,signal.SIGTERM);sent=True
  finally:os.close(fd)
deadline=time.monotonic()+12
while actual() and time.monotonic()<deadline:time.sleep(.25)
print(json.dumps(dict(checked_unix=time.time(),identity=identity,leader=actual(),signal_sent=sent,
 latest_joint_outer=(small(b/'latest-joint.json') or {}).get('outer'),
 stage=small(b/'logs/pipeline-status.json'),cleanup=small(b/'logs/cleanup-status.json'),
 exit=small(b/'logs/exit.json'),failure=small(b/'logs/failure.json'))))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--handle',type=Path,required=True)
    p.add_argument('--intent',type=Path,required=True);p.add_argument('--action',choices=('stop','observe'),required=True)
    a=p.parse_args();raw=a.intent.read_bytes()
    payload=dict(action=a.action,handle=json.loads(a.handle.read_text()),intent=json.loads(raw),
        intent_sha256=hashlib.sha256(raw).hexdigest(),operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-grpo-mixed30-stop-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect stop receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
