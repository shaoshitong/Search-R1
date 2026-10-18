"""Preserve the latest sealed warmup, stopping only at its next prefix boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from reproduction.actor50_probe100.control import route,ROOT

REMOTE=r'''
import hashlib,json,os,pathlib,signal,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base'])
def read(f):
 f=pathlib.Path(f);assert f.resolve()==f and not f.is_symlink() and f.stat().st_size<8*1024**2
 return json.loads(f.read_text())
assert read(b/'preparation/deployment.json')==h
assert p['intent']['scope']['handoff_warmup_execution']==h['execution_id']
assert p['intent']['scope']['new_actor_start_outer']==50
identity=read(b/'preparation/submit-receipt.json')['leader']
def actual():
 try:
  f=pathlib.Path('/proc')/str(identity['pid']);v=(f/'stat').read_text();v=v[v.rfind(')')+2:].split()
  now=dict(pid=identity['pid'],sid=int(v[3]),uid=f.stat().st_uid,start_ticks=int(v[19]),state=v[0])
  return now if all(now[k]==identity[k] for k in ('pid','sid','uid','start_ticks')) else None
 except (FileNotFoundError,ProcessLookupError):return None
deadline=time.monotonic()+48
while True:
 latest=read(b/'probe/latest.json');status=read(b/'logs/status.json');who=actual()
 safe=status['stage'].endswith('-prefix20') and status['cycle']==latest['cycle']+1
 progress=[b/'training'/('cycle-%04d'%status['cycle'])/('rank-%d-progress.json'%r) for r in range(4)]
 safe=safe and not any(f.exists() for f in progress)
 if p['action']=='observe' or not who or safe or time.monotonic()>=deadline:break
 time.sleep(.25)
signal_sent=False
if p['action']=='stop' and who and safe:
 cp=pathlib.Path(latest['checkpoint']);assert cp.is_file() and cp.stat().st_size==latest['bytes']
 value=dict(role='HUMAN_NEW_GRPO_GOAL_HANDOFF_LATEST_SEALED_WARMUP',checked_unix=time.time(),
  human_goal_created_unix=p['intent']['goal_created_unix'],intent_sha256=p['intent_sha256'],
  operator_sha256=p['operator_sha256'],run_id=h['run_id'],execution_id=h['execution_id'],
  latest=latest,status=status,no_unsealed_optimizer_writes=True,pipeline_identity=identity,
  scope='stop this independent warmup for actor50 GRPO continuation; preserve all outputs; no goal pause')
 target=b/'preparation/human-handoff-to-grpo50.json'
 if not target.exists():
  with target.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
 fd=os.pidfd_open(identity['pid'])
 try:
  assert actual() and actual()['uid']==os.getuid();signal.pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
 signal_sent=True
print(json.dumps(dict(checked_unix=time.time(),leader=actual(),safe_boundary=safe,
 signal_sent=signal_sent,latest=latest,status=status,
 cleanup=read(b/'logs/cleanup-status.json') if (b/'logs/cleanup-status.json').exists() else None,
 exit=read(b/'logs/exit.json') if (b/'logs/exit.json').exists() else None),allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stop','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);p.add_argument('--intent',type=Path,required=True);a=p.parse_args()
    raw=a.intent.read_bytes();payload=dict(action=a.action,handle=json.loads(a.handle.read_text()),
        intent=json.loads(raw),intent_sha256=hashlib.sha256(raw).hexdigest(),
        operator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-grpo-warmup-'+a.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect exact handoff receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
