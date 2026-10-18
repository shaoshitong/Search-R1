"""Latest human stop scope: seal probe epoch20, then signal only its owner."""
import hashlib
import json
import subprocess
import time

from .control import ROOT, route


REMOTE=r'''
import hashlib,json,os,pathlib,signal,sys,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base'])
assert h['run_id']=='81df904ad9584e29a35d703570b2bcef' and h['execution_id']=='a2b04d10010e486b991abcf8fb75df83'
def read(path):return json.loads(pathlib.Path(path).read_text())
assert read(b/'preparation/deployment.json')==h
assert p['intent']['scope']['stop_current_probe5_after_complete_epoch20']
submit=read(b/'preparation/submit-receipt.json');identity=submit['leader']
def same():
 try:
  path=pathlib.Path('/proc')/str(identity['pid']);raw=(path/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  now=dict(pid=identity['pid'],sid=int(v[3]),uid=path.stat().st_uid,start_ticks=int(v[19]))
  return all(now[k]==identity[k] for k in now)
 except (FileNotFoundError,ProcessLookupError):return False
deadline=time.monotonic()+48
target=b/'training/cycle-0010/complete.json'
while not target.exists():
 if not same():raise RuntimeError('Owner exited before requested complete epoch20')
 if time.monotonic()>deadline:raise TimeoutError('Epoch20 not complete yet; no signal sent')
 time.sleep(.25)
closed=read(target);assert closed['cycle']==10 and closed['completed_epochs']==20
assert closed['ema_updates']==closed['actual_adam_steps_per_rank'][0] and len(set(closed['actual_adam_steps_per_rank']))==1
curve=[json.loads(x) for x in (b/'metrics.jsonl').read_text().splitlines() if x]
assert len(curve)>=10 and curve[9]==closed['metrics']
latest=read(b/'probe/latest.json');assert latest['completed_epochs']>=20
status=read(b/'logs/status.json');progress=[]
for rank in range(4):
 f=b/'training/cycle-0011'/('rank-%d-progress.json'%rank)
 progress.append(read(f) if f.exists() else None)
value=dict(role='explicit_human_stop_at_complete_probe20_before_prefix20_reuse',checked_unix=time.time(),
 human_intent_sha256=p['intent_sha256'],run_id=h['run_id'],execution_id=h['execution_id'],
 requested_epoch=20,complete20_checkpoint=closed,latest_actual_epochs=latest['completed_epochs'],
 native_status_before=status,any_epoch21_real_updates=any(x is not None for x in progress),
 observed_cycle11_rank_progress=progress,owned_leader=identity,automatic_restart_authorized=False,
 preserve_all_source50_and_current_outputs=True,only_owned_leader_SIGTERM=True)
path=b/'preparation/user-stop-at20-before-prefix20.json'
if not path.exists():
 with path.open('x') as f:json.dump(value,f,sort_keys=True)
if same():
 fd=os.pidfd_open(identity['pid']);assert same()
 signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
 value['signal_sent']=True
else:value['signal_sent']=False
print(json.dumps(value))
'''


def main():
    handle=json.loads((ROOT/'preparation/actor50-probe100-resume-deployment-20261018.json').read_text())
    raw=(ROOT/'preparation/actor50-prefix20-human-intent-20261018.json').read_bytes()
    p=dict(handle=handle,intent=json.loads(raw),intent_sha256=hashlib.sha256(raw).hexdigest())
    result=subprocess.run(route(REMOTE),input=json.dumps(p).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-probe5-stop-at20-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n')
    if result.returncode:raise RuntimeError('Inspect stop receipt; do not assume stop: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(result.stdout))))


if __name__=='__main__':main()
