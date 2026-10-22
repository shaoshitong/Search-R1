"""Bounded small local follower of one durable remote metadata observer."""
import json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];C=R.parents[1]/'outputs/qwen7b-warmup-20261019/metadata-control.py';last=None
for _ in range(4320):
 r=subprocess.run(['python3',str(C),str(H/'read-live.py'),'--read-only','--timeout','25'],capture_output=True,text=True,cwd=R)
 if r.returncode:(H/'read-error-latest.json').write_text(json.dumps(dict(time=time.time(),error=r.stderr[-1400:],state_unknown=True))+'\n');time.sleep(45);continue
 x=json.loads(r.stdout);tmp=H/'actual-latest.tmp';tmp.write_text(json.dumps(x,sort_keys=True)+'\n');tmp.replace(H/'actual-latest.json');f=x['files'];latest=f.get('latest.json',{});summary=dict(time=x['time'],observer_live=x['actual_observer_identity_live'],replicas={n:dict(job_id=c['job_id'],state=c['scheduler']['state'],ranks={rank:r['actual_Adam_updates']for rank,r in c['progress_by_rank'].items()},startup_proven=c['startup_proven'],error=c.get('read_or_binding_error')or c.get('observer_error'))for n,c in latest.get('replicas',{}).items()},proofs={k:v for k,v in f.items()if k.startswith('startup-proof')},failures={k:v for k,v in f.items()if k.startswith('failure-')or k=='observer-failure.json'})
 (H/'actual-compact-latest.json').write_text(json.dumps(summary,sort_keys=True)+'\n');state=json.dumps({n:(c['state'],c['startup_proven'],bool(c['error']))for n,c in summary['replicas'].items()},sort_keys=True)
 if state!=last:print(json.dumps(summary),flush=True);last=state
 if summary['failures']or len(summary['proofs'])==2 or'complete.json'in f or'deadline.json'in f:print(json.dumps(dict(stop_reason='FIRST_ACTUAL_MILESTONE_OR_FAILURE_ROOT_HANDOFF')),flush=True);break
 time.sleep(30)
