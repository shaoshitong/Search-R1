"""Bounded foreground observation only; never submit, signal or recover a job."""
import json,subprocess,sys,time
from pathlib import Path
from reproduction.hpc3_prefix5.control import OUTPUT
observer=Path(__file__).with_name('hpc3-two-stage-cp61-observe-20261018.py')
for sample in range(160):
    result=subprocess.run([sys.executable,str(observer)],text=True,capture_output=True,timeout=40)
    if result.returncode:
        print(json.dumps(dict(sample=sample,status='READ_FAILED_NOT_TRAINING_TERMINAL',detail=result.stderr[-1500:])),flush=True)
    else:
        v=json.loads(result.stdout)
        def counts(key):return [(r or {}).get('completed_trajectories') for r in (v.get(key) or [])]
        row=dict(sample=sample,epoch=v['epoch'],job=v['job'],queue=v['queue'],stage=v.get('status'),
                 complete_outer=v['complete_outer'],actor=v['actor'],probe=v['probe'],prefix=counts('prefix'),
                 full_pool=counts('full_pool'),selected=v.get('selected'),updates=v.get('inflight_rank_updates'),
                 failures=v.get('failed_jobs'),errors=v.get('error_lines'))
        with (OUTPUT/'two-stage-cp61-follow-five.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if not v['queue'] or v.get('failed_jobs') or v.get('error_lines'):
            print(json.dumps(dict(status='ACTUAL_TERMINAL_OR_PHASE_FAILURE_NEEDS_ATTENTION',details=v)),flush=True);raise SystemExit(1)
        if v['complete_outer']>=66 and (v.get('status') or {}).get('outer',0)>=67 and any((x or 0)>0 for x in counts('prefix')):
            print(json.dumps(dict(status='COMPLETE62_TO66_AND_POSITIVE67_OBSERVED_NOT_FINAL200_ACCEPTANCE')),flush=True);raise SystemExit(0)
    time.sleep(30)
print(json.dumps(dict(status='BOUNDED_READ_ONLY_WAIT_EXPIRED_NOT_TRAINING_TERMINAL')),flush=True)
