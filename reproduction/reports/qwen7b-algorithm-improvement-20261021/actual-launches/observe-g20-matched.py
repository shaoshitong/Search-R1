"""Bounded read-only observer: never starts, edits, cancels, or resumes jobs."""
import json, subprocess, sys, time
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO=CONTROL/'work/qwen7b-probe-warmup-20261019'
HERE=Path(__file__).resolve().parent/'g20-matched-observer'
HERE.mkdir(exist_ok=True)
EXPECTED={'random':'743370','global-weighted-balanced':'743371'}
def atomic(path,value):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(path)
def main():
 previous=None;start=time.time()
 for ordinal in range(161):
  try:
   r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_g20_matched_suite','--control-root',str(CONTROL),'--action','status'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=80)
   if r.returncode:raise RuntimeError(r.stdout[-1200:]+r.stderr[-1200:])
   v=json.loads(r.stdout.splitlines()[-1]);parts=v['variants']
   assert {n:p.get('job_id') for n,p in parts.items()}==EXPECTED
   summary={'epoch':v['epoch'],'squeue':v.get('squeue'),'sacct':v.get('sacct'),'variants':{n:{k:p[k] for k in ('job_id','progress.json','next-update-replay.json','failure.json','interrupted.json','gpu-memory.json','complete.json') if k in p} for n,p in parts.items()}}
   atomic(HERE/'latest-good.json',summary)
   with (HERE/'observations.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
   cursors={n:p.get('progress.json',{}).get('cursor') for n,p in parts.items()}
   if cursors!=previous or ordinal%5==0:
    print(json.dumps(summary),flush=True);previous=cursors
   failures={n:{k:p[k] for k in ('failure.json','interrupted.json') if k in p} for n,p in parts.items() if 'failure.json' in p or 'interrupted.json' in p}
   if failures:atomic(HERE/'attention.json',dict(status='ACTUAL_G20_FAILURE_NO_AUTOMATIC_SUBMIT_OR_ROLLBACK',failures=failures));return 2
   if all('complete.json' in p for p in parts.values()):atomic(HERE/'complete-observation.json',dict(status='BOTH_G20_BUDGETS_AND_VALIDATIONS_HAVE_ACTUAL_COMPLETE_REQUIRE_INDEPENDENT_METRIC_AUDIT',actual=summary));return 0
   if not v.get('squeue') and v.get('sacct'):
    atomic(HERE/'attention.json',dict(status='JOBS_LEFT_QUEUE_WITHOUT_BOTH_COMPLETE_NO_BLIND_RETRY',actual=summary));return 3
  except Exception as e:
   failure=dict(epoch=time.time(),status='READ_ONLY_OBSERVATION_ERROR_PREVIOUS_GOOD_PRESERVED',type=type(e).__name__,message=str(e)[-2500:]);atomic(HERE/'last-observation-error.json',failure);print(json.dumps(failure),flush=True)
  if ordinal<160:time.sleep(45)
 atomic(HERE/'finite-window-ended.json',dict(status='READ_ONLY_WINDOW_ENDED_NO_JOB_MUTATION',elapsed_seconds=time.time()-start));return 4
if __name__=='__main__':raise SystemExit(main())
