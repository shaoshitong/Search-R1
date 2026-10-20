"""Bounded read-only watch of one existing job; never submits or restarts it."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--until-epoch',type=int,required=True);p.add_argument('--seconds',type=int,default=1800);a=p.parse_args()
    d=a.control_root/'outputs/qwen7b-warmup-20261019';lock=(d/'balanced-observer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (d/'balanced-observer-controller.json').write_text(json.dumps(dict(pid=os.getpid(),until_epoch=a.until_epoch,started=time.time(),job_id='742276',read_only=True)))
    end=time.monotonic()+a.seconds;failures=0
    while time.monotonic()<end:
        r=subprocess.run(['python3',str(d/'mixed-ft-status.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=55)
        if r.returncode:
            failures+=1;print(json.dumps(dict(status='OBSERVATION_FAILED_NO_RESTART',time=time.time(),failures=failures,details=r.stdout[-1000:]+r.stderr[-1000:])),flush=True)
            if failures>=3:return
        else:
            failures=0;v=json.loads((d/'mixed-ft-latest-status.json').read_text());assert v['fresh_control_snapshot'] and v['job_id']=='742276'
            progress=v.get('progress.json') or {};latest=v.get('latest.json') or {};metric=v.get('metrics-latest.json') or {};evaluations=v.get('evaluation_progress') or {}
            compact=dict(time=v['time'],job_id=v['job_id'],queue=v['queue'].strip(),epoch=progress.get('epoch'),batch=progress.get('batch'),batches=progress.get('batches'),steps=progress.get('steps'),checkpoint_epoch=latest.get('epoch'),checkpoint_cursor=latest.get('cursor'),checkpoint_step=latest.get('steps'),sealed_metric_epoch=metric.get('epoch'),evaluation={k:dict(epoch=x.get('epoch'),completed=x.get('completed_local_groups'),expected=x.get('expected_local_groups')) for k,x in evaluations.items()},complete_status=(v.get('complete.json') or {}).get('status'))
            with (d/'balanced-observer-history.jsonl').open('a') as f:f.write(json.dumps(compact)+'\n')
            print(json.dumps(compact),flush=True)
            if metric.get('epoch',0)>=a.until_epoch:
                print(json.dumps(dict(status='SEALED_EPOCH_METRIC_AVAILABLE',epoch=metric['epoch'],training_goal_completion_claimed=False)),flush=True);return
            if not v['queue'].strip():
                own=[line.split('|') for line in v.get('accounting','').splitlines() if line.split('|')[0]=='742276']
                if own and own[0][1].split()[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'):
                    print(json.dumps(dict(status='AUTHORITATIVE_JOB_TERMINAL_INSPECT_REQUIRED',accounting=own,goal_completion_claimed=False)),flush=True);return
        time.sleep(min(45,max(0,end-time.monotonic())))
    print(json.dumps(dict(status='BOUNDED_WATCH_FINISHED_NO_RESTART',time=time.time(),goal_completion_claimed=False)),flush=True)


if __name__=='__main__':main()
