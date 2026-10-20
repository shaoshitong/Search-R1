"""Finite authorized operator: receiver gate -> CPU acceptance -> four feature jobs.

Small summaries only. No token/password/model/raw outputs, priority edits,
job cancellation, dataset regeneration, or duplicate submission retries.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
MODULE='reproduction.operations.qwen7b_warmup.deploy_peer_newseed_replay'
ROOT=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-feature-pipeline'
ROOT.mkdir(exist_ok=True,parents=True)
START=time.time()


def event(value):
    value=dict(value,operator_epoch=time.time())
    with (ROOT/'observations.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
    (ROOT/'latest.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,sort_keys=True),flush=True)


def call(action,shard=None):
    args=[sys.executable,'-m',MODULE,'--control-root',str(CONTROL),'--action',action]
    if shard is not None:args+=['--shard',str(shard),'--submit-authorized']
    result=subprocess.run(args,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=150)
    values=[json.loads(s) for s in result.stdout.splitlines() if s.startswith('{')]
    if result.returncode or not values:
        event(dict(status='OPERATOR_ACTION_FAILED_NO_AUTOMATIC_DUPLICATE_RETRY',action=action,shard=shard,output=result.stdout[-4000:]))
        raise SystemExit(1)
    value=values[-1]
    if action!='status':event(dict(status='GUARDED_ACTION_RETURNED',action=action,shard=shard,result=value))
    return value


def observe():
    state=call('status');event(state)
    for role,stage in state['stages'].items():
        if stage.get('failure.json'):
            event(dict(status='ACTUAL_CPU_STAGE_FAILED_NEEDS_FIRST_CAUSE_FIX',role=role,failure=stage['failure.json']));raise SystemExit(2)
        if stage.get('launch') and not stage.get('same_process_birth') and not stage.get('complete'):
            event(dict(status='CPU_EXIT_WITHOUT_ACCEPTANCE',role=role,log_tail=stage.get('log_tail')));raise SystemExit(3)
    return state


def wait_until(predicate,maximum_hours=4):
    while True:
        value=observe()
        if predicate(value):return value
        if time.time()-START>maximum_hours*3600:
            event(dict(status='FINITE_OPERATOR_WAIT_ENDED_TASK_INCOMPLETE',elapsed=time.time()-START));raise SystemExit(4)
        time.sleep(50)


def main():
    state=wait_until(lambda v:v['receiver_complete'])
    call('ingest-prepare');call('ingest-start')
    wait_until(lambda v:bool(v['stages']['ingest'].get('complete')))
    call('features-start')
    wait_until(lambda v:bool(v['stages']['features'].get('complete')))
    call('finalize')
    for shard in range(4):call('submit',shard)
    progress={str(i):[] for i in range(4)}
    while True:
        state=observe()
        for key,own in state['shards'].items():
            data=own.get('progress.json',{});count=data.get('trajectories',0)
            if count and (not progress[key] or count!=progress[key][-1]):progress[key].append(count)
            if own.get('failure.json'):
                event(dict(status='ACTUAL_GPU_FEATURE_FAILURE_NEEDS_FIX',shard=key,failure=own['failure.json']));raise SystemExit(5)
        if all(len(progress[k])>=2 and state['shards'][k]['complete_chunks']>=1 for k in progress):
            event(dict(status='FOUR_REAL_GPU_FEATURE_SHARDS_TWO_INCREMENTS_AND_COMPLETE_CHUNK_VERIFIED',progress=progress,state=state))
            # Continue through completion; real output acceptance is stronger
            # than startup/progress and may arrive during this operator turn.
        if all('complete.json' in v for v in state['shards'].values()):
            event(dict(status='FOUR_ACTUAL_PEER_FEATURE_SHARDS_COMPLETE_REQUIRES_PARENT_BANK_MERGE',state=state));return
        if time.time()-START>4*3600:
            event(dict(status='FINITE_OPERATOR_WAIT_ENDED_TASK_INCOMPLETE',state=state));raise SystemExit(4)
        time.sleep(50)

if __name__=='__main__':main()
