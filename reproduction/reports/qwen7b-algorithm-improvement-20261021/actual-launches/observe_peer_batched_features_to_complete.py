"""Read-only continuation after reconciled jobs743345..348; no launch actions.

Observation failures have their own file and never replace latest-good state.
The previous submission operator is verified exited; existing writers stay live.
"""
import json,pathlib,subprocess,sys,time
REPO=pathlib.Path(__file__).resolve().parents[4]
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
OUT=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-batched-feature-pipeline'
EXPECTED={str(i):str(743345+i) for i in range(4)}
START=time.time()

def record(value,good=False):
    value=dict(value,observer_epoch=time.time())
    with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
    if good:
        for name in ('latest-good.json','latest.json'):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    else:(OUT/'latest-observation-error.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,sort_keys=True),flush=True)

if __name__=='__main__':
    while time.time()-START<4*3600:
        cmd=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_batched_replay','--control-root',str(CONTROL),'--action','status']
        p=subprocess.run(cmd,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=150)
        values=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
        if p.returncode or not values:
            record(dict(status='READ_ONLY_OBSERVATION_UNAVAILABLE_WRITERS_NOT_CHANGED',output=p.stdout[-2000:]));time.sleep(45);continue
        value=values[-1]
        if {k:v.get('job_id') for k,v in value['shards'].items()}!=EXPECTED:
            record(dict(status='OWNED_JOB_IDENTITY_MISMATCH_NO_MUTATIONS',actual=value));raise SystemExit(2)
        record(value,good=True)
        if any('failure.json' in own for own in value['shards'].values()):
            record(dict(status='ACTUAL_GPU_FAILURE_REQUIRES_ROOT_FIRST_CAUSE_FIX',actual=value));raise SystemExit(3)
        if all('complete.json' in own for own in value['shards'].values()):
            record(dict(status='FOUR_ACTUAL_BATCHED_PEER_SHARDS_COMPLETE_NEEDS_BANK_ACCEPTANCE',actual=value),good=True);break
        time.sleep(45)
