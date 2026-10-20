"""Finite accepted32 pipeline: rawG10 gate, prefix projection, four GPU shards.

Exclusive owner of these feature submissions. Normal queues are preserved;
unknown/failed starts are reported without duplicate attempts or hot changes.
"""
import json,pathlib,subprocess,sys,time
REPO=pathlib.Path(__file__).resolve().parents[4]
CONTROL=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
OUT=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-batched-feature-pipeline'
OUT.mkdir(parents=True,exist_ok=True);START=time.time()
MODULE='reproduction.operations.qwen7b_warmup.deploy_peer_batched_replay'

def event(value):
    value=dict(value,operator_epoch=time.time())
    with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
    (OUT/'latest.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,sort_keys=True),flush=True)

def call(action,shard=None):
    args=[sys.executable,'-m',MODULE,'--control-root',str(CONTROL),'--action',action,'--batch-size','32']
    if shard is not None:args+=['--shard',str(shard),'--submit-authorized']
    p=subprocess.run(args,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=150)
    values=[json.loads(s) for s in p.stdout.splitlines() if s.startswith('{')]
    if p.returncode or not values:event(dict(status='ACTION_FAILED_NO_BLIND_RETRY',action=action,shard=shard,output=p.stdout[-3000:]));raise SystemExit(2)
    result=values[-1]
    if action!='status':event(dict(status='GUARDED_ACTION_RETURNED',action=action,shard=shard,result=result))
    return result

def observe():
    value=call('status');event(value)
    for role,stage in value['stages'].items():
        if stage.get('failure.json'):event(dict(status='CPU_FAILURE_REQUIRES_FIRST_CAUSE_FIX',role=role,error=stage['failure.json']));raise SystemExit(3)
        if stage.get('launch') and not stage.get('same_process_birth') and not stage.get('complete'):event(dict(status='CPU_EXIT_WITHOUT_ACCEPTANCE',role=role,log=stage.get('log_tail')));raise SystemExit(4)
    for shard,own in value['shards'].items():
        if own.get('failure.json'):event(dict(status='GPU_FEATURE_FAILURE_REQUIRES_FIX',shard=shard,error=own['failure.json']));raise SystemExit(5)
    if time.time()-START>6*3600:event(dict(status='FINITE_WAIT_ENDED_INCOMPLETE'));raise SystemExit(6)
    return value

def wait_until(predicate):
    while True:
        value=observe()
        if predicate(value):return value
        time.sleep(45)

if __name__=='__main__':
    wait_until(lambda value:bool(value['stages']['ingest'].get('complete')))
    call('prepare-start')
    wait_until(lambda value:bool(value['stages']['features'].get('complete')))
    call('finalize')
    for shard in range(4):call('submit',shard)
    progress={str(i):[] for i in range(4)};startup_reported=False
    while True:
        value=observe()
        for shard,own in value['shards'].items():
            count=own.get('progress.json',{}).get('trajectories',0)
            if count and (not progress[shard] or progress[shard][-1]!=count):progress[shard].append(count)
        if not startup_reported and all(len(progress[k])>=2 and value['shards'][k]['complete_chunks'] for k in progress):
            event(dict(status='FOUR_ACTUAL_BATCHED_SHARDS_TWO_ADVANCES_AND_DURABLE_FEATURE_CHUNK_VERIFIED',progress=progress,state=value));startup_reported=True
        if all('complete.json' in own for own in value['shards'].values()):
            event(dict(status='FOUR_ACTUAL_BATCHED_SHARDS_COMPLETE_REQUIRES_INDEPENDENT_BANK_ACCEPTANCE',state=value));break
        time.sleep(45)
