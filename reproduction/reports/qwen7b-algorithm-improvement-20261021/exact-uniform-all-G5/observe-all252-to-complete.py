"""Finite read-only60s observer; never submits/restarts/changes a training job."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4]
CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent/'actual-observer'
OUT.mkdir(exist_ok=True)
started=time.monotonic();observations=0
while time.monotonic()-started<3*3600:
    p=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_exact_uniform_g5','--action','status','--control-root',str(CONTROL)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=65)
    if p.returncode:
        value=dict(status='READ_FAILED_NOT_CURRENT_PROGRESS',time=time.time(),returncode=p.returncode,error=p.stderr[-1200:])
    else:
        value=json.loads(p.stdout.splitlines()[-1]);(OUT/'latest-good.json').write_text(json.dumps(value,indent=2)+'\n')
    with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
    observations+=1;progress=value.get('progress.json',{});print(json.dumps(dict(observation=observations,time=value.get('time'),job_id=value.get('job_id'),queue=value.get('squeue'),cursor=progress.get('cursor'),steps=progress.get('steps'),loss=progress.get('loss'),seconds=progress.get('seconds'),next_update=value.get('next-update-replay.json',{}).get('status'),latest_cursor=value.get('latest.json',{}).get('cursor'),failure=value.get('failure.json'),complete=value.get('complete_ref'))),flush=True)
    if value.get('failure.json') or value.get('cpu-prepare-failure.json') or value.get('complete_ref'):break
    if value.get('sacct',{}).get('stdout') and any(x in value['sacct']['stdout'] for x in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):break
    time.sleep(60)
else:print(json.dumps(dict(status='FINITE_OBSERVATION_WINDOW_ENDED_NOT_EXPERIMENT_COMPLETE',observations=observations)),flush=True)
