"""Finite read-only observer for the actual separately recorded continuation."""
import argparse,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent/'actual-resume-observer'
p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--operator-commit',required=True);args=p.parse_args()
if not args.job_id.isdigit():p.error('Actual recorded job ID required')
OUT.mkdir(exist_ok=True)
(OUT/'observer-controller.json').write_text(json.dumps(dict(pid=os.getpid(),script=str(pathlib.Path(__file__).resolve()),job_id=args.job_id,operator_commit=args.operator_commit,time=time.time()))+'\n')
started=time.monotonic()
while time.monotonic()-started<5*3600:
    command=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_direct_hidden_resume','--control-root',str(CONTROL),'--commit',args.operator_commit,'--action','status']
    run=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=65)
    if run.returncode:state=dict(status='READ_FAILED_NOT_PROGRESS',time=time.time(),read_error=run.stderr[-1400:])
    else:
        state=json.loads(run.stdout.splitlines()[-1])
        if state['job_id']!=args.job_id:raise ValueError('Continuation observer cannot relabel the original job as the new allocation')
        (OUT/'latest-good.json').write_text(json.dumps(state,indent=2)+'\n')
    progress=state.get('progress',{});failure=state.get('failure') or state.get('continuation_failure')
    row=dict(time=state['time'],job_id=state.get('job_id'),original_job_id=state.get('original_job_id'),
        cursor=progress.get('cursor'),steps=progress.get('steps'),head_steps=progress.get('head_steps'),
        source_cursor=state.get('continuation_preparation',{}).get('source_cursor'),
        independent_first_resumed_CP=state.get('continuation_first-cp-independent-audit',{}).get('cursor'),
        replay=state.get('continuation_next-update-replay',{}).get('status'),
        handoff=state.get('continuation_handoff-complete',{}).get('status'),latest_cursor=state.get('latest',{}).get('cursor'),
        failure=failure,evaluations=[i for i in (256,512,768,938) if 'evaluation_'+str(i) in state],
        complete=bool(state.get('complete')),squeue=state.get('squeue'),sacct=state.get('sacct'),read_error=state.get('read_error'))
    with (OUT/'observations.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)
    if failure or any(s in state.get('sacct','') for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):break
    if state.get('complete') and '|COMPLETED|0:0|' in state.get('sacct',''):break
    time.sleep(60)
print(json.dumps(dict(status='FINITE_READ_ONLY_RESUME_OBSERVER_EXITED',independent_final_acceptance_still_required=True)),flush=True)
