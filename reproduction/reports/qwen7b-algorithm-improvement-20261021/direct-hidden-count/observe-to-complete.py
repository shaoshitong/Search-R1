"""Finite60s read-only joint observer; no submit/restart/priority actions."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent/'actual-observer';OUT.mkdir(exist_ok=True)
COMMIT='501cf80478b77f68c3e13a96d4659e3da0971b2e'
started=time.monotonic();count=0
while time.monotonic()-started<5*3600:
 args=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_direct_hidden_count_train','--control-root',str(CONTROL),'--commit',COMMIT,'--action','status']
 p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=65)
 if p.returncode:x=dict(status='READ_FAILED_NOT_PROGRESS',error=p.stderr[-1400:],time=time.time())
 else:
  x=json.loads(p.stdout.splitlines()[-1]);(OUT/'latest-good.json').write_text(json.dumps(x,indent=2)+'\n')
 progress=x.get('progress',{});row=dict(time=x.get('time'),job_id=x.get('job_id'),status=x.get('status'),squeue=x.get('squeue'),sacct=x.get('sacct'),cursor=progress.get('cursor'),steps=progress.get('steps'),head_steps=progress.get('head_steps'),loss=progress.get('loss'),seconds=progress.get('seconds'),replay=x.get('next-update-replay',{}).get('status'),independent_CP1=x.get('first-cp-independent-audit',{}).get('status'),latest_cursor=x.get('latest',{}).get('cursor'),failure=x.get('failure'),CPU_audit_failure=x.get('first-cp-audit-failure'),evaluations=[i for i in (256,512,768,938) if 'evaluation_'+str(i) in x],complete=bool(x.get('complete')),read_error=x.get('error'))
 with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
 print(json.dumps(row,sort_keys=True),flush=True);count+=1
 terminal=x.get('sacct','')
 if x.get('failure') or x.get('first-cp-audit-failure') or any(s in terminal for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):break
 if x.get('complete') and '|COMPLETED|0:0|' in terminal:break
 time.sleep(60)
print(json.dumps(dict(status='FINITE_READ_ONLY_JOINT_OBSERVER_EXITED',observations=count,independent_final_acceptance_still_required=True)),flush=True)
