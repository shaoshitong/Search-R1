"""Finite60s read-only observer of the single accepted append-only job."""
import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1]
OUT=pathlib.Path(__file__).parent/'actual-observer';OUT.mkdir(exist_ok=True)
COMMIT='b8c3f8a76fa195efc7c2e95cfa1a7d36669576ec'
started=time.monotonic();count=0
while time.monotonic()-started<5*3600:
 args=[sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_append_train','--control-root',str(CONTROL),'--commit',COMMIT,'--action','status']
 p=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=65)
 if p.returncode:x=dict(status='READ_FAILED_NOT_PROGRESS',returncode=p.returncode,error=p.stderr[-1400:],observed_epoch=time.time())
 else:
  x=json.loads(p.stdout.splitlines()[-1]);(OUT/'latest-good.json').write_text(json.dumps(x,indent=2)+'\n')
 progress=x.get('progress',{});row=dict(time=x.get('observed_epoch'),job_id=x.get('job_id'),status=x.get('status'),squeue=x.get('squeue'),sacct=x.get('sacct'),cursor=progress.get('cursor'),additional_completed=progress.get('additional_completed'),steps=progress.get('steps'),loss=progress.get('loss'),seconds=progress.get('seconds'),replay=x.get('next-update-replay',{}).get('status'),independent_CP257=x.get('first-cp-independent-audit',{}).get('status'),latest_cursor=x.get('latest',{}).get('cursor'),failure=x.get('failure'),CPU_audit_failure=x.get('first-cp-audit-failure'),evaluations={str(i):x['evaluation_'+str(i)]['ref'] for i in (512,768,938) if 'evaluation_'+str(i) in x},complete_ref=x.get('complete_ref'),read_error=x.get('error'))
 with (OUT/'observations.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
 print(json.dumps(row,sort_keys=True),flush=True);count+=1
 terminal=x.get('sacct','')
 if x.get('failure') or x.get('first-cp-audit-failure') or any(s in terminal for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):break
 if x.get('complete_ref') and '|COMPLETED|0:0|' in terminal:break
 time.sleep(60)
print(json.dumps(dict(status='FINITE_READ_ONLY_APPEND_OBSERVER_EXITED',observations=count,independent_final_acceptance_still_required=True)),flush=True)
