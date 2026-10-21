"""Finite read-only follow to CP1+full replay+five updates and next; no cron."""
import json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
start=time.time();previous=None
while time.time()-start<1800:
 p=subprocess.run(['python3','-m','reproduction.operations.qwen7b_warmup.deploy_direct_count_headlr','--action','status'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if p.returncode:raise RuntimeError('Read-only observer failed; no trainer mutation: '+p.stderr[-1800:])
 state=json.loads(p.stdout.splitlines()[-1]);assert state.get('job_id')=='743809'
 rows=state.get('first_seven_actual_updates',[]);progress=state.get('progress',{})
 compact=dict(epoch=state['epoch'],job_id='743809',squeue=state['squeue']['stdout'].strip(),sacct=state['sacct']['stdout'].strip(),cursor=progress.get('cursor'),
  legacy_Adam=progress.get('steps'),new_head_Adam=progress.get('head_steps'),first_actual_update_count=len(rows),
  first_GPU_replay=state.get('next-update-replay',{}).get('status'),first_CP1_CPU_audit=state.get('first-cp-independent-audit',{}).get('status'),
  latest_cursor=state.get('latest',{}).get('cursor'),failure=state.get('failure'))
 with (HERE/'actual-startup-observations.jsonl').open('a')as f:f.write(json.dumps(compact)+'\n')
 (HERE/'actual-startup-latest-good.json').write_text(json.dumps(state,indent=2)+'\n')
 key=(compact['squeue'],compact['cursor'],compact['first_GPU_replay'],compact['first_CP1_CPU_audit'])
 if key!=previous:print(json.dumps(compact),flush=True);previous=key
 if state.get('failure')or any(x in compact['sacct']for x in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL']):raise RuntimeError('Actual candidate failure preserved: '+json.dumps(compact))
 if len(rows)>=6 and compact['first_GPU_replay']=='PASSED'and compact['first_CP1_CPU_audit']=='JOINT_CP1_INDEPENDENT_FULL_SHA_NINE_COMPONENT_CPU_RELOAD_PASSED':
  first=rows[:6];assert [r['cursor']for r in first]==list(range(1,7))
  assert all(r['steps']==10464+r['cursor']and r['head_steps']==r['cursor']and r['joint_micro_replay_exact']for r in first)
  result=dict(status='MATCHED_HEADLR1E4_ACTUAL_REPLAY_CP1_AND_FIVE_PLUS_NEXT_ACCEPTED',first_six=first,first_replay=state['next-update-replay'],
   independent_CP1=state['first-cp-independent-audit'],submission=state['submission'],last_actual_observation=compact,elapsed_seconds=time.time()-start)
  (HERE/'actual-startup-accepted.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],last_actual_observation=compact)),flush=True);break
 time.sleep(30)
else:print(json.dumps(dict(status='FINITE_OBSERVATION_DEADLINE_WITHOUT_MUTATION',seconds=time.time()-start)),flush=True)
