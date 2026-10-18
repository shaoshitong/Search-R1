import json,time,shlex,subprocess
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import OUTPUT,PYTHON
intent=dict(observed_unix=time.time(),human_original_timestamp='not available in message metadata; observed 2026-10-18',instruction='最新的 iteration 这个结果能不能就是停下来…固定5120问题，先选1024问题全部5轨迹rollout，再选512 Mixed最高的问题；比如从60iter或者61iter resume',scope='Stop only current R3 job733342, retain newest complete CP61 if present, implement isolated two-stage successor; full200 goal stays active',later_instruction='latest follow-up asks for CP50/60 probe curves and does not revoke stop/modify authorization',job='733342',source_base='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
p=Path(OUTPUT)/'human-two-stage-stop-cp61-intent-20261018.json';p.write_text(json.dumps(intent,ensure_ascii=False,indent=2)+'\n')
code=r'''
import json,os,time,subprocess,hashlib
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
def rd(p):return json.loads(p.read_text())
c=rd(b/'preparation/launch-config.json');j=rd(b/'latest-joint.json');stage=rd(b/'logs/pipeline-status.json')
assert c['run_id']=='cc0de51f2d68495c9664aa888074839c' and c['execution_id']=='d914557a20dc4b1fa11dc4ad3df2cd0d'
assert j['outer']==61 and j['actor']['actual_adam_steps_per_rank']==[1220]*4 and j['actor']['scheduler_completed_outer']==61
assert j['probe']['actual_adam_steps_per_rank']==[550]*4
q=subprocess.run(['squeue','-h','-j','733342','-o','%i|%u|%T|%N'],text=True,capture_output=True)
assert q.returncode==0 and '733342|mou176|RUNNING|ACD1-8' in q.stdout
assert stage['outer']==62 and 'actor' not in stage['stage'], stage
intent=INTENT
intent.update(actual_latest_complete=61,actor_adam=[1220]*4,probe_adam=[550]*4,stage_before_stop=stage,queue_before_stop=q.stdout.strip(),stop_reason='NEW_HUMAN_TWO_STAGE_SELECTION_REQUEST_NOT_TRAINING_FAILURE',checkpoint_preserved=True)
p=b/'preparation/human-two-stage-stop-cp61-intent-20261018.json'
with p.open('x') as f:json.dump(intent,f,ensure_ascii=False,sort_keys=True)
r=subprocess.run(['scancel','733342'],text=True,capture_output=True)
out=dict(status='STOP_REQUEST_SENT' if r.returncode==0 else 'STOP_REQUEST_ERROR',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,latest_complete=61,actor_adam_per_rank=[1220]*4,probe_adam_per_rank=[550]*4,epoch=time.time(),intent_path=str(p))
with (b/'preparation/human-two-stage-stop-cp61-result-20261018.json').open('x') as f:json.dump(out,f,sort_keys=True)
print(json.dumps(out))
'''.replace('INTENT',repr(intent))
addr=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=addr,timeout=25)
(Path(OUTPUT)/'human-two-stage-stop-cp61-control-result.txt').write_bytes(r.stdout+r.stderr)
print(r.stdout.decode());print(r.stderr.decode());raise SystemExit(r.returncode)
