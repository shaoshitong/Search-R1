"""Retire only this task's waiting CPU coordinator, after both accepted CPs exist."""
from pathlib import Path
import json,time,os,signal,subprocess,hashlib
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');C=A/'selected-seed2-checkpoint-CPU-coordinator-20261023-v1';O=A/'seed2-matched-window-terminal-CPU-acceptance-20261023-v1'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def live(c):
 p=Path('/proc/%d/stat'%c['pid']);return p.exists()and p.read_text().rsplit(')',1)[1].split()[19]==c['start_ticks']
progress=read(C/'progress.json');ctrl=read(C/'controller.json')
assert progress['cell']=='window2018'and progress['stage']=='WAITING_ACTUAL_TERMINAL_NO_CP_OR_COMMON_LOCK_HELD','MSE has not finished; leave coordinator/CPU child untouched'
accepted=[]
for suite,cell,e in [('online-packed-current64-seed19137-plan2026102302-L15P80-8192-20261023-v1','current64',5),('online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1','mse-M5',7)]:
 p=O/suite/cell/('epoch-%02d'%e)/'accepted.json';v=read(p);assert v['status']=='TERMINAL_MATCHED_WINDOW_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED'and v['epoch']==e;accepted.append(proof(p))
child=read(C/'active-child.json');assert not live(child),'Never stop a live CPU validator'
b=A/'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v1/window2018';root=read(b/'root-submission.json');assert str(root['job_id'])=='750023';accounting=subprocess.check_output(['sacct','-X','-n','-P','-j','750023','--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15);assert'750023|FAILED|1:0'in accounting and not list(b.glob('checkpoint-epoch-*.pt'))and not(b/'controller.json').exists()
if live(ctrl):
 argv=Path('/proc/%d/cmdline'%ctrl['pid']).read_bytes().split(bytes([0]));assert str(C/'code-f0e68450cd10/selected_checkpoint_cpu_coordinator.py').encode()in argv;os.kill(ctrl['pid'],signal.SIGTERM)
 for unused in range(30):
  if not live(ctrl):break
  time.sleep(.1)
 assert not live(ctrl),'Own coordinator did not exit after SIGTERM'
result=dict(status='SUPERSEDED_BEFORE_ANY_WINDOW_UPDATE_AFTER_TWO_CPU_ACCEPTANCES',time=time.time(),coordinator_identity=ctrl,coordinator_identity_live=False,accepted_preserved=accepted,old_window_job=750023,old_window_state='FAILED1:0_CLI2018_REJECTED_NO_CP',CPU_validator_interrupted=False,training_or_Slurm_mutated=False);(C/'retired.json').write_text(json.dumps(result,sort_keys=True)+'\n');print(json.dumps(result))
