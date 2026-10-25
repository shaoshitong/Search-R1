"""Stop only the live job750491 explicitly cancelled by its human owner.
Preserve existing checkpoints and remote log evidence; do not submit or resume.
"""
import json, os, subprocess, time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023')
job=(base/'job.id').read_text().strip()
assert job=='750491', ('current job changed; inspect latest identity before cancellation',job)
run=json.loads((base/'logs/run-manifest.json').read_text())
assert run['source_commit']=='dde1452d2ec65fb3f4a0464d4beacf3b8136414c'
def cmd(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=35)
 return dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
queue=cmd(['squeue','-h','-j',job,'-o','%i|%u|%T|%N'])
assert queue['returncode']==0,queue
if queue['stdout'].strip():
 fields=queue['stdout'].strip().split('|');assert fields[0]==job and fields[1]=='jxu957',fields
sealpath=base/'preparation/rolling-checkpoints/latest.json'
seal=json.loads(sealpath.read_text()) if sealpath.exists() else None
record=dict(reason='explicit live human stop, not failure',instruction='停掉现在的训练吧',authorization_recorded_utc='2026-10-25T06:41:08Z',job=job,source_commit=run['source_commit'],run_id=run.get('run_id'),observed_unix=time.time(),before_queue=queue,latest_durable_seal=seal,training_resume_allowed=False,checkpoints_deleted=False)
evidence=base/'preparation/human-stop-20261025';evidence.mkdir(exist_ok=True)
for log in ['logs/training.log','logs/trainer.log']:
 p=base/log
 if p.is_file():
  with p.open('rb') as stream:
   stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));tail=stream.read()
  (evidence/(p.name+'.tail')).write_bytes(tail)
recordpath=evidence/'receipt.json'
recordpath.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
if queue['stdout'].strip():
 record['scancel']=cmd(['scancel',job]);assert record['scancel']['returncode']==0,record['scancel']
else: record['scancel']=dict(skipped=True,reason='already absent from queue; no duplicate cancellation')
record['after_queue']=cmd(['squeue','-h','-j',job,'-o','%i|%u|%T|%N'])
record['after_accounting']=cmd(['sacct','-j',job,'--noheader','--parsable2','-o','JobIDRaw,User,State,ExitCode,Elapsed,NodeList'])
record['checked_unix']=time.time();recordpath.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False))
