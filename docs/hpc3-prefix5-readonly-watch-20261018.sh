#!/usr/bin/env bash
# Bounded read-only supervision for job729624 through joint50 and real outer51 progress.
# Optional progress files may disappear during phase publication or cleanup.
# This supervisor never submits jobs, allocates GPUs, or changes training code.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json,shlex,subprocess,time
from reproduction.hpc3_prefix5.observe import query_for
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
base='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2';job='729624'
extension=r"""
result['base']=str(b)
result['checkpoint_import_pending']=joint is None
template=read(b/'preparation/resume-config-template.json')
result['resume_selected_outer']=template['resume_joint']['completed_outer'] if template else None
result['qualification_progress_ranks']=[]
if 'tail' in globals():
 import re
 result['qualification_progress_ranks']=sorted(set(map(int,re.findall('I1_I6_CPU_QUALIFIED_ACTOR_RANK=([0-3])',tail))))
qualification=read(pathlib.Path(template['resume_joint']['qualification_proof'])) if template else None
result['qualification']={key:qualification.get(key) for key in ['status','completed_outer','actual_adam_steps_per_rank','actor_rank_optimizer_rng_scheduler_verified','actor_driver_rng_verified','probe_model_optimizer_rank_rng_verified','payload_full_sha_read_once']} if qualification else None
result['phase_progress']=[]
result['optional_progress_read_races']=[]
import os
if status and status.get('outer'):
 for p in sorted(r.rglob('*progress*.json')):
  try:
   with p.open('rb') as handle:
    stat=os.fstat(handle.fileno());raw=handle.read(65537)
  except FileNotFoundError:
   result['optional_progress_read_races'].append(str(p.relative_to(r)))
   continue
  if len(raw)>65536:continue
  z=json.loads(raw)
  if not isinstance(z,dict):continue
  fields=['stage','rank','completed_trajectories','expected_trajectories','scored_trajectories','completed_questions','expected_questions','step','actual_adam_steps','total_loss','trajectory_bce','group_bce','actual_learning_rate']
  result['phase_progress'].append(dict(path=str(p.relative_to(r)),mtime=stat.st_mtime,**{k:z[k] for k in fields if k in z}))
print(json.dumps(result,allow_nan=False))
"""
q=query_for(base).replace('print(json.dumps(result,allow_nan=False))',extension)
previous_signature=None
for sample in range(40):
 address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
 remote=run(PYTHON+' -c '+shlex.quote(q),account='mou176',source_address=address,timeout=25)
 values=[]
 for line in remote.stdout.decode().splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if isinstance(v,dict) and v.get('job')==job:values.append(v)
 if not values:
  print(json.dumps(dict(sample=sample,returncode=remote.returncode,observation_error=(remote.stdout.decode()+remote.stderr.decode())[-1800:],training_terminal_not_inferred=True)),flush=True)
 else:
  v=values[-1]
  (OUTPUT/'observation-latest.json').write_text(json.dumps(v,indent=2)+'\n')
  with (OUTPUT/'observations.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  compact={key:v.get(key) for key in ['epoch','job','base','queue','complete_outer','status','actor','probe','checkpoint_import_pending','resume_selected_outer','qualification','qualification_progress_ranks','failed_jobs','rank_failures','inflight_rank_updates','error_lines','phase_progress']};compact['sample']=sample
  signature=json.dumps({k:val for k,val in compact.items() if k not in ('epoch','sample')},sort_keys=True)
  printed=compact if signature!=previous_signature else dict(sample=sample,epoch=v['epoch'],job=job,verified_live_queue=v['queue'],unchanged=True)
  previous_signature=signature
  print(json.dumps(printed),flush=True)
  with (OUTPUT/'cp9-recovery-live-watch.jsonl').open('a') as f:f.write(json.dumps(compact)+'\n')
  historical=v['resume_selected_outer'] if v['checkpoint_import_pending'] else v['complete_outer']
  for name in ['current.json','goal-acceptance-ledger.json']:
   p=OUTPUT/name;s=json.loads(p.read_text());s.update(checked_unix=time.time(),goal_status='active',full_goal_complete=False)
   if name=='current.json':s.update(formal_outers=historical,current_queue=v['queue'],last_remote_sample=v['epoch'],checkpoint_import_pending=v['checkpoint_import_pending'])
   else:s.update(actual_completed_outers=historical,latest_verified_job=v)
   p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
  if v['failed_jobs'] or v.get('error_lines') or (v.get('rank_failures') and any(v['rank_failures'])) or not any('|'+state+'|' in v['queue'] for state in ('PENDING','RUNNING')):break
  if v['complete_outer']>=50 and (v.get('status') or {}).get('outer',0)>=51 and any((p.get('completed_trajectories') or p.get('scored_trajectories') or 0)>0 for p in v['phase_progress']):break
 if sample<39:time.sleep(45)
PY
