#!/usr/bin/env bash
# Read-only native journal, actual Adam/skip accounting, and next-budget verification.
# Run from this isolated checkout: bash docs/hpc3-native-outer-acceptance-20261018.sh OUTER
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - "$@" <<'PY_NATIVE_AUDIT'
import json,shlex,subprocess,math,sys
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
from reproduction.hpc3_prefix5.observe import query_for
if len(sys.argv)!=2:raise SystemExit('Expected one completed outer number')
outer=int(sys.argv[1])
if not 2<=outer<=200:raise SystemExit('Outer must be in2..200')
extra=r"""
import hashlib
o=NATIVE_OUTER
path=b/'outers'/f'outer-{o:06d}'/'complete.json'
j=read(path,limit=128*1024**2);result['native_joint']=project_metadata(j);result['native_joint_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest() if j else None;result['native_joint_control_projection']='embedded source_actor_receipt only; complete original journal stays remote';result['next_budget']=read(b/'outers'/f'outer-{o+1:06d}'/'candidate-budget.json');result['native_rank_evidence']=[]
if j:
 prev=read(b/'outers'/f'outer-{o-1:06d}'/'complete.json',limit=128*1024**2)
 result['native_selection']=read(pathlib.Path(j['actor_selection_path']))
 for rank,receipt in enumerate(j['actor']['rank_receipts']):
  raw=pathlib.Path(receipt['ledger_path']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==receipt['ledger_sha256']
  events=[json.loads(line) for line in raw.splitlines() if line.strip()]
  rest=next(e for e in events if e.get('event')=='ACTUAL_RESTORE')
  writes=[e['actual_adam_steps'] for e in events if e.get('event')=='ACTUAL_ADAM_WRITE']
  before=prev['actor']['actual_adam_steps_per_rank'][rank];after=j['actor']['actual_adam_steps_per_rank'][rank]
  new=j['actor']['new_adam_steps_per_rank'][rank];skipped=j['actor']['skipped_update_opportunities'][rank]
  assert writes==list(range(before+1,after+1)) and len(writes)==new and new+skipped==20
  assert rest['actual_adam_before']==before and rest['restored_scheduler']==o-1
  monitors=sum(e.get('event')=='POLICY_MONITOR' for e in events)
  if skipped==0:assert monitors==40
  result['native_rank_evidence'].append(dict(rank=rank,restored_outer=rest['restored_completed_outer'],restored_adam=before,restored_scheduler=rest['restored_scheduler'],writes=writes,skipped=skipped,monitor_callbacks=monitors,native_ledger_sha_pass=True))
 nb=result['next_budget']
 if nb:
  assert nb['previous_outer']==o and nb['previous_joint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
  assert nb['previous_selected_mixed_questions']==j['rollout']['actual_selected_mixed_questions']
print(json.dumps(result,allow_nan=False))
""".replace('NATIVE_OUTER',str(outer))
q=query_for('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r2').replace('print(json.dumps(result,allow_nan=False))',extra)
r=run(PYTHON+' -c '+shlex.quote(q),account='mou176',source_address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip(),timeout=25)
if r.returncode:
 print('READ_OR_AUDIT_FAILED_NOT_TRAINING_TERMINAL',r.returncode,r.stdout.decode()[-1800:]);raise SystemExit(1)
d=json.loads(r.stdout.decode().strip());j=d['native_joint'];nb=d['next_budget']
if j:
 assert len(d['native_rank_evidence'])==4
 if not j['periodic_probe_update']:
  assert not d['native_selection']['outcome_labels_used']
  assert d['native_selection']['probe_before_checkpoint_sha256']==j['probe']['sha256']
 if nb:
  if nb['mode']=='fixed_probe_update':assert nb['questions']==2560
  else:
   assert nb['mode']=='adaptive';den=1.1 if nb['previous_selected_mixed_questions']==512 else nb['previous_selected_mixed_questions']/512
   assert nb['questions']==max(512,min(20480,4*math.ceil(nb['previous_questions']/(4*(den+1e-6)))))
(OUTPUT/f'progress-outer{outer}-and-live{outer+1}.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(epoch=d['epoch'],queue=d['queue'],native_joint_verified=bool(j),complete_outer=d['complete_outer'],stage=d['status'],actor=d['actor'],probe=d['probe'],next_budget=nb,prefix=[x['completed_trajectories'] if x else None for x in d.get('prefix',[])],errors=d['error_lines'],failures=d['failed_jobs']),ensure_ascii=False))
PY_NATIVE_AUDIT
