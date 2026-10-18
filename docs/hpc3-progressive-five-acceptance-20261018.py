"""Independent actual G5/Adam-ledger acceptance of resumed51..55 plus next56."""
import hashlib
import json
import shlex
import subprocess
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT

QUERY=r'''
import os,sys,json,time,hashlib
from pathlib import Path
from collections import defaultdict,Counter
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3')
os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1';os.environ['DYNRPO_PREFIX_TOKENS']='20'
sys.path.insert(0,str(b/'repo'))
def rd(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
if not (b/'outers/outer-000055/complete.json').exists():
 print(json.dumps(dict(status='WAITING_FOR_FIVE_REAL_COMPLETE_OUTERS')));raise SystemExit(0)
proof=b/'preparation/resumed51-55-and-positive56-acceptance.json'
if proof.exists():print(json.dumps(rd(proof)));raise SystemExit(0)
from reproduction.prefix_selection200.storage import actual_actor_boundary
from reproduction.prefix_selection200.adaptive_candidates import derive
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import actor_ledger
config=rd(b/'preparation/launch-config.json');j50=rd(b/'outers/outer-000050/complete.json')
accepted_probe=rd(b/'preparation/progressive-bootstrap-probe-full-sha-cpu.json')
assert accepted_probe['probe_checkpoint_sha256']==j50['probe']['sha256']
assert accepted_probe['probe_actual_cpu_deserialization'] and accepted_probe['probe_actual_adam_per_rank']==[400]*4
allocation=rd(b/'logs/slurm-allocation.json');assert allocation['total_physical_gpus']==4 and allocation['training']==allocation['auxiliary']
class SmallSeals:
 def file(self,path):
  p=Path(path);assert b in p.parents and p.resolve()==p
  before=p.stat();digest=sha(p);after=p.stat()
  assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  return dict(path=str(p),bytes=before.st_size,sha256=digest)
seals=SmallSeals();previous=[1000]*4;rows=[]
for outer in range(51,56):
 root=b/'outers'/('outer-%06d'%outer);joint=rd(root/'complete.json');actor=joint['actor'];probe=joint['probe']
 assert joint['run_id']==config['run_id'] and joint['execution_id']==config['execution_id']
 assert joint['source_commit']==config['source_commit'] and actual_actor_boundary(actor,outer)
 assert [a-n for a,n in zip(actor['actual_adam_steps_per_rank'],actor['new_adam_steps_per_rank'])]==previous
 assert actor['scheduler_completed_outer']==outer and probe['actual_adam_steps_per_rank']==[400]*4
 assert probe['sha256']==j50['probe']['sha256'] and joint['probe_metrics']['new_adam_steps_per_rank']==0
 ledger_proofs=[actor_ledger(seals,actor,outer,rank) for rank in range(4)]
 prefix=rd(root/'prefix/complete.json');score=rd(root/'probe-score/manifest.json');full=rd(root/'full/complete.json')
 selected=rd(root/'probe-score/selection.json');budget=rd(root/'candidate-budget.json')
 assert budget==derive(config,outer) and 512<=budget['questions']<=5120 and budget['questions']%4==0
 assert prefix['head_prefix_tokens']==20 and score['head_max_tokens']==293
 assert score['probe_before_checkpoint_sha256']==j50['probe']['sha256']
 assert prefix['questions']==budget['questions'] and prefix['trajectories']==budget['questions']*5
 assert full['questions']==512 and full['trajectories']==2560 and selected['outcome_labels_used'] is False
 assert joint['actor_selection_outcome_labels_used'] is False
 assert full['actor_version']==prefix['actor_version']==score['actor_version']==actor['sampling_actor_version']
 labels=Path(full['own_em']['path']);assert sha(labels)==full['own_em']['sha256']
 groups=defaultdict(list);seen=set()
 for raw in labels.read_text().splitlines():
  item=json.loads(raw);assert item['trajectory_id'] not in seen;seen.add(item['trajectory_id'])
  assert item['strict_em'] in (0,1) and item['actor_version']==full['actor_version']
  groups[item['qid']].append(item['strict_em'])
 assert len(groups)==512 and len(seen)==2560 and all(len(values)==5 for values in groups.values())
 hist=Counter(sum(values) for values in groups.values());mixed=sum(hist[k] for k in range(1,5))
 assert full['actual_selected_mixed_questions']==mixed and full['actual_selected_mixed_fraction']==mixed/512
 pool=rd(joint['pool_snapshot']);assert pool['next_outer']==outer+1 and pool['consumed_candidates']==budget['consumed_after']
 rows.append(dict(outer=outer,actual_adam_before=previous,actual_adam_after=actor['actual_adam_steps_per_rank'],
  new_actual_adam=actor['new_adam_steps_per_rank'],skipped=actor['skipped_update_opportunities'],scheduler=outer,
  candidates=budget['questions'],prefix_tokens=20,selected512_actual_k_histogram={str(k):hist[k] for k in range(6)},
  selected_mixed=mixed,selected_mixed_fraction=mixed/512,probe_actual_adam_per_rank=400,
  rank_ledger_sha256=[row['file']['sha256'] for row in ledger_proofs],joint_sha256=sha(root/'complete.json')))
 previous=actor['actual_adam_steps_per_rank']
status=rd(b/'logs/pipeline-status.json');next_root=b/'outers/outer-000056/prefix'
progress=[rd(next_root/('rank-%d-progress.json'%rank)) if (next_root/('rank-%d-progress.json'%rank)).exists() else None for rank in range(4)]
counts=[row.get('completed_trajectories',0) if row else 0 for row in progress]
if status['outer']<56 or not any(counts):
 print(json.dumps(dict(status='FIVE_OUTERS_ACCEPTED_WAITING_FOR_POSITIVE56',rows=rows)));raise SystemExit(0)
current_actor=rd(b/'actor/latest.json')
for item in current_actor['rank_states']:
 path=Path(item['path']);assert path.is_file() and path.stat().st_size==item['bytes']
value=dict(status='FIVE_COMPLETE_CP50_RESUMED_OUTERS_51_TO_55_AND_POSITIVE56_ACCEPTED',job=config['slurm_job'],
 runtime_source_commit=config['source_commit'],execution_id=config['execution_id'],total_physical_gpus=4,
 bootstrap_probe_full_cpu_readback=str(b/'preparation/progressive-bootstrap-probe-full-sha-cpu.json'),
 bootstrap_effective_parent=config['resume_joint'],actor_initial_actual_adam=[1000]*4,probe_frozen_actual_adam=[400]*4,
 independent_own_em_and_all20_rank_update_ledgers_sha_checked=True,actor_tensor_full_sha_repeated=False,
 rows=rows,next_outer=56,next_prefix_completed_per_rank=counts,current_actual_adam=current_actor['actual_adam_steps_per_rank'],
 operator_source_sha256=OPERATOR_SHA,checked_unix=time.time(),whole_200_goal_complete=False)
with proof.open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
'''

def main():
    code=QUERY.replace('OPERATOR_SHA',repr(hashlib.sha256(QUERY.encode()).hexdigest()))
    address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=55)
    if r.returncode:raise RuntimeError(r.stdout.decode()[-4000:]+r.stderr.decode()[-1000:])
    value=next(json.loads(line) for line in reversed(r.stdout.decode().splitlines()) if line.lstrip().startswith('{'))
    (OUTPUT/'progressive-cp50-five-complete-acceptance.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))

if __name__=='__main__':main()
