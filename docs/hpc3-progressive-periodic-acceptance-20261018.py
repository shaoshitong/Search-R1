"""Accept one non-transition P10 and actual next-outer head/Adam use.

Only scalar receipts cross SSH. The separate periodic CPU operator must have
qualified the final probe payload; this journal check never claims actor CPU
readback or full hashes of an actively growing next-outer update ledger.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON

QUERY = r'''
import os,sys,json,time,hashlib,datetime
from pathlib import Path
from collections import defaultdict,Counter
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r3');n=TARGET_N
os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1'
os.environ['DYNRPO_PREFIX_TOKENS']=str(20 if n<100 else 40 if n<150 else 80)
sys.path.insert(0,str(b/'repo'))
def rd(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
def wait(stage):print(json.dumps(dict(status=stage,outer=n)));raise SystemExit(0)
root=b/'outers'/('outer-%06d'%n);joint_path=root/'complete.json'
if not joint_path.exists():wait('WAITING_FOR_ACTUAL_P10_JOINT')
proof=b/'preparation'/('progressive-periodic-%06d-actual-next-use.json'%n)
if proof.exists():print(json.dumps(rd(proof)));raise SystemExit(0)
cpu=b/'preparation'/('live-probe-%06d-v1-full-sha-cpu.json'%n)
if not cpu.exists():wait('WAITING_FOR_FINAL_PROBE_FULL_SHA_CPU_PROOF')
c=rd(b/'preparation/launch-config.json');j=rd(joint_path);previous=rd(b/'outers'/('outer-%06d'%(n-1))/'complete.json')
from reproduction.prefix_selection200.progressive_prefix import boundary,actor_prefix,before_fit
from reproduction.prefix_selection200.adaptive_candidates import derive
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import actor_ledger
assert j['outer']==n and j['periodic_probe_update'] and j['runtime_source_commit']=='89c97ff2a5a60c7c3d6e1b0bf825d122f0ff874c'
assert j['run_id']==c['run_id'] and j['execution_id']==c['execution_id']
head_before=before_fit(n)['actual_adam_per_rank'];head_after=boundary(n)['actual_adam_per_rank']
assert j['probe']['actual_adam_steps_per_rank']==[head_after]*4
assert j['probe_metrics']['new_adam_steps_per_rank']==150 and j['probe_metrics']['epochs']==3
qualified=rd(cpu)
assert qualified['status']=='PERIODIC_PROBE_FULL_SHA_CPU_READBACK_PASS'
assert qualified['source_joint_sha256']==sha(joint_path) and qualified['runtime_commit']==j['runtime_source_commit']
assert qualified['probe']['file']['sha256']==j['probe']['sha256'] and qualified['probe']['finite_model']
assert qualified['probe']['actual_adam_steps_per_rank']==[head_after]*4 and qualified['cuda_initialized'] is False
for seal in qualified['verified_files']:
 s=Path(seal['path']).stat()
 assert (s.st_ino,s.st_size,s.st_mtime_ns)==(seal['inode'],seal['bytes'],seal['mtime_ns'])
 assert str(os.statvfs(seal['path']).f_fsid)==str(seal['shared_filesystem_fsid'])
class Seals:
 def file(self,path):
  p=Path(path);assert b in p.parents and p.resolve()==p
  return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
seals=Seals();a=j['actor']
assert [x-y for x,y in zip(a['actual_adam_steps_per_rank'],a['new_adam_steps_per_rank'])]==previous['actor']['actual_adam_steps_per_rank']
assert a['scheduler_completed_outer']==n and a['complete_outer']
rank_ledgers=[actor_ledger(seals,a,n,rank) for rank in range(4)]
prefix=rd(root/'prefix/complete.json');pool=rd(root/'full-pool/complete.json');selected=rd(root/'full/complete.json')
assert prefix['head_prefix_tokens']==actor_prefix(n) and prefix['questions']==pool['questions']==2560
assert prefix['trajectories']==pool['trajectories']==12800
assert pool['actor_version']==prefix['actor_version']==j['rollout']['actor_version']==a['sampling_actor_version']==j['probe_metrics']['actor_version']
assert j['actor_selection_outcome_labels_used'] is True
assert rd(Path(j['actor_selection_path']))['outcome_labels_used'] is True
def own_labels(receipt,count):
 own=receipt['own_em'];assert Path(own['path']).stat().st_size==own['bytes'] and sha(own['path'])==own['sha256']
 groups=defaultdict(list);ids={}
 with Path(own['path']).open() as f:
  for raw in f:
   row=json.loads(raw);tid=row['trajectory_id']
   assert tid not in ids and type(row['strict_em']) is int and row['strict_em'] in (0,1)
   assert row['actor_version']==pool['actor_version']
   ids[tid]=row['strict_em'];groups[row['qid']].append(row['strict_em'])
 assert len(ids)==count*5 and len(groups)==count and all(len(v)==5 for v in groups.values())
 histogram=Counter(sum(v) for v in groups.values())
 return ids,{str(k):histogram[k] for k in range(6)},sum(histogram[k] for k in range(1,5))
all_ids,pool_hist,pool_mixed=own_labels(pool,2560)
selected_ids,selected_hist,selected_mixed=own_labels(selected,512)
assert all(all_ids[k]==v for k,v in selected_ids.items())
assert selected_mixed==j['rollout']['actual_selected_mixed_questions'] and selected_mixed/512==j['rollout']['actual_selected_mixed_fraction']
fit=root/'probe-update';epoch_rows=[rd(fit/('epoch-%03d.json'%e)) for e in (1,2,3)]
assert [r['actual_adam_steps_per_rank'] for r in epoch_rows]==[[head_before+50*e]*4 for e in (1,2,3)]
for row in epoch_rows:
 assert Path(row['path']).stat().st_size==row['bytes'] and row['prefix_tokens']==actor_prefix(n)
updates=[]
for rank in range(4):
 p=fit/('rank-%d-updates.jsonl'%rank);rows=[json.loads(raw) for raw in p.read_text().splitlines()]
 assert len(rows)==150 and [r['step'] for r in rows]==list(range(1,151))
 assert [r['actual_adam_steps'] for r in rows]==list(range(head_before+1,head_after+1))
 assert [r['epoch'] for r in rows]==[e for e in (1,2,3) for _ in range(50)]
 updates.append(dict(rank=rank,actual_updates=150,sha256=sha(p)))
curve=[json.loads(raw) for raw in (fit/'global-loss-updates.jsonl').read_text().splitlines()]
assert len(curve)==150
assert datetime.datetime.fromisoformat(a['saved_utc'])<=datetime.datetime.fromisoformat(curve[0]['observed_utc'])
next_root=b/'outers'/('outer-%06d'%(n+1));score_path=next_root/'probe-score/manifest.json'
if not score_path.exists():wait('WAITING_FOR_ACTUAL_NEXT_NEW_HEAD_SCORE')
score=rd(score_path);next_prefix=rd(next_root/'prefix/complete.json')
assert score['probe_before_checkpoint_sha256']==j['probe']['sha256']
assert score['head_max_tokens']==273+actor_prefix(n+1)
assert next_prefix['actor_version']==score['actor_version']==a['actor_version']
assert next_prefix['head_prefix_tokens']==actor_prefix(n+1)
nb=rd(next_root/'candidate-budget.json');assert nb==derive(c,n+1)
stage=rd(b/'logs/pipeline-status.json');next_joint=next_root/'complete.json'
if next_joint.exists():
 next_a=rd(next_joint)['actor'];paths=[Path(r['ledger_path']) for r in next_a['rank_receipts']]
else:
 marker='outer-%06d-actor-'%(n+1)
 if not stage['stage'].startswith(marker):wait('WAITING_FOR_NEXT_ACTUAL_ACTOR_RESTORE_AND_WRITE')
 attempt=stage['stage'][len(marker):];assert len(attempt)==32 and all(k in '0123456789abcdef' for k in attempt)
 directory=b/'actor/outers'/('outer-%04d-attempt-%s'%(n+1,attempt))
 paths=[directory/('rank-%d-updates.jsonl'%rank) for rank in range(4)]
next_evidence=[]
for rank,p in enumerate(paths):
 if not p.exists():wait('WAITING_FOR_ALL_FOUR_NEXT_RESTORE_AND_WRITES')
 raw=p.read_bytes();lines=raw.splitlines(keepends=True)
 if lines and not lines[-1].endswith(b'\n'):lines.pop()
 rows=[json.loads(line) for line in lines if line.strip()]
 restores=[r for r in rows if r['event']=='ACTUAL_RESTORE'];writes=[r for r in rows if r['event']=='ACTUAL_ADAM_WRITE']
 if not restores or not writes:wait('WAITING_FOR_ALL_FOUR_NEXT_RESTORE_AND_WRITES')
 restore=restores[0];assert restore['restored_completed_outer']==restore['restored_scheduler']==n
 assert restore['actual_adam_before']==a['actual_adam_steps_per_rank'][rank]
 assert restore['actual_model_master_dtypes']==['torch.float32']
 assert writes[0]['actual_adam_steps']==a['actual_adam_steps_per_rank'][rank]+1
 assert all(r['rank']==rank and r['outer']==n+1 for r in writes)
 next_evidence.append(dict(rank=rank,restored_completed_outer=n,restored_adam=restore['actual_adam_before'],
  actual_writes_observed=len(writes),last_actual_adam=writes[-1]['actual_adam_steps'],
  ledger_complete_at_observation=next_joint.exists(),observed_complete_lines_sha256=hashlib.sha256(b''.join(lines)).hexdigest()))
result=dict(status='P10_THREE_EPOCH_CPU_QUALIFIED_AND_NEXT_REAL_NEW_HEAD_ACTOR_USE_ACCEPTED',outer=n,
 checked_unix=time.time(),runtime_source_commit=j['runtime_source_commit'],operator_source_sha256=OPERATOR_SHA,
 actor_actual_adam=a['actual_adam_steps_per_rank'],actor_new_adam=a['new_adam_steps_per_rank'],actor_skipped=a['skipped_update_opportunities'],
 actor_rank_ledger_sha256=[r['file']['sha256'] for r in rank_ledgers],student_saved_before_first_probe_update=True,
 probe_actual_before=head_before,probe_actual_after=head_after,probe_updates_per_rank=150,epochs=3,prefix_tokens=actor_prefix(n),
 epoch_boundaries=[head_before+50*e for e in (1,2,3)],epoch_files_size_and_receipts_checked=True,epoch_payload_full_sha_repeated=False,
 probe_cpu_proof=str(cpu),probe_cpu_proof_sha256=sha(cpu),head_sha256=j['probe']['sha256'],
 probe_rank_update_ledgers=updates,full_pool_questions=2560,full_pool_k_histogram=pool_hist,full_pool_mixed=pool_mixed,
 selected512_k_histogram=selected_hist,selected512_mixed=selected_mixed,probe_data_actor_version=pool['actor_version'],
 actual_next_head_score_sha256=sha(score_path),next_candidate_budget=nb,next_actual_actor_restore_and_writes=next_evidence,
 actual_actor_tensor_cpu_readback_performed=False,full_200_goal_complete=False)
with proof.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(report=result,mirror=dict(path=str(proof),sha256=sha(proof)))))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer',type=int,choices=[n for n in range(60,200,10) if n not in (100,150)],required=True)
    args=parser.parse_args()
    code=QUERY.replace('TARGET_N',str(args.outer)).replace('OPERATOR_SHA',repr(hashlib.sha256(QUERY.encode()).hexdigest()))
    address=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip()
    r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=address,timeout=55)
    if r.returncode:raise RuntimeError('Periodic audit failed, no training action taken: '+r.stdout.decode()[-2000:]+r.stderr.decode()[-2000:])
    value=next(json.loads(line) for line in reversed(r.stdout.decode().splitlines()) if line.lstrip().startswith('{'))
    (OUTPUT/('progressive-periodic-%06d-acceptance.json'%args.outer)).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))

if __name__=='__main__':main()
