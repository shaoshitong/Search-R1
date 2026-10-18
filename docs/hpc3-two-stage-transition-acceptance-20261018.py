"""Accept actual post100/150 Prefix and reward-grounding use in complete101/151.

Requires the separate full milestone CPU proof. Does not replace fixed512 QA,
run extra inference, alter training, or claim pre-fit tensor equality beyond
the recorded checked runtime migration. Only small audit metadata crosses SSH.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run

QUERY = r'''
import hashlib,json,math,os,time,sys
from pathlib import Path
b=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4');n=TARGET_OUTER
def rd(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
 return h.hexdigest()
def pending(status):print(json.dumps(dict(status=status,outer=n)));raise SystemExit(0)
assert rd(b/'preparation/source-publication.json')['commit']=='2000d46a530c64d78be7802bf11175a68b65effe'
assert rd(b/'preparation/slurm-submission.json')['job']=='733898'
root=b/'outers'/('outer-%06d'%n);next_root=b/'outers'/('outer-%06d'%(n+1))
if not (root/'complete.json').is_file():pending('PENDING_ACTUAL_TRANSITION_JOINT')
if not (next_root/'complete.json').is_file():pending('PENDING_ACTUAL_NEXT_COMPLETE_OUTER')
cpu_path=b/'preparation'/('live-milestone-%06d-v1-full-sha-cpu.json'%n)
if not cpu_path.is_file():pending('PENDING_SEPARATE_FULL_MILESTONE_CPU_PROOF')
qualified=rd(cpu_path);j=rd(root/'complete.json');next_j=rd(next_root/'complete.json')
c=rd(b/'preparation/launch-config.json');a=j['actor'];next_a=next_j['actor']
assert qualified['status']=='LIVE_IMMUTABLE_MILESTONE_FULL_SHA_CPU_READBACK_PASS'
assert qualified['source_joint_sha256']==sha(root/'complete.json')
assert qualified['operator_script_sha256']==CPU_OPERATOR_SHA
assert qualified['validator_runtime_commit']==j['runtime_source_commit']==next_j['runtime_source_commit']=='2000d46a530c64d78be7802bf11175a68b65effe'
for seal in qualified['verified_files']:
 stat=Path(seal['path']).stat()
 assert (stat.st_ino,stat.st_size,stat.st_mtime_ns,int(os.statvfs(seal['path']).f_fsid))==(seal['inode'],seal['bytes'],seal['mtime_ns'],seal['shared_filesystem_fsid'])
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['DYNRPO_PROGRESSIVE_PREFIX']='cp50_prefix20_40_80_three_epoch_v1'
os.environ['DYNRPO_PREFIX_TOKENS']=str(40 if n==100 else 80)
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200 import grounding
from reproduction.prefix_selection200.progressive_prefix import actor_prefix,after_boundary,before_fit,boundary
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import actor_ledger
from reproduction.prefix_selection200.interaction_protocol import CONTRACT
assert j['run_id']==next_j['run_id']==c['run_id'] and j['execution_id']==next_j['execution_id']==c['execution_id']
assert j['outer']==n and next_j['outer']==n+1
grounding.validate_joint(c,j);grounding.validate_joint(c,next_j)
refresh=rd(root/'grounding-refresh.json');state=refresh['grounding']
assert refresh['required_before_outer']==n+1 and refresh['run_id']==c['run_id']
assert state==next_j['grounding']==next_a['grounding_state'] and state['round']==n
assert state['source_actor_version']==a['actor_version'] and state['version']!=j['grounding']['version']
grounding.validate_actor_transition(dict(c,grounding_state=state,grounding_probe_version=state['version']),a,n+1)
manifest=rd(state['manifest_path']);groot=Path(state['manifest_path']).parent
assert manifest['encoder_sha256']=={name:item['sha256'] for name,item in a['hf_files'].items()}
assert manifest['calibration_sha256']==c['grounding_calibration_sha256']
assert sha(groot/'heads/calibration-report.json')==manifest['calibration_report_sha256']
assert sha(groot/'dumps/calibration-label-proof.json')==manifest['dump_sha256']['calibration-label-proof.json']
assert rd(groot/'heads/calibration-report.json')==manifest['calibration_report']
assert rd(groot/'dumps/calibration-label-proof.json')==manifest['label_proof']
labels=manifest['label_proof'];report=manifest['calibration_report']
assert labels['question_count']==report['unique_question_count']==300 and labels['group_size']==5
assert set(labels['question_ids'])==set(c['grounding_calibration_question_ids'])
assert labels['grounding_source_actor_version']==a['actor_version'] and labels['interaction_protocol']==CONTRACT
assert labels['calibration_protocol']['rollout_max_new']==500 and labels['calibration_protocol']['correction_feedback'] is True
assert labels['interaction_execution']['actual_compute_dtype']=='torch.bfloat16'
assert labels['interaction_config_sha256']==sha(groot/'interaction-config.json')
assert report['require_all_cv_folds'] is True and report['configured_cv_folds']==2
import joblib,numpy as np
ground_heads=[]
for key,name in (('boundary','boundary_probe.pkl'),('commit','commit_probe.pkl')):
 p=Path(state[key]);assert sha(p)==state['head_sha256'][name]
 fitted=joblib.load(p);clf=fitted['clf']
 assert list(clf.classes_)==[0,1] and math.isfinite(float(fitted['auc']))
 assert np.isfinite(clf.coef_).all() and np.isfinite(clf.intercept_).all()
 ground_heads.append(dict(kind=key,path=str(p),sha256=sha(p),auc=float(fitted['auc']),finite_binary_classifier=True))
reward=next_a['reward'];assert reward['actual_reward_world_size']==4 and reward['trajectories']==2560
assert reward['prefix_correctness_head_used_for_reward'] is False and reward['version']==state['version']
assert sorted(r['rank'] for r in reward['ranks'])==list(range(4)) and len({r['probe_url'] for r in reward['ranks']})==4
assert all(r['version']==state['version'] and r['prefix_correctness_head_used_for_reward'] is False for r in reward['ranks'])
class Seals:
 def file(self,path):
  p=Path(path);assert b in p.parents and p.resolve()==p
  return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
ledgers=[actor_ledger(Seals(),next_a,n+1,rank) for rank in range(4)]
assert [x-y for x,y in zip(next_a['actual_adam_steps_per_rank'],next_a['new_adam_steps_per_rank'])]==a['actual_adam_steps_per_rank']
assert next_a['scheduler_completed_outer']==n+1 and next_a['complete_outer']
head=j['probe'];assert next_j['probe']['sha256']==head['sha256']
assert qualified['probe']['file']['sha256']==head['sha256']
assert head['actual_adam_steps_per_rank']==[boundary(n)['actual_adam_per_rank']]*4
score=rd(next_root/'probe-score/manifest.json');prefix=rd(next_root/'prefix/complete.json')
assert score['probe_before_checkpoint_sha256']==head['sha256'] and score['head_max_tokens']==273+after_boundary(n)
assert prefix['head_prefix_tokens']==actor_prefix(n+1)==after_boundary(n) and prefix['actor_version']==a['actor_version']
assert prefix['questions']==5120 and prefix['trajectories']==25600
for rank in prefix['ranks']:
 assert rank['actor_hidden_layer']==27 and rank['actual_compute_dtype']=='torch.bfloat16' and rank['independent_encoder_forward'] is False
migration=rd(b/'probe'/('prefix-migration-%03d.json'%n));change=migration['readout_migration']
previous=rd(b/'outers'/('outer-%06d'%(n-1))/'complete.json')
assert migration['source_receipt']['sha256']==previous['probe']['sha256']
assert migration['source_actual_adam']==migration['target_actual_adam']==[before_fit(n)['actual_adam_per_rank']]*4
assert migration['all_block_parameters_equal'] is True and migration['all_native_buffers_retained'] is True
assert change==dict(old_prefix_tokens=actor_prefix(n),new_prefix_tokens=after_boundary(n),question_slots_preserved=273,
 block_parameters_reinitialized=False,output_old_coordinates_preserved=True,new_output_coordinates_zero=True,optimizer_steps_reset=False,rank_rng_reset=False)
data=b/'probe-refreshes'/('round-%06d'%n)/'outers'/root.name
reservation=rd(data/'probe-only-reservation.json');assert reservation['student_checkpoint_outer']==n and reservation['student_actor_version']==a['actor_version'] and reservation['actor_pool_advanced'] is False
refresh_prefix=rd(data/'prefix/complete.json');pool=rd(data/'full-pool/complete.json');diagnostic=rd(data/'diagnostic-complete.json')
assert refresh_prefix['questions']==pool['questions']==2560 and refresh_prefix['head_prefix_tokens']==after_boundary(n)
assert refresh_prefix['actor_version']==pool['actor_version']==diagnostic['actor_version']==a['actor_version']
refresh_score=rd(data/'probe-score/manifest.json')
assert refresh_score['probe_before_checkpoint_sha256']==migration['migrated_receipt']['sha256']
assert j['probe_metrics']['actor_version']==a['actor_version'] and j['probe_metrics']['epochs']==3 and j['probe_metrics']['new_adam_steps_per_rank']==150
result=dict(status='ACTUAL_PREFIX_AND_GROUNDING_TRANSITION_USED_BY_COMPLETE_NEXT_OUTER_PASS',outer=n,checked_unix=time.time(),
 job='733898',run_id=c['run_id'],runtime_commit=j['runtime_source_commit'],operator_query_sha256=QUERY_SHA,
 source_joint_sha256=sha(root/'complete.json'),next_joint_sha256=sha(next_root/'complete.json'),
 milestone_cpu_proof=str(cpu_path),milestone_cpu_proof_sha256=sha(cpu_path),milestone_cpu_stat_seals_reused=True,
 old_prefix=actor_prefix(n),new_prefix=after_boundary(n),probe_head_sha256=head['sha256'],probe_adam=head['actual_adam_steps_per_rank'],
 runtime_checked_migration_sha256=sha(b/'probe'/('prefix-migration-%03d.json'%n)),pre_fit_tensors_independently_recompared=False,
 new_prefix_data_actor_version=pool['actor_version'],grounding_round=n,grounding_version=state['version'],
 grounding_manifest_sha256=sha(state['manifest_path']),adoption_sha256=sha(state['adoption_receipt']),grounding_heads=ground_heads,
 next_actor_actual_adam=next_a['actual_adam_steps_per_rank'],next_actor_new_adam=next_a['new_adam_steps_per_rank'],
 next_actor_skips=next_a['skipped_update_opportunities'],next_actor_ledger_sha256=[r['file']['sha256'] for r in ledgers],
 fresh_calibration_feature_payload_hash_scan=False,fixed512_QA_accepted_by_this_operator=False,full_goal_complete=False)
proof=b/'preparation'/('transition-%06d-actual-next-complete.json'%n)
if proof.exists():
 old=rd(proof);assert old['source_joint_sha256']==result['source_joint_sha256'] and old['next_joint_sha256']==result['next_joint_sha256'] and old['operator_query_sha256']==result['operator_query_sha256'];result=old
else:
 with proof.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(report=result,mirror=dict(path=str(proof),sha256=sha(proof)))))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int, choices=(100, 150), required=True)
    args = parser.parse_args()
    cpu_sha = hashlib.sha256(Path(__file__).with_name('hpc3-two-stage-milestone-readback-20261018.py').read_bytes()).hexdigest()
    code = QUERY.replace('TARGET_OUTER', str(args.outer)).replace('CPU_OPERATOR_SHA', repr(cpu_sha)).replace('QUERY_SHA', repr(hashlib.sha256(QUERY.encode()).hexdigest()))
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    response = run(PYTHON + ' -c ' + shlex.quote(code), account='mou176', source_address=address, timeout=55)
    if response.returncode:
        raise RuntimeError('Transition audit failed; no training action taken: ' + response.stdout.decode()[-1200:] + response.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in reversed(response.stdout.decode().splitlines()) if line.lstrip().startswith('{'))
    (OUTPUT / ('two-stage-transition-%06d-acceptance.json' % args.outer)).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
