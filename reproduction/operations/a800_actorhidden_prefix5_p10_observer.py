"""Read one bounded P10 actor/native-prefix5 snapshot without runtime changes.

The operator is published separately from the running training source. It
extends the existing control observer with strict four-rank P10 clocks, scalar
completed-outer history and already-produced QA metrics. It never opens model,
checkpoint, tensor, feature, trajectory, question or evaluation-row payloads.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/Users/shaoshitong/Documents/Search-G1')
CONTROL = Path(__file__).with_name('a800_actorhidden_prefix5_p10_control.py')
MAX_OUTPUT = 128 * 1024


EXTRA = r'''
import math
def p10_clock(outer):
 assert type(outer) is int and 0<=outer<=200,'Invalid completed actor boundary'
 last=min(outer,199)//10*10
 return last,last//10*50
def numeric(value):
 assert type(value) in (int,float) and math.isfinite(value),'Nonfinite or nonnumeric scalar metadata'
 return value
def optional_numeric(value):return None if value is None else numeric(value)
def unit(value):
 value=numeric(value);assert 0<=value<=1,'Score outside [0,1]'
 return value
def hash_value(value):
 assert isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'Invalid producer hash metadata'
 return value
def strict_probe(value,outer):
 last,updates=p10_clock(outer)
 assert value['run_id']==h['run_id'] and value['completed_outer']==last,'Probe completed-outer clock differs'
 assert value['actual_adam_steps_per_rank']==[updates]*4,'Four-rank P10 probe Adam counters differ'
 return last,updates
def strict_actor(value,outer):
 assert value['run_id']==h['run_id'] and value['completed_outer']==outer
 assert value['actual_adam_steps_per_rank']==[20*outer]*4 and value['scheduler_completed_outer']==outer
 receipts=value.get('rank_receipts')
 if receipts is not None:
  assert len(receipts)==4 and {row['rank'] for row in receipts}==set(range(4))
  assert all(row['outer']==outer and row['new_adam_steps']==20 and row['opportunities']==20
   and row['actual_adam_steps']==20*outer and row['scheduler_completed_outer']==outer for row in receipts)
def safe_metrics(value):
 if value is None:return None
 keys=('count','mixed_count','nonmixed_count','accuracy','balanced_accuracy','tp','fp','fn','tn',
  'trajectory_bce','group_bce','total_loss','mean_q')
 return {key:optional_numeric(value[key]) for key in keys if key in value}
def safe_probe_metrics(value):
 if value is None:return None
 result={key:value[key] for key in ('status','outer','last_update_outer','new_adam_steps_per_rank','actual_adam_steps_per_rank',
  'actual_learning_rate','unselected_training_questions','train_loop_seconds') if key in value}
 for key in ('train','test'):
  if key in value:result[key]=safe_metrics(value[key])
 before=value.get('before_update')
 if isinstance(before,dict):result['before_update']={key:safe_metrics(before[key]) for key in ('train','test') if key in before}
 for key in ('train_online_trajectory_bce','train_online_group_bce','train_online_total_loss'):
  if key in value:result[key]=numeric(value[key])
 return result
if v.get('actor_latest') is not None:
 strict_actor(v['actor_latest'],v['actor_latest']['completed_outer'])
if v.get('probe_latest') is not None:
 assert type(v['probe_latest']['completed_outer']) is int
 probe_outer=v['probe_latest']['completed_outer']
 assert probe_outer==0 or (probe_outer%10==0 and 0<probe_outer<200)
 strict_probe(v['probe_latest'],probe_outer)
if v.get('latest_joint') is not None:
 v['latest_joint']['probe_metrics']=safe_probe_metrics(v['latest_joint'].get('probe_metrics'))

latest=joint
n=0 if latest is None else latest['outer']
assert type(n) is int and 0<=n<=200
v['complete_outer']=n
v['probe_clock_contract']=dict(period=10,updates_per_period=50,skip_update_at_outer200=True,
 last_update_outer=p10_clock(n)[0],actual_adam_steps_per_rank=[p10_clock(n)[1]]*4,
 correctness_requires_recorded_probe_metadata=True)
v['history_columns']=['outer','actor_adam_per_rank','probe_completed_outer','probe_adam_per_rank',
 'reward_mean','reward_min','reward_max','selected_mixed_fraction','selected_mixed_count',
 'selected_strict_correct','actual_em_selection','candidate_mixed_count','fallback_count',
 'all_correct_groups','all_wrong_groups','mixed_groups','joint_sha256']
v['complete_outer_scalar_history']=[];v['probe_update_history']=[]
actor_versions={}
for outer in range(1,n+1):
 root=b/'outers'/('outer-%06d'%outer)
 completed_joint=document(root,'complete.json')
 assert completed_joint is not None,'A previously committed outer metadata is missing; preserve unknown'
 assert completed_joint['run_id']==h['run_id'] and completed_joint['source_commit']==h['source_commit']
 assert completed_joint['outer']==outer
 actor,probe=completed_joint['actor'],completed_joint['probe']
 strict_actor(actor,outer);last,probe_updates=strict_probe(probe,outer)
 clock=completed_joint.get('probe_boundary')
 if clock is not None:
  assert clock['last_update_outer']==last and clock['actual_adam_per_rank']==probe_updates
 metrics=completed_joint.get('probe_metrics') or {}
 assert metrics['outer']==outer and metrics['actual_adam_steps_per_rank']==[probe_updates]*4
 expected_new=50 if outer<200 and outer%10==0 else 0
 assert metrics['new_adam_steps_per_rank']==expected_new
 assert completed_joint.get('periodic_probe_update') is (expected_new==50)
 reward=actor['reward'];selection=completed_joint['rollout']
 assert selection['questions']==512 and selection['trajectories']==2560
 mean=numeric(reward['score_mean'])
 # Min/max can be absent in a producer; absence is unknown, never fabricated0.
 minimum=optional_numeric(reward.get('score_min'));maximum=optional_numeric(reward.get('score_max'))
 if minimum is None or maximum is None:
  reward_ranks=reward.get('ranks',[])
  if len(reward_ranks)==4 and {row.get('rank') for row in reward_ranks}==set(range(4)):
   if all('score_min' in row and 'score_max' in row for row in reward_ranks):
    minimum=min(numeric(row['score_min']) for row in reward_ranks)
    maximum=max(numeric(row['score_max']) for row in reward_ranks)
 if minimum is not None and maximum is not None:assert minimum<=mean<=maximum
 fraction=unit(selection['actual_selected_mixed_fraction'])
 mixed_count=selection.get('actual_selected_mixed_questions')
 if mixed_count is not None:
  assert type(mixed_count) is int and 0<=mixed_count<=512 and abs(fraction-mixed_count/512)<=1e-12
 counts=reward.get('advantage_summary',{}).get('actual_group_counts',{})
 group_counts=[counts.get(key) for key in ('all_correct','all_wrong','mixed')]
 if all(value is not None for value in group_counts):
  assert all(type(value) is int and value>=0 for value in group_counts) and sum(group_counts)==512
 actual_em=selection.get('actor_selection_outcome_labels_used',False)
 assert type(actual_em) is bool
 candidate_mixed=selection.get('actual_mixed_candidate_count')
 fallback=selection.get('fallback_count')
 if actual_em:
  assert expected_new==50 and type(candidate_mixed) is int and 0<=candidate_mixed<=2560
  assert type(fallback) is int and 0<=fallback<=512
 v['complete_outer_scalar_history'].append([outer,20*outer,last,probe_updates,mean,minimum,maximum,
  fraction,mixed_count,selection.get('selected_strict_correct'),actual_em,candidate_mixed,fallback,
  *group_counts,sha(root/'complete.json')])
 actor_versions[outer]=actor['actor_version']
 if expected_new:
  v['probe_update_history'].append(safe_probe_metrics(metrics))
if latest is not None:
 strict_probe(latest['probe'],n)
 assert actor_versions[n]==latest['actor']['actor_version']

v['qa_milestones']=[]
for outer in (5,50,100,150,200):
 root=b/'outers'/('outer-%06d'%outer)
 qa=document(root,'validation/complete.json')
 if qa is None:
  v['qa_milestones'].append(dict(outer=outer,status='NOT_GENERATED_UNKNOWN_NOT_ZERO'))
  continue
 assert outer<=n and qa['run_id']==h['run_id'] and qa['actor_version']==actor_versions[outer]
 assert qa['qa_count']==512 and len(qa['output_shards'])==4 and len(qa['rank_receipts'])==4
 sources={}
 for name,values in qa['per_source'].items():
  sources[name]=dict(qa_count=values['qa_count'],em=unit(values['em']),f1=unit(values['f1']))
 assert {key:value['qa_count'] for key,value in sources.items()}==dict(nq=256,hotpotqa=256)
 v['qa_milestones'].append(dict(outer=outer,status='EXISTING_NATIVE_QA_COMPLETE',qa_count=512,
  em=unit(qa['em']),f1=unit(qa['f1']),per_source=sources,
  question_source_sha256=hash_value(qa['question_source_sha256']),
  output_shards=[dict(bytes=row['bytes'],sha256=hash_value(row['sha256'])) for row in qa['output_shards']],
  rank_receipt_sha256=[hash_value(row['sha256']) for row in qa['rank_receipts']],
  metadata_sha256=sha(root/'validation/complete.json')))

pipeline=v.get('pipeline_process')
v['pipeline_terminal_error']=None
if pipeline is not None and pipeline['alive'] is False:
 path=b/'logs/native-pipeline.log'
 if path.exists():
  info=regular(path);remaining=MAX_METADATA-metadata_bytes
  full_hash=None;classes=[];coverage='not_read_oversized_log'
  if info.st_size<=remaining:
   digest=hashlib.sha256();tail=b''
   with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(64*1024),b''):
     metadata_bytes+=len(chunk);assert metadata_bytes<=MAX_METADATA
     digest.update(chunk);tail=(tail+chunk)[-65536:]
   full_hash=digest.hexdigest();coverage='full_log_hash_last64KiB_error_class_only'
   tail=tail.decode(errors='replace')
   for line in tail.splitlines():
    match=re.match(r'^(?:[A-Za-z_][A-Za-z0-9_]*\.)*([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)|KeyboardInterrupt|SystemExit):',line)
    if match:classes.append(match.group(1))
  v['pipeline_terminal_error']=dict(log_bytes=info.st_size,log_sha256=full_hash,
   traceback_error_classes=classes[-4:],coverage=coverage,log_body_returned=False)
v['metadata_bytes_read']=metadata_bytes
v['operator_contract']='a800_actorhidden_prefix5_p10_bounded_scalar_observer_v1'
v['no_new_evaluation_or_model_readback']=True
'''


def controller():
    spec = importlib.util.spec_from_file_location('actorhidden_p10_control_for_observer', CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote_code(module):
    marker = '\nemit(v)\n'
    if module.OBSERVE.count(marker) != 1 or not module.OBSERVE.endswith(marker):
        raise ValueError('Published control observer shape changed; do not execute an unbound splice')
    return module.OBSERVE[:-len(marker)] + '\n' + EXTRA + '\nemit(v)\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handle', type=Path, required=True)
    args = parser.parse_args()
    if not args.handle.is_absolute():
        parser.error('--handle must be an absolute deployment JSON path')
    h = json.loads(args.handle.read_text())
    module = controller()
    payload = dict(handle=h)
    started = time.time()
    operator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    control_sha = hashlib.sha256(CONTROL.read_bytes()).hexdigest()
    try:
        result = subprocess.run(module.route(remote_code(module)), input=json.dumps(payload).encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
        record = dict(action='observe',checked_unix=time.time(),elapsed_seconds=time.time()-started,
            exit_code=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    except subprocess.TimeoutExpired as error:
        record = dict(action='observe',checked_unix=time.time(),elapsed_seconds=time.time()-started,
            exit_code=None,timed_out=True,observation_unknown=True,
            stdout=(error.stdout or b'').decode(errors='replace'),
            stderr=(error.stderr or b'').decode(errors='replace'))
    record.update(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
        operator_sha256=operator_sha,control_sha256=control_sha,remote_mutations=False)
    out=ROOT/'preparation'/('a800-actorhidden-prefix5-p10-native-observation-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(record,indent=2)+'\n')
    if record.get('timed_out') or record['exit_code']:
        raise RuntimeError('Bounded observation unknown; retain running trainer and receipt '+str(out))
    if len(result.stdout)>MAX_OUTPUT:
        raise ValueError('Observation output oversized; preserve unknown receipt '+str(out))
    print(json.dumps(dict(receipt=str(out),operator_sha256=operator_sha,
                         result=json.loads(result.stdout))))


if __name__=='__main__':main()
