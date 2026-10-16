#!/usr/bin/env python3
"""Pure local original200 curve extension. No SSH, training, evaluation or legacy report writers."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path

FIELDS = {
 'training_reward_mean':'critic/rewards/mean', 'training_score_mean':'critic/score/mean',
 'raw_within_tree_reward_population_variance':'tree_variance/within_root_raw_reward_population_variance',
 'raw_group_reward_population_variance':'tree_variance/group_raw_reward_population_variance',
 'pure_intra_adv_population_variance':'tree_variance/intra_adv_population_variance',
 'total_actual_adv_population_variance':'tree_variance/group_actual_adv_population_variance',
 'mixed_reward_question_fraction':'tree_variance/mixed_reward_question_fraction',
 'same_reward_question_fraction':'tree_variance/same_reward_question_fraction',
 'mixed_reward_tree_fraction':'tree_variance/mixed_reward_tree_fraction',
 'same_reward_tree_fraction':'tree_variance/same_reward_tree_fraction',
 'generation_wall_seconds':'timing_s/gen'}
OLD = {'LEARNING_CURVE_COMPLETE_THROUGH190.json':(1194660,'5a03840da2fa65bd1c00c0465148796db36bba0262448b631157d6e98dcf92da'),
       'LEARNING_CURVE_COMPLETE_THROUGH190.csv':(208230,'9379ba15d35a210095a6efc970506c97c786c7ecb9c6c7606035136a88f75a6b')}
NAME='LEARNING_CURVE_COMPLETE_THROUGH200_ORIGINAL_RECORDS'
READ={};CREATED={}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode()
def number(value):return type(value) in (int,float) and math.isfinite(value)
def path(name):
 p=Path(name);assert '..' not in p.parts
 p=p if p.is_absolute() else W/p if p.parts[0] in ('outputs','work') else B/p
 assert not p.is_symlink();p=p.resolve(strict=True);assert p.is_relative_to(W);return p
def ref(p):
 p=path(p);raw=p.read_bytes();return dict(path=str(p.relative_to(W)),bytes=len(raw),sha256=sha(raw))
def read(q,mirrors=True):
 p=path(q.get('path') or q['file']);raw=p.read_bytes();assert len(raw)==q['bytes'] and sha(raw)==q['sha256']
 if mirrors:
  assert p.is_relative_to(B)
  for m in M:assert(m/p.relative_to(B)).read_bytes()==raw
 name=str(p);assert name not in READ or READ[name]==raw;READ[name]=raw
 return raw
def bound(q,mirrors=True):return json.loads(read(q,mirrors))
def expand(value):
 if not isinstance(value,dict) or value.get('storage')!='complete_original_field_parts':return value
 out={} if value['container_type']=='mapping' else []
 for q in value['parts']:
  part=bound(q);assert part['schema']=='cumulative_scalar_field_part_v1'
  if isinstance(out,dict):assert not set(out).intersection(part['value']);out.update(part['value'])
  else:out.extend(part['value'])
 assert len(out)==value['original_entries'];return out
def publish(name,raw):
 for root in [B]+M:
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
  try:
   with p.open('xb') as stream:stream.write(raw)
  except FileExistsError:assert p.read_bytes()==raw,'Immutable output differs: '+str(p)
 CREATED[name]=raw;return dict(path=str((B/name).relative_to(W)),bytes=len(raw),sha256=sha(raw))
def check():
 result=[]
 for round_number in (1,2):
  for name,raw in READ.items():assert Path(name).read_bytes()==raw
  for name,raw in CREATED.items():
   for root in [B]+M:assert(root/name).read_bytes()==raw
  for name,(size,digest) in OLD.items():
   for root in [B]+M:
    raw=(root/name).read_bytes();assert(len(raw),sha(raw))==(size,digest)
  result.append(dict(round=round_number,utc=datetime.now(timezone.utc).isoformat(),unchanged_inputs=len(READ),new_three_copy_files=len(CREATED)))
 return result
def raw_top_values(text):
 decoder=json.JSONDecoder();i=1;out={}
 while True:
  while text[i].isspace():i+=1
  if text[i]=='}':break
  key,end=decoder.raw_decode(text,i);i=end
  while text[i].isspace():i+=1
  assert text[i]==':';i+=1
  while text[i].isspace():i+=1
  start=i;_,i=decoder.raw_decode(text,i);out[key]=text[start:i]
  while text[i].isspace():i+=1
  if text[i]==',':i+=1
  else:assert text[i]=='}';break
 return out
def append_raw_array(old,values):
 assert old.startswith('[') and old.endswith(']') and json.loads(old)
 result=old[:-1]+','+','.join(enc(v).decode().strip() for v in values)+']'
 assert json.loads(result)==json.loads(old)+values
 assert result.startswith(old[:-1]+',');return result

def main():
 global W,B,M
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workspace',required=True,type=Path)
 parser.add_argument('--pins',required=True,type=Path);parser.add_argument('--pins-bytes',required=True,type=int);parser.add_argument('--pins-sha256',required=True)
 args=parser.parse_args();assert __debug__;W=args.workspace.resolve(strict=True);B=W/'outputs/tree-nested-branch-handoff-20261013'
 M=[W/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
 manifest=bound(dict(path=str(args.pins),bytes=args.pins_bytes,sha256=args.pins_sha256));assert manifest['schema']=='tree_final200_original_curve_inputs_v1'
 assert path(manifest['source_command']['path'])==Path(__file__).resolve();read(manifest['source_command'],False)
 oldraw=read(dict(path=str(B/'LEARNING_CURVE_COMPLETE_THROUGH190.json'),bytes=OLD['LEARNING_CURVE_COMPLETE_THROUGH190.json'][0],sha256=OLD['LEARNING_CURVE_COMPLETE_THROUGH190.json'][1]));old=json.loads(oldraw);spans=raw_top_values(oldraw.decode())
 csvraw=read(dict(path=str(B/'LEARNING_CURVE_COMPLETE_THROUGH190.csv'),bytes=OLD['LEARNING_CURVE_COMPLETE_THROUGH190.csv'][0],sha256=OLD['LEARNING_CURVE_COMPLETE_THROUGH190.csv'][1]));assert csvraw.endswith(b'\n')
 assert [p['outer'] for p in old['training_points']]==list(range(1,191));assert [d['outer'] for d in old['completed_dev512_points']]==list(range(5,191,5))
 capture=bound(manifest['capture']);assert capture['status']=='ORIGINAL200_RECORDS_READ' and capture['remote_exit_code']==0
 rows=expand(capture['rows']);controls=expand(capture['controls']);assert [r['outer'] for r in rows]==list(range(1,201));assert [d['outer'] for d in controls]==list(range(5,201,5))
 assert controls[0]['value']['phase']=='acceptance';assert all(d['value']['phase']=='periodic' for d in controls[1:])
 costs=bound(manifest['cost_totals']);assert costs['through_effective_outer']==200
 cost_summary=bound(manifest['cost_summary']);assert cost_summary['status']=='ORIGINAL200_COST_AGGREGATED_NOT_FINAL_ACCEPTANCE'
 variance=bound(manifest['variance']);assert variance['status']=='PASS_ACTUAL_ORIGINAL_SCALAR_COVERAGE_THROUGH200'
 assert all(v['actual_covered_outers']==list(range(1,201)) and not v['unknown_outers'] for v in variance['coverage'].values())
 rowparts={}
 for q in capture['rows']['parts']:
  for i,row in enumerate(bound(q)['value']):rowparts[row['outer']]=(q,i)
 cumulative={str(r):0 for r in range(8)};new=[];devs=[];historical_point_field_absence=[]
 for row in rows:
  n=row['outer'];m=row['metric'];ranks=row['rank_records'];assert [r['rank'] for r in ranks]==list(range(8))
  for rank,r in enumerate(ranks):cumulative[str(rank)]+=r['successful_optimizer_calls']
  if n<=190:
   for key,field in FIELDS.items():
    if key in old['training_points'][n-1] and old['training_points'][n-1][key] is not None:assert old['training_points'][n-1][key]==m[field]
    else:historical_point_field_absence.append(dict(outer=n,field=key,original_point_not_rewritten=True))
   continue
  scalar=ref(B/f'final_original_costs200/SCALARS_OUTER{n:04d}.json');s=bound(scalar);assert s['iteration_metrics']==[m] and s['outer']==n
  assert s['evidence_mode']=='RETROSPECTIVE_SOURCE_READBACK' and s['historical_live_poll_created'] is False
  assert row['attempt_record']['actual_completed_optimizer_calls_per_rank']==cumulative
  p=dict(outer=n,source_commit=row['source_commit'],config_sha256=manifest['config_sha256'],run_directory=row['run'],**{key:m[field] for key,field in FIELDS.items()})
  assert all(number(p[key]) for key in FIELDS)
  rankfields={'actual_outer_adam_calls_per_rank':'successful_optimizer_calls','actual_outer_skipped_opportunities_per_rank':'opportunities_skipped','actual_outer_processed_opportunities_per_rank':'opportunities_processed','recorded_finite_grad_norm_counts_per_rank':'finite_grad_norms','recorded_nonzero_grad_norm_counts_per_rank':'nonzero_grad_norms','recorded_local_shard_delta_counts_per_rank':'local_shard_sample_delta'}
  p.update({key:{str(r['rank']):r.get(field) for r in ranks} for key,field in rankfields.items()});p['actual_cumulative_adam_calls_per_rank']=dict(cumulative)
  dev=next((d for d in controls if d['outer']==n),None);testing=m.get('timing_s/testing');wall=m['timing_s/step'];q,i=rowparts[n]
  p.update(guard_skip=None,step_wall_seconds=wall,normal_dev_testing_seconds=testing,
    training_archive_wall_seconds_excluding_current_dev=wall-testing if dev is not None and number(testing) else wall,
    archive_time_origin='Original step wall minus its recorded dev subset only at actual completed dev outer.',
    current_outer_completed_dev512=None,source_receipt=None,source_receipt_sha256=None,
    source_poll=None,source_poll_sha256=None,input_summary=scalar['path'],input_summary_sha256=scalar['sha256'],
    cumulative_cost_source=manifest['cost_totals'] if n==200 else None,whole_run_cost_source=manifest['cost_totals'],
    source_metric_file=scalar['path'],source_metric_file_sha256=scalar['sha256'],source_metric_selector=f'iteration_metrics[step={n}]',
    source_rank_summary_file=q['file'],source_rank_summary_file_sha256=q['sha256'],source_rank_summary_selector=f'value[{i}].rank_records',
    evidence_mode='RETROSPECTIVE_SOURCE_READBACK',original_metric_line_sha256=row['metric_line']['line_sha256'],
    controller_rollout_costs=row['controller']['costs'],candidate_outcome_counts={key:value['candidate_count'] for key,value in row['branch']['candidate_outcomes'].items()},
    no_historical_live_FULL_or_CUM_created=True)
  if dev is not None:
   value=dev['value'];metrics=value['metrics'];assert value['status']=='VALIDATION_COST_COMPLETE' and value['evaluated_rows']==value['dataset_rows']==512
   controlref=ref(B/f'final_original_costs200/DEV_OUTER{n:04d}_ORIGINAL.json');assert read(controlref).decode()==dev['raw_summary_utf8'];assert(controlref['bytes'],controlref['sha256'])==(dev['source']['bytes'],dev['source']['sha256'])
   p['current_outer_completed_dev512']=dict(rows=512,f1=metrics['val/f1/all'],em=metrics['val/em/all'],ledger_sha256=value['ledger_sha256'])
   d=dict(outer=n,source_commit=row['source_commit'],run_directory=row['run'],source_receipt=None,source_receipt_sha256=None,
    source_original_dev_summary=controlref['path'],source_original_dev_summary_sha256=controlref['sha256'],original_control_source=dev['source'],
    status=value['status'],rows=512,phase=value['phase'],f1=metrics['val/f1/all'],em=metrics['val/em/all'],
    nq_f1=metrics['val/f1/nq'],nq_em=metrics['val/em/nq'],hotpotqa_f1=metrics['val/f1/hotpotqa'],hotpotqa_em=metrics['val/em/hotpotqa'],
    ledger_path=Path(dev['source']['path']).name,ledger_sha256=value['ledger_sha256'],normal_dev_testing_wall_seconds=testing,
    source_original_dev_selector='Original COMPLETE JSON control',evidence_mode='RETROSPECTIVE_SOURCE_READBACK');devs.append(d)
  new.append(p)
 assert [p['outer'] for p in new]==list(range(191,201)) and [d['outer'] for d in devs]==[195,200]
 assert cumulative==expand(costs['successful_optimizer_accounting'])['direct_observed_calls_per_rank']
 fresh=copy.deepcopy(old['fresh_chain']);fresh['effective_source_segments'][-1]['actual_outers']=list(range(6,201))
 top=dict(summary_schema='tree_nested_learning_curve_original_records_through200_v1',created_at_utc=manifest['created_at_utc'],
  scope='Actual original-file scalar extension191..200 plus original COMPLETE dev195/200. Old190 array object text and CSV prefix are exact. Missing historical live FULL/CUM remain absent.',
  training_outer_range=[1,200],training_point_count=200,completed_dev512_point_count=40,target_outer=200,fresh_chain=fresh,
  prior_curve=ref(B/'LEARNING_CURVE_COMPLETE_THROUGH190.json'),prior_csv=ref(B/'LEARNING_CURVE_COMPLETE_THROUGH190.csv'),
  original_capture=manifest['capture'],original_costs=manifest['cost_totals'],original_variance=manifest['variance'],actual_input_manifest=ref(args.pins),
  source1ed_actual_prior_preserved=[7,7],source1ed_new_calls=0,remote_operations=False,STATUS_REPORT_alias_canonical_modified=False,
  historical_point_field_absence=historical_point_field_absence,
  complete_original_training_series200={key:[row['metric'][field] for row in rows] for key,field in FIELDS.items()},
  original_final51713_completion_claimed=False,independent_final_acceptance_pending=True,
  limits=['Historical outer1 eight finite parameter samples have zero delta and prove no sampled nonzero change; original nonzero gradients and successful calls are separate.',
          'Whole-run cost200 is bound only as whole-run cost; no per-round CUM191..199 is fabricated.',
          'Interrupted original final and new fast final costs are outside this training/dev curve and remain independently accounted.'])
 training_raw=append_raw_array(spans['training_points'],new);dev_raw=append_raw_array(spans['completed_dev512_points'],devs)
 pieces=[json.dumps(k)+':'+json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False) for k,v in top.items()]
 pieces.extend(['"training_points":'+training_raw,'"completed_dev512_points":'+dev_raw]);curve_raw=('{'+','.join(pieces)+'}\n').encode();curve=json.loads(curve_raw)
 assert curve['training_points'][:190]==old['training_points'] and curve['completed_dev512_points'][:38]==old['completed_dev512_points']
 existing=list(csv.DictReader(io.StringIO(csvraw.decode())));assert len(existing)==228;header=list(existing[0]);buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=header,lineterminator='\n')
 def cell(value):return json.dumps(value,ensure_ascii=False,separators=(',',':')) if isinstance(value,(dict,list)) else value
 for p in new:
  r={k:p.get(k) for k in header};r.update(row_type='training',dev_rows=(p['current_outer_completed_dev512'] or {}).get('rows'),dev_f1=(p['current_outer_completed_dev512'] or {}).get('f1'),dev_em=(p['current_outer_completed_dev512'] or {}).get('em'),cumulative_cost_source_path=(p['cumulative_cost_source'] or {}).get('path'),cumulative_cost_source_sha256=(p['cumulative_cost_source'] or {}).get('sha256'))
  writer.writerow({k:cell(v) for k,v in r.items()})
  d=next((d for d in devs if d['outer']==p['outer']),None)
  if d is not None:
   r={k:None for k in header};r.update(row_type='dev512',outer=d['outer'],normal_dev_testing_seconds=d['normal_dev_testing_wall_seconds'],dev_rows=512,dev_f1=d['f1'],dev_em=d['em'],source_commit=d['source_commit'],source_metric_file=d['source_original_dev_summary'],source_metric_file_sha256=d['source_original_dev_summary_sha256'],input_summary=d['source_original_dev_summary'],input_summary_sha256=d['source_original_dev_summary_sha256'],ledger_sha256=d['ledger_sha256']);writer.writerow(r)
 resultcsv=csvraw+buffer.getvalue().encode();allcsv=list(csv.DictReader(io.StringIO(resultcsv.decode())));assert resultcsv.startswith(csvraw) and len(allcsv)==240 and sum(r['row_type']=='training' for r in allcsv)==200 and sum(r['row_type']=='dev512' for r in allcsv)==40
 products=[publish(NAME+'.json',curve_raw),publish(NAME+'.csv',resultcsv)]
 binding=dict(schema='tree_final200_original_curve_binding_v1',status='ACTUAL_LOCAL_CURVE200_ORIGINAL_RECORDS_COMPLETE_NOT_FINAL_ACCEPTANCE',products=products,
  original_capture=manifest['capture'],cost_totals=manifest['cost_totals'],variance=manifest['variance'],input_manifest=ref(args.pins),generator=ref(Path(__file__)),
  old190_curve_bytes_preserved=True,old190_csv_prefix_bytes=len(csvraw),old190_csv_prefix_sha256=sha(csvraw),
  old_training_array_body_bytes=len(spans['training_points'][:-1].encode()),old_training_array_body_sha256=sha(spans['training_points'][:-1].encode()),
  old_dev_array_body_bytes=len(spans['completed_dev512_points'][:-1].encode()),old_dev_array_body_sha256=sha(spans['completed_dev512_points'][:-1].encode()),
  training_points=200,dev_points=40,csv_data_rows=240,remote_operations=False,old190_no_mutation=True)
 products.append(publish(NAME+'_BINDING.json',enc(binding)));observations=check()
 receipt=dict(schema='tree_final200_original_curve_publication_v1',status=binding['status'],products=products,observations=observations,source_inputs=[ref(name) for name in READ],remote_operations=False)
 products.append(publish(NAME+'_PUBLICATION.json',enc(receipt)));check();print(json.dumps(dict(status=binding['status'],products=products,training_points=200,dev_points=40,csv_data_rows=240)))
if __name__=='__main__':main()
