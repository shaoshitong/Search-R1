"""Once-only final scalar/CPU-state delivery audit, executed only after completion.

No model forward, GPU work, raw predictions or model tensors are returned to Mac.
Uses the frozen experiment environment with CUDA visibility disabled for torch.load.
All paths/identities come from the existing task-local current control receipt.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

WORK = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
REPO = Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix-only-four-h1008-20261016')
REMOTE = r'''
import json,pathlib,hashlib,datetime,os
import torch
base=pathlib.Path(__BASE__)
expected=__EXPECTED__
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
config=read(base/'preparation/launch-config.json')
assert config['base']==str(base) and config['run_id']==expected['run_id'] and config['source_commit']==expected['commit']
summary=dict(role='multi_turn_prefix_only_four_arm_complete_metrics_cpu_state_audit',base=str(base),run_id=config['run_id'],
 source_commit=config['source_commit'],observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups={},
 no_model_forward=True,semantic_reload_performed=False,no_raw_predictions_downloaded=True,
 metric_scope='Fixednew512questions/G5 natural diagnostic; training metrics are balanced replacement draw statistics')
def logit_auc(predictions):
 pairs=sorted((float(z),int(y)) for _,z,y in predictions)
 npos=sum(y for _,y in pairs);nneg=len(pairs)-npos
 if not npos or not nneg:raise ValueError('AUROC requires both actual classes')
 start=0;positive_rank_sum=0.;unique=0;ties=0
 while start<len(pairs):
  stop=start+1
  while stop<len(pairs) and pairs[stop][0]==pairs[start][0]:stop+=1
  positive_rank_sum+=sum(y for _,y in pairs[start:stop])*((start+1+stop)/2.)
  unique+=1;ties+=(stop-start)*(stop-start-1)//2;start=stop
 return dict(auroc=(positive_rank_sum-npos*(npos+1)/2.)/(npos*nneg),positive=npos,negative=nneg,
  method='Mann-Whitney average tied ranks on raw logits; no sigmoid; fixed epoch20',unique_raw_logits=unique,genuine_logit_tied_pairs=ties)
plans={};scan_shas=set();scans={};diagnostic_identities=set()
for arm in (10,20,40,80):
 root=base/'groups'/('prefix-'+str(arm))/'train'
 metrics=rows(root/'metrics.jsonl');progress=rows(root/'progress.jsonl');result=read(root/'RESULT.json');claim=read(root/'run-claim.json');binding=claim['binding']
 assert [r['epoch'] for r in metrics]==list(range(21))
 for r in metrics:
  assert r['split']=='new_512_question_diagnostic' and r['actual_adam_steps_per_rank']==[r['epoch']*100]*2
  for kind in ('raw','prior_corrected'):
   m=r[kind];assert m['count']==2560 and m['positive']==559 and m['negative']==2001
  assert r['natural_training_positive_prior']==binding['natural_training_positive_prior'] and r['corrected_logit_shift']==binding['corrected_logit_shift']
 epochs=[r for r in progress if r['stage']=='EPOCH_COMPLETE'];assert [r['epoch'] for r in epochs]==list(range(1,21))
 for r in epochs:assert r['train_draw_count']==25600 and r['train_draw_positive']==12800 and r['actual_adam_steps_per_rank']==[r['epoch']*100]*2
 assert binding['prefix_tokens']==arm and binding['sampling']=='balanced' and binding['balanced_global_positive']==128
 assert binding['global_batch']==256 and binding['local_batch']==128 and binding['world_size']==2 and binding['accumulation']==1 and binding['epochs']==20
 assert result['actual_adam_steps_per_rank']==[2000]*2 and result['epoch']==20 and result['diagnostic_evaluations']==21 and result['actor_updates']==0 and result['binding_sha256']==claim['binding_sha256']
 plans[str(arm)]=[r['sampler_plan_sha256'] for r in epochs]
 prediction_path=root/'diagnostic-epoch-20.pt'
 with prediction_path.open('rb') as f:
  before_pred=os.fstat(f.fileno());hp=hashlib.sha256()
  for chunk in iter(lambda:f.read(8*1024**2),b''):hp.update(chunk)
  f.seek(0);prediction_doc=torch.load(f,map_location='cpu',weights_only=False);after_pred=os.fstat(f.fileno())
  assert (before_pred.st_dev,before_pred.st_ino,before_pred.st_size,before_pred.st_mtime_ns)==(after_pred.st_dev,after_pred.st_ino,after_pred.st_size,after_pred.st_mtime_ns)
 assert prediction_doc['epoch']==20 and prediction_doc['binding_sha256']==claim['binding_sha256']
 predictions=prediction_doc['predictions'];assert len(predictions)==2560 and len({v[0] for v in predictions})==2560
 assert all(len(v)==3 and isinstance(v[0],str) and type(v[2])is int and v[2]in(0,1) for v in predictions)
 assert sum(v[2] for v in predictions)==559
 identity_sha=hashlib.sha256(json.dumps(sorted((v[0],v[2]) for v in predictions),separators=(',',':')).encode()).hexdigest()
 diagnostic_identities.add(identity_sha)
 ranking=logit_auc(predictions)
 ranking.update(prediction_file_sha256=hp.hexdigest(),prediction_file_bytes=before_pred.st_size,identity_and_labels_sha256=identity_sha,no_predictions_exported=True)
 del predictions,prediction_doc
 cache=binding['cache_manifest'];spec=cache['spec']
 assert cache['representation']=='multi_turn_retained_policy_prefix_layer27_v1' and spec['prefix_length']==arm and spec['head_max_tokens']==80
 assert spec['prefix_window']=='multi_turn_actual_kept_policy_tokens' and spec['question_hidden_stored'] is False and spec['environment_hidden_stored'] is False
 scanpath=pathlib.Path(cache['inputs_dir'])/'scan-complete.json';scan_sha=sha(scanpath)
 assert scan_sha==cache['inputs_scan_sha256'];scan_shas.add(scan_sha)
 if scan_sha not in scans:
  scan=read(scanpath)
  assert scan['count']==28160 and scan['split_counts']==dict(train=25600,test=2560)
  assert scan['representation']=='multi_turn_retained_policy_prefix_layer27_v1' and scan['prefix_window']=='multi_turn_actual_kept_policy_tokens'
  scans[scan_sha]=dict(path=str(scanpath),sha256=scan_sha,encoder_global_max_tokens=scan['encoder_global_max_tokens'],
   head_max_tokens=scan['head_max_tokens'],max_policy_prefix_tokens=scan['max_policy_prefix_tokens'],prefix_coverage=scan['prefix_coverage'],
   position_provenance=scan['position_provenance'],prefix_source=scan['prefix_source'],split_proof=scan['split_proof'])
 checkpoint=root/'head-epoch-20.pt'
 with checkpoint.open('rb') as f:
  before=os.fstat(f.fileno());h=hashlib.sha256()
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  f.seek(0);cp=torch.load(f,map_location='cpu',weights_only=False);after=os.fstat(f.fileno())
  assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert cp['epoch']==20 and cp['binding']==binding and cp['binding_sha256']==claim['binding_sha256'] and cp['actual_adam_steps_per_rank']==[2000]*2
 states=cp['optimizer']['state'];steps=[int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in states.values()]
 assert len(steps)>0 and set(steps)=={2000} and len(states)==sum(len(g['params']) for g in cp['optimizer']['param_groups'])
 rng=cp['rank_rng'];assert sorted(r['rank'] for r in rng)==list(range(2)) and all(r['actual_adam_steps']==2000 for r in rng)
 for r in rng:assert all(k in r for k in ('python','numpy','torch_cpu','torch_cuda','sampler')) and r['torch_cpu'].device.type=='cpu' and r['torch_cuda'].device.type=='cpu'
 audit=dict(path=str(checkpoint),bytes=before.st_size,full_sha256=h.hexdigest(),cpu_load_performed=True,semantic_reload_performed=False,
  epoch=cp['epoch'],binding_matches_claim=True,optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),
  optimizer_moment_dtypes=sorted({str(v[k].dtype) for v in states.values() for k in ('exp_avg','exp_avg_sq')}),
  model_parameter_dtypes=sorted({str(v.dtype) for v in cp['model'].values()}),rng_ranks=sorted(r['rank'] for r in rng),
  rng_field_names=sorted(set.intersection(*(set(r) for r in rng))),actual_adam_steps_per_rank=cp['actual_adam_steps_per_rank'])
 summary['groups'][str(arm)]=dict(metrics=metrics,progress=progress,result=result,checkpoint_audit=audit,
  binding_summary={k:binding[k] for k in ('prefix_tokens','seed','sampling','global_batch','local_batch','world_size','accumulation','epochs','steps_per_epoch','planned_adam_steps_per_rank','natural_training_positive_prior','corrected_logit_shift','shared_max_tokens')},
  feature_manifest_sha256=sha(pathlib.Path(binding['cache'])/'manifest.json'),input_scan_sha256=scan_sha,fixed_epoch20_ranking=ranking,
  all_21_test_counts_verified=True,balanced_draws_verified=True,total_training_draws=sum(r['train_draw_count'] for r in epochs),
  total_positive_training_draws=sum(r['train_draw_positive'] for r in epochs),
  metric_files_sha256={n:sha(root/n) for n in ('metrics.jsonl','progress.jsonl','RESULT.json','run-claim.json')})
 del cp,states,rng
summary['same_global_sampler_plan_per_epoch_across_four_arms']=all(p==plans['10'] for p in plans.values())
assert summary['same_global_sampler_plan_per_epoch_across_four_arms'] and len(scan_shas)==1 and len(diagnostic_identities)==1
summary['same_epoch20_test_identity_and_labels_across_four_arms']=True
summary['same_test_count_and_positive_at_all_84_evaluations']=True
summary['input_scan_summaries']=scans
out=base/'preparation'/'final-metric-audit-__TAG__.json'
with out.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(summary=summary,remote_summary_path=str(out),remote_summary_sha256=sha(out))))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',default=str(WORK/'prefix-only-four-current-control.json'))
    p.add_argument('--management-receipt',default=str(WORK/'qprobe3-management-current.json'))
    p.add_argument('--output',default=str(WORK/'prefix-only-four-final-metrics.json'))
    p.add_argument('--execute-after-completion',action='store_true',help='Explicitly run the once-only remote final audit after root confirms completion')
    a=p.parse_args()
    if not a.execute_after_completion:p.error('Prepared only; root must confirm actual four-arm completion before execution')
    output=Path(a.output)
    if output.exists():raise FileExistsError('Existing audit output; inspect it instead of replaying')
    ctl=json.loads(Path(a.control).read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    script=REMOTE.replace('__BASE__',repr(ctl['base'])).replace('__EXPECTED__',repr(dict(run_id=ctl['run_id'],commit=ctl['commit']))).replace('__TAG__',tag)
    wrapper="import json,pathlib,subprocess,os,sys\nbase=pathlib.Path("+repr(ctl['base'])+")\nc=json.loads((base/'preparation/launch-config.json').read_text())\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(base/'repo'))\nr=subprocess.run([str(pathlib.Path(c['environment_root'])/'bin/python'),'-'],input="+repr(script.encode())+",capture_output=True,env=env,timeout=210)\nsys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)\n"
    sys.path.insert(0,str(REPO))
    from reproduction.h1008_qprobe.management_transport import remote
    socket=json.loads(Path(a.management_receipt).read_text())['socket']
    res=json.loads(remote(wrapper,WORK/('prefix-only-four-final-metrics-transport-'+tag+'.json'),control_path=socket))
    with output.open('x') as f:json.dump(res,f,indent=2,allow_nan=False)
    print(json.dumps(dict(local_path=str(output),local_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),remote_path=res['remote_summary_path'],remote_sha256=res['remote_summary_sha256'],
     groups={k:dict(final=v['metrics'][-1],result=v['result'],checkpoint=v['checkpoint_audit'],ranking=v['fixed_epoch20_ranking']) for k,v in res['summary']['groups'].items()})))


if __name__=='__main__':main()
