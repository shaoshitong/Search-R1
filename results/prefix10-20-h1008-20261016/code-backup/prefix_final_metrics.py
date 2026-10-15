import json,pathlib,sys,datetime,hashlib
repo=pathlib.Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix10-20-h1008-20261016');sys.path.insert(0,str(repo))
from reproduction.h1008_qprobe.management_transport import remote
work=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work');ctl=json.loads((work/'prefix10-20-current-control.json').read_text());tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
script=r'''
import json,pathlib,hashlib,datetime,os,collections
import torch
base=pathlib.Path('__BASE__')
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
summary=dict(role='prefix10_20_complete_small_metrics_and_cpu_checkpoint_audit',base=str(base),observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups={},no_model_forward=True,semantic_reload_performed=False,raw_predictions_downloaded=False)
plans={}
for arm in (10,20):
 root=base/'groups'/('prefix-'+str(arm))/'train';metrics=rows(root/'metrics.jsonl');progress=rows(root/'progress.jsonl');result=read(root/'RESULT.json');claim=read(root/'run-claim.json');binding=claim['binding']
 assert [r['epoch'] for r in metrics]==list(range(21))
 for r in metrics:
  assert r['split']=='new_512_question_diagnostic' and r['actual_adam_steps_per_rank']==[r['epoch']*100]*4
  for kind in ('raw','prior_corrected'):
   m=r[kind];assert m['count']==2560 and m['positive']==559 and m['negative']==2001
  assert r['natural_training_positive_prior']==binding['natural_training_positive_prior'] and r['corrected_logit_shift']==binding['corrected_logit_shift']
 epochs=[r for r in progress if r['stage']=='EPOCH_COMPLETE'];assert [r['epoch'] for r in epochs]==list(range(1,21))
 for r in epochs:assert r['train_draw_count']==25600 and r['train_draw_positive']==12800 and r['actual_adam_steps_per_rank']==[r['epoch']*100]*4
 assert binding['sampling']=='balanced' and binding['balanced_global_positive']==128 and binding['global_batch']==256 and binding['local_batch']==64 and binding['world_size']==4 and binding['accumulation']==1 and binding['epochs']==20
 assert result['actual_adam_steps_per_rank']==[2000]*4 and result['epoch']==20 and result['diagnostic_evaluations']==21 and result['actor_updates']==0 and result['binding_sha256']==claim['binding_sha256']
 plans[str(arm)]=[r['sampler_plan_sha256'] for r in epochs]
 checkpoint=root/'head-epoch-20.pt'
 with checkpoint.open('rb') as f:
  before=os.fstat(f.fileno());h=hashlib.sha256()
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  f.seek(0);cp=torch.load(f,map_location='cpu',weights_only=False);after=os.fstat(f.fileno())
  assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert cp['epoch']==20 and cp['binding']==binding and cp['binding_sha256']==claim['binding_sha256'] and cp['actual_adam_steps_per_rank']==[2000]*4
 states=cp['optimizer']['state'];steps=[int(v['step'].item()) if hasattr(v['step'],'item') else int(v['step']) for v in states.values()]
 assert len(steps)>0 and set(steps)=={2000}
 assert len(states)==sum(len(g['params']) for g in cp['optimizer']['param_groups'])
 rng=cp['rank_rng'];assert sorted(r['rank'] for r in rng)==list(range(4)) and all(r['actual_adam_steps']==2000 for r in rng)
 for r in rng:assert all(k in r for k in ('python','numpy','torch_cpu','torch_cuda','sampler')) and r['torch_cpu'].device.type=='cpu' and r['torch_cuda'].device.type=='cpu'
 audit=dict(path=str(checkpoint),bytes=before.st_size,full_sha256=h.hexdigest(),cpu_load_performed=True,semantic_reload_performed=False,epoch=cp['epoch'],binding_matches_claim=True,optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),optimizer_moment_dtypes=sorted({str(v[k].dtype) for v in states.values() for k in ('exp_avg','exp_avg_sq')}),model_parameter_dtypes=sorted({str(v.dtype) for v in cp['model'].values()}),rng_ranks=sorted(r['rank'] for r in rng),rng_field_names=sorted(set.intersection(*(set(r) for r in rng))),actual_adam_steps_per_rank=cp['actual_adam_steps_per_rank'])
 summary['groups'][str(arm)]=dict(metrics=metrics,progress=progress,result=result,checkpoint_audit=audit,binding_summary={k:binding[k] for k in ('prefix_tokens','seed','sampling','global_batch','local_batch','world_size','accumulation','epochs','steps_per_epoch','planned_adam_steps_per_rank','natural_training_positive_prior','corrected_logit_shift','shared_max_tokens')},all_21_test_counts_verified=True,balanced_draws_verified=True,total_training_draws=sum(r['train_draw_count'] for r in epochs),total_positive_training_draws=sum(r['train_draw_positive'] for r in epochs),metric_files_sha256={n:sha(root/n) for n in ('metrics.jsonl','progress.jsonl','RESULT.json','run-claim.json')})
 del cp,states,rng
summary['same_global_sampler_plan_per_epoch_across_arms']=plans['10']==plans['20'];assert summary['same_global_sampler_plan_per_epoch_across_arms']
summary['same_test_count_and_positive_at_all_42_evaluations']=True
out=base/'preparation'/'final-metric-audit-__TAG__.json'
with out.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(summary=summary,remote_summary_path=str(out),remote_summary_sha256=sha(out))))
'''.replace('__BASE__',ctl['base']).replace('__TAG__',tag)
wrapper="import json,pathlib,subprocess,os,sys\nbase=pathlib.Path("+repr(ctl['base'])+")\nc=json.loads((base/'preparation/launch-config.json').read_text())\nenv=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(base/'repo'))\nr=subprocess.run([str(pathlib.Path(c['environment_root'])/'bin/python'),'-'],input="+repr(script.encode())+",capture_output=True,env=env,timeout=210)\nsys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);sys.exit(r.returncode)\n"
socket=json.loads((work/'qprobe3-management-current.json').read_text())['socket'];res=json.loads(remote(wrapper,work/('prefix10-20-final-metrics-transport-'+tag+'.json'),control_path=socket))
p=work/'prefix10-20-final-metrics.json'
with p.open('x') as f:json.dump(res,f,indent=2,allow_nan=False)
print(json.dumps(dict(local_path=str(p),local_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),remote_path=res['remote_summary_path'],remote_sha256=res['remote_summary_sha256'],same_plan=res['summary']['same_global_sampler_plan_per_epoch_across_arms'],groups={k:dict(final=v['metrics'][-1],epoch0=v['metrics'][0],checkpoint=v['checkpoint_audit'],draws=v['total_training_draws'],positive_draws=v['total_positive_training_draws'],curve=[dict(epoch=r['epoch'],raw_accuracy=r['raw']['accuracy'],raw_bce=r['raw']['bce'],corrected_accuracy=r['prior_corrected']['accuracy'],corrected_bce=r['prior_corrected']['bce']) for r in v['metrics']]) for k,v in res['summary']['groups'].items()})))
