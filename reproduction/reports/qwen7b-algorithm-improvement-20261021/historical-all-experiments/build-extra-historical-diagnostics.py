"""Local archived diagnostics; no remote scoring, models, or final-pool access."""
from pathlib import Path
import json,hashlib,time
H=Path(__file__).resolve().parent;ROOT=H.parents[5];O=ROOT/'outputs/qwen7b-warmup-20261019';rows=[];proofs={}
def read(path):
 data=path.read_bytes();proofs[str(path)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());return json.loads(data)
def base(expt,checkpoint,state,path):return dict(experiment=expt,checkpoint=checkpoint,state=state,train_BA=None,test_BA=None,top_mixed=None,top_k=None,pool_questions=2048,pool_identity='already viewed fixed-HF2048 ownG5 /663 Mixed',pool_mixed=663,train_pool=None,source=str(path),source_sha256=proofs[str(path)]['sha256'],comparability_note='',status='ARCHIVED_ACTUAL_METRIC')
def parse(m):
 p=m.get('pool',m);t=m.get('selected512',{});return dict(test_BA=p.get('mixed_balanced_accuracy',p.get('BA')),AUC=p.get('mixed_roc_auc',p.get('AUC')),top_mixed=t.get('mixed_questions',p.get('Top512')),top_k=512,Mixed_recall=p.get('mixed_recall',p.get('TPR')),nonMixed_recall=p.get('nonmixed_recall',p.get('TNR')))
p=O/'e2-e4-fixed-cpu-actual.json';v=read(p)['complete']
for item in v['results']:
 r=base('E2-E4-fixed-ensemble/'+item['recipe']['id'],'frozen E2 and E4','EMA prediction ensemble',p);r.update(parse(item));r.update(training_budget=dict(new_updates=0),comparability_note='Exactly6 preregistered CPU combinations; no fitting/threshold tuning/new probe. E2/E4 numerical HF backend,663 Mixed pool. Not an independent trained model; all jointgates failed.');rows.append(r)
p=O/'reports/peer-within1-final-independent-audit.json';v=read(p);m=v['evaluations']['selection2048FreshG5HF'];r=base('peer8192-rank/within1','cursor256','EMA',p);r.update(parse(m));r.update(checkpoint_ref=v['final_checkpoint'],train_pool='same8192 genuine G10/B32/4uniform G5 views per question',training_budget=dict(questions=8192,updates=256,source_Adam=10464,final_Adam=10720),job_id=v['job_id'],all_correct_FPR=m['FPRs']['allcorrect'],all_wrong_FPR=m['FPRs']['allwrong'],diagnostic512_BA=v['evaluations']['diagnostic512']['BA'],comparability_note='Actual third matched peer ranking fork, +within_rank1 on authentic distinguishable pairs; same256 updates and E1 state as no-rank/inter1/bothrank1. Raw/fixed-train BA absent.');rows.append(r)
p=O/'reports/readout-contribution-final-delivery.json';v=read(p)['audit']
for model,value in v['models'].items():
 for component,item in value['components'].items():
  m=item['metrics'];r=base('frozen-question-prefix-contribution/'+model+'/'+component,'frozen source checkpoint','frozen EMA ablation',p);r.update(parse(m));r.update(training_budget=dict(new_updates=0),job_id=v['job_id'],all_correct_FPR=m['AC_FPR'],all_wrong_FPR=m['AW_FPR'],comparability_note='No training, only fixed input/readout logit decomposition diagnostic. legacy is original BF16 production; shadow_full is FP32 shadow; question/prefix are fixed components, not new learned models. Prefix hidden states already contain question context; this does not isolate semantic information. No threshold fitting.');rows.append(r)
p=O/'reports/long-prefix-final-delivery.json';horizon=read(p)
result=dict(status='ARCHIVED_EXTRA_DIAGNOSTICS_NO_NEW_SCORING',time=time.time(),rows=rows,source_proofs=proofs,diagnostics_without_BA_or_Top=[dict(experiment='causal-prefix-horizon',status=horizon['status'],trained_probe_updates=0,source=str(p),source_sha256=proofs[str(p)]['sha256'],questions=8192,trajectories=81920,table=horizon['table'],note='Only exact causal-input collision counts, not trained horizon models or accuracy; no BA/Top invented.')],final4096_read=False,new_GPU_or_remote_evaluation=False)
(H/'extra-historical-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=len(rows),no_metric_diagnostic_families=1)))
