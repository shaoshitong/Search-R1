"""Frozen linear-readout contribution diagnostic, never a promotion candidate.

The production forward is unchanged. A prehook consumes only its current
microbatch's already-masked flat hidden states and computes FP32 shadows.
Question includes the original bias; prefix has no bias. Prefix representations
still retain causal question context. This is not a question-free input test.
"""
import argparse,fcntl,gzip,json,math,os,platform,shlex,time,traceback
from pathlib import Path
import numpy as np
if __package__:
 from . import score_peer_g10_calibration as s
else:
 import score_peer_g10_calibration as s
TARGET=s.A/'frozen-readout-question-prefix-diagnostic-v1'
STATUS='FROZEN_E4_PEER256_QUESTION_PREFIX_READOUT_DIAGNOSTIC_COMPLETE'
JOB='q7b-readout-contribution'


def sigmoid(z):
 a=np.asarray(z,dtype=np.float64);return np.exp(-np.logaddexp(0.,-a))
def mixed(p):
 p=np.asarray(p,dtype=np.float64)
 if p.ndim!=2 or p.shape[1]!=5 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Five finite own probabilities required')
 return 1-np.prod(p,axis=1)-np.prod(1-p,axis=1)
def corr(x,y):
 x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
 if len(x)!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Paired finite observations required')
 x=x-x.mean();y=y-y.mean();d=float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
 return float(np.dot(x,y)/d) if d else None
def auc(scores,truth):
 a=np.asarray(scores);y=np.asarray(truth,dtype=bool);pos=int(y.sum());neg=len(y)-pos
 if not pos or not neg:return None
 order=np.argsort(a,kind='stable');a=a[order];y=y[order];r=0.;i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[j]==a[i]:j+=1
  r+=float(y[i:j].sum())*(i+1+j)/2;i=j
 return (r-pos*(pos+1)/2)/(pos*neg)
def metrics(scores,labels,qids):
 k=np.asarray(labels).sum(1);y=(k>0)&(k<5);p=np.asarray(scores)>=.5
 tp=int((y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum());fp=int((~y&p).sum())
 order=sorted(range(len(qids)),key=lambda i:(-float(scores[i]),str(qids[i])))[:512]
 return dict(questions=len(qids),true_Mixed=int(y.sum()),K_histogram=np.bincount(k,minlength=6).tolist(),BA=.5*(tp/(tp+fn)+tn/(tn+fp)),TPR=tp/(tp+fn),TNR=tn/(tn+fp),AW_FPR=float(p[k==0].mean()),AC_FPR=float(p[k==5].mean()),AUC=auc(scores,y),Top512=int(y[order].sum()),Top512_fraction=float(y[order].mean()),confusion=dict(tp=tp,fn=fn,tn=tn,fp=fp),threshold=.5,diagnostic_only=True,calibration_fitted=False)


class ContributionHook:
 """Does not return/replace input or output; releases large tensors in hook."""
 def __init__(self,output,question_tokens=273,prefix_tokens=20,dimension=3584):
  if output.in_features!=(question_tokens+prefix_tokens)*dimension or output.out_features!=1:raise ValueError('Exact flattened native readout required')
  self.split=question_tokens*dimension;self.parts=None;self.handle=output.register_forward_pre_hook(self.capture)
 def capture(self,module,args):
  import torch
  if self.parts is not None:raise ValueError('Previous microbatch must be consumed first')
  x=args[0].detach()
  with torch.inference_mode(),torch.autocast(device_type=x.device.type,enabled=False):
   x=x.float();w=module.weight.detach().float();bias=module.bias.detach().float() if module.bias is not None else None
   full=torch.nn.functional.linear(x,w,bias);question=torch.nn.functional.linear(x[:,:self.split],w[:,:self.split],bias);prefix=torch.nn.functional.linear(x[:,self.split:],w[:,self.split:],None)
   self.parts={name:value.flatten().cpu().tolist() for name,value in (('shadow_full',full),('question',question),('prefix',prefix))}
 def pop(self):
  if self.parts is None:raise ValueError('Readout was not called')
  result=self.parts;self.parts=None;return result
 def close(self):self.handle.remove();self.parts=None


def summarize(rows):
 labels=np.asarray([r['strict_em'] for r in rows],dtype=np.int64);qids=[r['qid'] for r in rows];truth=(labels.sum(1)>0)&(labels.sum(1)<5)
 arrays={name:np.asarray([r[name+'_logits'] for r in rows],dtype=np.float64) for name in ('legacy','shadow_full','question','prefix')}
 if any(z.shape!=labels.shape or not np.isfinite(z).all() for z in arrays.values()):raise ValueError('Finite aligned G5 decomposition required')
 probabilities={name:(np.asarray([r['legacy_p_j'] for r in rows]) if name=='legacy' else sigmoid(z)) for name,z in arrays.items()};scores={name:(np.asarray([r['legacy_q'] for r in rows]) if name=='legacy' else mixed(p)) for name,p in probabilities.items()}
 result=dict(components={},scores={name:metrics(value,labels,qids) for name,value in scores.items()})
 for name,z in arrays.items():
  centered=z-z.mean(1,keepdims=True);within=float(np.mean(centered**2));between=float(np.var(z.mean(1)));gaps=[];pgaps=[];pairwins=0.;pairs=0;aucs=[]
  for i,y in enumerate(labels):
   if 0<y.sum()<5:
    c=z[i,y==1];w=z[i,y==0];difference=c[:,None]-w[None,:];wins=float((difference>0).sum()+.5*(difference==0).sum());pairwins+=wins;pairs+=difference.size;aucs.append(wins/difference.size);gaps.append(float(c.mean()-w.mean()));p=probabilities[name][i];pgaps.append(float(p[y==1].mean()-p[y==0].mean()))
  result['components'][name]=dict(within_question_population_variance=within,between_question_mean_variance=between,total_population_variance=float(z.var()),within_fraction=within/(within+between) if within+between else None,mean_absolute_logit=float(np.abs(z).mean()),RMS_logit=float(np.sqrt(np.mean(z*z))),corr_logit_trajectory_EM=corr(z,labels),corr_probability_trajectory_EM=corr(probabilities[name],labels),corr_question_mean_logit_trueMixed=corr(z.mean(1),truth),corr_within_variance_trueMixed=corr(np.var(z,axis=1),truth),corr_q_trueMixed=corr(scores[name],truth),within_correct_minus_wrong_logit=float(np.mean(gaps)),within_correct_minus_wrong_probability=float(np.mean(pgaps)),within_pair_micro_AUC=pairwins/pairs,within_question_macro_AUC=float(np.mean(aucs)),mixed_questions=len(gaps),eligible_within_pairs=pairs)
 q=arrays['question'];p=arrays['prefix'];f=arrays['shadow_full'];cov=float(np.mean((q-q.mean())*(p-p.mean())));den=float(f.var());qc=q-q.mean(1,keepdims=True);pc=p-p.mean(1,keepdims=True);within_cov=float(np.mean(qc*pc));within_den=float(np.mean((f-f.mean(1,keepdims=True))**2));absq=float(np.abs(q).mean());absp=float(np.abs(p).mean())
 reconstruction=np.abs(f-q-p);rounding=np.abs(f-arrays['legacy'])
 result['decomposition']=dict(question_includes_original_bias=True,prefix_has_no_bias=True,prefix_hidden_retains_question_context=True,
  full_FP32_minus_sum_abs_max=float(reconstruction.max()),full_FP32_minus_sum_abs_mean=float(reconstruction.mean()),shadow_FP32_minus_legacy_abs_max=float(rounding.max()),shadow_FP32_minus_legacy_abs_mean=float(rounding.mean()),bitexact_claim=False,
  mean_abs_question_share=absq/(absq+absp),mean_abs_prefix_share=absp/(absq+absp),question_prefix_global_covariance=cov,
  signed_total_variance_question_share=(float(q.var())+cov)/den if den else None,signed_total_variance_prefix_share=(float(p.var())+cov)/den if den else None,
  signed_within_variance_question_share=(float(np.mean(qc*qc))+within_cov)/within_den if within_den else None,signed_within_variance_prefix_share=(float(np.mean(pc*pc))+within_cov)/within_den if within_den else None,
  corr_legacy_q_question_only_q=corr(scores['legacy'],scores['question']),corr_legacy_q_prefix_only_q=corr(scores['legacy'],scores['prefix']))
 return result


def render(code,config_sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=@@JOB@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/logs/scoring-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@RUNTIME@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u @@ENTRY@@ --action run --config-sha256 @@SHA@@
'''.replace('@@JOB@@',JOB).replace('@@BASE@@',str(TARGET)).replace('@@RUNTIME@@',shlex.quote(str(s.TRAIN/('code-'+s.PIN[:12])))).replace('@@ENTRY@@',shlex.quote(str(Path(code)/Path(__file__).name))).replace('@@SHA@@',config_sha)


def prepare():
 s.runtime(s.read(s.TRAIN/'source-publication.json'));e2=s.read(s.A/'E2-same-HF-baseline-20261021/config.json');cal=s.read(s.A/'peer8192-frozen-calibration-scores-v2/config.json');compatible=s.read(s.A/'peer8192-frozen-calibration-scores-v2/CPU-compatible-inputs.json')
 baselines={v['name']:v['external_predictions'] for v in compatible['sources']};checkpoints=cal['checkpoints']
 if {v['name'] for v in checkpoints}!={'E4','peer256'}:raise ValueError('Exactly E4 and original peer256 required')
 groups=s.read(s.checked(e2['groups']))
 if len(groups)!=2048 or any(len(g['trajectory_ids'])!=5 for g in groups):raise ValueError('Only same-HF frozen2048 G5 allowed')
 for d in checkpoints:
  if Path(d['path']).stat().st_size!=d['bytes']:raise ValueError('Source checkpoint size changed')
  s.checked(baselines[d['name']])
 pub=s.read(TARGET/'source-publication.json');code=Path(pub['code_root'])
 for name,item in pub['files'].items():s.checked(dict(path=str(code/name),**item))
 c=dict(role='FROZEN_READOUT_CONTRIBUTION_DIAGNOSTIC_ONLY',checkpoints=checkpoints,baselines=baselines,model_dir=cal['model_dir'],model_config=cal['model_config'],groups=e2['groups'],features=e2['features'],fixed_G5_sha256=e2['fixed_G5_sha256'],validation_complete=e2['validation_complete'],source_publication=s.proof(TARGET/'source-publication.json'),pinned_runtime_publication=s.proof(s.TRAIN/'source-publication.json'),micro_groups=8,question_tokens=273,prefix_tokens=20,dimension=3584,question_bias=True,prefix_bias=False,actor_updates=0,probe_updates=0,optimizer_constructed=False,weights_fitted=False,final_pool_read=False,calibration_grid_constructed=False)
 s.immutable(TARGET/'config.json',c);script=render(code,s.proof(TARGET/'config.json')['sha256']);p=TARGET/'scoring.sbatch'
 if p.exists() and p.read_text()!=script:raise ValueError('Existing immutable sbatch differs')
 if not p.exists():p.write_text(script)
 subprocess_check=['bash','-n',str(p)]
 import subprocess
 subprocess.run(subprocess_check,check=True)
 out=dict(status='FROZEN_READOUT_DIAGNOSTIC_PREPARED',config=s.proof(TARGET/'config.json'),sbatch=s.proof(p),questions=2048,models=['E4','peer256'],tensor_payloads_loaded=False);s.immutable(TARGET/'preparation.json',out);return out


def run(config_sha):
 if platform.system()!='Linux' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Owned remote Slurm GPU required')
 cfg=s.proof(TARGET/'config.json')
 if cfg['sha256']!=config_sha:raise ValueError('Frozen config changed')
 c=s.read(cfg['path']);s.runtime(s.read(s.checked(c['pinned_runtime_publication'])));pub=s.read(s.checked(c['source_publication']))
 for name,item in pub['files'].items():s.checked(dict(path=str(Path(pub['code_root'])/name),**item))
 from reproduction.operations.qwen7b_warmup import probe_ablation_train as u,evaluate_frozen_validation as e
 from reproduction.operations.qwen7b_warmup.model import Probe
 from reproduction.operations.qwen7b_warmup.replay_feature_reader import ReplayFeatureReader
 import torch
 if torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0) or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('One deterministic H100 required')
 groups=s.read(s.checked(c['groups']));manifests=s.read(s.checked(c['features']));s.checked(c['validation_complete']);s.checked(c['model_config']);device=torch.device('cuda',0);torch.cuda.set_device(device)
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
 model=Probe(c['model_dir'],273,20).to(device).eval().requires_grad_(False);reader=ReplayFeatureReader(manifests,26);results={}
 s.atomic(TARGET/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],config=cfg,status='FROZEN_NATIVE_FLATTEN_READOUT_ACCEPTED',input_features=model.output.in_features,question_split=273*3584,time=time.time()))
 try:
  for d in c['checkpoints']:
   name=d['name'];out=TARGET/name;out.mkdir(exist_ok=True);state=torch.load(s.checked(d),map_location='cpu',mmap=True,weights_only=False)
   if name=='E4':e.validate_checkpoint_metadata(d,s.read(s.checked(d['source_receipt'])),s.read(s.checked(d['source_config'])),state)
   elif d['sha256']!=s.CP_SHA or state['cursor']!=256 or state['steps']!=10720 or state['actor_updates']!=0 or state['accepted_hashes']!=d['accepted_hashes']:raise ValueError('Original accepted peer256 differs')
   ema=u.model_mapping_hash(state['ema']);native=state['ema_buffers'];model.load_state_dict(state['ema'],strict=True);u.restore_buffers(model,native);before=u.model_hash(model);buffers_before=u.tensor_hash(u.buffers(model))
   if before!=ema or (name=='peer256' and state['accepted_hashes']['ema']!=ema) or (name=='E4' and any(x['ema_sha256']!=ema for x in state['rank_states'])):raise ValueError('Loaded frozen EMA differs')
   del state
   with gzip.open(s.checked(c['baselines'][name]),'rt') as f:baseline={row['qid']:row for row in map(json.loads,f)}
   if set(baseline)!={g['qid'] for g in groups}:raise ValueError('Sealed legacy source has another cohort')
   hook=ContributionHook(model.output);rows=[];started=time.monotonic();io=compute=0.;p_errors=[];q_errors=[];tmp=out/'decomposition.jsonl.gz.tmp'
   try:
    with gzip.open(tmp,'wt') as stream,torch.inference_mode():
     for start in range(0,len(groups),8):
      part=groups[start:start+8];tick=time.monotonic();batch=reader.get([t for g in part for t in g['trajectory_ids']],device);torch.cuda.synchronize(device);io+=time.monotonic()-tick;tick=time.monotonic()
      with torch.autocast('cuda',dtype=torch.bfloat16):legacy=model(batch).reshape(len(part),5)
      own=hook.pop();z=legacy.cpu().tolist();p=legacy.sigmoid().cpu().tolist();legacy_q=u.group_event_scores(legacy)['mixed_probability'].cpu().tolist();components={k:np.asarray(v).reshape(len(part),5).tolist() for k,v in own.items()}
      for i,g in enumerate(part):
       old=baseline[g['qid']]
       if old['trajectory_ids']!=g['trajectory_ids'] or old['strict_em']!=g['labels']:raise ValueError('Legacy ownIDs/EM differ')
       row=dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],strict_em=g['labels'],legacy_logits=z[i],legacy_p_j=p[i],legacy_q=legacy_q[i],**{k+'_logits':v[i] for k,v in components.items()})
       if not all(math.isfinite(v) for k in ('legacy','shadow_full','question','prefix') for v in row[k+'_logits']):raise ValueError('Nonfinite diagnostic logit')
       p_errors.extend(abs(a-b) for a,b in zip(p[i],old['p_j']));q_errors.append(abs(row['legacy_q']-old['predicted_mixed']));stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
      del batch,legacy,own,components;torch.cuda.synchronize(device);compute+=time.monotonic()-tick
      s.atomic(TARGET/'progress.json',dict(model=name,completed_questions=len(rows),expected_questions=2048,completed_trajectories=5*len(rows),elapsed_seconds=time.monotonic()-started,feature_io_seconds=io,compute_seconds=compute,actor_updates=0,probe_updates=0,time=time.time()))
   finally:hook.close()
   destination=out/'decomposition.jsonl.gz';tmp.replace(destination);summary=summarize(rows);summary['sealed_legacy_reproduction']=dict(source=c['baselines'][name],ownIDs_EM_exact=True,p_abs_mean=float(np.mean(p_errors)),p_abs_max=max(p_errors),q_abs_mean=float(np.mean(q_errors)),q_abs_max=max(q_errors),p_exact_matches=sum(x==0 for x in p_errors),p_total=len(p_errors),bitexact_probabilities=all(x==0 for x in p_errors),source_baseline_unchanged=True)
   if u.model_hash(model)!=before or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Read-only diagnostic changed parameters/gradients')
   u.restore_buffers(model,native)
   if u.tensor_hash(u.buffers(model))!=buffers_before or s.proof(d['path'])!={k:d[k] for k in ('path','bytes','sha256')}:raise ValueError('Source buffers/file changed')
   report=dict(status='ONE_FROZEN_NATIVE_READOUT_CONTRIBUTION_DIAGNOSTIC_COMPLETE',checkpoint=d,config=cfg,rows=s.proof(destination),summary=summary,ema_sha256=ema,ema_buffers_sha256=buffers_before,parameters_unchanged=True,questions=2048,trajectories=10240,elapsed_seconds=time.monotonic()-started,feature_io_seconds=io,compute_seconds=compute,actor_updates=0,probe_updates=0,optimizer_constructed=False,diagnostic_only=True,final_pool_read=False);s.immutable(out/'complete.json',report);results[name]=report
   print(json.dumps(dict(model=name,status=report['status'],questions=2048)),flush=True)
  result=dict(status=STATUS,config=cfg,models=results,question_tokens=273,prefix_tokens=20,dimension=3584,flatten_width=293*3584,parameters_unchanged=True,normal_exit=True,actor_updates=0,probe_updates=0,optimizer_constructed=False,diagnostic_only=True,calibration_fitted=False,final_pool_read=False,time=time.time());s.immutable(TARGET/'complete.json',result);return result
 finally:reader.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','run'),required=True);p.add_argument('--config-sha256');a=p.parse_args();TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
 with (TARGET/'scoring.lock').open('a+') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:r=prepare() if a.action=='prepare' else run(a.config_sha256);print(json.dumps(r if a.action=='prepare' else dict(status=r['status'])),flush=True)
  except BaseException as error:s.atomic(TARGET/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
