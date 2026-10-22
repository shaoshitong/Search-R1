"""Future-only common fixed trainB2048 diagnostics; no dev selection or final.

The GPU entry adds E1 EMA predictions and a fixed32 viewed-dev adapter replay.
Z E10/raw and selected P80 predictions may be reused only after exact own-G5
identity/label/metadata proof checks; never substitute old trainA diagnostics.
"""
import argparse,copy,fcntl,gc,gzip,hashlib,inspect,json,os,time,traceback
from pathlib import Path
from . import score_frozen_final4096 as native
from . import online_packed_full169028_MSE_P80 as packing
B,A=packing.B,packing.A
BASE=A/'selected-state-fixed-trainB2048-unified-20261023-v1'
FULL=A/'full169028-policy80-physical-ownG10-20261023-v2/fixture'
CAUSAL=A/'online-mixed-layer-prefix-20261022-v1/fixture-complete.json'
TRAIN_SHA='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b'
CAUSAL_SHA='2aa8f1de7dbb64412bb03088803b1c69d4e61fa63476e8c10536d2bd20911d5d'
DEV_SHA='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'
Z={'Z_seed1':A/'online-packed-full169028-M3to10-P20-20261022-v1/z_mse_anchor','Z_seed2':A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1'}
E1_EVAL=A/'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/complete.json'
E1_DEV=A/'joint-mixed-direct-official-169028-b128-v1/evaluations/epoch-01/EMA-fixed_selection2048.jsonl.gz'
STATUS='FIXED_TRAIN_B2048_ONLY_SELECTED_STATE_UNIFIED_DIAGNOSTICS_ROOT_AUTHORIZED'
REPLAY_TOLERANCE=.001
read,proof,checked,atomic=packing.read,packing.proof,packing.checked,packing.atomic


def valid_ref(ref):
 return isinstance(ref,dict)and set(('path','bytes','sha256'))<=set(ref)and isinstance(ref['path'],str)and ref['path'].startswith(str(B)+'/')and type(ref['bytes'])is int and ref['bytes']>0 and isinstance(ref['sha256'],str)and len(ref['sha256'])==64


def descriptor(x):
 if x.get('status')!=STATUS or x.get('root_authorized_actual_execution')is not True:raise ValueError('Unselected template cannot execute')
 expected=dict(train_questions=2048,fixed_M=5,fixed_train_SHA256=TRAIN_SHA,trainA_used=False,epoch_or_threshold_selected_on_trainB=False,threshold_fitted=False,final4096_read=False,adapter_replay=dict(model='E1_EMA',questions=32,selection='first32_sorted_existing_fixed_dev_qids',purpose='adapter_consistency_only_no_selection',absolute_probability_tolerance=REPLAY_TOLERANCE))
 if any(x.get(k)!=v for k,v in expected.items()):raise ValueError('Exact fixedtrainB, no selection/tuning and predeclared32 adapter replay required')
 for key in('source_publication','full_fixture','causal_fixture','E1_dev_evaluation','E1_dev_predictions'):
  if not valid_ref(x.get(key)):raise ValueError('Source proof required: '+key)
 if x['full_fixture']['path']!=str(FULL/'fixture-complete.json')or x['causal_fixture']['path']!=str(CAUSAL)or x['causal_fixture']['sha256']!=CAUSAL_SHA:raise ValueError('Exact accepted authentic P80 and native causal fixtures required')
 if x['E1_dev_evaluation']['path']!=str(E1_EVAL)or x['E1_dev_evaluation']['sha256']!='3d832d00b2d40eaf5ffad4568c1785c84270b0a81e033c57e6c7030a93cd0de8'or x['E1_dev_predictions']['path']!=str(E1_DEV)or x['E1_dev_predictions']['sha256']!='1d821ef5059710213681b4eab3f0b55ad499485ae0b2a774a7684ffc6e9910ea':raise ValueError('Replay original existing same-CP E1 predictions only')
 models=x.get('models',[])
 if [m.get('name')for m in models[:3]]!=['E1_EMA','Z_seed1','Z_seed2']:raise ValueError('E1 EMA plus both Z E10/raw replications required')
 baseline=models[0]
 if (baseline.get('checkpoint'),baseline.get('state'),baseline.get('mode'),baseline.get('family'),baseline.get('layer'),baseline.get('prefix'),baseline.get('threshold'))!=(native.E1,'ema','score','native_direct',27,20,.5):raise ValueError('Exact legacy E1 EMA adapter and fixed .5 diagnostic threshold required')
 for model in models[1:]:
  for key in('checkpoint','CPU_acceptance','training_publication','training_complete','evaluation','saved_predictions'):
   if not valid_ref(model.get(key)):raise ValueError('Complete source/CPU proof required: '+key)
  if model['name']in Z:
   parent=Z[model['name']]
   if (model.get('state'),model.get('epoch'),model.get('family'),model.get('mode'),model.get('layer'),model.get('prefix'),model.get('threshold'))!=('raw',10,'packed_Z','reuse_saved_predictions',21,20,.08):raise ValueError('Both Z replications fixed to E10/raw, no new selection')
   paths=dict(checkpoint=parent/'checkpoint-epoch-10.pt',training_complete=parent/'complete.json',evaluation=parent/'evaluation-epoch-10.json',saved_predictions=parent/'epoch-10-fixed_train2048_G5-predictions.jsonl.gz')
   if any(model[k]['path']!=str(v)for k,v in paths.items()):raise ValueError('Wrong Z epoch or training prediction source')
  elif (model.get('family'),model.get('mode'),model.get('layer'),model.get('prefix'),model.get('threshold'))!=('packed_P80_MSE','reuse_saved_predictions',15,80,.08)or model.get('state')not in('raw','EMA')or type(model.get('epoch'))is not int or not 1<=model['epoch']<=10:raise ValueError('Optional selected P80 uses existing same-trainB MSE predictions only')
 if len({m['name']for m in models})!=len(models):raise ValueError('Repeated model names')
 return x


def template(publication=None):
 ref=lambda p:dict(path=str(p),bytes=None,sha256=None)
 x=dict(status='FUTURE_ONLY_NOT_AUTHORIZED',required_status=STATUS,root_authorized_actual_execution=False,source_publication=publication,train_questions=2048,fixed_M=5,fixed_train_SHA256=TRAIN_SHA,trainA_used=False,epoch_or_threshold_selected_on_trainB=False,threshold_fitted=False,final4096_read=False,full_fixture=ref(FULL/'fixture-complete.json'),causal_fixture=dict(path=str(CAUSAL),bytes=None,sha256=CAUSAL_SHA),E1_dev_evaluation=ref(E1_EVAL),E1_dev_predictions=ref(E1_DEV),adapter_replay=dict(model='E1_EMA',questions=32,selection='first32_sorted_existing_fixed_dev_qids',purpose='adapter_consistency_only_no_selection',absolute_probability_tolerance=REPLAY_TOLERANCE),models=[dict(name='E1_EMA',checkpoint=native.E1,state='ema',mode='score',family='native_direct',layer=27,prefix=20,epoch=1,threshold=.5)])
 for name,parent in Z.items():
  x['models'].append(dict(name=name,checkpoint=ref(parent/'checkpoint-epoch-10.pt'),CPU_acceptance=ref(A/'ACTUAL_CPU_ACCEPTANCE.json'),training_publication=ref((parent.parent if name=='Z_seed1'else parent)/'source-publication.json'),training_complete=ref(parent/'complete.json'),evaluation=ref(parent/'evaluation-epoch-10.json'),saved_predictions=ref(parent/'epoch-10-fixed_train2048_G5-predictions.jsonl.gz'),state='raw',epoch=10,family='packed_Z',mode='reuse_saved_predictions',layer=21,prefix=20,threshold=.08))
 return x


def explicit(group):
 labels=group['labels'];assert len(labels)==len(group['trajectory_ids'])==5;mean=sum(labels)/5
 return dict(group,M=5,true_mixed=0<sum(labels)<5,target_variance=mean*(1-mean))


def identity(rows):
 return [dict(qid=r['qid'],trajectory_ids=r['trajectory_ids'],labels=r['labels'])for r in sorted(rows,key=lambda r:r['qid'])]


def match_saved(rows,views):
 if len(rows)!=2048 or len({r['qid']for r in rows})!=2048 or identity(rows)!=identity(views):raise ValueError('Saved fixed trainA/other pool cannot substitute exact trainB ownG5')
 for r in rows:
  if r['true_mixed']!=(0<sum(r['labels'])<5):raise ValueError('Saved Mixed label differs from actual ownG5')
 return sorted(rows,key=lambda r:r['qid'])


def find_ref(value,wanted):
 if isinstance(value,dict):return all(value.get(k)==wanted[k]for k in('path','bytes','sha256'))or any(find_ref(v,wanted)for v in value.values())
 if isinstance(value,list):return any(find_ref(v,wanted)for v in value)
 return False


def flattened(record,n=80):
 tokens=[]
 for seg in record['segments']:
  take=min(n-len(tokens),seg['policy_length']);tokens+=seg['context_tokens'][seg['request_length']:seg['request_length']+take]
  if len(tokens)==n:break
 return tokens


def compare_causal_compact(causal,compact,labels):
 if set(causal)!=set(compact)or set(causal)!=set(labels):raise ValueError('Exact causal/compact own trajectory coverage required')
 for tid,r in causal.items():
  s=compact[tid]
  if r['label']!=labels[tid]or r['qid']!=s['qid']or r['prompt']!=s['prompt']or flattened(r)!=s['policy_token_ids']:raise ValueError('Actual own EM/prompt/real P80 causal-to-compact source differs')
 return dict(trajectories=len(causal),own_EM_question_prompt_and_true_P80_exact=True,native_keeps_actual_causal_segments=True,compact_not_fabricated_into_native_context=True)


def rank_values(values):
 import numpy as np
 order=np.argsort(values,kind='stable');r=np.empty(len(values));i=0
 while i<len(order):
  j=i+1
  while j<len(order)and values[order[j]]==values[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2;i=j
 return r


def metrics(rows,variance):
 import numpy as np
 rows=sorted(rows,key=lambda r:r['qid']);assert len(rows)==2048;y=np.array([r['true_mixed']for r in rows],bool);s=np.array([r['score']for r in rows]);target=np.array([r['target_variance']for r in rows]);assert np.isfinite(s).all();threshold=.08 if variance else.5;p=s>=threshold;pos=y.sum();neg=len(y)-pos;top=sorted(rows,key=lambda r:(-r['score'],r['qid']))[:512];out=dict(questions=2048,Mixed_questions=int(pos),Mixed_BA=float(.5*(p[y].mean()+(~p[~y]).mean())),Mixed_recall=float(p[y].mean()),Mixed_AUC=float((rank_values(s)[y].sum()-pos*(pos-1)/2)/(pos*neg)),nonMixed_FPR=float(p[~y].mean()),threshold=threshold,threshold_fitted=False,confusion=dict(tp=int((p&y).sum()),tn=int((~p&~y).sum()),fp=int((p&~y).sum()),fn=int((~p&y).sum())),Top512=dict(Mixed=sum(r['true_mixed']for r in top),Mixed_fraction=sum(r['true_mixed']for r in top)/512,selected_qids_SHA256=packing.digest([r['qid']for r in top])),classwise={})
 for name,mask in [('all_wrong',np.array([sum(r['labels'])==0 for r in rows])),('Mixed',y),('all_correct',np.array([sum(r['labels'])==5 for r in rows]))]:out['classwise'][name]=dict(questions=int(mask.sum()),predicted_Mixed_fraction=float(p[mask].mean()))
 if variance:out.update(loss_type='variance_MSE(v)',fixed_diagnostic_loss=float(((s-target)**2).mean()),score_semantics='population_EM_variance_v')
 else:
  q=np.clip(s,np.finfo(float).eps,1-np.finfo(float).eps);out.update(loss_type='direct_Mixed_BCE',fixed_diagnostic_loss=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()),score_semantics='direct_Mixed_probability')
 return out


def reused(model,views):
 accept=read(checked(model['CPU_acceptance']));assert str(accept['status']).endswith('ACCEPTED')and find_ref(accept,model['checkpoint']);done=read(checked(model['training_complete']));assert done['epochs']==10 and str(done['status']).endswith('COMPLETE');checked(model['training_publication']);evaluation=read(checked(model['evaluation']));assert evaluation['epoch']==model['epoch']
 source_fixture=accept.get('source_fixture')or accept.get('binding',{}).get('fixture')or accept.get('accepted_input',{}).get('fixture');fixture=read(checked(source_fixture));assert fixture['fixed_train_G5']['sha256']==TRAIN_SHA
 if model['family']=='packed_Z':assert (accept['loss'],accept['layer'],accept['prefix'],accept['cursor'])==('z_mse_anchor',21,20,13210)and accept['source_training_publication']==model['training_publication']
 if 'source_complete'in accept:assert accept['source_complete']==model['training_complete']
 with gzip.open(checked(model['saved_predictions']),'rt')as f:rows=match_saved([json.loads(line)for line in f],views)
 field=model['state']+('_predicted_variance'if model['family']=='packed_Z'else'_score');factor=1 if model['family']=='packed_Z'else .25
 for row in rows:row['score']=row[field]*factor
 return rows,dict(saved_prediction_identity_exact=True,mode='reuse_exact_saved_trainB_predictions',checkpoint=model['checkpoint'],CPU_acceptance=model['CPU_acceptance'],source_evaluation=model['evaluation'],saved_predictions=model['saved_predictions'],source_evaluation_fixed_training_metrics=evaluation['metrics']['fixed_train2048_G5'][model['state']],no_GPU_forward_for_reused_model=True)


def replay_alignment(rows,saved):
 import math
 if len(rows)!=32 or len(saved)!=32 or identity(rows)!=identity(saved):raise ValueError('Exact predeclared32 viewed ownG5 replay identity required')
 orig={r['qid']:r for r in saved};d=[abs(1/(1+math.exp(-r['logit']))-orig[r['qid']]['predicted_mixed'])for r in rows]
 return dict(questions=32,selection='first32_sorted_existing_fixed_dev_qids',same_E1_EMA_CP_and_own_G5=True,absolute_probability_delta_mean=sum(d)/32,absolute_probability_delta_max=max(d),absolute_probability_tolerance=REPLAY_TOLERANCE,passed=max(d)<=REPLAY_TOLERANCE,used_for_model_or_threshold_selection=False)


def run(path,sha):
 assert os.uname().sysname=='Linux'and os.environ.get('SLURM_JOB_ID')and proof(path)['sha256']==sha;cfg=descriptor(read(path));pub=read(checked(cfg['source_publication']));code=Path(__file__).resolve().parents[3];assert pub['code_root']==str(code)
 for name,ref in pub['files'].items():checked(dict(ref,path=str(code/name)))
 full=read(checked(cfg['full_fixture']));accept=read(FULL/'accepted.json');assert accept['fixture']==cfg['full_fixture']and accept['actual_physical_execution_and_seal_source_exact']is True and accept['fresh_full_P20_prefix_prompt_and_own_EM_exact']is True;checked(accept['physical_source_contract']);checked(accept['P20_alignment']);assert full['pilot_train_own_policy80_differing_rows']==0
 assert full['fixed_train_G5']['sha256']==TRAIN_SHA and full['validation_groups']['sha256']==DEV_SHA;views=[explicit(v)for v in read(checked(full['fixed_train_G5']))];assert len(views)==2048 and sum(v['true_mixed']for v in views)==888
 dev=read(checked(full['validation_groups']));replay=[explicit(dev[q])for q in sorted(dev)[:32]];causal_fixture=read(checked(cfg['causal_fixture']));contract=read(checked(full['source_contract']));pilot=read(checked(contract['pilot_fixture']));pilot_accept=read(checked(contract['pilot_acceptance']));assert pilot_accept['fixture']==contract['pilot_fixture']and pilot_accept['source_policy80_exact_match']is True
 checked(cfg['E1_dev_evaluation']);checked(cfg['E1_dev_predictions'])
 import torch
 import torch.distributed as dist
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl');guard=(BASE/('score-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(BASE/'complete.json').exists();mine=sorted(views,key=lambda r:r['qid'])[rank::world];myreplay=replay[rank::world];wanted={t:y for v in mine+myreplay for t,y in zip(v['trajectory_ids'],v['labels'])};causal={};compact={}
 for ref in causal_fixture['token_shards']:
  with gzip.open(checked(ref),'rt')as f:
   for line in f:
    r=json.loads(line)
    if r['trajectory_id']in wanted:assert r['trajectory_id']not in causal;causal[r['trajectory_id']]=r
 for ref in pilot['token_shards']:
  with gzip.open(checked(ref),'rt')as f:
   for line in f:
    r=json.loads(line)
    if r['trajectory_id']in wanted:assert r['trajectory_id']not in compact;compact[r['trajectory_id']]=r
 alignment=compare_causal_compact(causal,compact,wanted);assert len(causal)==len(wanted)==5200
 if rank==0:atomic(BASE/'controller.json',dict(time=time.time(),job_id=os.environ['SLURM_JOB_ID'],questions=2048,role='fixed_trainB_not_trainA',fixed_train_SHA256=TRAIN_SHA,models=[m['name']for m in cfg['models']],replay_questions=32,development2048_rescored=False,final4096_read=False,actor_updates=0,probe_updates=0,epoch_or_threshold_selected_on_trainB=False))
 started=time.time();out=native.score_variant(dict(checkpoint=native.E1,state_key='ema',layer=27,prefix=20,name='E1_EMA_trainB_and_fixed32_replay'),mine+myreplay,causal,device,BASE/('progress-rank-%d.json'%rank),rank);gathered=[None]*world;dist.all_gather_object(gathered,dict(rows=out,alignment=alignment))
 if rank==0:
  import math
  allrows=[r for x in gathered for r in x['rows']];mainqid={v['qid']for v in views};base_rows=[r for r in allrows if r['qid']in mainqid];replay_rows=sorted([r for r in allrows if r['qid']not in mainqid],key=lambda r:r['qid']);assert len(base_rows)==2048 and len(replay_rows)==32
  for row in base_rows:row['score']=1/(1+math.exp(-row['logit']))
  with gzip.open(cfg['E1_dev_predictions']['path'],'rt')as f:saved={r['qid']:r for r in(json.loads(line)for line in f)}
  expected=sorted(saved)[:32];assert expected==[r['qid']for r in replay_rows];replay_result=replay_alignment(replay_rows,[saved[q]for q in expected]);results={'E1_EMA':dict(metrics=metrics(base_rows,False),mode='actual_same_E1_EMA_frozen_causal_prefill',checkpoint=native.E1)};predictions={'E1_EMA':base_rows}
  for model in cfg['models'][1:]:
   rows,source=reused(model,views);results[model['name']]=dict(metrics=metrics(rows,True),source=source);predictions[model['name']]=rows
  for name,rows in predictions.items():
   target=BASE/(name+'-fixed-trainB2048-predictions.jsonl.gz')
   with gzip.open(target,'wt')as f:
    for row in sorted(rows,key=lambda r:r['qid']):f.write(json.dumps(row)+'\n')
   results[name]['predictions']=proof(target)
  atomic(BASE/'complete.json',dict(status='FIXED_TRAIN_B2048_UNIFIED_SELECTED_STATE_DIAGNOSTICS_COMPLETE'if replay_result['passed']else'E1_LEGACY_ADAPTER_REPLAY_ALIGNMENT_FAILED',time=time.time(),config=proof(path),fixed_train_G5=full['fixed_train_G5'],role='fixed_trainB_not_old_trainA',pool=dict(questions=2048,Mixed=888,all_wrong=950,all_correct=210),models=results,E1_adapter_fixed32_viewed_replay=replay_result,input_alignment_by_rank=[x['alignment']for x in gathered],existing_trainA_history_preserved=True,epoch_or_threshold_selected_on_trainB=False,development2048_rescored=False,new_rollouts=0,actor_updates=0,probe_updates=0,final4096_read=False,seconds=time.time()-started))
 dist.barrier();dist.destroy_process_group()


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);args=p.parse_args()
 try:run(args.config,args.config_sha)
 except BaseException:
  if BASE.is_dir():atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(time=time.time(),traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
