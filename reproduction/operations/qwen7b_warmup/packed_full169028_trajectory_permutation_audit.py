"""Future-only viewed NQ ownG5 order audit for accepted full169028 L15/P80 MSE.

Staging and metadata tests never load checkpoints or read predictions. A later
root-bound decision fixes the same epoch/state for both seeds before inference.
Only original/reverse/cyclic1 are reported; never select a favorable order.
"""
import argparse,copy,fcntl,gc,gzip,hashlib,inspect,json,os,time,traceback
from pathlib import Path
from . import packed_trajectory_permutation_audit as protocol
from . import online_packed_full169028_MSE_P80 as training
B,A=training.B,training.A
BASE=A/'full169028-L15P80-MSE-viewed-permutation-audit-20261023-v1'
FIXTURE=A/'full169028-policy80-physical-ownG10-20261023-v2/fixture'
PARENTS={
 'seed1':A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3',
 'seed2':A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3'}
STATUS='FULL169028_L15P80_MSE_VIEWED_G5_PERMUTATION_AUDIT_SELECTED_BEFORE_INFERENCE'
ORDERS=protocol.ORDERS;SEPARATOR=protocol.SEPARATOR
read,proof,checked,atomic=training.read,training.proof,training.checked,training.atomic
DEV_SHA='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'
SCORE_TOLERANCE=0.001


def reference(ref):
 return isinstance(ref,dict)and set(('path','bytes','sha256'))<=set(ref)and isinstance(ref['path'],str)and ref['path'].startswith(str(A)+'/')and type(ref['bytes'])is int and ref['bytes']>0 and isinstance(ref['sha256'],str)and len(ref['sha256'])==64 and all(c in'0123456789abcdef'for c in ref['sha256'])


def descriptor(value):
 if value.get('status')!=STATUS or value.get('root_authorized_actual_execution')is not True:raise ValueError('Future-only descriptor is not an authorized bound audit')
 expected=dict(layer=15,prefix=80,head_max_positions=2048,objective='mse',training_questions=169028,M=5,orders=list(ORDERS),questions=2048,score_semantics='4*predicted_population_variance',variance_threshold=.08,score_alignment_absolute_tolerance=SCORE_TOLERANCE,epoch_state_chosen_before_order_audit=True,choose_favorable_trajectory_order=False,final4096_read=False)
 if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact viewed ownG5/P80/MSE geometry and fixed order protocol required')
 if type(value.get('epoch'))is not int or not 1<=value['epoch']<=10 or value.get('state')not in('raw','EMA')or not isinstance(value.get('selected_at'),(int,float)):raise ValueError('Select actual epoch/state before auditing')
 if not reference(value.get('audit_source_publication'))or not reference(value.get('fixture'))or value['fixture']['path']!=str(FIXTURE/'fixture-complete.json'):raise ValueError('Exact accepted physical-v2 fixture and audit source required')
 replicas=value.get('replications',[])
 if [r.get('seed_id')for r in replicas]!=['seed1','seed2']:raise ValueError('Both training seeds must be audited at the same selected epoch/state')
 for r in replicas:
  parent=PARENTS[r['seed_id']]
  for key in('checkpoint','CPU_acceptance','training_complete','training_publication','source_evaluation','source_saved_predictions','ready'):
   if not reference(r.get(key)):raise ValueError('Exact replication source proof required: '+key)
  expected_paths=dict(checkpoint=parent/('checkpoint-epoch-%02d.pt'%value['epoch']),training_complete=parent/'complete.json',training_publication=parent/'source-publication.json',source_evaluation=parent/('evaluation-epoch-%02d.json'%value['epoch']),source_saved_predictions=parent/('epoch-%02d-fixed_validation2048_G5-predictions.jsonl.gz'%value['epoch']),ready=parent/'ready.json')
  if any(r[k]['path']!=str(p)for k,p in expected_paths.items()):raise ValueError('Replication path/epoch mismatch')
 return value


def future_template(publication=None):
 x=dict(status='FUTURE_ONLY_UNSELECTED_NOT_AUTHORIZED',required_status=STATUS,root_authorized_actual_execution=False,selected_at=None,epoch=None,state=None,layer=15,prefix=80,head_max_positions=2048,objective='mse',training_questions=169028,M=5,orders=list(ORDERS),questions=2048,score_semantics='4*predicted_population_variance',variance_threshold=.08,score_alignment_absolute_tolerance=SCORE_TOLERANCE,epoch_state_chosen_before_order_audit=True,choose_favorable_trajectory_order=False,fixture=dict(path=str(FIXTURE/'fixture-complete.json'),bytes=None,sha256=None),audit_source_publication=publication,final4096_read=False,replications=[])
 for seed in PARENTS:
  x['replications'].append(dict(seed_id=seed,**{k:dict(path=None,bytes=None,sha256=None)for k in('checkpoint','CPU_acceptance','training_complete','training_publication','source_evaluation','source_saved_predictions','ready')}))
 return x


def find_ref(value,wanted):
 if isinstance(value,dict):return all(value.get(k)==wanted[k]for k in('path','bytes','sha256'))or any(find_ref(v,wanted)for v in value.values())
 if isinstance(value,list):return any(find_ref(v,wanted)for v in value)
 return False


def evidence(config,replica):
 """Only called by the later authorized GPU audit; not by staging/import/tests."""
 fixture=read(checked(config['fixture']));accepted_ref=proof(FIXTURE/'accepted.json');accepted=read(checked(accepted_ref));ready=read(checked(replica['ready']))
 assert fixture['status']==training.FIXTURE_STATUS and fixture['training_questions']==169028 and fixture['max_real_policy_prefix']==80
 assert accepted['fixture']==config['fixture']==ready['fixture']and ready['independent_acceptance']==accepted_ref
 p20=training.full_acceptance_alignment_ref(accepted);physical=training.full_acceptance_physical_source_ref(accepted);training.validate_physical_source_contract(read(checked(physical)));checked(p20)
 assert ready['P20_alignment']==p20 and ready['physical_source_contract']==physical
 assert fixture['validation_groups']['sha256']==DEV_SHA
 accept=read(checked(replica['CPU_acceptance']));assert accept['status']=='TERMINAL_FULL169028_P80_MSE_CPU_SHA_MODEL_ADAM_RNG_PLAN_PHYSICAL_GATE_ACCEPTED'
 assert accept['checkpoint']==replica['checkpoint']and accept['epoch']==config['epoch']and all(find_ref(accept,r)for r in(config['fixture'],accepted_ref,p20,physical))
 complete=read(checked(replica['training_complete']));assert complete['epochs']==10 and str(complete['status']).endswith('COMPLETE')
 publication=read(checked(replica['training_publication']));assert publication['commit']=='7a8ff6ae8a33324846d85f77ffc28e9ab699c028'
 for name in('online_packed_full169028_MSE_P80.py','packed_variance_probe.py'):
  key='reproduction/operations/qwen7b_warmup/'+name;assert proof(Path(__file__).parent/name)['sha256']==publication['files'][key]['sha256'],'Exact training pack/probe source required'
 ev=read(checked(replica['source_evaluation']));assert ev['epoch']==config['epoch']and ev['cursor']==1321*config['epoch'];checked(replica['source_saved_predictions'])
 return fixture,dict(fixture=config['fixture'],independent_acceptance=accepted_ref,physical_source_contract=physical,P20_alignment=p20,CPU_acceptance=replica['CPU_acceptance'])


def compact(row):
 tokens=row['policy_token_ids'];assert row['prompt']and 0<len(tokens)<=80
 return dict(qid=row['qid'],trajectory_id=row['trajectory_id'],prompt=row['prompt'],policy_token_ids=tokens)


def input_identity(view,records):
 packed=training.pack_group(view,records,SEPARATOR,80)
 return dict(qid=view['qid'],unordered_own_ID_EM_tokens=training.digest(sorted((tid,y,records[tid]['prompt'],records[tid]['policy_token_ids'])for tid,y in zip(view['trajectory_ids'],view['labels']))),ordered_ID_EM=training.digest(list(zip(view['trajectory_ids'],view['labels']))),packed_input_sha256=training.digest(packed),target_variance=view['target_variance'])


def saved_alignment(rows,saved,state):
 if len(rows)!=2048 or len(saved)!=2048 or {r['qid']for r in rows}!=set(saved):raise ValueError('Exact fixed2048 question coverage required')
 delta=[]
 for r in rows:
  s=saved[r['qid']]
  if (r['trajectory_ids'],r['labels'],r['true_mixed'])!=(s['trajectory_ids'],s['labels'],s['true_mixed']):raise ValueError('Original G5 trajectory/EM order differs from saved evaluation')
  delta.append(abs(r[state+'_original']*4-s[state+'_score']))
 return dict(questions=2048,question_trajectory_EM_order_exact=True,score_semantics='4*predicted_population_variance',absolute_score_delta_mean=sum(delta)/len(delta),absolute_score_delta_max=max(delta),absolute_score_tolerance=SCORE_TOLERANCE,passed=max(delta)<=SCORE_TOLERANCE)


def checkpoint_metadata(blob,config,replica):
 b=blob['binding'];seed=9137 if replica['seed_id']=='seed1'else 19137;plan=2026102227 if seed==9137 else 2026102302
 required=dict(architecture='one_question_packed_matched_objective_v1',training_scope='full169028_fresh_official_actor_not_pilot_resume',objective='mse',fixed_m=5,layer=15,capture_layer=14,prefix=80,head_max_positions=2048,training_questions=169028,updates_per_epoch=1321,EMA_decay=.9995,lr=5e-6,init_seed=seed,train_plan_seed=plan,diagnostic_seed=2026102228,fixture=config['fixture'])
 assert all(b.get(k)==v for k,v in required.items())and blob['epoch']==config['epoch']and blob['epoch_complete']is True and blob['next_epoch']==config['epoch']+1 and blob['within_next_epoch_question_position']==0 and blob['execution_phase']=='epoch_complete_pre_evaluation'
 assert all(blob[k]==1321*config['epoch']for k in('cursor','actual_Adam_updates','EMA_updates'))and blob['actor_updates']==0 and len(blob['RNG_by_rank'])==2
 assert blob['init_seed']==seed and blob['train_plan_seed']==plan
 return b


def run(path,sha):
 assert os.uname().sysname=='Linux'and os.environ.get('SLURM_JOB_ID')and proof(path)['sha256']==sha
 cfg=descriptor(read(path));assert cfg['selected_at']<=time.time();pub=read(checked(cfg['audit_source_publication']));code=Path(__file__).resolve().parents[3];assert pub['code_root']==str(code)
 for name,ref in pub['files'].items():checked(dict(ref,path=str(code/name)))
 import numpy as np
 import torch
 import torch.distributed as dist
 from torch import nn
 from transformers import AutoModel,AutoTokenizer
 from .packed_variance_probe import PackedVarianceProbe,PackedVarianceHead
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==2;device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device);dist.init_process_group('nccl')
 guard=(BASE/('audit-rank-%d.lock'%rank)).open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(BASE/'complete.json').exists()
 fixture=read(checked(cfg['fixture']));groups=read(checked(fixture['validation_groups']));assert len(groups)==2048
 views=[protocol.reorder(groups[q],'original')for q in sorted(groups)];assert sum(v['true_mixed']for v in views)==663;mine=views[rank::world];wanted={t for v in mine for t in v['trajectory_ids']};records={}
 train_shards={ref['path']for receipt in fixture['RAW_chunk_receipts']for ref in read(checked(receipt))['token_shards']}
 dev_shards=[(i,ref)for i,ref in enumerate(fixture['token_shards'])if ref['path']not in train_shards];assert len(dev_shards)==2 and {i%2 for i,_ in dev_shards}=={0,1}
 for i,ref in dev_shards:
  if i%world!=rank:continue
  with gzip.open(checked(ref),'rt')as f:
   for line in f:
    row=json.loads(line)
    if row['trajectory_id']in wanted:assert row['role']!='train'and row['trajectory_id']not in records;records[row['trajectory_id']]=compact(row)
 assert set(records)==wanted
 model_dir=B/'assets/qwen25-7b-instruct-a09a35458c70';official=read(model_dir/'complete.json');assert official['revision']=='a09a35458c702b33eeacc393d103063234e8bc28'and official['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'
 tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True);assert tok.encode('\n### trajectory\n',add_special_tokens=False)==SEPARATOR
 actor=AutoModel.from_pretrained(str(model_dir),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False);block=copy.deepcopy(actor.layers[15]).float();rotary=copy.deepcopy(actor.rotary_emb).float();actor.layers=nn.ModuleList(list(actor.layers[:15]));actor.norm=nn.Identity();torch.cuda.empty_cache()
 if rank==0:atomic(BASE/'controller.json',dict(job_id=os.environ['SLURM_JOB_ID'],config=proof(path),epoch=cfg['epoch'],state=cfg['state'],world_size=2,layer=15,prefix=80,head_max_positions=2048,fixed_G5_questions=2048,replications=['seed1','seed2'],orders=list(ORDERS),question_prompt_occurrences_per_group=1,optimizer_created=False,actor_updates=0,probe_updates=0,final4096_read=False,time=time.time()))
 state=cfg['state'];results={};started=time.time()
 for replica in cfg['replications']:
  seed=replica['seed_id'];out=BASE/seed;out.mkdir(exist_ok=True);input_proofs=None
  if rank==0:_,input_proofs=evidence(cfg,replica);checked(replica['checkpoint'])
  dist.barrier();probe=PackedVarianceProbe(copy.deepcopy(block),copy.deepcopy(rotary),'cache_position'in inspect.signature(block.forward).parameters,PackedVarianceHead(max_positions=2048));blob=torch.load(replica['checkpoint']['path'],map_location='cpu',mmap=True,weights_only=False);binding=checkpoint_metadata(blob,cfg,replica);probe.load_state_dict(blob[state],strict=True);del blob;gc.collect();probe.to(device).eval().requires_grad_(False);rows={v['qid']:dict(v)for v in mine};identities={}
  with torch.inference_mode():
   for order in ORDERS:
    own=[protocol.reorder(v,order)for v in mine];identities[order]=[input_identity(v,records)for v in own]
    for offset in range(0,len(own),8):
     part=own[offset:offset+8];packed=[training.pack_group(v,records,SEPARATOR,80)for v in part];width=max(len(r['input_ids'])for r in packed);ids=torch.full((len(part),width),151643,dtype=torch.long,device=device);mask=torch.zeros_like(ids,dtype=torch.bool);pos=torch.zeros_like(ids);types=torch.zeros_like(ids)
     for j,r in enumerate(packed):
      n=len(r['input_ids']);ids[j,:n]=torch.tensor(r['input_ids'],device=device);mask[j,:n]=torch.tensor(r['attention_mask'],device=device);pos[j,:n]=torch.tensor(r['position_ids'],device=device);types[j,:n]=torch.tensor(r['token_types'],device=device)
     features=actor(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True).last_hidden_state.detach()
     with torch.autocast('cuda',dtype=torch.bfloat16):z=probe(dict(features=features,attention_mask=mask,position_ids=pos,token_types=types));scores=torch.sigmoid(z).cpu().tolist()
     for v,logit,s in zip(part,z.float().cpu().tolist(),scores):rows[v['qid']][state+'_'+order]=s/4;rows[v['qid']][state+'_'+order+'_logit']=logit
     atomic(BASE/('progress-rank-%d.json'%rank),dict(seed_id=seed,order=order,own_questions=offset+len(part),expected_own_questions=1024,epoch=cfg['epoch'],state=state,actor_updates=0,probe_updates=0,time=time.time(),seconds=time.time()-started));del features,z
  for order in ORDERS[1:]:
   assert [i['unordered_own_ID_EM_tokens']for i in identities[order]]==[i['unordered_own_ID_EM_tokens']for i in identities['original']]
   assert [i['target_variance']for i in identities[order]]==[i['target_variance']for i in identities['original']]
  gathered=[None]*world;dist.all_gather_object(gathered,dict(rows=list(rows.values()),identities=identities))
  if rank==0:
   merged=sorted([r for x in gathered for r in x['rows']],key=lambda r:r['qid']);saved={}
   with gzip.open(replica['source_saved_predictions']['path'],'rt')as f:
    for line in f:s=json.loads(line);assert s['qid']not in saved;saved[s['qid']]=s
   aligned=saved_alignment(merged,saved,state);metrics={order:protocol.summary(merged,state+'_'+order)for order in ORDERS};comparison={order:protocol.compare(merged,state,order)for order in ORDERS[1:]}
   for order in ORDERS[1:]:comparison[order]['Top512_true_Mixed_change']=metrics[order]['Top512']['Mixed']-metrics['original']['Top512']['Mixed'];comparison[order]['Top512_true_Mixed_fraction_change']=comparison[order]['Top512_true_Mixed_change']/512
   target=out/'dev2048-permutation-predictions.jsonl.gz'
   with gzip.open(target,'wt')as f:
    for row in merged:f.write(json.dumps(row)+'\n')
   result=dict(status='VIEWED_ORDER_AUDIT_COMPLETE'if aligned['passed']else'ORIGINAL_SAVED_EVALUATION_ALIGNMENT_FAILED',epoch=cfg['epoch'],state=state,checkpoint=replica['checkpoint'],input_proofs=input_proofs,original_vs_saved_evaluation=aligned,source_saved_evaluation_metrics=read(replica['source_evaluation']['path'])['metrics']['fixed_validation2048_G5'][state],metrics=metrics,comparisons_to_original=comparison,predictions=proof(target),identity={order:training.digest(sorted([r for x in gathered for r in x['identities'][order]],key=lambda r:r['qid']))for order in ORDERS},same_own_ID_EM_P80_tokens_and_targets_all_orders=True,final4096_read=False);atomic(out/'complete.json',result);results[seed]=result
  dist.barrier();del probe;gc.collect();torch.cuda.empty_cache()
 if rank==0:
  aligned=all(r['original_vs_saved_evaluation']['passed']for r in results.values());atomic(BASE/'complete.json',dict(status='FULL169028_L15P80_MSE_VIEWED_G5_PERMUTATION_AUDIT_COMPLETE'if aligned else'ORIGINAL_SAVED_EVALUATION_ALIGNMENT_FAILED',time=time.time(),config=proof(path),epoch=cfg['epoch'],state=state,replications=results,orders=list(ORDERS),questions=2048,layer=15,prefix=80,head_max_positions=2048,fixed_M=5,score_semantics='population_variance_v; source saved scores are4v',variance_threshold=.08,threshold_fitted=False,order_selected_by_audit=False,optimizer_created=False,actor_updates=0,probe_updates=0,final4096_read=False,seconds=time.time()-started,interpretation='Viewed NQ development order sensitivity only; no new-question or final confirmation.'))
 dist.barrier();dist.destroy_process_group()


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True);args=p.parse_args()
 try:run(args.config,args.config_sha)
 except BaseException:
  if BASE.is_dir():atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(time=time.time(),traceback=traceback.format_exc()))
  raise
if __name__=='__main__':main()
