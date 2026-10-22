"""Read only completed fullP80 dev2048 prediction files; emit aggregates only.

No CP, tensors, final, new inference, Slurm, training mutation or remote writes.
An epoch is included only if its atomically published evaluation JSON exists.
"""
import gzip,hashlib,json,math,os,re,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
ROOTS={
 'seed1':A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3',
 'seed2':A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3'}
FIXTURE=A/'full169028-policy80-physical-ownG10-20261023-v2/fixture/fixture-complete.json'
DEV_SHA='50f1e3a6927d02c0f8cadf1feb5b4229b075b5d099fe31fd0de5c2ae3b149292'
SOURCE='7a8ff6ae8a33324846d85f77ffc28e9ab699c028'

def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):assert proof(ref['path'])==ref;return Path(ref['path'])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def identity(rows):return [dict(qid=r['qid'],trajectory_ids=r['trajectory_ids'],labels=r['labels'])for r in sorted(rows,key=lambda r:r['qid'])]

def run():
 assert os.uname().sysname=='Linux';started=time.time();fixture=read(FIXTURE);assert fixture['status']=='FULL169028_AUTHENTIC_POLICY80_OWN_G10_FIXED2048_READY';ref=fixture['validation_groups'];assert ref['sha256']==DEV_SHA;groups=read(checked(ref));assert len(groups)==2048 and all(len(g['trajectory_ids'])==len(g['labels'])==5 for g in groups.values())
 expected=identity(list(groups.values()));pool_counts=dict(all_wrong=sum(sum(g['labels'])==0 for g in groups.values()),Mixed=sum(0<sum(g['labels'])<5 for g in groups.values()),all_correct=sum(sum(g['labels'])==5 for g in groups.values()));assert pool_counts==dict(all_wrong=1055,Mixed=663,all_correct=330);pool_sha=digest(expected);rows=[];sources={};completed={}
 for seed,root in ROOTS.items():
  pub=read(root/'source-publication.json');assert pub['commit']==SOURCE;job=read(root/'root-submission.json')['job_id'];evals=sorted(root.glob('evaluation-epoch-*.json'));epochs=[]
  for path in evals:
   match=re.fullmatch(r'evaluation-epoch-(\d\d)\.json',path.name)
   if not match:continue
   ep=int(match[1]);assert 0<=ep<=10;evref=proof(path);ev=read(path);assert ev['epoch']==ep and ev['cursor']==ev['actual_Adam_updates']==ev['EMA_updates']==ep*1321;bind=ev['binding'];assert (bind['layer'],bind['prefix'],bind['fixed_m'],bind['training_questions'],bind['objective'],bind['head_max_positions'])==(15,80,5,169028,'mse',2048);assert bind['init_seed']==(9137 if seed=='seed1'else 19137)and bind['train_plan_seed']==(2026102227 if seed=='seed1'else 2026102302)
   predictions=root/('epoch-%02d-fixed_validation2048_G5-predictions.jsonl.gz'%ep);pref=proof(predictions)
   with gzip.open(predictions,'rt')as f:rr=[json.loads(line)for line in f]
   assert len(rr)==2048 and len({r['qid']for r in rr})==2048 and identity(rr)==expected
   for r in rr:assert r['true_mixed']==(0<sum(r['labels'])<5)and r['M']==5 and all(math.isfinite(r[state+'_score'])and 0<=r[state+'_score']<=1 for state in('raw','EMA'))
   for state in('raw','EMA'):
    top=sorted(rr,key=lambda r:(-r[state+'_score'],r['qid']))[:512];hist={str(k):sum(sum(r['labels'])==k for r in top)for k in range(6)};counts=dict(all_wrong=hist['0'],Mixed=sum(hist[str(k)]for k in range(1,5)),all_correct=hist['5']);assert sum(counts.values())==512
    saved=ev['metrics']['fixed_validation2048_G5'][state]['Top512'];assert saved['Mixed']==counts['Mixed']and abs(saved['Mixed_fraction']-counts['Mixed']/512)<1e-12;errors=512-counts['Mixed'];rows.append(dict(seed=seed,job_id=job,epoch=ep,state=state,Top512=counts,EM_count_histogram=hist,Mixed_fraction=counts['Mixed']/512,selected_nonMixed_questions=errors,error_composition=dict(all_wrong_fraction=counts['all_wrong']/errors if errors else None,all_correct_fraction=counts['all_correct']/errors if errors else None),evaluation_Top512_exact_match=True,selected_QID_SHA256=digest([r['qid']for r in top]),fixed_pool_identity_SHA256=pool_sha))
   assert proof(path)==evref and proof(predictions)==pref,'A completed artifact changed during read';sources[seed+':epoch%02d'%ep]=dict(evaluation=evref,predictions=pref);epochs.append(ep)
  completed[seed]=dict(epochs=epochs,status='COMPLETED_EVALUATIONS_AGGREGATED'if epochs else'NO_COMPLETED_EVALUATION_YET',source_publication=proof(root/'source-publication.json'),job_id=job)
 return dict(status='FULL_P80_EXISTING_COMPLETED_DEV_TOP512_CLASS_COMPOSITION_COMPLETE',time=time.time(),completed_epoch_snapshot=completed,rows=rows,source_proofs=sources,pool=dict(questions=2048,own_G5=True,strict_EM_counts=pool_counts,metadata=ref,identity_SHA256=pool_sha,all_seed_epoch_ordered_qid_own_trajectory_EM_exact=True,domain='NQ_already_repeatedly_viewed_development'),ranking='actual saved raw_score or EMA_score descending, ties qid; never EM or classification threshold',e0_is_untrained_initialization_not_completed_training=True,checkpoint_or_model_tensor_opened=False,individual_predictions_exported=False,final4096_read=False,new_inference=False,Slurm_called=False,remote_files_written=False,training_changed=False,candidate_epoch_state_selected=False,seconds=time.time()-started)
print(json.dumps(run(),sort_keys=True,allow_nan=False))
