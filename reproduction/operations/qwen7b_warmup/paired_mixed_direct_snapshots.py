"""Finite descriptive paired CPU analysis of existing direct-head predictions.

Reuses the published 1a0f5b4 question-bootstrap primitives. Direct probabilities
are validated directly; they are never given fictitious trajectory p_j values.
"""
import argparse,fcntl,gzip,hashlib,importlib.util,json,math,os,random,resource,time
from pathlib import Path
PAIRS=[('CP128_raw','E0'),('CP128_EMA','E0'),('CP512_raw','E0'),('CP512_raw','CP128_raw'),('CP512_EMA','E0'),('CP512_EMA','CP128_EMA'),('CP512_raw','E2'),('CP512_raw','E4'),('CP512_EMA','E2'),('CP512_EMA','E4')]
NAMES=['E0','CP128_raw','CP128_EMA','CP512_raw','CP512_EMA','E2','E4']
def read(p):return json.loads(Path(p).read_bytes())
def normalized_direct(ref,p):
 with gzip.open(p.checked(ref),'rt')as f:rows=[json.loads(line)for line in f]
 by={}
 for r in rows:
  qid=r['qid'];tids=r['trajectory_ids'];ys=r['strict_em'];q=r['predicted_mixed'];mixed=0<sum(ys)<5
  if qid in by or len(tids)!=5 or len(set(tids))!=5 or len(ys)!=5 or any(type(v)is not int or v not in(0,1)for v in ys)or not math.isfinite(q)or not 0<=q<=1 or r['true_mixed']is not mixed:raise ValueError('Direct own-G5/probability schema differs')
  by[qid]=dict(qid=qid,trajectory_ids=tids,strict_em=ys,score=q,mixed=mixed)
 return by

def scope(models,qids,name,p):
 aligned={key:[by[qid]for qid in qids]for key,by in models.items()};point={key:p.metrics(rows)for key,rows in aligned.items()};columns={a+'__minus__'+b:{key:[]for key in p._deltas(point[a],point[b])}for a,b in PAIRS};rng=random.Random(p.SEED+int(hashlib.sha256(name.encode()).hexdigest()[:8],16));skipped=0
 for rep in range(p.REPLICATES):
  indices=[rng.randrange(len(qids))for unused in qids]
  try:result={key:p.metrics([rows[i]for i in indices])for key,rows in aligned.items()}
  except ValueError:skipped+=1;continue
  for a,b in PAIRS:
   for key,value in p._deltas(result[a],result[b]).items():columns[a+'__minus__'+b][key].append(value)
  if(rep+1)%500==0:print(json.dumps(dict(stage='paired_bootstrap',scope=name,replicates=rep+1,epoch=time.time())),flush=True)
 if skipped==p.REPLICATES:raise ValueError('No valid two-class bootstrap replicate')
 comparisons={a+'__minus__'+b:dict(candidate_minus_baseline=p._deltas(point[a],point[b]),percentile_95_CI={key:[p._quantile(v,.025),p._quantile(v,.975)]for key,v in columns[a+'__minus__'+b].items()},paired_question_resamples=True)for a,b in PAIRS}
 return dict(qids_sha256=p.digest(qids),models=point,comparisons=comparisons,bootstrap_replicates_requested=p.REPLICATES,bootstrap_replicates_used=p.REPLICATES-skipped,single_class_replicates_skipped=skipped)

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);args=a.parse_args();guard=Path(args.output).with_suffix('.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(180,180));started=time.time();manifest=read(args.manifest)
 helper=Path(__file__).with_name('paired_question_stability.py');assert hashlib.sha256(helper.read_bytes()).hexdigest()==manifest['protocol_module']['sha256'];spec=importlib.util.spec_from_file_location('published_question_bootstrap',str(helper));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
 assert manifest['role']==p.ROLE and manifest['questions']==2048 and manifest['bootstrap_replicates']==p.REPLICATES and set(manifest['predictions'])==set(NAMES+['E0_EMA'])
 audits={name:read(p.checked(ref))for name,ref in manifest['accepted_metadata'].items()};direct=audits['direct_identity_audit'];assert direct['status']=='E0_CP128_CP512_DIRECT_MIXED_FIXED512_2048_INDEPENDENT_CPU_METRIC_AUDIT_PASSED'and direct['all12_predictions_same_own_G5_IDs_and_EM']and direct['final4096_read']is False
 # Freeze partition using IDs alone before consuming labels or probabilities.
 with gzip.open(p.checked(manifest['predictions']['E0']),'rt')as f:qids=sorted([json.loads(line)['qid']for line in f],key=str)
 assert len(qids)==2048;frozen=p.freeze_strata(qids);models={key:normalized_direct(manifest['predictions'][key],p)for key in NAMES if key not in('E2','E4')};assert normalized_direct(manifest['predictions']['E0_EMA'],p)==models['E0']
 for key in('E2','E4'):
  with gzip.open(p.checked(manifest['predictions'][key]),'rt')as f:models[key]=p._validated([json.loads(line)for line in f])
 for key,by in models.items():
  assert set(by)==set(qids)
  for qid in qids:assert all(by[qid][field]==models['E0'][qid][field]for field in('trajectory_ids','strict_em','mixed')),'Own-G5/EM mismatch '+key
 for phase,kind,key in [('E0','raw','E0'),('E0','EMA','E0_EMA'),('CP128','raw','CP128_raw'),('CP128','EMA','CP128_EMA'),('CP512','raw','CP512_raw'),('CP512','EMA','CP512_EMA')]:assert manifest['predictions'][key]==direct['prediction_refs'][phase][kind]['fixed_selection2048']
 print(json.dumps(dict(stage='all_seven_models_actual_own_G5_EM_aligned',questions=2048,strata_sha256=frozen['strata_sha256'],epoch=time.time())),flush=True)
 scopes={'all_questions':scope(models,qids,'all_questions',p)}
 for i,subset in enumerate(frozen['strata']):name='hash_stratum_%d'%i;scopes[name]=scope(models,subset,name,p)
 result=dict(status='DESCRIPTIVE_DIRECT_E0_CP128_CP512_AND_E2_E4_PAIRED_QUESTION_STABILITY_COMPLETE',seed=p.SEED,strata=frozen,selection_fraction=.25,threshold=.5,scopes=scopes,comparisons=[dict(candidate=a,baseline=b)for a,b in PAIRS],source_manifest=manifest,source_manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),confidence_interval='pointwise percentile 2.5/97.5; descriptive, not multiplicity-corrected',question_resampling_with_replacement=True,selection_recomputed_per_resample=True,parameters_fitted=False,threshold_selected=False,candidate_selected=False,external_final_read=False,generalization_claim=False,GPU_initialized=False,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 output=Path(args.output)
 if output.exists():raise ValueError('Preserve existing result')
 temporary=output.with_name(output.name+'.partial');temporary.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');os.replace(str(temporary),str(output));print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],peak_RSS_KiB=result['peak_RSS_KiB'])),flush=True)
if __name__=='__main__':main()
