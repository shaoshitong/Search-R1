"""One actual published E1 ownG5 plan vs canonicalG10; metadata only.

No sampler is rerun. No rollout, tensor, checkpoint or final pool is opened.
The goal6 counting functions are reused from the immutable accepted CPU helper.
"""
import argparse,hashlib,importlib.util,json,math,resource,time,traceback
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
BASE=A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3'
OUT=A/'fullMSE-750190-actual-E1-sampler-diagnostic-20261023-v1'
GOAL6=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1/code-693a81dfbbae/accept_full169028_L15P80_MSE_checkpoint.py'
GOAL6_SHA='40e308e6a3c65bc0132c7b249939059775f4513d867576e8df74fa7aabc61448'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(ref):
 assert proof(ref['path'])==ref;return Path(ref['path'])
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def atomic(p,v):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-sha',required=True);args=parser.parse_args();assert proof(__file__)['sha256']==args.source_sha;started=time.time();OUT.mkdir(exist_ok=True);assert str(read(BASE/'root-submission.json')['job_id'])=='750190';pub=read(BASE/'source-publication.json');controller=read(BASE/'controller.json');binding=controller
 assert all(binding.get(k)==v for k,v in dict(training_questions=169028,init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228,fixed_m=5,sampling_independent_of_EM=True,class_weighting=False).items())
 ready=read(BASE/'ready.json');assert ready['publication']==proof(BASE/'source-publication.json');fixture=read(checked(binding['fixture']));accepted=read(checked(ready['independent_acceptance']));assert accepted['fixture']==binding['fixture']and accepted['actual_physical_execution_and_seal_source_exact']and accepted['fresh_full_P20_prefix_prompt_and_own_EM_exact']
 groups=read(checked(fixture['train_groups']));assert len(groups)==169028
 canonical_path=A/'joint169028-native-training-bank-v1/groups.json';raw=read(canonical_path);canonical=raw['train']if'train'in raw else raw
 if isinstance(canonical,list):canonical={g['qid']:g for g in canonical}
 assert len(canonical)==169028 and set(canonical)==set(groups)
 for q,g in groups.items():
  c=canonical[q];assert g['trajectory_ids']==c['trajectory_ids']and g['labels']==c['labels']and g['question_hash']==c['question_hash']
 del canonical,raw
 planpath=BASE/'sampling-plan-epoch-01.json';before=planpath.stat();saved=read(planpath);rows=saved['rows'];assert all(saved[k]==v for k,v in dict(epoch=1,init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228).items());assert digest(rows)==saved['plan_sha256'];assert len(rows)==169028 and len({r['qid']for r in rows})==169028 and{r['qid']for r in rows}==set(groups)
 assert proof(GOAL6)['sha256']==GOAL6_SHA;spec=importlib.util.spec_from_file_location('immutable_goal6_counter',GOAL6);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);counts=helper.new_sampling_diagnostic();K5_by_K10={k:[0]*6 for k in range(11)};inclusion=[0]*10;position_counts=[[0]*10 for _ in range(5)];global_ids=set();seen_selected=0;observed_classes=dict(all_wrong=0,Mixed=0,all_correct=0)
 for r in rows:
  q=r['qid'];g=groups[q];assert g['qid']==q and len(g['trajectory_ids'])==len(g['labels'])==10 and len(set(g['trajectory_ids']))==10 and all(y in(0,1)for y in g['labels']);own=dict(zip(g['trajectory_ids'],g['labels']));ids=r['trajectory_ids'];labels=r['labels'];assert r['M']==len(ids)==len(labels)==5 and len(set(ids))==5 and labels==[own[t]for t in ids];assert not(global_ids&set(ids));global_ids.update(ids);seen_selected+=5;positive=sum(labels);k10=sum(g['labels']);mean=positive/5;assert r['true_mixed']==(0<positive<5)and r['target_variance']==mean*(1-mean)and r['target4variance']==4*r['target_variance'];helper.add_sampling_question(counts,g['labels'],labels);K5_by_K10[k10][positive]+=1;observed_classes['all_wrong'if positive==0 else'all_correct'if positive==5 else'Mixed']+=1
  for position,t in enumerate(ids):index=g['trajectory_ids'].index(t);inclusion[index]+=1;position_counts[position][index]+=1
 diagnostic=helper.finish_sampling_diagnostic(counts);diagnostic['source']='Existing actual sampling-plan-epoch-01.json only; no plan regeneration';N=len(rows);expected_classes=dict(all_wrong=0.,Mixed=0.,all_correct=0.);expectation_rows={};total_chi2=0.;expected_hard_var=0.
 for k10,observed in K5_by_K10.items():
  n=counts[k10]['questions'];expected=[]
  for k5 in range(6):
   ways=(math.comb(k10,k5)if k10>=k5 else 0)*(math.comb(10-k10,5-k5)if 10-k10>=5-k5 else 0);e=n*ways/252;expected.append(e)
   if not ways:assert observed[k5]==0
   elif e:total_chi2+=(observed[k5]-e)**2/e
  assert abs(sum(expected)-n)<1e-8;expected_classes['all_wrong']+=expected[0];expected_classes['all_correct']+=expected[5];expected_classes['Mixed']+=sum(expected[1:5]);p=(expected[0]+expected[5])/n if n else 0.
  if 0<k10<10:expected_hard_var+=n*p*(1-p)
  expectation_rows[str(k10)]=dict(questions=n,observed_K5_0_to5=observed,exact_uniform_5of10_expected_K5_0_to5=expected)
 expected_hard=diagnostic['expected_G10Mixed_to_G5_homogeneous'];hard=diagnostic['observed_G10Mixed_to_G5_homogeneous'];after=planpath.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 result=dict(status='ACTUAL_FIRST_EPOCH_169028_UNIQUE_OWN_G5_CANONICAL_LABELS_AND_NATURAL_HARD_NEGATIVES_VERIFIED',job_id='750190',epoch=1,init_seed=19137,train_plan_seed=2026102302,questions=N,unique_questions=N,trajectory_visits=seen_selected,unique_selected_trajectory_ids=len(global_ids),canonical_ownG10_exact_for_every_question=True,actual_saved_plan_SHA_matches=True,actual_plan_sha256=saved['plan_sha256'],observed_G5_classes={k:dict(questions=v,fraction=v/N)for k,v in observed_classes.items()},natural_random_G5_exact_expected_classes=expected_classes,hard_negative_diagnostic=diagnostic,hard_negative_fraction_of_all_questions=hard/N,hard_negative_fraction_of_G10Mixed=hard/diagnostic['G10_Mixed_questions'],homogeneous_hard_negative_expected_variance=expected_hard_var,homogeneous_hard_negative_standardized_residual=(hard-expected_hard)/math.sqrt(expected_hard_var),K10_by_K5_hypergeometric=expectation_rows,descriptive_hypergeometric_Pearson_chi2=total_chi2,chi2_pvalue_not_claimed=True,trajectory_index_inclusion_counts=inclusion,trajectory_index_inclusion_expected_each=N*.5,trajectory_index_inclusion_zscores=[(n-N*.5)/math.sqrt(N*.25)for n in inclusion],trajectory_position_index_counts=position_counts,question_order_is_lexically_sorted=[r['qid']for r in rows]==sorted(groups),question_order_SHA256=digest([r['qid']for r in rows]),source_actual_plan=proof(planpath),source_canonical_G10=proof(canonical_path),source_fixture=binding['fixture'],source_train_groups=fixture['train_groups'],source_accepted_fixture=ready['independent_acceptance'],source_physical_contract=ready['physical_source_contract'],source_P20_alignment=ready['P20_alignment'],source_training_publication=proof(BASE/'source-publication.json'),source_goal6_helper=proof(GOAL6),rank_progress={str(rank):{k:read(BASE/('progress-rank-%d.json'%rank)).get(k)for k in('cursor','epoch','time')}for rank in(0,1)},plan_regenerations=0,RAW_reads=0,checkpoint_or_tensor_reads=0,final4096_read=False,Slurm_mutated=False,training_source_or_labels_modified=False,seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,time=time.time());atomic(OUT/'result.json',result);print(json.dumps(dict(status=result['status'],receipt=proof(OUT/'result.json'),questions=N,observed_G5_classes=result['observed_G5_classes'],hard_negative_questions=hard,expected_hard_negative_questions=expected_hard,seconds=result['seconds'])))
if __name__=='__main__':
 try:main()
 except BaseException:
  OUT.mkdir(exist_ok=True);atomic(OUT/'failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
