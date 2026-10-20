"""Independent integer reduction of sealed horizon aggregate units; no raw/tensors."""
import argparse,hashlib,json,os,time
from pathlib import Path
BASE=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-long-prefix-causal-diagnostic-v1')

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for v in iter(lambda:f.read(4194304),b''):h.update(v)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def verified(v):
 if ref(v['path'])!=v:raise ValueError('Independent source SHA/size failed')
 return read(v['path'])
def reduce_values(rows,key=None):
 x=rows[0]
 if isinstance(x,dict):
  assert all(set(v)==set(x) for v in rows)
  return {k:reduce_values([v[k] for v in rows],k) for k in x}
 if isinstance(x,list):
  assert all(len(v)==len(x) for v in rows)
  return [reduce_values([v[i] for v in rows]) for i in range(len(x))]
 if key=='G':assert len(set(rows))==1;return x
 return max(rows) if key=='max_selected_context_tokens' else sum(rows)
def close(a,b):
 if a is None or b is None:assert a is b
 else:assert abs(a-b)<=1e-14*max(1,abs(a),abs(b))
def check_scope(raw,reported,expected):
 g=raw['G'];n=len(expected);hist=[sum(sum(v)==k for v in expected) for k in range(g+1)];mixed=sum(hist[1:-1]);cw=sum(k*(g-k)*hist[k] for k in range(g+1));pairs=n*g*(g-1)//2
 assert raw['questions']==n and raw['trajectories']==n*g and raw['K_histogram']==hist and raw['Mixed']==mixed and raw['correct_wrong_pairs']==cw and raw['all_pairs']==pairs
 for k in ('G','questions','trajectories','Mixed','all_pairs','correct_wrong_pairs','K_histogram'):assert raw[k]==reported[k]
 previous={};short=-1;selected=-1;primary=[]
 for length in (5,20,40,80,160):
  key=str(length);value=raw['lengths'][key];public=reported['lengths'][key];cov=value['coverage'];assert cov['short_trajectories']>=short and cov['selected_policy_tokens']>=selected;short=cov['short_trajectories'];selected=cov['selected_policy_tokens'];assert sum(cov['short_terminal_reason'].values())==short
  assert {k:v for k,v in public['coverage'].items() if k in cov}==cov
  close(public['coverage']['short_fraction'],short/(n*g));close(public['coverage']['multi_turn_fraction'],cov['multi_turn_trajectories']/(n*g));close(public['coverage']['past_retrieval_fraction'],cov['past_retrieval_trajectories']/(n*g))
  for name,s in value['representations'].items():
   assert sum(s['distinct_histogram'])==n and s['distinct_histogram'][0]==0 and 0<=s['conflict_pairs']<=min(s['duplicate_pairs'],cw) and s['duplicate_pairs']<=pairs
   assert s['any_duplicate_questions']==n-s['distinct_histogram'][g] and s['all_identical_questions']==s['distinct_histogram'][1]
   assert s['excess_duplicate_trajectories']==sum((g-i)*z for i,z in enumerate(s['distinct_histogram']))
   if name in previous:assert s['conflict_pairs']<=previous[name][0] and s['duplicate_pairs']<=previous[name][1]
   previous[name]=(s['conflict_pairs'],s['duplicate_pairs']);published=public['representations'][name]
   for k,v in s.items():assert published[k]==v
   close(published['duplicate_pair_fraction'],s['duplicate_pairs']/pairs);close(published['conflict_fraction_of_correct_wrong_pairs'],s['conflict_pairs']/cw if cw else None);close(published['distinguishable_correct_wrong_pair_fraction'],1-s['conflict_pairs']/cw if cw else None);close(published['any_duplicate_question_fraction'],s['any_duplicate_questions']/n);close(published['Mixed_question_conflict_fraction'],s['Mixed_questions_with_conflict']/mixed if mixed else None)
  full=value['representations']['full_causal_input'];tok=value['representations']['policy_tokens'];assert full['duplicate_pairs']<=tok['duplicate_pairs'] and full['conflict_pairs']<=tok['conflict_pairs']
  primary.append(dict(prefix=length,duplicate_pairs=full['duplicate_pairs'],all_pairs=pairs,duplicate_fraction=full['duplicate_pairs']/pairs,opposite_EM_same_causal_input_pairs=full['conflict_pairs'],correct_wrong_pairs=cw,distinguishable_correct_wrong_fraction=1-full['conflict_pairs']/cw if cw else None,Mixed_questions_with_conflict=full['Mixed_questions_with_conflict'],Mixed_questions=mixed,short_trajectories=short,short_fraction=short/(n*g),multi_turn_trajectories=cov['multi_turn_trajectories'],past_retrieval_trajectories=cov['past_retrieval_trajectories'],policy_token_only_conflict_pairs=tok['conflict_pairs'],same_policy_tokens_but_distinct_causal_context_clusters=value['token_context_ambiguous_clusters']))
 return primary

def audit(base=BASE):
 before=ref(base/'complete.json');done=verified(before);assert done['status']=='FROZEN8192_TRUE_G10_LONG_CAUSAL_PREFIX_AGGREGATES_COMPLETE';cfg=verified(done['config']);identity=verified(cfg['identity']);raw=verified(cfg['raw_complete']);assert raw['outputs']['groups']==cfg['raw_groups'];canonical_groups={g['qid']:g for g in verified(cfg['raw_groups'])['train']};groups=[]
 for row in identity['groups']:
  g=canonical_groups[row['qid']];assert row['trajectory_ids']==g['trajectory_ids'] and row['question_hash']==g['question_hash'] and len(g['trajectory_ids'])==10;groups.append(g)
 assert len(groups)==8192 and len({g['qid'] for g in groups})==8192 and done['P20_tokens_positions_contexts_exact_matches']==81920
 expected={w['index']:w for w in cfg['work_plan'] if w['event']=='newseed20261021'};units=[];seen=set();p20=0;source_paths=set()
 for item in done['aggregate_units']:
  unit=verified(item);work=unit['work'];assert work==expected[work['index']] and work['index'] not in seen;seen.add(work['index']);assert unit['config']==done['config'] and unit['only_aggregates'] is True and unit['source_full_and_prefix_SHA_verified'] is True
  assert unit['questions']==work['selected_questions'] and unit['trajectories']==10*unit['questions'] and unit['P20_tokens_positions_contexts_exact_matches']==unit['trajectories'];p20+=unit['P20_tokens_positions_contexts_exact_matches']
  for path,stamp in unit['source_stamps'].items():
   actual=Path(path).stat();assert stamp==[actual.st_dev,actual.st_ino,actual.st_size,actual.st_mtime_ns,actual.st_ctime_ns];source_paths.add(path)
  units.append(unit['statistics'])
 assert seen==set(expected) and p20==81920;combined=reduce_values(units);labels={name:[g['labels'][start:end] for g in groups] for name,start,end in [('authentic_G10',0,10),('original_G5',0,5),('newseed_G5',5,10)]};table={name:check_scope(combined[name],done['statistics'][name],labels[name]) for name in labels};previous=verified(cfg['P20_reference'])
 for key,name in [('policy_token_only','policy_tokens'),('full_causal_input','full_causal_input')]:
  z=combined['authentic_G10']['lengths']['20']['representations'][name];assert z['duplicate_pairs']==previous[key]['identical_input_pairs'] and z['conflict_pairs']==previous[key]['identical_input_opposite_EM_pairs'] and z['Mixed_questions_with_conflict']==previous[key]['true_G10_Mixed_questions_with_input_conflict']
 assert ref(base/'complete.json')==before
 out=dict(status='INDEPENDENT_SEALED_HORIZON_AGGREGATES_AND_P20_REFERENCE_PASSED',complete=before,config=done['config'],P20_reference=cfg['P20_reference'],questions=8192,trajectories=81920,units=len(units),all_unit_fullSHA_fresh=True,source_files_stamps_unchanged=len(source_paths),source_full_raw_bulkSHA_repeated=False,own_K_and_correct_wrong_pair_counts_recomputed=True,P20_previous_statistics_exact=True,collision_monotonicity_passed=True,table=table,no_model_tensor_or_validation_or_final_read=True,only_aggregate_metadata_read=True,time=time.time());p=base/'independent-audit.json'
 if p.exists():assert {k:v for k,v in read(p).items() if k!='time'}=={k:v for k,v in out.items() if k!='time'}
 else:p.write_text(json.dumps(out,sort_keys=True))
 return dict(audit=ref(p),summary=read(p))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=BASE);a=p.parse_args();print(json.dumps(audit(a.base),sort_keys=True))
