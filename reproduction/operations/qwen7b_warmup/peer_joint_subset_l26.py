"""SHA-bound accepted train subset -> unchanged qualified HF L26/P20 capture.

Supports canonical repeated text, unequal four-way shards, and an explicitly
accepted identity-only subset. No rollout, label selection, actor/probe update.
"""
import argparse,fcntl,gzip,json,os,shutil,time
from pathlib import Path
from . import peer_audited122_l26_replay as core
from . import replay_probe_layers as r
from . import peer_newseed_ingest as raw
from .validation_fixedg5_replay import batching_binding
from .replay_bank_prepare import q1_model_binding

def config(path,sha):
 ref=r.fingerprints(r.remote(path));assert ref['sha256']==sha
 c=r.read(ref['path']);base=r.remote(c['target'])
 if Path(ref['path'])!=base/'cohort-config.json' or not base.name.startswith('peer-joint124194-') or c['status']!='FROZEN_ACCEPTED_TRAIN_SUBSET_L26_CONFIG':raise ValueError('Exact published isolated training subset config required')
 if c['questions']<4 or c['trajectories']!=10*c['questions'] or c['prefix_tokens']!=20 or c['question_max_tokens']!=273 or c['batch_size']!=32 or c['max_context_tokens']!=16384:raise ValueError('Qualified capture geometry/budget differs')
 for key in ('raw_complete','raw_groups','source_events'):r.checked_file(c[key])
 if c.get('identity_subset'):r.checked_file(c['identity_subset'])
 return ref,c,base

def selected(c):
 accepted=r.read(c['raw_complete']['path'])
 if accepted['status']!=raw.STATUS or accepted['strict_EM_mismatches'] or accepted['actor_updates']or accepted['outputs']['groups']!=c['raw_groups']or accepted['outputs']['source-events']!=c['source_events']:raise ValueError('Actual accepted ownEM/rawG10 source required')
 gs=r.read(c['raw_groups']['path'])['train'];by={g['qid']:g for g in gs}
 if len(by)!=len(gs):raise ValueError('Duplicate source qid')
 if c.get('identity_subset'):
  identity=r.read(c['identity_subset']['path'])
  if identity['role']!='train'or identity['labels_consulted']or identity['scores_consulted']or not identity['external_final_qid_hash_exclusions_retained']or identity['source_raw_complete']!=c['raw_complete']or identity['source_raw_groups']!=c['raw_groups']:raise ValueError('Accepted label-blind train subset lineage differs')
  wanted=identity['groups'];gs=[]
  for g in wanted:
   real=by[g['qid']]
   if any(real[k]!=g[k]for k in ('qid','question_hash','trajectory_ids')):raise ValueError('Subset differs from genuine accepted own G10')
   gs.append(real)
 if len(gs)!=c['questions']:raise ValueError('Declared actual training subset budget differs')
 qs=set();ts=set();hs=set();rows=[];domains={}
 for g in gs:
  q,h,ids=g['qid'],g['question_hash'],g['trajectory_ids'];dom=json.loads(q)[0]
  if q in qs or len(ids)!=10 or len(set(ids))!=10 or ts.intersection(ids)or dom not in ('nq','hotpotqa'):raise ValueError('Authentic train qid/ten own IDs must be unique')
  qs.add(q);ts.update(ids);hs.add(h);domains[dom]=domains.get(dom,0)+1;rows.append(dict(qid=q,question_hash=h,trajectory_ids=ids))
 return rows,gs,domains,len(rows)-len(hs)

def prepare(path,sha,code_root,commit,golden):
 ref,c,base=config(path,sha);rows,truth,domains,hash_repeats=selected(c);code=r.remote(code_root)
 if code.parent!=base:raise ValueError('Independent immutable code root required')
 base.mkdir(exist_ok=True)
 with(base/'prepare.lock').open('a')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if(base/'preparation.json').exists():
   v=r.read(base/'preparation.json');assert v['config']==ref and v['source_commit']==commit;return v
  free=shutil.disk_usage(base).free
  if free<c['minimum_free_bytes']:raise ValueError('Declared shared-space reserve insufficient')
  identity=dict(role='train',groups=rows,source_ingest=c['raw_complete'],source_groups=c['raw_groups'],identity_subset=c.get('identity_subset'),summary=dict(questions=len(rows),trajectories=len(rows)*10,domains=domains,canonical_train_repeated_text_questions=hash_repeats,labels_consulted=False,scores_consulted=False,original_warmup_unseen_claim=False,former_internal5120_not_held_if_trained=True,final_outcomes_read=False))
  raw.immutable(base/'identity-plan.json',identity);iref=r.fingerprints(base/'identity-plan.json');parts=[rows[i::4]for i in range(4)];owners={t:i for i,part in enumerate(parts)for g in part for t in g['trajectory_ids']};wanted={t:g['qid']for g in rows for t in g['trajectory_ids']}
  events=r.read(c['source_events']['path']);sources=[];seenpath=set()
  for event in events['events']:
   for batch in event['batches']:
    v=batch['prefix']
    if v['path']not in seenpath:seenpath.add(v['path']);sources.append(v)
  batching=batching_binding(r.fingerprints(golden),32);q1,binding=q1_model_binding();streams=[];paths=[];counts=[0]*4;bad=[0]*4;stats={};seen=set()
  try:
   for i in range(4):
    d=base/('shard-%02d'%i);(d/'inputs').mkdir(parents=True,exist_ok=True);(d/'logs').mkdir(exist_ok=True);dst=d/'inputs/prefix.jsonl.gz';tmp=dst.with_suffix('.gz.partial')
    if dst.exists()or tmp.exists():raise ValueError('Unknown prior prefix projection preserved')
    paths.append((tmp,dst));streams.append(gzip.open(tmp,'xb'))
   for source in sources:
    with gzip.open(r.checked_file(source)['path'],'rt')as inp:
     for line in inp:
      row=json.loads(line);tid=row['trajectory_id']
      if tid not in wanted:continue
      q=wanted[tid];i=owners[tid]
      if tid in seen or row['qid']!=q or any(k in row for k in ('strict_em','labels','reward_model','ground_truth','response_token_ids')):raise ValueError('Only exact causal own prefix records allowed')
      v=core.geometry(row);dom=json.loads(q)[0];st=stats.setdefault(dom,dict(trajectories=0,max_prompt=0,max_context=0,short_policy=0,overlength=0));st['trajectories']+=1;st['max_prompt']=max(st['max_prompt'],v['prompt']);st['max_context']=max(st['max_context'],v['context']);st['short_policy']+=int(v['policy']<20)
      if v['overlength']:bad[i]+=1;st['overlength']+=1
      else:r.normalize_prefix(row,tid,q,True);streams[i].write(r.canonical(row)+b'\n');counts[i]+=1
      seen.add(tid)
      if len(seen)%512==0:raw.atomic(base/'prepare-progress.json',dict(status='ACTUAL_ACCEPTED_SUBSET_PREFIX20_GEOMETRY_ADVANCING',trajectories=len(seen),expected=len(wanted),counts=counts,bad=bad,geometry=stats,time=time.time()))
  finally:
   for f in streams:f.close()
  if seen!=set(wanted):raise ValueError('Missing genuine selected prefix IDs')
  selections=[];blocked=[]
  for i,(tmp,dst)in enumerate(paths):
   os.replace(str(tmp),str(dst))
   if bad[i]:blocked.append(dict(shard=i,overlength=bad[i]));continue
   if counts[i]!=len(parts[i])*10:raise ValueError('Actual unequal shard coverage differs')
   sel=dict(role='peer_newseed_train_G10',model_revision=r.REVISION,questions=len(parts[i]),trajectories=[dict(qid=g['qid'],trajectory_id=t)for g in parts[i]for t in g['trajectory_ids']],real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,capture_layers=list(r.LAYERS),persist_layers=[26],prefix_files=[r.fingerprints(dst)],model_files=q1['model_files'],source_model_binding=binding,source_ingest=c['raw_complete'],question_plan=iref,domain_scope=dict(kind='accepted_canonical_training_subset',overall_question_counts=domains,canonical_train_text_repeats_retained=True,unseen_previous_training_claim=False),source_commit=commit,expected_code_root=str(code),shard=i,shards=4,batching=batching,geometry_checked_no_truncation=True,actor_updates=0,new_responses_generated=0)
   p=base/('shard-%02d'%i)/'inputs/selection.json';raw.immutable(p,sel);selections.append(r.fingerprints(p))
  done=dict(status='ACCEPTED_TRAIN_SUBSET_GEOMETRY_FOUR_SHARDS_PREPARED'if not blocked else'ACCEPTED_TRAIN_SUBSET_OVERLENGTH_SHARDS_PRESERVED',config=ref,identity_plan=iref,raw_complete=c['raw_complete'],raw_groups=c['raw_groups'],source_events=c['source_events'],source_commit=commit,code_root=str(code),batching=batching,selections=selections,blocked_shards=blocked,questions=len(rows),trajectories=len(wanted),domains=domains,geometry=stats,canonical_train_repeated_text_questions=hash_repeats,compute_layers=list(r.LAYERS),persist_layers=[26],writer_layout='lossless_question_base_xor_policy_v1',free_space_bytes=free,minimum_free_bytes=c['minimum_free_bytes'],actor_updates=0,probe_updates=0,new_rollouts_generated=0,final_pool_read=False,time=time.time());raw.immutable(base/'preparation.json',done);return done

def worker(path,sha,preparation,preparation_sha,pair):
 ref,c,base=config(path,sha)
 if pair not in(0,1)or os.environ.get('SLURM_NTASKS')!='2':raise ValueError('Two isolated one-GPU tasks required')
 i=pair*2+int(os.environ['SLURM_LOCALID']);p=r.fingerprints(preparation)
 if p['sha256']!=preparation_sha:raise ValueError('Preparation SHA differs')
 done=r.read(p['path']);assert done['config']==ref;own={r.read(x['path'])['shard']:x for x in done['selections']}[i];n=r.read(own['path'])['questions']
 return core.capture(own['path'],own['sha256'],base/('shard-%02d'%i)/'capture',expected_questions=n,complete_status='JOINT_TRAIN_SUBSET_REAL_G10_L26_SHARD_COMPLETE')

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','worker'),required=True);p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--preparation');p.add_argument('--preparation-sha256');p.add_argument('--pair',type=int);a=p.parse_args()
 try:v=prepare(a.config,a.config_sha256,a.code_root,a.commit,a.golden)if a.action=='prepare'else worker(a.config,a.config_sha256,a.preparation,a.preparation_sha256,a.pair)
 except BaseException as e:
  _,c,b=config(a.config,a.config_sha256);raw.atomic(b/'failure.json',dict(type=type(e).__name__,error=str(e),time=time.time()));raise
 print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
