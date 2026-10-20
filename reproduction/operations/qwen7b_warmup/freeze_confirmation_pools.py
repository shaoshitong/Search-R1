"""Freeze two untouched NQ question pools remotely before any probe scoring.

Questions/answers stay remote. This does not generate trajectories or submit
jobs. Existing train/dev/stage/final and the repeatedly viewed2048 are excluded.
"""
import argparse,hashlib,json,random,time,unicodedata
from pathlib import Path

SOURCE_SHA='10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().strip().rstrip('?').split())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)

def main():
 p=argparse.ArgumentParser();p.add_argument('--training-base',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();B=a.training_base;b=a.base
 assert B.is_absolute() and b.is_absolute() and b.parent==B and b.name=='algorithm-improvement-20261021'
 old=B/'selection-2048-20261020';source=old/'source/nq-dev-bcafb8dd07d4.jsonl';assert sha(source)==SOURCE_SHA
 acceptance=read(old/'preparation/question-plan-acceptance.json');assert acceptance['source_sha256']==SOURCE_SHA
 c=read(B/'config.json');assert c['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
 if (b/'confirmation/manifest.json').exists():
  previous=read(b/'confirmation/manifest.json');assert previous['source_sha256']==SOURCE_SHA
  for f in previous['pools'].values():assert sha(f['path'])==f['sha256']
  print(json.dumps(dict(status='EXISTING_FROZEN_CONFIRMATION_POOLS_VERIFIED',manifest=previous)));return
 used=read(old/'inputs/questions.json')['rows'];assert len(used)==2048
 used_source={r['source_question_id'] for r in used};used_text={r['canonical_question_sha256'] for r in used};assert len(used_source)==2048
 exclusion=set();exclusion_files={};common=Path(c['canonical_train']).parent
 for split in ('train','validation','stage','final'):
  path=common/(split+'.jsonl');digest=sha(path);assert digest==acceptance['excluded_jsonl_sha256'][split]
  exclusion_files[split]=dict(path=str(path),sha256=digest)
  for line in path.open():exclusion.add(norm(json.loads(line)['question']))
 available=[];seen=set();removed=dict(viewed_validation=0,canonical_overlap=0,duplicate_text=0)
 for line in source.open():
  row=json.loads(line);key='nq:dev:'+str(row['id']);normalized=norm(row['question']);h=hashlib.sha256(normalized.encode()).hexdigest()
  if key in used_source or h in used_text:removed['viewed_validation']+=1;continue
  if normalized in exclusion:removed['canonical_overlap']+=1;continue
  if h in seen:removed['duplicate_text']+=1;continue
  seen.add(h);available.append(dict(source_question_id=key,canonical_question_sha256=h,source_row=row))
 assert len(available)>=4096
 random.Random(20261021).shuffle(available);plans={};sets=[]
 for index,name in enumerate(('confirmation2048','stability2048')):
  selected=available[index*2048:(index+1)*2048];ids={r['source_question_id'] for r in selected};texts={r['canonical_question_sha256'] for r in selected}
  assert len(ids)==len(texts)==2048 and not ids&used_source and not texts&used_text
  assert all(not ids&old_ids and not texts&old_texts for old_ids,old_texts in sets);sets.append((ids,texts))
  payload=dict(role='UNTOUCHED_FINAL_CONFIRMATION_NOT_FOR_TUNING',questions=selected,source_sha256=SOURCE_SHA,seed=20261021,generation_not_started=True,selection_before_any_rollout_or_probe_score=True,final_grouping='FIXED_FIVE_REAL_TRAJECTORIES; no EM-aware subgroup selection',actor_updates=0)
  dest=b/'confirmation'/(name+'.json');atomic(dest,payload)
  plans[name]=dict(path=str(dest),sha256=sha(dest),questions=2048,ordered_source_id_sha256=hashlib.sha256(json.dumps([r['source_question_id'] for r in selected],separators=(',',':')).encode()).hexdigest())
 manifest=dict(status='TWO_DISJOINT_UNTOUCHED_2048_QUESTION_POOLS_FROZEN',source_sha256=SOURCE_SHA,source_path=str(source),source_revision='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c',source_commit=a.source_commit,pools=plans,remaining_unassigned=len(available)-4096,removed=removed,excluded_canonical=exclusion_files,excluded_viewed2048_sha256=sha(old/'inputs/questions.json'),score_or_generated_EM_observed=False,fit_or_tuning_allowed=False,no_reserved_final_QA_rows_used=True,old_data_or_jobs_changed=False,time=time.time())
 atomic(b/'confirmation/manifest.json',manifest);print(json.dumps(manifest))

if __name__=='__main__':main()
