"""Remote CPU metadata: exact planned qid closure; freeze already sealed delta."""
import pathlib,json,hashlib,time,fcntl,os,resource
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';T=A/'hpc3-tail-newseedG5-20261021-v1';O=A/'hpc3-tail-completion-delta01-v1';O.mkdir(exist_ok=True)
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canon(v)).hexdigest()
def load(p,sha=None):
 p=pathlib.Path(p);before=p.stat();raw=p.read_bytes();after=p.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);ref=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());assert sha is None or sha==ref['sha256'];return json.loads(raw),ref
def checked(ref):
 v,actual=load(ref['path'],ref['sha256']);assert'bytes'not in ref or actual['bytes']==ref['bytes'];return v,actual
def once(p,v):
 raw=canon(v)+b'\n'
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
with(O/'prepare.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(O/'metadata-complete.json').exists():print(json.dumps(load(O/'metadata-complete.json')[0]));raise SystemExit(0)
 c,cref=load(T/'config.json','368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835')
 # Exact literal producer SHA; no model/tensor files are opened.
 manifest,mref=load(T/'inputs/manifest.json','e6f4f19cd89813da805f7b6d3fc9e818ccef05eb5fba9d02af6f82b4e1c58cab');assert len(manifest['batches'])==241
 bank,bref=load(B/'bank/complete.json','6f7a5510eb092bd4a414b13f4f7a8a7f1dbe836dbfdd55fd5db97a19ba2c303e');canonical,gref=load(B/'bank/groups.json',bank['groups']['sha256']);canonical={g['qid']:g for g in canonical['train']};assert len(canonical)==169028
 main,mcref=load(A/'joint124194-native-training-bank-v1/complete.json','a8fe2c2ea823087d50ac86b283eb361b51ae7ba5ba0571860193214338200801');mg,mgref=checked(main['groups']);mainq={g['qid']:g['question_hash']for g in mg['train']};del mg;assert len(mainq)==124194
 d,dc=load(A/'peer-newseed-ingest-audited138-after971-20261021-51057f37/complete.json','e7056d4eb952151b95241dc87ac732fb641daa5aea633f09de4373f536d69ed3');dg,dgref=checked(d['outputs']['groups']);donorq={g['qid']:g['question_hash']for g in dg['train']};assert len(donorq)==17664;del dg
 old,oldref=load(A/'hpc3-tail9344-old5-qualified-g10-v1/complete.json','0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd');og,ogref=checked(old['outputs']['groups']);priorq={g['qid']:g['question_hash']for g in og['train']};del og;assert len(priorq)==9344
 ev,_=checked(old['outputs']['source-events']);priorseals={x['seal']['path']:x['seal']for e in ev['events']if e['event']=='HPC3_newseed20261021'for x in e['batches']};assert len(priorseals)==73
 excluded,exref=checked(main['evaluation_exclusions']);eq={g['qid']for g in excluded['groups']};eh={g['question_hash']for g in excluded['groups']};assert len(eq)==6656
 whole={};sealed={};frozen=[];workers={0:dict(batches=0,questions=0),1:dict(batches=0,questions=0)}
 for item in manifest['batches']:
  value,iref=checked(item);rows=value['rows'];assert len(rows)==item['questions'];qids=[]
  for row in rows:
   q,h,o=row['qid'],row['question_hash'],row['source_ordinal'];assert q not in whole and q in canonical and h==canonical[q]['question_hash']and o==canonical[q]['source_ordinal']and row['candidate_ordinal']==o and q not in eq and h not in eh
   whole[q]=h;qids.append(q)
  p=pathlib.Path(item['path']);sealpath=T/'sampling'/p.parent.name/'batches'/p.stem/'complete.json'
  if not sealpath.exists():continue
  seal,sref=load(sealpath);assert seal['execution_run_id']==c['execution_run_id']and seal['input_sha256']==iref['sha256']and seal['execution_job']==('745518','745519')[item['worker']]and seal['questions']==len(rows)and seal['trajectories']==5*len(rows)
  sealed.update({q:whole[q]for q in qids});w=workers[item['worker']];w['batches']+=1;w['questions']+=len(rows)
  if str(sealpath)in priorseals:assert sref==priorseals[str(sealpath)];continue
  frozen.append(dict(input=iref,seal=sref,worker=item['worker'],questions=len(rows),ordered_qid_sha256=digest(qids)))
 assert len(whole)==30754 and set(priorq)<=set(whole)and set(priorq)<=set(sealed)
 for name,values in [('main',mainq),('donor',donorq),('whole_tail',whole),('qualified_tail73',priorq)]:assert all(q in canonical and canonical[q]['question_hash']==h for q,h in values.items()),name
 S=set;union=S(mainq)|S(donorq)|S(whole);missing=S(canonical)-union;extra=union-S(canonical);overlap=S(donorq)&S(whole);assert not missing and not extra and len(overlap)==3584 and not(S(mainq)&S(whole))and not(S(mainq)&S(donorq))
 actual=S(mainq)|S(donorq)|S(sealed);qualified=S(mainq)|S(donorq)|S(priorq);domains={}
 for q in whole:dom=json.loads(q)[0];domains[dom]=domains.get(dom,0)+1
 frozen_ref=once(O/'frozen-seals.json',dict(role='fixed_new_sealed_G5_delta_excluding_prior73',producer_config=cref,whole_plan_inputs=mref,prior_qualified_G10=oldref,batches=frozen,questions=sum(x['questions']for x in frozen),trajectories=5*sum(x['questions']for x in frozen),prior_batches=73,current124194_fit_expanded=False,full_qualification_pending=True,created_epoch=time.time()))
 result=dict(status='WHOLE241_EXACT_CANONICAL_QID_COVERAGE_AND_NEW_SEALED_DELTA_FROZEN',epoch=time.time(),producer_config=cref,whole_plan_inputs=mref,canonical_bank_complete=bref,canonical_groups=gref,main_complete=mcref,main_groups=mgref,donor_complete=dc,donor_groups=dgref,prior_tail_complete=oldref,prior_tail_groups=ogref,true6656_exclusions=exref,planned_tail_batches=241,planned_tail_questions=30754,planned_tail_trajectories=153770,planned_tail_domains=domains,donor_vs_whole_tail_qid_intersection=len(overlap),donor_vs_first73_qid_intersection=len(S(donorq)&S(priorq)),main_vs_donor_qid_overlap=0,main_vs_tail_qid_overlap=0,planned_union_questions=len(union),planned_union_missing_canonical_questions=len(missing),planned_union_extraneous_questions=len(extra),planned_union_qid_sha256=digest(sorted(union)),canonical_qid_sha256=digest(sorted(canonical)),actual_sealed_own_questions=len(sealed),actual_sealed_own_trajectories=5*len(sealed),actual_generated_union_questions=len(actual),actual_generated_missing_canonical_questions=len(S(canonical)-actual),full_qualified_union_questions=len(qualified),full_qualified_missing_canonical_questions=len(S(canonical)-qualified),current_workers=workers,next_delta_batches=len(frozen),next_delta_questions=sum(x['questions']for x in frozen),next_delta_frozen=frozen_ref,whole_G5_source_choice_required=True,matching_tid_does_not_prove_matching_payload=True,current124194_fit_expanded=False,model_or_tensor_loaded=False,new_GPU_jobs=0)
 ref=once(O/'metadata-complete.json',result);print(json.dumps(dict(result,metadata_complete=ref),sort_keys=True))
