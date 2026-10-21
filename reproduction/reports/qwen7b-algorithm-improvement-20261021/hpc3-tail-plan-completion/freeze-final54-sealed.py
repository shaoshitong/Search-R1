"""Metadata-only final54 scope, waits for all54 real atomic seals.

Never follows beyond the frozen241 plan, changes a writer, or opens rollout data.
"""
import pathlib,json,hashlib,time,fcntl,os
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';T=A/'hpc3-tail-newseedG5-20261021-v1';O=A/'hpc3-tail-completion-final54-v1';P=A/'hpc3-tail-completion-delta01-v1';O.mkdir(exist_ok=True)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def load(p,sha=None):
 p=pathlib.Path(p);b=p.stat();raw=p.read_bytes();e=p.stat();assert(b.st_ino,b.st_size,b.st_mtime_ns)==(e.st_ino,e.st_size,e.st_mtime_ns);ref=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());assert sha is None or sha==ref['sha256'];return json.loads(raw),ref
def checked(ref):
 v,r=load(ref['path'],ref['sha256']);assert'bytes'not in ref or r['bytes']==ref['bytes'];return v,r
def once(p,v):
 raw=canonical(v)+b'\n'
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
with(O/'freeze.lock').open('a+')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(O/'frozen-seals.json').exists():print(json.dumps(dict(status='FINAL54_ALREADY_FROZEN_NO_REFREEZE',frozen=load(O/'frozen-seals.json')[1])));raise SystemExit(0)
 config,cref=load(T/'config.json','368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835');manifest,mref=load(T/'inputs/manifest.json','e6f4f19cd89813da805f7b6d3fc9e818ccef05eb5fba9d02af6f82b4e1c58cab');first,firstref=load(A/'hpc3-tail9344-old5-qualified-g10-v1/complete.json','0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd');events,_=checked(first['outputs']['source-events']);priorpaths={g['input']['path']for e in events['events']if e['event']=='HPC3_newseed20261021'for g in e['batches']};assert len(priorpaths)==73
 middle,middleref=load(P/'complete.json','f789b74c3f3b9fc7c813fdf4f958c148a8c2741cd7e37f7e2be5917d1dfb10a0');assert middle['raw_G10_full_qualified']and middle['questions']==14592 and middle['new5_batches']==114 and middle['strict_EM_mismatches']==0;frozen,_=checked(middle['frozen']);priorpaths.update(g['input']['path']for g in frozen['batches']);assert len(priorpaths)==187
 items=[g for g in manifest['batches']if g['path']not in priorpaths];assert len(items)==54 and sum(g['questions']for g in items)==6818
 scope=once(O/'planned-final54-inputs.json',dict(role='fixed_last54_of_own241_not_yet_payload_qualification',producer_config=cref,whole_plan_inputs=mref,first73_qualified=firstref,middle114_qualified=middleref,batches=items,questions=6818,trajectories=34090,do_not_follow_new_cohort=True,current124194_fit_expanded=False));sealed=[];pending=[];seen=set();workers={0:dict(batches=0,questions=0),1:dict(batches=0,questions=0)}
 for item in items:
  batch,iref=checked(item);qids=[g['qid']for g in batch['rows']];assert len(qids)==len(set(qids))==item['questions']and not set(qids)&seen;seen.update(qids);p=pathlib.Path(item['path']);dest=T/'sampling'/p.parent.name/'batches'/p.stem/'complete.json'
  if not dest.exists():pending.append(item['path']);continue
  seal,sref=load(dest);assert seal['input_sha256']==iref['sha256']and seal['execution_run_id']==config['execution_run_id']and seal['execution_job']==('745518','745519')[item['worker']]and seal['actor_optimizer_updates']==0 and seal['questions']==len(qids)and seal['trajectories']==5*len(qids)
  sealed.append(dict(input=iref,seal=sref,worker=item['worker'],questions=len(qids),ordered_qid_sha256=hashlib.sha256(canonical(qids)).hexdigest()));workers[item['worker']]['batches']+=1;workers[item['worker']]['questions']+=len(qids)
 assert len(seen)==6818
 if pending:print(json.dumps(dict(status='WAITING_FOR_ALL_FINAL54_SEALS_NO_AUDIT_STARTED',epoch=time.time(),planned_scope=scope,sealed_batches=len(sealed),sealed_questions=sum(g['questions']for g in sealed),pending_batches=len(pending),workers=workers,model_or_tensor_loaded=False,new_GPU_jobs=0)));raise SystemExit(0)
 fref=once(O/'frozen-seals.json',dict(role='fixed_final54_sealed_G5_excluding_prior73_and_middle114',producer_config=cref,whole_plan_inputs=mref,prior_qualified_G10=firstref,additional_prior_qualified_G10=middleref,batches=sealed,questions=6818,trajectories=34090,prior_batches=187,current124194_fit_expanded=False,full_qualification_pending=True,created_epoch=time.time()));print(json.dumps(dict(status='ALL_FINAL54_REAL_SEALS_FROZEN_PENDING_G10_AUDIT',epoch=time.time(),frozen=fref,planned_scope=scope,questions=6818,new5_trajectories=34090,workers=workers)))
