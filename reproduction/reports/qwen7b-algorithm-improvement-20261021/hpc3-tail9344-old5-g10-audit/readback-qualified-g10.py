"""Terminal small receipt/metadata SHA readback; no raw or tensor rescan."""
import pathlib,json,hashlib,fcntl,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');base=A/'hpc3-tail9344-old5-qualified-g10-v1';new=A/'hpc3-tail73-full-new5-audit-v1'
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(ref):
 assert proof(ref['path'])==ref;return json.loads(pathlib.Path(ref['path']).read_bytes())
p=base/'complete.json';ref=proof(p);assert ref['sha256']=='0898b54516068d5b2c0fd1e2d95befb47d58ed2e96ca1065fc9b556387abbbfd';done=checked(ref);n=checked(done['new5_full_qualification']);assert n['new5_full_source_qualified']and n['strict_EM_recomputed']==46720
for r in done['old5_batch_audits']:checked(r)
groups=checked(done['outputs']['groups'])['train'];events=checked(done['outputs']['source-events']);contract=checked(done['outputs']['source-resolution-contract']);assert len(groups)==9344 and len({g['qid']for g in groups})==9344 and len({t for g in groups for t in g['trajectory_ids']})==93440 and all(len(g['trajectory_ids'])==len(g['labels'])==10 for g in groups)
assert done['raw_G10_full_qualified']and done['strict_EM_mismatches']==done['actual_same_question_old_new_seed_collisions']==0 and done['old5_strict_EM_recomputed']==46720 and done['new5_strict_EM_recomputed']==46720 and not done['CUDA_initialized']and not done['model_tensors_loaded']and not done['feature_bank_created']and not done['current124194_fit_expanded']
locks=[]
for name in('audit.lock','launch.lock'):
 with(base/name).open('a+')as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(name)
print(json.dumps(dict(status='FIXED9344_RAW_G10_FULL_QUALIFICATION_SMALL_RECEIPTS_AND_METADATA_FRESH_SHA_PASSED',epoch=time.time(),complete=ref,outputs=done['outputs'],questions=9344,trajectories=93440,strict_EM_mismatches=0,old_new_native_seed_collisions=0,old5_batches=len(done['old5_batch_audits']),CPU_locks_released=locks,contract=contract,feature_bank_created=False,current124194_fit_expanded=False,raw_response_or_model_tensor_rehashed_in_terminal_readback=False),sort_keys=True))
