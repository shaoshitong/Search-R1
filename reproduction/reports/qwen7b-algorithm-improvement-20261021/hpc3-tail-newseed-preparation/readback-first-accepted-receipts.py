"""Remote, metadata-only re-read after two actual CPU first-batch audits."""
import hashlib,json,pathlib,time,subprocess
BASE=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
tail=BASE/'hpc3-tail-newseedG5-20261021-v1';audit=BASE/'hpc3-tail-firstbatch-audit-v1'
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
config=proof(tail/'config.json');assert config['sha256']=='368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835'
pub=read(tail/'source-publication.json');assert pub['commit']=='2590d0b272013264b6d727a6a38493a790ee1305'
codepub=read(audit/'source-publication.json');assert codepub['source_commit']=='8599d1fdc7ff8538be3478fb236b05446d7b8598'and proof(pathlib.Path(codepub['code']['path']))==codepub['code']
workers=[]
for w,job in enumerate(['745518','745519']):
 assert(tail/('worker-%d.job.id'%w)).read_text().strip()==job
 p=audit/('worker-%d-first.json'%w);v=read(p);assert v['job_id']==job and v['strict_EM_recomputed']==640 and v['strict_EM_mismatches']==0 and v['questions']==128 and v['trajectories']==640 and v['actor_updates']==0
 assert v['status']=='HPC3_OWN_FIRST_BATCH_P20_G5_STRICT_EM_CAUSAL_NATIVE_SEEDS_INDEPENDENTLY_PASSED'and v['real_G5_ownIDs_and_seeds_verified']and v['Prefix20_causal_positions_verified']and not v['CUDA_initialized']and not v['model_tensors_loaded']and v['auditor_source']==codepub['code']
 assert proof(pathlib.Path(v['seal']['path']))==v['seal']
 h=read(audit/('worker-%d-actual-hardware.json'%w));assert h['job_id']==job and h['GPU_metadata_complete']and len(h['read_only_proc_GPU_metadata'])==2 and all('H100'in x['Model']for x in h['read_only_proc_GPU_metadata'])
 old=audit/('worker-%d-step.log'%w);progress=read(tail/'workers'/('worker-%d'%w)/'progress.json')
 workers.append(dict(worker=w,job_id=job,accepted_receipt=proof(p),hardware=proof(audit/('worker-%d-actual-hardware.json'%w)),actual_H100_GPUs=2,first_seal=v['seal'],strict_EM_recomputed=v['strict_EM_recomputed'],strict_EM_mismatches=0,K5_histogram=v['K5_histogram'],initial_GPU0_nvidia_probe_failure_preserved=proof(old)if old.exists()else None,current_completed_batches=progress['completed_batches'],current_sealed_questions=progress['questions'],current_sealed_trajectories=progress['trajectories']))
print(json.dumps(dict(status='BOTH_OWN_FIRST_REAL_G5_BATCHES_INDEPENDENTLY_ACCEPTED_AND_SMALL_RECEIPTS_FRESH_SHA_VERIFIED',epoch=time.time(),namespace=str(tail),config=config,generation_source_publication=proof(tail/'source-publication.json'),independent_audit_publication=proof(audit/'source-publication.json'),workers=workers,potentialOverlap=True,whole_tail_complete=False,actor_updates=0,new_GPU_jobs=0,model_tensor_or_raw_rollout_loaded=False),sort_keys=True))
