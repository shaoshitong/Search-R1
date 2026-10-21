"""Read-only frozen unsealed-donor batch metadata, never a cross-cluster claim."""
import importlib.util,json,subprocess,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORKER='''import pathlib,json,hashlib,time
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
c=json.loads((B/'config.json').read_bytes());m=json.loads((B/'inputs/manifest.json').read_bytes());assert c['seed']==20261021 and c['run_id']=='70188b8e50fd4b45bc420ce58014f46e'
assert c['sampling_attempt_id']=='independent-seed20261021-g5-20261020'and c['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
remaining=[];sealed=[]
for item in m['batches']:
 p=pathlib.Path(item['path']);rel=p.relative_to(B);batch=int(p.stem.split('-')[-1]);shard=int(p.parent.name.split('-')[-1]);done=B/'sampling'/p.parent.name/'batches'/p.stem/'complete.json'
 if done.exists():
  v=json.loads(done.read_bytes());assert v['run_id']==c['run_id']and v['questions']==item['questions']and v['input_sha256']==item['sha256'];sealed.append(dict(relative=str(rel),questions=v['questions'],complete=proof(done)))
 else:
  actual=proof(p);assert actual['sha256']==item['sha256'];remaining.append(dict(relative=str(rel),questions=item['questions'],input_bytes=actual['bytes'],input_sha256=actual['sha256'],source_shard=shard,source_batch=batch,original_partition_stride=2,original_batch_questions=128))
keys=['run_id','seed','sampling_attempt_id','model','model_revision','capture_prefix_tokens','question_max_tokens','feature_layer','feature_dimension','actor_hidden_native','actor_optimizer_updates','source_seed','source_run_id']
last=max((B/'sampling').glob('shard-*/batches/batch*/complete.json'),key=lambda p:p.stat().st_mtime);cfg=last.parent/'config.json';v=json.loads(cfg.read_bytes());runtimekeys=['rollout_memory_utilization','rollout_utilization_cap','rollout_free_reserve_gib','rollout_max_num_seqs','rollout_max_batched_tokens','rollout_enforce_eager','rollout_retry_initialization_oom','retrieval_threads_per_engine','retrieval_timeout']
files={}
for rel in ['reproduction/operations/qwen7b_resample/worker.py','reproduction/operations/qwen7b_warmup/sampling.py','reproduction/prefix_selection200/rollout.py','reproduction/prefix_selection200/interaction_protocol.py','reproduction/prefix_selection200/contract.py']:
 p=B/'code'/rel;files[rel]=proof(p)if p.exists()else dict(missing=True)
out=dict(role='HPC3_INDEPENDENT_TAIL_G5_PREPARATION_NO_DONOR_CLAIM',epoch=time.time(),source_base=str(B),source_config=proof(B/'config.json'),source_manifest=proof(B/'inputs/manifest.json'),source_identity={k:c[k]for k in keys if k in c},actual_last_sealed_config=proof(cfg),actual_generation_settings={k:v[k]for k in runtimekeys if k in v},source_code=files,remaining_batches=remaining,sealed_batch_count=len(sealed),sealed_questions=sum(x['questions']for x in sealed),questions=sum(x['questions']for x in remaining),trajectories=5*sum(x['questions']for x in remaining),sealed_membership_sha256=hashlib.sha256(json.dumps(sealed,sort_keys=True,separators=(',',':')).encode()).hexdigest(),potentialOverlap=True,shared_donor_atomic_claim=False,local_two_workers_disjoint=True,frozen_main_cohort_questions=124194,frozen_main_cohort_expanded=False,source_modified=False,payload_transferred_to_Mac=False)
assert out['sealed_questions']+out['questions']==169028
print(json.dumps(out,sort_keys=True))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('h100-10','python3 -c '+shlex.quote(WORKER)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-800:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'source-tail-plan.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items()if k!='remaining_batches'},sort_keys=True))
