#!/usr/bin/env bash
# Independent CPU row readback for actual completed15; no training, HF forward, or submissions.
set -euo pipefail
cd '/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941'
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' QPROBE_SOURCE=/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941 /data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014/env/bin/python - <<'PY'
import os,json,time,hashlib,stat
from pathlib import Path
import torch
from reproduction.common8.behavior_runtime import cast_token_fields,row_hash,read_gate_receipt
from reproduction.common8.behavior_trace import document_sha256
from reproduction.hpc3_qprobe.verify_goal import need
b=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014')
run=json.loads((b/'logs/run-manifest.json').read_text());step=15
row=next(x for x in [json.loads(s) for s in (b/'logs/actual-updates.jsonl').read_text().splitlines()] if x['step']==step)
aid=row['attempt_id'];p=b/'artifacts/question-probe/attempts'/aid/'real-rollout.pt'
need(os.getuid()==14090 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU owner only')
need(p.resolve()==p and not p.is_symlink(),'unsafe raw path')
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
 before=os.fstat(stream.fileno());need(stat.S_ISREG(before.st_mode) and before.st_uid==14090,'unsafe raw file or owner')
 h=hashlib.sha256()
 for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
 need(h.hexdigest()=='717e0a81b02acbc38572d54b3b527acc962666110095105f97f14a10e9782953','raw15 previously independent canonical-label-bound SHA differs')
 stream.seek(0);raw=torch.load(stream,map_location='cpu',weights_only=False);after=os.fstat(stream.fileno())
 need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'raw mutated while loading')
binding=read_gate_receipt(b,run,step,update_record=row)
gate_path=b/'logs'/('behavior-probability-attempt-'+aid+'.json');gate=json.loads(gate_path.read_text())
ident=raw['identity'];out=raw['output']
need(ident['run_id']==run['run_id'] and ident['source_sha256']==run['source_sha256'] and ident['slurm_job_id']==gate['slurm_job_id']=='719225' and ident['logical_update']==step and ident['attempt_id']==aid and ident['policy_snapshot_id']==gate['policy_snapshot_id'],'actual raw identity vs gate differs')
parents=[]
for i,value in enumerate(raw['parents'].non_tensor_batch['index']):
 if hasattr(value,'item'):value=value.item()
 parents.append(str(i)+'-'+document_sha256(value))
need(parents==ident['parent_group_ids'],'actual parent occurrence/index digest differs')
expected_ids=[run['run_id']+'-'+str(step)+'-attempt-'+aid+'-'+parent+'-sample-'+str(sample) for parent in parents for sample in range(5)]
need(expected_ids==ident['trajectory_ids']==gate['trajectory_ids']==out.non_tensor_batch['behavior_trajectory_id'].tolist(),'actual ordered G5 raw/actor/gate trajectory IDs differ')
cast_token_fields(out)
out.batch['old_log_probs']=out.batch['sampling_log_probs'].clone()
need(torch.isfinite(out.batch['old_log_probs']).all().item(),'actual saved sampling LP nonfinite')
actual_rows={tid:row_hash(out,i) for i,tid in enumerate(expected_ids)}
need(actual_rows==gate['row_tensor_sha256'],'actual raw row tensor bytes vs accepted actor gate differ')
upstream={tid:parents[i//5].split('-',1)[1] for i,tid in enumerate(expected_ids)}
need(upstream==gate['upstream_group_sha256'],'actual original parent group bytes vs accepted actor gate differ')
print(json.dumps(dict(epoch=time.time(),step=15,actual_cpu_only=True,model_weights_loaded=False,new_generation=0,actor_or_head_updates_performed=0,private_trace_reconstructed=False,actual_saved_raw_full_SHA=h.hexdigest(),gate_sha256=binding['proof_sha256'],actual_raw2560_row_tensor_hashes_equal_to_actor_gate=True,actual_raw2560_ordered_IDs_equal=True,actual_raw512_parent_index_group_digests_equal=True,actual_saved_sampling_LP_clone_equal_and_finite=True,raw_row_tensor_combined_sha256=document_sha256(actual_rows),scope='all actual row fields + ordered identities + parent index hashes, paired with independentcanonical-gold/terminalEM raw15 proof; private trace and per-token context not yet independently reconstructed')))
PY

