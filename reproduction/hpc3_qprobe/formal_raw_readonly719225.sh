#!/usr/bin/env bash
# Pure CPU completed11..14 raw/gold readback; not a complete200 goal consumer.
set -euo pipefail
cd '/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941'
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' QPROBE_SOURCE=/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941 /data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014/env/bin/python - <<'PY'
import os,json,time,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from search_g1.mixed_runtime import strict_vectors
from search_g1.qprobe_runtime import _complete_generation
from search_g1.full_state_restore import equal_tree
from search_g1.qprobe_features import sha_file,sha_json,original_question_and_id
from search_g1.qprobe_selection import PredictQuestion
from reproduction.hpc3_qprobe.verify_goal import verify_selection,need
b=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014')
run=json.loads((b/'logs/run-manifest.json').read_text())
need(os.getuid()==14090 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU-only owner required')
train=b/'data/train.parquet';train_sha=sha_file(train)
need(train_sha==run['input_sha256']['train.parquet'],'canonical train bytes changed')
frame=pd.read_parquet(train,columns=['data_source','extra_info','prompt','reward_model'])
tokenizer=AutoTokenizer.from_pretrained(run['initial_model'],local_files_only=True,trust_remote_code=False)
for name in ('search_g1/signals.py','search_r1/qa_protocol.py','verl/utils/reward_score/qa_em.py'):
 need(sha_file(Path(name))==run['source_sha256'][name],'pinned scorer source differs')
tokenizer_proofs={}
for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
 value=sha_file(Path(run['initial_model'])/name)
 need(value==run['initial_model_sha256'][name],'bound tokenizer bytes differ');tokenizer_proofs[name]=value
def normal(v):
 if isinstance(v,np.ndarray):return normal(v.tolist())
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:normal(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [normal(x) for x in v]
 return v
rows={r['step']:r for r in [json.loads(x) for x in (b/'logs/actual-updates.jsonl').read_text().splitlines()]}
for step in [11,12,13,14]:
 aid=rows[step]['attempt_id'];d=b/'artifacts/question-probe/attempts'/aid;p=d/'real-rollout.pt';s0=p.stat()
 need(p.is_file() and not p.is_symlink() and s0.st_uid==14090,'unsafe raw rollout')
 raw_sha=sha_file(p);raw=torch.load(p,map_location='cpu',weights_only=False)
 q=json.loads((d/'selection.json').read_text());a=json.loads((d/'rollout-audit.json').read_text())
 ev=json.loads((b/'artifacts/question-probe/completed'/(aid+'.json')).read_text())
 _complete_generation(raw['output'],raw['identity'])
 ident=raw['identity']
 need(ident['run_id']==run['run_id'] and ident['source_sha256']==run['source_sha256'] and ident['slurm_job_id']=='719225' and ident['logical_update']==step and ident['attempt_id']==aid and ident['policy_snapshot_id']==q['policy_snapshot_id'],'raw actual producer/generation identity differs')
 vectors=strict_vectors(raw['parents'],raw['output'],tokenizer)
 need(len(vectors)==512 and all(len(v)==5 for v in vectors),'decoded actual G5 budget differs')
 equal_tree(vectors,raw['strict_correctness']);equal_tree(vectors,[g['strict_correctness'] for g in a['groups']])
 need(raw['selected_ordinals']==[g['ordinal'] for g in a['groups']],'raw selected group order differs')
 need(ident['trajectory_ids']==[t for g in a['groups'] for t in g['trajectory_ids']],'raw trajectory identities differ')
 measured=verify_selection(q,a,run['question_probe']['seed']);equal_tree(measured,ev['prequential'])
 need(ev['outer']==step and ev['prediction_version']==q['head_version'] and ev['head_update']['update_count']==step and ev['label_policy_snapshot_id']==q['policy_snapshot_id'] and ev['successful_actor_snapshot_id']==run['run_id']+':'+str(step),'actual actor/head transaction identity differs')
 nt=raw['parents'].non_tensor_batch;wanted={str(v) for v in nt['candidate_stable_id']}
 need(len(wanted)==512,'raw parent qids duplicate');wanted_pairs={tuple(json.loads(v)) for v in wanted}
 matches={v:[] for v in wanted}
 for src,extra,prompt,reward in frame.itertuples(index=False,name=None):
  idx=extra.get('index') if isinstance(extra,dict) else None
  if idx is not None and (str(src),str(idx)) not in wanted_pairs:continue
  question,qid=original_question_and_id(dict(data_source=src,extra_info=extra,prompt=prompt))
  if qid in wanted:matches[qid].append((question,reward))
 bound=[]
 for i,g in enumerate(a['groups']):
  row=q['questions'][g['ordinal']];qid=str(nt['candidate_stable_id'][i]);question=str(nt['candidate_original_question'][i]);point=PredictQuestion(g['ordinal'],qid,question)
  need(qid==row['qid']==g['question_id'] and point.question_hash==row['question_hash'] and point.content_hash==row['content_sha256']==g['content_sha256'],'raw selected qid/text/content differs')
  gold=normal(nt['reward_model'][i]['ground_truth']['target'])
  need(matches[qid] and any(t==question and normal(r['ground_truth']['target'])==gold for t,r in matches[qid]),'raw parent question/gold absent from canonical SHA-bound train')
  bound.append(dict(qid=qid,question_hash=point.question_hash,target=gold))
 s1=p.stat();need((s0.st_ino,s0.st_size,s0.st_mtime_ns)==(s1.st_ino,s1.st_size,s1.st_mtime_ns),'raw changed during read')
 counts=[sum(v) for v in vectors]
 print(json.dumps(dict(epoch=time.time(),step=step,uid=os.getuid(),actual_cpu_only=True,model_weights_loaded=False,new_trajectories_generated=0,actor_or_head_updates_performed=0,raw_path=str(p),raw_bytes=s1.st_size,raw_sha256=raw_sha,actual_complete_generation_metadata_verified=True,pinned_scorer_sources_equal=True,tokenizer_files_sha256=tokenizer_proofs,actual_raw_decode_strict_terminal_em_equal=True,actual_raw_selected_groups_equal=True,actual_raw_trajectory_ids_equal=True,actual_prequential_head_event_equal=True,actual_actor_head_transaction_identity_equal=True,canonical_train_path=str(train.resolve()),canonical_train_sha256=train_sha,canonical_train_rows=len(frame),actual_all512_parent_qid_text_content_gold_equal=True,bound_parent_gold_digest=sha_json(bound),questions=512,trajectories=2560,strict_correct=sum(counts),K_distribution={str(k):counts.count(k) for k in range(6)},head_version=ev['head_version'],source_commit=run['source_commit'])),flush=True)
 del raw,vectors,nt;gc.collect()

PY

