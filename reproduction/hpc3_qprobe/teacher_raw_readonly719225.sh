#!/usr/bin/env bash
# Pure CPU raw-cache readback; see raw_readback719225.md. No training or submissions.
set -euo pipefail
cd '/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941'
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES='' QPROBE_SOURCE=/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/source/1843babdd96b387dd10d91e3207c9ea9489b6941 /data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014/env/bin/python - <<'PY'
import os,json,time,gc,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from verl import DataProto
from search_g1.qprobe_features import sha_file,sha_json,original_question_and_id,question_text_sha
from search_g1.qprobe_selection import QuestionPool
from search_g1.qprobe_warmup import TeacherCache,moments,verify_frozen
from search_g1.mixed_runtime import strict_vectors
from search_g1.qprobe_runtime import _complete_generation
from search_g1.full_state_restore import equal_tree
from reproduction.hpc3_qprobe.verify_goal import need
b=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014')
run=json.loads((b/'logs/run-manifest.json').read_text());root=Path(run['question_probe']['teacher_warmup_cache']).resolve()
need(os.getuid()==14090 and os.environ.get('CUDA_VISIBLE_DEVICES')=='','actual CPU-only owner required')
identity_path=root/'identity.json';complete_path=root/'complete.json'
identity=json.loads(identity_path.read_text());complete=json.loads(complete_path.read_text())
need(sha_file(identity_path)==complete['identity_sha256'],'teacher identity SHA changed')
need(complete['questions']==5120 and complete['trajectories']==25600 and identity['training_world_size']==4 and len(identity['questions'])==5120,'teacher actual budgets differ')
need(identity['model_revision']==run['model_revision'] and identity['model_files_sha256']==run['initial_model_sha256'],'actual official teacher identity differs')
for name,expected in identity['scorer_sources'].items():need(sha_file(Path(name))==expected,'original teacher scorer bytes changed')
for name,expected in identity['rollout_sources'].items():need(sha_file(Path(name))==expected,'original teacher rollout bytes changed')
need(identity['seed']==run['question_probe']['seed'],'canonical teacher seed differs')
train=b/'data/train.parquet';train_sha=sha_file(train)
need(train_sha==run['input_sha256']['train.parquet']==identity['train_parquet_sha256'],'teacher canonical train bytes changed')
frame=pd.read_parquet(train,columns=['data_source','extra_info','prompt','reward_model'])
pool=QuestionPool(5120);canonical=[]
for index in np.random.default_rng(identity['seed']).permutation(len(frame)):
 row=frame.iloc[int(index)];question,qid=original_question_and_id(row,'prompt')
 if pool.reserve(qid,question) is not None:
  canonical.append((int(index),question,qid,row['reward_model']))
 if len(canonical)==5120:break
rebuilt=[dict(qid=x[2],question_hash=question_text_sha(x[1])) for x in canonical]
equal_tree(rebuilt,identity['questions'])
tokenizer=AutoTokenizer.from_pretrained(run['initial_model'],local_files_only=True,trust_remote_code=False)
for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
 need(sha_file(Path(run['initial_model'])/name)==identity['model_files_sha256'][name],'actual official teacher tokenizer bytes changed')
cache=object.__new__(TeacherCache);cache.root=root;cache.identity=identity
# Bypass only the writer constructor/lock. validate_chunk remains the original pure reader.
for index in [0,1,2,3,4,5,6,7,8,9]:
 chunk_path=root/('chunk-%02d.json'%index);record=complete['chunks'][index]
 need(record['relative_path']==chunk_path.name and sha_file(chunk_path)==record['sha256'],'teacher chunk receipt SHA differs')
 value=json.loads(chunk_path.read_text());cache.validate_chunk(index,value)
 raw_path=root/value['raw_rollout']['relative_path'];before=raw_path.stat()
 need(raw_path.is_file() and not raw_path.is_symlink() and raw_path.resolve().parent==root and before.st_uid==14090,'unsafe teacher raw owner/path')
 raw=torch.load(raw_path,map_location='cpu',weights_only=False)
 _complete_generation(raw['output'],raw['identity'])
 expected=identity['questions'][index*512:(index+1)*512]
 equal_tree(raw['question_bindings'],expected)
 equal_tree(raw['contracts'],value['contracts']);equal_tree(raw['frozen_workers'],value['frozen_workers'])
 frozen=verify_frozen(raw['frozen_workers']['before'],raw['frozen_workers']['after'],4)
 ident=raw['identity'];need(ident['logical_update']==0 and ident['teacher_warmup'] is True,'teacher actual outer identity differs')
 need(ident['slurm_job_id']==('718509' if index<2 else '718650'),'actual authorized teacher producer job differs')
 for name,digest in identity['scorer_sources'].items():need(ident['source_sha256'][name]==digest,'raw teacher scorer namespace differs')
 for name,digest in identity['rollout_sources'].items():need(ident['source_sha256'][name]==digest,'raw teacher rollout namespace differs')
 parent=DataProto.from_dict(tensors={'qprobe_source_index':torch.tensor([x[0] for x in canonical[index*512:(index+1)*512]])},non_tensors={'reward_model':np.array([x[3] for x in canonical[index*512:(index+1)*512]],dtype=object)})
 vectors=strict_vectors(parent,raw['output'],tokenizer)
 need(len(vectors)==512 and all(len(v)==5 for v in vectors),'teacher actual decoded G5 budget differs')
 need(ident['trajectory_ids']==[tid for g in value['groups'] for tid in g['trajectory_ids']],'teacher raw trajectory IDs differ')
 for offset,(vector,group) in enumerate(zip(vectors,value['groups'])):
  equal_tree(vector,group['strict_em']);equal_tree({k:group[k] for k in ('qid','question_hash')},expected[offset])
  count,mean,std=moments(vector)
  need((count,mean,std)==(group['correct_count'],group['mean'],group['std']),'teacher raw decoded moments differ')
  independent_std=math.sqrt(sum((v-mean)**2 for v in vector)/4)
  need(math.isclose(independent_std,std,rel_tol=1e-14,abs_tol=1e-14),'teacher sample std ddof1 differs')
  need(group['run_id']==ident['run_id'] and group['policy_snapshot_id']==ident['policy_snapshot_id'] and group['source_outer']==0,'teacher actual group provenance differs')
 after=raw_path.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'teacher raw changed during read')
 counts=[sum(v) for v in vectors]
 print(json.dumps(dict(epoch=time.time(),chunk=index,uid=os.getuid(),actual_cpu_only=True,cache_writer_constructed=False,cache_lock_taken=False,model_weights_loaded=False,generated_trajectories=0,actor_or_head_updates_performed=0,raw_sha256=value['raw_rollout']['sha256'],raw_bytes=before.st_size,chunk_sha256=record['sha256'],identity_sha256=complete['identity_sha256'],canonical_train_sha256=train_sha,canonical_train_rows=len(frame),canonical_all5120_seed_order_qid_text_rebuilt_equal=True,actual_all512_raw_strict_terminal_EM_K_mean_std_ddof1_equal=True,actual_frozen_four_rank_and_raw_provenance_equal=True,questions=512,trajectories=2560,strict_correct=sum(counts),K_distribution={str(k):counts.count(k) for k in range(6)},teacher_run_id=ident['run_id'])),flush=True)
 del raw,vectors,parent;gc.collect()

PY

