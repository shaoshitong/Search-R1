"""Bind Prefix5/P10 to four physical GPUs shared by training and retrieval."""
import argparse,csv,hashlib,io,json,os,pathlib,subprocess,time,uuid
BASE=pathlib.Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
COMMON=pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')
OFFICIAL=pathlib.Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725')
REUSED=pathlib.Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
R17=pathlib.Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016-r17')
from reproduction.prefix_selection200.gpu_layout import SHARED


def shared_devices(selected, by_index):
 if len(selected)!=4 or len(set(selected))!=4:
  raise ValueError('Expected exactly four physical Slurm GPUs, shared by training and retrieval')
 uuids=[by_index[x] for x in selected]
 if len(set(uuids))!=4:raise ValueError('Actual four GPU UUIDs must be distinct')
 return dict(gpu_resource_layout=SHARED,physical_gpu_indices=list(selected),physical_gpu_uuids=uuids,
  auxiliary_gpu_indices=list(selected),auxiliary_gpu_uuids=list(uuids),grounding_physical_gpus=list(selected),
  grounding_physical_gpu=selected[-1],grounding_refit_visible_device=selected[-1])


def visible_allocation(visible, assigned, by_index):
 """Slurm job GPU IDs are global; CUDA/NVML IDs can be cgroup-local."""
 by_uuid={value:key for key,value in by_index.items()}
 values=visible if visible!=[''] else assigned
 selected=[]
 for value in values:
  if value.isdigit() and int(value) in by_index:selected.append(int(value))
  elif value in by_uuid:selected.append(by_uuid[value])
  else:raise ValueError('Visible GPU absent from NVML inventory: '+repr(dict(visible=visible,slurm_job_gpus=assigned,nvml=by_index)))
 if len(selected)!=4 or len(set(selected))!=4:
  raise ValueError('Exactly four distinct Slurm-visible GPUs required: '+repr(dict(visible=visible,slurm_job_gpus=assigned,nvml=by_index)))
 return selected


def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()


def once(path,value):
 with pathlib.Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())


def adaptive_policy(config):
 from reproduction.prefix_selection200.adaptive_candidates import policy
 source=pathlib.Path(config['source_question_pool'])
 if sha(source)!=config['source_question_pool_sha256']:raise ValueError('Adaptive eligible pool changed')
 pool=json.loads(source.read_text())
 if pool['eligible_unique_questions']!=len(pool['records']):raise ValueError('Adaptive distinct pool size differs')
 return policy(len(pool['records']))


def prepare():
 a=BASE/'reference-assets';p=BASE/'preparation';p.mkdir(exist_ok=True)
 done=p/'assets-ready.json'
 if done.exists():
  prior=json.loads(done.read_text());publication=json.loads((p/'source-publication.json').read_text())
  template=json.loads((p/'launch-config-template.json').read_text());desired=adaptive_policy(template)
  policy_changed=template.get('adaptive_candidates')!=desired
  layout_changed=template.get('gpu_resource_layout')!=SHARED
  if prior['commit']!=publication['commit'] or policy_changed or layout_changed:
   if (p/'submit-intent.json').exists():raise ValueError('Cannot rebind a claimed runtime')
   template.update(source_commit=publication['commit'],adaptive_candidates=desired,gpu_resource_layout=SHARED,
    grounding_physical_gpus=[0,1,2,3],grounding_physical_gpu=3,grounding_refit_visible_device=3)
   (p/'launch-config-template.json').write_text(json.dumps(template,sort_keys=True))
   prior.update(commit=publication['commit'],code_rebind_before_launch_unix=time.time(),verified_scientific_inputs_unchanged=True,
    adaptive_candidates=desired,adaptive_policy_changed=policy_changed,gpu_resource_layout=SHARED,
    total_physical_gpus=4);done.write_text(json.dumps(prior,sort_keys=True))
  return prior
 accepted=json.loads((BASE/'receive-complete.json').read_text())
 if accepted['status']!='REFERENCE_ASSETS_ALL_TARGET_SHA_VERIFIED':raise ValueError('Target assets have not passed SHA')
 publication=json.loads((p/'source-publication.json').read_text());template=json.loads((a/'reference-launch-config.json').read_text())
 c=dict(template,base=str(BASE),run_id=publication['run_id'],execution_id=publication['execution_id'],source_commit=publication['commit'],execution_backend='slurm_hpc3',
  environment_root=str(BASE/'env'),model_dir=str(OFFICIAL),official_model_dir=str(OFFICIAL),encoder_model_dir=str(OFFICIAL),actor_model_dir=str(OFFICIAL),actor_version=publication['run_id']+'-official0',
  canonical_train=str(COMMON/'train.parquet'),validation_questions=str(COMMON/'validation.parquet'),final_questions=str(COMMON/'final.parquet'),common_manifest_path=str(COMMON/'manifest.json'),
  source_question_pool=str(a/'source-pool.json'),fixed_test_identity_path=str(a/'fixed512-identities.json'),composite_manifest_path=str(a/'composite-manifest.json'),
  probe_source_checkpoint=str(a/'warmup-epoch5-ema.pt'),probe_source_sha256=sha(a/'warmup-epoch5-ema.pt'),probe_source_bytes=(a/'warmup-epoch5-ema.pt').stat().st_size,
  fixed_test_features=str(a/'feature-provenance-only'),
  retrieval_index='/data/user/jxu957/experiments/topd-20261008/bundle/retrieval/wiki/e5_Flat.index',
  retrieval_corpus='/data/user/jxu957/experiments/topd-20261008/bundle/retrieval/wiki/wiki-18.jsonl',
  retrieval_encoder=str(REUSED/'source/assets-root/assets/e5-base-v2'),retriever_gpu_lib=str(BASE/'retriever-gpu-lib'),retrieval_data_cache=str(BASE/'retrieval-cache'),
  grounding_calibration_questions=str(a/'calibration-300.parquet'),grounding_calibration_data=str(a/'calibration-300.parquet'),
  gpu_resource_layout=SHARED,grounding_physical_gpus=[0,1,2,3],grounding_refit_visible_device=3,grounding_physical_gpu=3,python_header_cpath='/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/bootstrap-tools/python-dev/usr/include/python3.10',
  rollout_enforce_eager=True,pool_state_path=str(BASE/'question-pool/state.json'),prefix_probe_receipt=str(BASE/'probe/latest.json'),
  candidate_schedule=dict(effective_from_outer=1,legacy_candidates=2560,ordinary_candidates=5120,update_candidates=2560,probe_update_period=10,probe_update_at_final=False,selected_questions=512))
 # Scientific source files, labels and EMA remain byte-identical. Only the paths
 # inside the grounding manifest are rebound to their already SHA-verified copy.
 ground=json.loads((a/'grounding/version-manifest.json').read_text());ground.update(encoder=str(OFFICIAL),boundary=str(a/'grounding/boundary_probe.pkl'),commit=str(a/'grounding/commit_probe.pkl'))
 target=p/'grounding-initial-manifest.json';once(target,ground)
 c.update(grounding_manifest=str(target),grounding_manifest_sha256=sha(target),grounding_initial_manifest=str(target),grounding_initial_manifest_sha256=sha(target),grounding_probe_version=ground['version'])
 feature=a/'feature-provenance-only';feature.mkdir(exist_ok=True)
 (feature/'manifest.json').write_bytes((a/'feature-manifest.json').read_bytes())
 checked=[]
 for key,digest in [('canonical_train','canonical_train_sha256'),('validation_questions','validation_questions_sha256'),('final_questions','final_questions_sha256'),('common_manifest_path','common_manifest_sha256'),('source_question_pool','source_question_pool_sha256'),('fixed_test_identity_path','fixed_test_identity_sha256'),('grounding_calibration_questions','grounding_calibration_sha256')]:
  actual=sha(c[key])
  if actual!=c[digest]:raise ValueError('HPC3 existing input differs: '+key)
  checked.append(dict(role=key,sha256=actual))
 for name,digest in c['model_files_sha256'].items():
  if sha(OFFICIAL/name)!=digest:raise ValueError('Official actor file differs: '+name)
 for key,name in [('boundary','boundary_probe.pkl'),('commit','commit_probe.pkl')]:
  if sha(ground[key])!=ground['head_sha256'][name]:raise ValueError('Grounding head differs')
 from reproduction.prefix_selection200.pool import fixed_test_identities
 fixed_test_identities(c)
 c['adaptive_candidates']=adaptive_policy(c)
 from reproduction.prefix_selection200.candidate_schedule import candidate_count
 assert candidate_count(c,1)==5120
 once(p/'launch-config-template.json',c)
 result=dict(status='HPC3_PREFIX5_TARGET_ASSETS_READY',commit=publication['commit'],checked_unix=time.time(),fixed_warmup_model_only=True,online_probe_adam_initial=0,checked=checked,full_warmup_cache_copied=False,adaptive_candidates=c['adaptive_candidates'])
 once(done,result);return result


def run():
 if not os.environ.get('SLURM_JOB_ID'):raise ValueError('A real Slurm allocation is required')
 prepare();p=BASE/'preparation';c=json.loads((p/'launch-config-template.json').read_text())
 if (p/'submit-intent.json').exists():raise ValueError('Existing execution claim: never restart blindly')
 visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assigned=os.environ.get('SLURM_JOB_GPUS','').split(',')
 mapping=list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True))))
 by_index={int(x[0]):x[1].strip() for x in mapping}
 selected=visible_allocation(visible,assigned,by_index)
 c.update(shared_devices(selected,by_index),slurm_job=os.environ['SLURM_JOB_ID'],slurm_node=os.uname().nodename)
 train,aux=c['physical_gpu_indices'],c['auxiliary_gpu_indices']
 from reproduction.prefix_selection200.launch import validate_config
 validate_config(c,BASE);once(p/'launch-config.json',c)
 once(p/'submit-intent.json',dict(config_sha256=sha(p/'launch-config.json'),run_id=c['run_id'],source_commit=c['source_commit'],job=os.environ['SLURM_JOB_ID'],created_unix=time.time()))
 once(BASE/'logs/slurm-allocation.json',dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,uid=os.getuid(),training=train,auxiliary=aux,assigned_uuids=[by_index[x] for x in selected],slurm_global_job_gpu_ids=assigned,initial_cuda_visible_devices=visible,total_physical_gpus=4,gpu_resource_layout=SHARED,time=time.time()))
 from reproduction.prefix_selection200.launch import run as launch
 return launch(BASE)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
 print(json.dumps(prepare() if args.action=='prepare' else run()),flush=True)
