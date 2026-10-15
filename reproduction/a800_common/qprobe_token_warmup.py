"""Prepare then run only a fresh all-token binary-head warmup on qualified G5 labels.

CPU preparation authenticates the original teacher label producer separately
from the new token representation and scans all canonical question lengths.
Run encodes only the same5120 questions, unloads the official frozen encoder,
then fits a fresh head for10epochs. Explicit feature-only recovery authenticates
one completed encoder artifact and fits a fresh head in a new process without
loading the encoder or inheriting Head/Adam state. No formal training starts.
"""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
TEACHER='/data/shaoshitong/Search-G1-qprobe-shared/teacher-warmup-v4-sigma3-fullouter'
RESUME_OUTPUT='/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31/warmup'
RESUME_SOURCE='bed0f379f6f65f0695fdc2e24b72818edf1c61c9'
RESUME_METADATA_SHA={
    'prepared.json':'449cbfcaa514bca3eb368f395ebfe7ad770685a31e6fcaadcb755ad2c57dace9',
    'run-failure.json':'3f26f96ddab67a086e2a3c627a4018b9b2917130f096eecdf0e6224645f38900',
    'run-claim.json':'3ad2df1b753bf865f4dfcc0e9da303c652e27d4b444b99f75bb6106c10046a73'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def read(path):
    path=Path(path);assert path==path.resolve() and not path.is_symlink() and path.stat().st_uid==os.getuid()
    assert path.stat().st_size<=128*1024**2
    return json.loads(path.read_bytes())


def remote(path):
    path=Path(path);assert path.is_absolute() and path.resolve()==path and str(path).startswith(('/data/','/mnt/','/backup/'))
    return path


def once(path,value):
    raw=canonical(value)+b'\n';fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def progress(output,value):
    event=dict(value,observed_unix=time.time());temporary=output/'progress.partial.json'
    with temporary.open('w') as stream:json.dump(event,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,output/'progress.json')
    print(json.dumps(event,allow_nan=False),flush=True)


def source_binding(args):
    repo=Path(__file__).resolve().parents[2]
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==args.source_commit
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain']).strip()
    from reproduction.a800_common.deploy import published_source
    from reproduction.common8.source_files import runtime_files
    mapping=published_source(repo,args.source_commit)
    assert mapping=={str(path.relative_to(repo)):sha(path) for path in runtime_files(repo)}
    publication=read(args.source_publication)
    assert publication['source_commit']==publication['remote_commit']==args.source_commit
    assert publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256']==hashlib.sha256(canonical(mapping)).hexdigest()
    return mapping


def assets(args,mapping):
    from search_g1.qprobe_features import COMMON_MANIFEST_SHA,OFFICIAL_REVISION,original_question_and_id,question_text_sha
    from reproduction.common8.prepare_run import verify_model
    import pyarrow.parquet as pq
    base=remote(args.source_run_base);run=read(base/'logs/run-manifest.json')
    assert run['model_revision']==OFFICIAL_REVISION and run['initial_model']==run['reference_model']=='/data/search-g1/assets/qwen2.5-3b-instruct'
    assert run['input_sha256']['manifest.json']==COMMON_MANIFEST_SHA==sha(base/'data/manifest.json')
    assert sha(base/'data/train.parquet')==run['input_sha256']['train.parquet']
    manifest=read(base/'data/manifest.json');assert manifest['splits']['train']['sha256']==run['input_sha256']['train.parquet']
    table=pq.read_table(base/'data/train.parquet',columns=['prompt','data_source','extra_info'])
    rows=[original_question_and_id(row) for row in table.to_pylist()]
    assert len(rows)==169028 and len({qid for _,qid in rows})==169028
    official=base/'preparation/official-model-manifest-original.json'
    model=remote(run['initial_model']);hashes=verify_model(model,official);assert hashes==run['initial_model_sha256']
    old_features=read(base/'question-features/manifest.json')
    assert old_features['count']==169028 and old_features['feature_dim']==2048
    assert old_features['spec_sha']==hashlib.sha256(canonical(old_features['spec'])).hexdigest()
    assert old_features['spec']['encoder_revision']==OFFICIAL_REVISION and old_features['spec']['feature_layer']==27
    assert old_features['spec']['pooling']=='last_nonpadding_prompt_token'
    assert old_features['spec']['train_parquet_sha256']==run['input_sha256']['train.parquet']
    assert old_features['spec']['encoder_files_sha256']==hashlib.sha256(canonical(hashes)).hexdigest()
    cfg=read(base/'logs/resolved-training-config.json');config=cfg['resolved_config']
    assert cfg['run_id']==run['run_id'] and cfg['execution_id']==run['execution_id']
    assert cfg['resolved_config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return base,run,rows,hashes,old_features,config


def qualified_labels(args,run,rows,old_features,config,mapping,output):
    """Read-only verification of ORIGINAL v4 identity, never substitute token spec."""
    import fcntl
    import numpy as np
    from search_g1.qprobe_features import question_text_sha
    from search_g1.qprobe_selection import QuestionPool
    from search_g1.qprobe_warmup import TeacherCache,teacher_identity,STATUS
    teacher=remote(TEACHER);assert sha(teacher/'identity.json')==args.teacher_identity_sha256
    cached=read(teacher/'identity.json');complete=read(teacher/'complete.json')
    pool=QuestionPool(5120);chosen=[]
    seed=config['question_probe']['seed']
    for index in np.random.default_rng(seed).permutation(len(rows)):
        question,qid=rows[int(index)]
        if pool.reserve(qid,question) is not None:chosen.append((int(index),question,qid))
        if len(chosen)==5120:break
    assert len(chosen)==5120
    expected=teacher_identity(run,feature_spec_sha=old_features['spec_sha'],sampling_config=config['actor_rollout_ref']['rollout'],
        agent_config={key:config['data'][key] for key in ('max_start_length','max_prompt_length','max_response_length','max_obs_length')},
        max_turns=config['max_turns'],seed=seed,questions=[dict(qid=qid,question_hash=question_text_sha(q)) for _,q,qid in chosen])
    assert cached==expected, 'Original qualified teacher identity differs; never alias a new token spec'
    for name,digest in dict(expected['scorer_sources'],**expected['rollout_sources']).items():
        assert mapping.get(name)==run['source_sha256'][name]==digest, 'Teacher scientific source changed'
    # Bypass constructor's writer/create path. Existing complete cache is read-only.
    checker=TeacherCache.__new__(TeacherCache);checker.root=teacher;checker.identity=expected
    with (teacher/'writer.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        chunks=[]
        for index in range(10):
            chunk=checker.read_chunk(index);assert chunk is not None;chunks.append(chunk)
            progress(output,dict(stage='original_teacher_full_sha',completed_chunks=index+1,total_chunks=10))
        groups=[group for chunk in chunks for group in chunk['groups']]
        tids=[tid for group in groups for tid in group['trajectory_ids']]
        assert len(groups)==5120 and len(tids)==len(set(tids))==25600
        protocol={key:chunks[0]['contracts'][key] for key in ('sampling_contract_sha','strict_scoring_contract_sha','retriever_version')}
        assert all({key:chunk['contracts'][key] for key in protocol}==protocol for chunk in chunks)
        actual=dict(status=STATUS,questions=5120,trajectories=25600,identity_sha256=sha(teacher/'identity.json'),protocol=protocol,
            chunks=[dict(relative_path='chunk-%02d.json'%index,sha256=sha(teacher/('chunk-%02d.json'%index))) for index in range(10)])
        assert complete==actual and sha(teacher/'identity.json')==args.teacher_identity_sha256
    assert all(group['source_outer']==0 for group in groups)
    provenance=dict(teacher_cache=TEACHER,teacher_identity_sha256=args.teacher_identity_sha256,teacher_complete_sha256=sha(teacher/'complete.json'),
        original_feature_spec_sha256=old_features['spec_sha'],original_feature_manifest_sha256=sha(Path(args.source_run_base)/'question-features/manifest.json'),
        original_teacher_identity_preserved=True,full_raw_and_chunk_sha_verified=True,questions=5120,trajectories=25600,
        full_raw_rollout_bytes=sum(chunk['raw_rollout']['bytes'] for chunk in chunks),
        source_scientific_sha256=dict(expected['scorer_sources'],**expected['rollout_sources']),protocol=protocol,
        new_token_spec_not_substituted_for_original_spec=True,trajectories_regenerated=False,old_head_inherited=False)
    return chosen,groups,provenance


def task_config(args):
    config=dict(source_commit=args.source_commit,teacher_identity_sha256=args.teacher_identity_sha256,source_run_base=str(args.source_run_base),
        auxiliary_weight=args.auxiliary_weight,target_statistic=args.target_statistic,top_k=512,exploration_questions=0,
        epochs=10,batch_size=256,learning_rate=3e-4,head_seed=args.head_seed,feature_dtype=args.feature_dtype)
    if args.resume_token_feature_output is not None:config['resume_token_feature_output']=str(args.resume_token_feature_output)
    return config


def prepare(args,output,mapping):
    if args.resume_token_feature_output is not None:return prepare_head_only(args,output,mapping)
    import numpy as np
    from transformers import AutoTokenizer
    from search_g1.qprobe_token_features import scan_token_lengths
    base,run,rows,hashes,old_features,config=assets(args,mapping)
    chosen,groups,provenance=qualified_labels(args,run,rows,old_features,config,mapping,output)
    tokenizer=AutoTokenizer.from_pretrained(run['initial_model'],local_files_only=True,trust_remote_code=False)
    assert tokenizer.pad_token_id is not None
    scan_started=time.monotonic()
    lengths,token_hashes,scan=scan_token_lengths(tokenizer,rows,progress=lambda event:progress(output,event))
    scan_elapsed=time.monotonic()-scan_started
    model_config=read(Path(run['initial_model'])/'config.json');assert model_config['hidden_size']==2048
    assert scan['global_max_tokens']<=model_config['max_position_embeddings'], 'Global Bmax exceeds official context; no truncation permitted'
    np.save(output/'global-token-lengths.npy',lengths,allow_pickle=False)
    with (output/'global-token-lengths.npy').open('rb') as stream:os.fsync(stream.fileno())
    from search_g1.qprobe_features import question_text_sha
    with (output/'global-token-bindings.jsonl').open('x') as stream:
        for (question,qid),length,token_hash in zip(rows,lengths,token_hashes):
            stream.write(json.dumps(dict(qid=qid,question_hash=question_text_sha(question),length=int(length),token_ids_sha256=token_hash),sort_keys=True,ensure_ascii=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    once(output/'global-scan.json',scan)
    prepared=dict(stage='TOKEN_BINARY_WARMUP_CPU_PREPARED',task_config=task_config(args),source_sha256=mapping,
        source_map_sha256=hashlib.sha256(canonical(mapping)).hexdigest(),label_producer_provenance=provenance,global_scan=scan,
        work_cost=dict(cpu_tokenizer_rows=169028,encoder_question_rows=5120,encoder_fraction_of_canonical_train=5120/169028,
            measured_cpu_scan_seconds=scan_elapsed,measured_cpu_scan_rows_per_second=169028/max(scan_elapsed,1e-9),
            packed_feature_bytes=int(sum(int(lengths[index]) for index,_,_ in chosen))*2048*(2 if args.feature_dtype=='bfloat16' else 4),
            dense_padding_saved_on_disk=False,encoder_full_train_forward_performed=False),
        prepared_files={name:dict(sha256=sha(output/name),bytes=(output/name).stat().st_size) for name in ('global-token-lengths.npy','global-token-bindings.jsonl','global-scan.json')},
        no_gpu_model_loaded=True,formal_training_started=False,checked_unix=time.time())
    once(output/'prepared.json',prepared)
    print(json.dumps(dict(stage=prepared['stage'],output_dir=str(output),prepared_sha256=sha(output/'prepared.json'),
        global_max_tokens=scan['global_max_tokens'],teacher_identity_sha256=provenance['teacher_identity_sha256'],formal_training_started=False)),flush=True)



def resume_encoded_features(args,output,mapping):
    """Authenticate one completed encoder artifact; no encoder/head state load."""
    import numpy as np
    from reproduction.a800_common.deploy import published_source
    from reproduction.common8.source_files import runtime_files
    from search_g1.qprobe_features import question_text_sha,sha_json
    from search_g1.qprobe_token_features import TokenFeatureCache
    parent=remote(args.resume_token_feature_output);assert str(parent)==RESUME_OUTPUT
    for name,expected in RESUME_METADATA_SHA.items():assert sha(parent/name)==expected
    previous=read(parent/'prepared.json');failure=read(parent/'run-failure.json');claim=read(parent/'run-claim.json')
    assert previous['stage']=='TOKEN_BINARY_WARMUP_CPU_PREPARED' and previous['task_config']['source_commit']==RESUME_SOURCE
    assert failure['failure_type']=='AssertionError' and any(frame['file']=='qprobe_token_warmup.py' and frame['function']=='run' and frame['line']==246 for frame in failure['frames'])
    keys=('pid','start_ticks','sid','uid');expected_runner=(145986,614484752,145985,os.getuid())
    assert tuple(claim[key] for key in keys)==tuple(failure[key] for key in keys)==expected_runner
    assert claim['task_config']==previous['task_config']
    current_config={key:value for key,value in task_config(args).items() if key not in ('source_commit','resume_token_feature_output')}
    assert current_config=={key:value for key,value in previous['task_config'].items() if key!='source_commit'}
    prior_repo=parent.parent/'repo';old_map=published_source(prior_repo,RESUME_SOURCE)
    assert subprocess.check_output(['git','-C',str(prior_repo),'rev-parse','HEAD'],text=True).strip()==RESUME_SOURCE
    assert old_map==previous['source_sha256']=={str(path.relative_to(prior_repo)):sha(path) for path in runtime_files(prior_repo)}
    assert previous['source_map_sha256']==hashlib.sha256(canonical(old_map)).hexdigest()
    publication=read(parent.parent/'preparation/source-publication.json')
    assert publication['source_commit']==publication['remote_commit']==RESUME_SOURCE and publication['remote_fresh_object_store_disk_git_hash_equal'] is True
    assert publication['source_map_sha256']==previous['source_map_sha256']
    runner_name='reproduction/a800_common/qprobe_token_warmup.py'
    assert set(mapping)==set(old_map) and all(mapping[name]==old_map[name] for name in mapping if name!=runner_name), 'Resume changed scientific/core/feature source'
    original_lines=(prior_repo/runner_name).read_text().splitlines()
    assert 'assert torch.cuda.memory_allocated()==0' in original_lines[245]
    for name in ('fresh-head.pt','fresh-head.partial.pt','warmup-metrics.json','ready.json','partial-head.pt','partial-head.json'):
        assert not (parent/name).exists() and not (parent/name).is_symlink()
    parent_log=parent.parent/'logs'
    assert sha(parent_log/'run-exit.json')=='6af047d6fa270727d86d4e1ecbd96df5cfd22fa37ecab2b745c4e2502457bbd0'
    assert sha(parent_log/'run-launch.json')=='b28f6fb7d3da7a6058451f2dc526142565cdc805a331e7ff8f153d0e362f3414'
    assert read(parent_log/'run-exit.json')['exit_code']==1
    worker=read(parent_log/'run-launch.json')['process']
    assert tuple(worker[key] for key in keys)==(145985,614484748,145985,os.getuid())
    def process(pid):
        path=Path('/proc')/str(pid);raw=(path/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(fields[19]),sid=int(fields[3]),uid=path.stat().st_uid,state=fields[0])
    for record in (worker,claim):
        try:actual=process(record['pid'])
        except (FileNotFoundError,ProcessLookupError):continue
        assert actual['state'] in ('Z','X') or any(actual[key]!=record[key] for key in keys), 'Original encoder process still live'
    try:current_leader=process(worker['pid'])
    except (FileNotFoundError,ProcessLookupError):current_leader=None
    reused_session=current_leader is not None and current_leader['pid']==current_leader['sid'] and current_leader['start_ticks']!=worker['start_ticks']
    if not reused_session:
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:actual=process(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            except PermissionError:
                try:assert path.stat().st_uid!=os.getuid()
                except FileNotFoundError:pass
                continue
            assert actual['state'] in ('Z','X') or (actual['sid'],actual['uid'])!=(worker['sid'],worker['uid']), 'Original encoder session survivor'
    for name,item in previous['prepared_files'].items():assert (parent/name).stat().st_size==item['bytes'] and sha(parent/name)==item['sha256']
    base,origin,rows,hashes,old_features,config=assets(args,mapping)
    chosen,groups,provenance=qualified_labels(args,origin,rows,old_features,config,mapping,output)
    assert provenance==previous['label_producer_provenance']
    scan=read(parent/'global-scan.json');assert scan==previous['global_scan']
    lengths=np.load(parent/'global-token-lengths.npy',allow_pickle=False)
    assert lengths.dtype==np.int32 and lengths.shape==(169028,) and int(lengths.max())==scan['global_max_tokens']
    bindings=[json.loads(line) for line in (parent/'global-token-bindings.jsonl').read_text().splitlines()]
    assert len(bindings)==169028
    for index,((question,qid),record) in enumerate(zip(rows,bindings)):
        assert (record['qid'],record['question_hash'],record['length'])==(qid,question_text_sha(question),int(lengths[index]))
    cache=TokenFeatureCache(parent/'token-features');cache.verify_all();assert cache.manifest['count']==5120
    qids=[qid for _,_,qid in chosen];assert cache.ids==qids
    for index,question,qid in chosen:
        item=cache.index[qid];record=bindings[index]
        assert all(item[key]==record[key] for key in ('qid','question_hash','length','token_ids_sha256'))
    tokenizer_files={name:value for name,value in hashes.items() if name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
    template=dict(question_extraction=old_features['spec']['question_extraction'],prompt_key='prompt',added_template=None,tokenizer_add_special_tokens=True,truncation=False)
    expected_spec=dict(encoder_id=old_features['spec']['encoder_id'],encoder_revision=origin['model_revision'],encoder_files_sha256=sha_json(hashes),tokenizer_sha256=sha_json(tokenizer_files),
        train_manifest_sha256=origin['input_sha256']['manifest.json'],train_parquet_sha256=origin['input_sha256']['train.parquet'],split='train',question_extraction=old_features['spec']['question_extraction'],
        representation='all_nonpadding_question_tokens_packed_v1',feature_layer=27,feature_dim=2048,encoder_compute_dtype='bfloat16',feature_dtype=args.feature_dtype,
        prompt_template_sha256=sha_json(template),prompt_spec=template,token_feature_producer_sha256=old_map['search_g1/qprobe_token_features.py'],original_label_feature_spec_sha256=old_features['spec_sha'],
        global_max_tokens=scan['global_max_tokens'],global_train_rows=169028,truncation=False,tokenizer_add_special_tokens=True,
        global_scan_sha256=sha(parent/'global-scan.json'),teacher_identity_sha256=provenance['teacher_identity_sha256'],teacher_complete_sha256=provenance['teacher_complete_sha256'])
    assert cache.spec==expected_spec
    assert cache.manifest['total_tokens']*2048*(2 if args.feature_dtype=='bfloat16' else 4)==previous['work_cost']['packed_feature_bytes']
    for start in range(0,5120,64):
        batch=cache.get_batch(qids[start:start+64],device='cpu');del batch
        progress(output,dict(stage='resume_token_feature_full_validation',verified_questions=min(start+64,5120),total_questions=5120))
    recovery=dict(source_output=str(parent),source_commit=RESUME_SOURCE,parent_prepared_sha256=sha(parent/'prepared.json'),
        parent_failure_sha256=sha(parent/'run-failure.json'),parent_run_claim_sha256=sha(parent/'run-claim.json'),
        parent_source_map_sha256=previous['source_map_sha256'],feature_cache_dir=str(cache.path),feature_manifest_sha256=sha(cache.path/'manifest.json'),
        token_feature_spec_sha256=cache.spec_sha,feature_files=cache.manifest['files'],global_metadata_files=previous['prepared_files'],
        old_encoder_failure_preserved=True,old_head_initialized=False,old_head_adam_updates=0,
        old_encoder_process_births_exited=True,old_encoder_session_exited=True,owned_cuda_release_required_from_operations=True,
        token_features_full_sha_verified=True,scientific_source_identical_except_runner=True,
        old_head_inherited=False,encoder_reencoded=False,formal_training_started=False)
    return cache,qids,groups,provenance,scan,recovery


def prepare_head_only(args,output,mapping):
    cache,qids,groups,provenance,scan,recovery=resume_encoded_features(args,output,mapping)
    once(output/'global-scan.json',scan)
    prepared=dict(stage='TOKEN_BINARY_HEAD_ONLY_CPU_PREPARED',task_config=task_config(args),source_sha256=mapping,
        source_map_sha256=hashlib.sha256(canonical(mapping)).hexdigest(),label_producer_provenance=provenance,
        global_scan=scan,parent_feature_resume=recovery,prepared_files={'global-scan.json':dict(sha256=sha(output/'global-scan.json'),bytes=(output/'global-scan.json').stat().st_size)},
        no_gpu_model_loaded=True,encoder_reencoded=False,headfit_only=True,formal_training_started=False,checked_unix=time.time())
    once(output/'prepared.json',prepared)
    print(json.dumps(dict(stage=prepared['stage'],output_dir=str(output),prepared_sha256=sha(output/'prepared.json'),
        global_max_tokens=scan['global_max_tokens'],feature_manifest_sha256=recovery['feature_manifest_sha256'],
        headfit_only=True,formal_training_started=False)),flush=True)


def run_head_only(args,output,mapping,gpu):
    import torch
    assert torch.cuda.device_count()==1
    prepared=read(output/'prepared.json')
    assert prepared['stage']=='TOKEN_BINARY_HEAD_ONLY_CPU_PREPARED' and prepared['task_config']==task_config(args) and prepared['source_sha256']==mapping
    assert sha(output/'global-scan.json')==prepared['prepared_files']['global-scan.json']['sha256']
    cache,qids,groups,provenance,scan,recovery=resume_encoded_features(args,output,mapping)
    assert recovery==prepared['parent_feature_resume'] and provenance==prepared['label_producer_provenance']
    assert torch.cuda.memory_allocated()==0, 'Head-only process has unexpected pre-head Torch allocation'
    progress(output,dict(stage='head_only_features_qualified',encoded_questions_this_process=0,head_optimizer_updates=0,encoder_loaded_in_this_process=False))
    return fit_and_save(args,output,prepared,gpu,cache,qids,groups,provenance,headfit_only=True,recovery=recovery)


def gpu_admission(index):
    import fcntl
    assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    selected=[row for row in csv.reader(io.StringIO(raw)) if int(row[0].strip())==index]
    assert len(selected)==1 and int(selected[0][2])<=1024 and int(selected[0][3])==0, 'Selected GPU is occupied; no foreign stop permitted'
    uuid=selected[0][1].strip();root=Path('/data/shaoshitong/Search-G1-qprobe-shared/token-feature-gpu-locks');root.mkdir(parents=True,exist_ok=True)
    lock=(root/(uuid+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    assert not any(row and row[0].strip()==uuid for row in csv.reader(io.StringIO(apps)))
    from reproduction.a800_common.precompute_qprobe import bind_cuda_uuid
    from reproduction.common8.native_allocation import cuda_uuids
    actual=bind_cuda_uuid(uuid,cuda_uuids)
    return lock,dict(physical_index=index,physical_gpu_uuid=uuid,actual_cuda_uuids=actual)


def run(args,output,mapping,gpu):
    if args.resume_token_feature_output is not None:return run_head_only(args,output,mapping,gpu)
    import numpy as np
    import torch
    from transformers import AutoModel,AutoTokenizer
    from search_g1.qprobe_features import OFFICIAL_ENCODER,OFFICIAL_REVISION,COMMON_MANIFEST_SHA,QUESTION_EXTRACTION_RULE,sha_json,question_text_sha
    from search_g1.qprobe_token_features import TokenFeatureCache,TokenFeatureCacheWriter,extract_question_token_features
    from search_g1.qprobe_token_model import TokenBinaryQProbeConfig,TokenBinaryWarmup
    assert torch.cuda.device_count()==1
    prepared=read(output/'prepared.json');assert prepared['task_config']==task_config(args) and prepared['source_sha256']==mapping
    for name,item in prepared['prepared_files'].items():assert (output/name).stat().st_size==item['bytes'] and sha(output/name)==item['sha256']
    base,origin,rows,hashes,old_features,config=assets(args,mapping)
    chosen,groups,provenance=qualified_labels(args,origin,rows,old_features,config,mapping,output)
    assert provenance==prepared['label_producer_provenance']
    lengths=np.load(output/'global-token-lengths.npy',allow_pickle=False);assert lengths.shape==(169028,) and int(lengths.max())==prepared['global_scan']['global_max_tokens']
    bindings=[json.loads(line) for line in (output/'global-token-bindings.jsonl').read_text().splitlines()]
    assert len(bindings)==len(rows)
    for ordinal,((question,qid),record) in enumerate(zip(rows,bindings)):
        assert record['qid']==qid and record['question_hash']==question_text_sha(question) and record['length']==int(lengths[ordinal])
    records=[bindings[index] for index,_,_ in chosen]
    tokenizer_files={name:value for name,value in hashes.items() if name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')}
    template=dict(question_extraction=QUESTION_EXTRACTION_RULE,prompt_key='prompt',added_template=None,tokenizer_add_special_tokens=True,truncation=False)
    spec=dict(encoder_id=OFFICIAL_ENCODER,encoder_revision=OFFICIAL_REVISION,encoder_files_sha256=sha_json(hashes),tokenizer_sha256=sha_json(tokenizer_files),
        train_manifest_sha256=COMMON_MANIFEST_SHA,train_parquet_sha256=origin['input_sha256']['train.parquet'],split='train',question_extraction=QUESTION_EXTRACTION_RULE,
        representation='all_nonpadding_question_tokens_packed_v1',feature_layer=27,feature_dim=2048,encoder_compute_dtype='bfloat16',feature_dtype=args.feature_dtype,
        prompt_template_sha256=sha_json(template),prompt_spec=template,token_feature_producer_sha256=mapping['search_g1/qprobe_token_features.py'],original_label_feature_spec_sha256=old_features['spec_sha'],
        global_max_tokens=prepared['global_scan']['global_max_tokens'],global_train_rows=169028,truncation=False,tokenizer_add_special_tokens=True,
        global_scan_sha256=sha(output/'global-scan.json'),teacher_identity_sha256=provenance['teacher_identity_sha256'],teacher_complete_sha256=provenance['teacher_complete_sha256'])
    tokenizer=AutoTokenizer.from_pretrained(origin['initial_model'],local_files_only=True,trust_remote_code=False);assert tokenizer.pad_token_id is not None
    model=AutoModel.from_pretrained(origin['initial_model'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16).to('cuda:0')
    model.requires_grad_(False);model.eval();writer=TokenFeatureCacheWriter(output/'token-features',spec,records)
    for start in range(0,5120,args.encode_batch_size):
        batch=chosen[start:start+args.encode_batch_size];questions=[q for _,q,_ in batch];qids=[qid for _,_,qid in batch]
        tokens=tokenizer(questions,padding=False,truncation=False,add_special_tokens=True)['input_ids']
        for row,(index,_,_) in enumerate(batch):
            assert len(tokens[row])==bindings[index]['length'] and hashlib.sha256(np.asarray(tokens[row],dtype='<i8').tobytes()).hexdigest()==bindings[index]['token_ids_sha256']
        features=extract_question_token_features(model,tokenizer,questions,feature_layer=27,max_tokens=spec['global_max_tokens'],device='cuda:0')
        writer.append(qids,features);progress(output,dict(stage='token_feature_encoding',completed_questions=writer.written,total_questions=5120))
    feature_manifest=writer.finish();del model;gc.collect();torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated()==0, 'Frozen encoder still allocated before fresh-head fit'
    progress(output,dict(stage='official_encoder_unloaded',torch_allocated_bytes=0,encoded_questions=5120))
    cache=TokenFeatureCache(output/'token-features',expected_spec_sha=feature_manifest['spec_sha'])
    qids=[qid for _,_,qid in chosen];assert cache.ids==qids
    return fit_and_save(args,output,prepared,gpu,cache,qids,groups,provenance)


def save_remote_state(path,state):
    import torch
    temporary=path.with_suffix('.partial.pt')
    assert not path.exists() and not temporary.exists()
    torch.save(state,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    os.rename(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def fit_and_save(args,output,prepared,gpu,cache,qids,groups,provenance,*,headfit_only=False,recovery=None):
    """Fresh Head/Adam only; preserve actual partial writes on fit failure."""
    import numpy as np
    import torch
    from search_g1.qprobe_token_model import TokenBinaryQProbeConfig,TokenBinaryWarmup
    assert len(qids)==len(groups)==5120 and cache.ids==qids
    counts=np.asarray([group['correct_count'] for group in groups],dtype=np.int64)
    head_config=TokenBinaryQProbeConfig(max_tokens=cache.spec['global_max_tokens'],feature_dim=2048,hidden_size=128,attention_heads=4,epochs=10,batch_size=256,
        learning_rate=3e-4,auxiliary_weight=args.auxiliary_weight,target_statistic=args.target_statistic,seed=args.head_seed,top_k=512)
    fresh=TokenBinaryWarmup(head_config,device='cuda:0')
    assert fresh.update_count==fresh._adam_steps()==0 and fresh.fitted is False and not fresh.optimizer.state
    progress(output,dict(stage='fresh_head_initialized',head_optimizer_updates=0,head_adam_steps=0,headfit_only=headfit_only,
        encoder_loaded_in_this_process=not headfit_only,encoded_questions_this_process=0 if headfit_only else 5120,old_head_inherited=False))
    try:
        metrics=fresh.fit(cache,cache.lengths.copy(),counts,qids,progress_callback=lambda event:progress(output,event))
        assert fresh.update_count==fresh._adam_steps()==200 and fresh.fitted is True
        assert len(metrics['epoch_history'])==11 and len(metrics['minibatch_history'])==200
    except BaseException:
        # State remains partial: no rollback, counters reset, or implicit retry.
        partial=output/'partial-head.pt'
        save_remote_state(partial,dict(token_feature_spec_sha256=cache.spec_sha,label_producer_provenance=provenance,
            parent_feature_resume=recovery,head=fresh.state_dict()))
        once(output/'partial-head.json',dict(stage='TOKEN_BINARY_PARTIAL_HEAD_PRESERVED',head_optimizer_updates=fresh.update_count,
            head_adam_steps=fresh._adam_steps(),fitted=fresh.fitted,head_sha256=sha(partial),head_bytes=partial.stat().st_size,
            headfit_only=headfit_only,old_head_inherited=False,rollback_performed=False,automatic_retry=False,formal_training_started=False))
        raise
    once(output/'warmup-metrics.json',metrics)
    state=dict(token_feature_spec_sha256=cache.spec_sha,label_producer_provenance=provenance,parent_feature_resume=recovery,head=fresh.state_dict())
    head_path=output/'fresh-head.pt';save_remote_state(head_path,state)
    loaded=torch.load(head_path,map_location='cpu',weights_only=False)
    assert loaded['token_feature_spec_sha256']==cache.spec_sha and loaded['head']['update_count']==200
    assert loaded['head']['config']==state['head']['config'] and loaded['head']['fitted'] is True
    for name,value in state['head']['model'].items():
        actual=loaded['head']['model'][name];assert torch.equal(value.detach().cpu(),actual) and bool(torch.isfinite(actual).all())
    ready=dict(stage='TOKEN_BINARY_FRESH_HEAD_WARMUP10_COMPLETE',source_commit=args.source_commit,source_map_sha256=prepared['source_map_sha256'],
        task_config=task_config(args),**gpu,questions=5120,teacher_trajectories_reused=25600,trajectories_generated=0,global_max_tokens=cache.spec['global_max_tokens'],
        token_feature_spec_sha256=cache.spec_sha,token_feature_manifest_sha256=sha(cache.path/'manifest.json'),feature_cache_dir=str(cache.path),
        original_teacher_feature_spec_sha256=provenance['original_feature_spec_sha256'],teacher_identity_sha256=provenance['teacher_identity_sha256'],
        feature_representation='all_question_tokens_no_pooling_no_truncation',old_pooled_vector_features_read=False,old_head_inherited=False,
        encoder_unloaded_before_head_fit=not headfit_only,headfit_only=headfit_only,encoder_loaded_in_this_process=not headfit_only,
        encoded_questions_this_process=0 if headfit_only else 5120,encoder_process_exited_before_head_fit=headfit_only,
        parent_feature_resume=recovery,head_optimizer_updates=200,head_adam_steps=200,epoch_records=11,minibatch_records=200,
        head_sha256=sha(head_path),head_bytes=head_path.stat().st_size,metrics_sha256=sha(output/'warmup-metrics.json'),prepared_sha256=sha(output/'prepared.json'),
        metrics_scope='same5120_training_fit_only_no_independent_validation',formal_training_started=False,automatic_formal_200_launch=False,checked_unix=time.time())
    once(output/'ready.json',ready);print(json.dumps(ready,allow_nan=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run-base','output-dir','source-publication'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--teacher-identity-sha256',required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--physical-index',type=int)
    parser.add_argument('--resume-token-feature-output',type=Path,default=None)
    parser.add_argument('--encode-batch-size',type=int,default=4);parser.add_argument('--feature-dtype',choices=('bfloat16','float32'),default='bfloat16')
    parser.add_argument('--target-statistic',choices=('sample_std','sample_variance'),default='sample_std')
    parser.add_argument('--auxiliary-weight',type=float,default=0.3);parser.add_argument('--head-seed',type=int,default=1729)
    args=parser.parse_args();output=remote(args.output_dir);origin=remote(args.source_run_base)
    assert output!=origin and origin not in output.parents and Path(TEACHER) not in output.parents
    if args.resume_token_feature_output is not None:
        parent=remote(args.resume_token_feature_output);assert str(parent)==RESUME_OUTPUT
        assert output!=parent and parent.parent not in output.parents and output not in parent.parents
    assert args.encode_batch_size>0 and args.auxiliary_weight>=0 and args.auxiliary_weight<float('inf')
    assert os.environ.get('SLURM_JOB_ID') is None, 'This entry requires explicit local one-GPU admission'
    assert args.prepare_only or args.physical_index is not None
    mapping=source_binding(args)
    if args.prepare_only:output.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=''
    else:assert output.is_dir() and (output/'prepared.json').is_file() and not (output/'ready.json').exists()
    for name in ('cache/hf','cache/triton','cache/inductor','tmp'):(output/name).mkdir(parents=True,exist_ok=True)
    os.environ.update(HF_HOME=str(output/'cache/hf'),XDG_CACHE_HOME=str(output/'cache'),TMPDIR=str(output/'tmp'),
        TRITON_CACHE_DIR=str(output/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    stat=(Path('/proc')/str(os.getpid())/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    birth=dict(pid=os.getpid(),start_ticks=int(fields[19]),sid=int(fields[3]),uid=os.getuid(),hostname=socket.gethostname())
    once(output/('prepare-claim.json' if args.prepare_only else 'run-claim.json'),dict(task_config=task_config(args),**birth,no_automatic_retry=True,claimed_unix=time.time()))
    lock=None
    try:
        if args.prepare_only:prepare(args,output,mapping)
        else:
            lock,gpu=gpu_admission(args.physical_index);run(args,output,mapping,dict(birth,**gpu))
    except BaseException as failure:
        frames=[dict(file=Path(frame.filename).name,line=frame.lineno,function=frame.name) for frame in traceback.extract_tb(failure.__traceback__)[-8:]]
        once(output/('prepare-failure.json' if args.prepare_only else 'run-failure.json'),dict(failure_type=type(failure).__name__,frames=frames,**birth,no_automatic_retry=True,checked_unix=time.time()))
        # Keep all partial remote artifacts for diagnosis; never emit raw QA/logs.
        print(json.dumps(dict(stage='TOKEN_WARMUP_FAILED',failure_type=type(failure).__name__,output_dir=str(output),formal_training_started=False)),flush=True)
        return 1
    finally:
        if lock is not None:lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
