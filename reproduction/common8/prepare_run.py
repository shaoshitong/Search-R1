"""Bind an entirely fresh run to the exact canonical shared data and official model."""
import argparse
import hashlib
import json
import os
import time
import uuid
import fcntl
from contextlib import contextmanager
from pathlib import Path
from reproduction.common8.verify_data import digest, probe_question_id, verify
from reproduction.common8.formal_contract import CONTRACT_KIND, TARGET_UPDATES,SAFETY_MODE, assert_formal_run

REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
OFFICIAL_MANIFEST_SHA='7f53ce6c010ac076b1ec482031bea7c9e6b8d0a096a61b880fe8ad4971bc9e44'


def write_method(base, commit, source, restoration=False):
    """Freeze prospective method before the fresh run starts; never claim health."""
    from reproduction.common8.formal_inline_safety import METHOD_ROLE
    from reproduction.common8.formal_contract import FORMAL_COUNTING, FORMAL_SAFEGUARDS
    from reproduction.common8.policy_calibration_plan import canonical_bytes, document_sha256
    method=dict(schema_version=1, role=METHOD_ROLE, source_commit=commit,
        source_map_sha256=document_sha256(source), policy_safety_mode=SAFETY_MODE,
        target_updates=TARGET_UPDATES, warmup_steps=57,
        same_process_after_five=True, fresh_states=not restoration,
        optimizer_counting=FORMAL_COUNTING, safeguards=FORMAL_SAFEGUARDS,
        created_unix=time.time(), negative_tail_semantics='prospective operational candidate; not calibrated health',
        production_admission=False, healthy_checkpoint_accepted=False)
    path=Path(base)/'preparation/formal-inline-method.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(canonical_bytes(method));f.flush();os.fsync(f.fileno())
    return method


@contextmanager
def calibration_lock(base):
    directory = Path(base) / 'preparation'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'calibration-derive.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _empty_calibration_targets(base):
    for name in ('calibration-300.parquet', 'audit.json'):
        if (Path(base) / 'data' / name).exists():
            raise FileExistsError('Existing calibration derivative must not be replaced: ' + name)


def _publish_calibration_locked(base, selected, audit, parquet_writer):
    """Publish complete derivatives; audit is the last committed file."""
    base = Path(base)
    _empty_calibration_targets(base)
    token = uuid.uuid4().hex
    parquet = base / 'data/calibration-300.parquet'
    audit_path = base / 'data/audit.json'
    staged_parquet = parquet.with_name('.calibration-300.' + token + '.partial.parquet')
    staged_audit = audit_path.with_name('.audit.' + token + '.partial.json')
    try:
        parquet_writer(selected, staged_parquet)
        with staged_parquet.open('rb') as stream:
            os.fsync(stream.fileno())
        with staged_audit.open('w') as stream:
            json.dump(audit, stream, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        _empty_calibration_targets(base)
        os.replace(staged_parquet, parquet)
        os.replace(staged_audit, audit_path)
        descriptor = os.open(parquet.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        for path in (staged_parquet, staged_audit):
            if path.exists(): path.unlink()


def derive_calibration(base):
    with calibration_lock(base):
        _empty_calibration_targets(base)
        return _derive_calibration_locked(base)


def _derive_calibration_locked(base):
    import numpy as np
    import pyarrow.parquet as pq
    proof=verify(base/'data')
    table=pq.read_table(base/'data/train.parquet')
    sources=table['data_source'].to_pylist()
    ids=[probe_question_id(x) for x in table['prompt'].to_pylist()]
    rng=np.random.RandomState(42)
    chosen=[]; seen=set()
    for source in ('nq','hotpotqa'):
        candidates=[i for i,s in enumerate(sources) if s==source]
        count=0
        for i in rng.permutation(candidates):
            i=int(i)
            if ids[i] in seen:continue
            chosen.append(i);seen.add(ids[i]);count+=1
            if count==150:break
        assert count==150
    chosen.sort()
    audit=dict(calibration_count=300,calibration_question_ids=[ids[i] for i in chosen],
        original_train_row_indices=chosen, original_train_sha256=proof['products']['train']['sha256'],
        semantics='train-derived probe fitting subset; every canonical training row retained')
    _publish_calibration_locked(base, table.take(chosen), audit, pq.write_table)
    assert verify(base/'data')==proof, 'Canonical shared files changed during derivative creation'
    return audit


def verify_model(model, official_path):
    official=json.loads(Path(official_path).read_text())
    from reproduction.easy2560_joint4.model7b import FORMAT as SEVEN_FORMAT, verify as verify_seven
    if official.get('format') == SEVEN_FORMAT:
        return verify_seven(model, official)
    if official.get('format') == 'hpc3-official-qwen-aa8e725-nine-files-v1':
        from reproduction.hpc3_qprobe.model_manifest import verify as verify_hpc3_model
        return verify_hpc3_model(model, official)
    assert digest(official_path)==OFFICIAL_MANIFEST_SHA
    official=json.loads(official_path.read_text())
    assert official['repo']=='Qwen/Qwen2.5-3B-Instruct'
    assert official['revision']==REVISION
    index=json.loads((model/'model.safetensors.index.json').read_text())
    names=set(index['weight_map'].values())|{'model.safetensors.index.json','config.json',
        'tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json'}
    assert len(names)==9
    records={r['file']:r for r in official['files']}
    hashes={}
    for name in sorted(names):
        record=records[name];path=model/name
        assert path.stat().st_size==record['bytes']
        actual=digest(path)
        assert actual==record['sha256']
        if record.get('expected_lfs_sha256'):
            assert actual==record['expected_lfs_sha256']
        else:
            content=path.read_bytes()
            git=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            assert git==record['git_blob_sha1']
        hashes[name]=actual
    config=json.loads((model/'config.json').read_text())
    assert config['model_type']=='qwen2' and config['hidden_size']==2048
    return hashes


def prepare(base,model,official_path,commit,allow_existing_gpu_processes=False,rollout_memory_utilization=0.1,local_retrieval_assets=None,restore_receipt=None,execution_site=None,actor_micro_batch_per_rank=None,actor_logical_micro_batch_per_rank=None,easy2560_selection=False,easy2560_settings=None,dynrpo7b_settings=None):
    base,model,official_path=map(Path,(base,model,official_path))
    assert execution_site in (None,'hpc3','h1008')
    target=base/'logs/run-manifest.json'
    assert not target.exists(), 'Inspect the existing run rather than create a duplicate'
    data_proof=verify(base/'data')
    model_hashes=verify_model(model,official_path)
    restore_source=None
    if restore_receipt:
        import shutil
        from search_g1.full_state_restore import source
        _,restore_source,_,origin,restore_parent=source(restore_receipt)
        for name in ('calibration-300.parquet','audit.json'):
            assert not (base/'data'/name).exists()
            shutil.copy2(origin/'data'/name,base/'data'/name)
        audit=json.loads((base/'data/audit.json').read_text())
    else:
        audit=derive_calibration(base)
    repo=base/'repo'
    from reproduction.common8.source_files import runtime_files
    source={str(p.relative_to(repo)):digest(p) for p in runtime_files(repo)}
    budget=json.loads((repo/'reproduction/common8/budget.json').read_text())
    assert budget['model_revision']==REVISION and budget['fresh_actor']
    official = json.loads(official_path.read_text())
    from reproduction.easy2560_joint4.model7b import FORMAT as SEVEN_FORMAT, REPO as SEVEN_REPO, REVISION as SEVEN_REVISION
    model_revision = REVISION
    if official.get('format') == SEVEN_FORMAT:
        assert (easy2560_selection or dynrpo7b_settings is not None) and execution_site == 'hpc3'
        model_revision = SEVEN_REVISION
        budget.update(model_id=SEVEN_REPO, model_revision=model_revision)
    assert budget['common_manifest_sha256']==data_proof['manifest_sha256']
    inputs={name:digest(base/'data'/name) for name in
        ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet',
         'exclusions-audit.json','calibration-300.parquet','audit.json')}
    method = write_method(base, commit, source,restoration=bool(restore_receipt))
    result=dict(policy_safety_required='actor_rollout_ref.actor.policy_safety.enabled=true' in json.loads((repo/'reproduction/common8/recipe.json').read_text()),profile='common8',run_id=uuid.uuid4().hex,started_unix=time.time(),
        mixed_first_selection=True,actor_micro_batch_per_rank=2,completed_updates=0,initial_model=str(model.resolve()),reference_model=str(model.resolve()),
        initial_model_sha256=model_hashes,model_revision=model_revision, budget=budget,
        source_commit=commit,source_sha256=source,
        source_snapshot_semantics='disk fingerprint before launch; not proof of loaded modules',
        input_sha256=inputs,calibration_question_ids=audit['calibration_question_ids'],
        contract_kind=CONTRACT_KIND,target_updates=TARGET_UPDATES,policy_safety_mode=SAFETY_MODE,
        save_interval=50,acceptance_update=5,initial_probe_version=None,
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
        train_log=str(base/'logs/search-g1-common8-training.log'),
        optimizer_semantics='fresh Adam, scheduler, RNG and sampler; official fresh actor; no prior checkpoint or diagnostic state',
        retrieval_placement='pending actual resource and retrieval-equivalence admission',
        probe_visible_device=None)
    if execution_site in ('hpc3','h1008'):result['execution_site']=execution_site
    if easy2560_selection:
        assert execution_site=='hpc3' and actor_micro_batch_per_rank in (4,8,16,32)
        assert actor_logical_micro_batch_per_rank==4
        result.update(easy2560_selection=True,data_selection_mode='mixed_first',
            actor_logical_micro_batch_per_rank=4,selection_candidate_questions=2560,
            selection_selected_questions=512,selection_fixed_candidate_budget=True,
            all_correct_virtual_zero=False)
        if easy2560_settings is not None:
            result.update(easy2560_settings)
    else:
        assert easy2560_settings is None
    if dynrpo7b_settings is not None:
        assert not easy2560_selection and execution_site=='hpc3'
        result.update(dynrpo7b_settings)
    if execution_site=='h1008':
        from reproduction.common8.precision_contract import NATIVE_BF16_PROFILE
        assert os.environ.get('SEARCH_G1_EXECUTION_PROFILE')==NATIVE_BF16_PROFILE, 'H1008 must bind its BF16/FP32 execution recipe explicitly'
        if restore_receipt:
            assert restore_parent.get('execution_site')=='h1008' and restore_parent.get('actor_micro_batch_per_rank')==4, 'H1008 recovery must belong to the new same-budget experiment'
        assert actor_micro_batch_per_rank in (None,4), 'Fresh H1008 budget binds local micro4'
        assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE','8')=='8'
    if os.environ.get('SEARCH_G1_EXECUTION_PROFILE'):
        from reproduction.common8.precision_contract import NATIVE_BF16_PROFILE, NATIVE4_BF16_PROFILE, SAMPLING_JOURNAL_POLICY
        size=int(os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE','8'))
        assert size in (4,8) and (size==8 or execution_site=='hpc3')
        profile=NATIVE_BF16_PROFILE if size==8 else NATIVE4_BF16_PROFILE
        assert os.environ['SEARCH_G1_EXECUTION_PROFILE'] == profile and (restore_receipt or execution_site in ('hpc3','h1008'))
        result.update(precision_profile=profile, behavior_probability_policy=SAMPLING_JOURNAL_POLICY,
            execution_backend='slurm', training_world_size=size, actor_micro_batch_per_rank=4 if execution_site=='h1008' else 2 if size==8 else 4,
            vllm_gpu_memory_utilization=rollout_memory_utilization, followup_checkpoint_required=True)
        if easy2560_selection or dynrpo7b_settings is not None:
            assert size==4
            result['actor_micro_batch_per_rank']=actor_micro_batch_per_rank
        result['budget'].update(world_size=size,compute_dtype='bfloat16', rollout_dtype='bfloat16', allow_tf32=None)
        from reproduction.common8.formal_contract import formal_mathematics
        result['budget'].update(formal_mathematics(result))
    if execution_site in ('hpc3','h1008'):
        result['execution_site']=execution_site
        from reproduction.common8.topology import validate_base
        validate_base(base.resolve(),result)
    if restore_receipt:
        from search_g1.qprobe_checkpoint import PAIR_STATUS
        if easy2560_selection:
            from search_g1.easy2560_restore import STATUS as EASY_STATUS
            assert restore_source.get('status')==EASY_STATUS
            assert restore_source['world_size']==result['training_world_size']==4
            # Easy bind below validates the original physical16 CP5 and only
            # permits downward physical fallback, with unchanged logical4 loss,
            # global mini128 and every non-physical budget/input field.
        elif restore_source.get('status')==PAIR_STATUS:
            # This CLI is the native eight-rank route. Four-rank local sources
            # use the local preparation entry; changing shard layout is forbidden.
            assert restore_source['world_size']==result['training_world_size'], 'Checkpoint shard layout must match actual native topology'
            import copy
            result['budget']=copy.deepcopy(restore_parent['budget'])
            for key in ('training_world_size','actor_micro_batch_per_rank','precision_profile',
                        'behavior_probability_policy'):
                if key in restore_parent:result[key]=copy.deepcopy(restore_parent[key])
        from search_g1.full_state_restore import bind
        bind(result,restore_receipt)
    from reproduction.common8.shared_slurm import bind as bind_shared_slurm
    bind_shared_slurm(result,base)
    from reproduction.common8.native_ray import prepare as prepare_ray
    prepare_ray(result)
    assert type(allow_existing_gpu_processes) is bool
    assert 0.0 < rollout_memory_utilization <= 0.4
    result['vllm_gpu_memory_utilization'] = rollout_memory_utilization
    if local_retrieval_assets is not None:
        assets = Path(local_retrieval_assets).resolve()
        assert assets.is_dir() and (assets/'e5_Flat.index').is_file() and (assets/'wiki-18.jsonl').is_file()
        result['local_retrieval_assets'] = str(assets)
    if allow_existing_gpu_processes:
        import subprocess,csv,io
        from reproduction.a800.owned_processes import identity
        raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],text=True)
        result['shared_gpu_policy'] = 'preserve_bound_existing_processes'
        result['shared_gpu_cohosts'] = [dict(application=[r[0].strip(),r[1].strip()],
            process_identity=identity(int(r[1]))) for r in csv.reader(io.StringIO(raw)) if r]
    assert_formal_run(result)
    from reproduction.common8.formal_inline_safety import verify_method
    verify_method(method,result)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--model',required=True);p.add_argument('--official-manifest',required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--allow-existing-gpu-processes',action='store_true')
    p.add_argument('--rollout-memory-utilization',type=float,default=0.1)
    p.add_argument('--local-retrieval-assets');p.add_argument('--restore-receipt')
    p.add_argument('--execution-site',choices=('hpc3','h1008'))
    p.add_argument('--actor-micro-batch-per-rank',type=int,choices=(2,4));a=p.parse_args()
    print(json.dumps(prepare(a.base,a.model,a.official_manifest,a.source_commit,
        a.allow_existing_gpu_processes,a.rollout_memory_utilization,a.local_retrieval_assets,a.restore_receipt,a.execution_site,a.actor_micro_batch_per_rank)))
