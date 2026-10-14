"""Prepare a fresh A800 manifest from official weights and shared canonical data.

Preparation creates no GPU admission, fitted probe, training claim or success
marker. Runtime startup must bind the real local PID birth, host and GPU UUIDs.
"""
import argparse
import copy
import json
import os
import socket
import stat
import subprocess
import time
import uuid
from pathlib import Path
from reproduction.common8.prepare_run import REVISION,derive_calibration,verify_model
from reproduction.common8.verify_data import digest,verify
from reproduction.common8.source_files import runtime_files
from reproduction.common8.formal_contract import CONTRACT_KIND,TARGET_UPDATES,SAFETY_MODE,assert_formal_run,formal_mathematics
from reproduction.common8.topology import validate_base
from reproduction.common8.precision_contract import BF16_PROFILE,SAMPLING_JOURNAL_POLICY


def prepare(base,model,official_path,commit,initial_probes_source_base=None,restore_receipt=None):
    base,model,official_path=map(lambda p:Path(p).resolve(),(base,model,official_path))
    identity=dict(profile='common8',training_world_size=4,execution_backend='ssh_local',
        slurm_job_id=None,execution_id=uuid.uuid4().hex,precision_profile=BF16_PROFILE,
        behavior_probability_policy=SAMPLING_JOURNAL_POLICY,actor_micro_batch_per_rank=4)
    validate_base(base,identity)
    if initial_probes_source_base is not None:
        initial_probes_source_base=Path(initial_probes_source_base)
        validate_base(initial_probes_source_base,identity)
        assert initial_probes_source_base!=base and initial_probes_source_base.is_dir()
        assert initial_probes_source_base.stat().st_uid==os.getuid(), 'Initial-probe source owner differs'
    assert os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE')=='4'
    assert model==Path('/data/search-g1/assets/qwen2.5-3b-instruct').resolve()
    target=base/'logs/run-manifest.json'
    assert not target.exists(), 'Inspect this existing attempt; never overwrite its identity'
    assert not (base/'logs/process-ledger.json').exists() and not (base/'logs/training-launch-claim.json').exists()
    repo=base/'repo';assert repo.is_dir() and (repo/'.git').exists()
    actual_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert actual_commit==commit, 'Source commit differs from this fresh clone'
    # Shared refit helpers resolve their executable/header paths through the
    # fresh base. These links are new-run aliases, never copies or mutations of
    # the existing environment and bootstrap resources.
    for name,shared in (('env','/data/search-g1/env'),('bootstrap-tools','/data/search-r1/bootstrap-tools')):
        link=base/name;destination=Path(shared)
        assert destination.is_dir()
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve()==destination.resolve(), 'Existing run resource differs: '+name
        else:link.symlink_to(destination,target_is_directory=True)
    data_proof=verify(base/'data');model_hashes=verify_model(model,official_path)
    if restore_receipt:
        assert initial_probes_source_base is None, 'A complete pair and fresh probe adoption are mutually exclusive'
        from search_g1.qprobe_checkpoint import read_source
        _,restore_source,_,origin,restore_parent=read_source(restore_receipt)
        assert restore_source['world_size']==4, 'Local restoration requires the exact four-rank source layout'
        import shutil
        for name in ('calibration-300.parquet','audit.json'):
            assert not (base/'data'/name).exists()
            shutil.copy2(origin/'data'/name,base/'data'/name)
        audit=json.loads((base/'data/audit.json').read_text())
    else:audit=derive_calibration(base)
    source={str(p.relative_to(repo)):digest(p) for p in runtime_files(repo)}
    budget=copy.deepcopy(json.loads((repo/'reproduction/common8/budget.json').read_text()))
    budget.update(formal_mathematics(identity))
    assert budget['model_revision']==REVISION and budget['fresh_actor']
    assert budget['common_manifest_sha256']==data_proof['manifest_sha256']
    assert budget['world_size']==4 and budget['ppo_micro_batch_size']==16
    assert budget['per_rank_actor_micro_batch']==4 and budget['ppo_mini_batch_size']==128
    assert budget['per_rank_log_prob_micro_batch']==8
    inputs={name:digest(base/'data'/name) for name in
        ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet',
         'exclusions-audit.json','calibration-300.parquet','audit.json')}
    # Ray's Unix sockets include its timestamp/session/PID suffix. Keep this
    # private real root short; a symlink alias of the long base is insufficient.
    ray_tmp_dir=Path('/tmp')/('g-'+identity['execution_id'])
    assert ray_tmp_dir.parent.resolve()==ray_tmp_dir.parent, 'Ray temporary parent must be a real directory'
    ray_tmp_dir.mkdir(mode=0o700,exist_ok=False)
    ray_stat=ray_tmp_dir.lstat()
    assert stat.S_ISDIR(ray_stat.st_mode) and ray_tmp_dir.resolve()==ray_tmp_dir
    assert ray_stat.st_uid==os.getuid() and stat.S_IMODE(ray_stat.st_mode)==0o700
    assert len(os.fsencode(str(ray_tmp_dir/'session_0000-00-00_00-00-00_000000_2147483647'/'sockets/plasma_store')))<107
    ray_identity=dict(device=ray_stat.st_dev,inode=ray_stat.st_ino,uid=ray_stat.st_uid,mode=0o700)
    fd=os.open(ray_tmp_dir.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    result=dict(identity,run_id=uuid.uuid4().hex,started_unix=time.time(),
        execution_host=socket.gethostname(),completed_updates=0,
        ray_tmp_dir=str(ray_tmp_dir),ray_tmp_dir_identity=ray_identity,
        ray_local_resources=dict(num_cpus=16,object_store_memory_bytes=8*1024**3,
            scope='local scheduling and shared-object storage; training batches and precision unchanged'),
        policy_safety_required=True,initial_model=str(model),reference_model=str(model),
        initial_model_sha256=model_hashes,model_revision=REVISION,budget=budget,
        source_commit=commit,source_sha256=source,
        source_snapshot_semantics='disk fingerprint before launch; not proof of loaded modules',
        input_sha256=inputs,calibration_question_ids=audit['calibration_question_ids'],
        contract_kind=CONTRACT_KIND,target_updates=TARGET_UPDATES,policy_safety_mode=SAFETY_MODE,
        save_interval=50,acceptance_update=5,initial_probe_version=None,
        checkpoint_dir=str(base/'checkpoints/search-g1-common8'),
        train_log=str(base/'logs/search-g1-common8-training.log'),
        optimizer_semantics='fresh Adam, scheduler, RNG and sampler; official fresh actor; no prior checkpoint or diagnostic state',
        gpu_placement_policy='launch_time_four_idle_excluding_auxiliary_v1',
        gpu_placement_bound=False,
        gpu_placement_semantics='unbound preparation placeholders; actual idle physical indices and UUIDs are selected and persisted immediately before launch',
        training_physical_indices=[0,1,2,3],auxiliary_physical_indices=[5,6],
        retriever_physical_indices=[5,6],probe_physical_index=6,
        excluded_physical_indices=[4,7],probe_visible_device=None,
        retrieval_placement='pending actual local asset, allocation and retrieval-equivalence admission',
        retrieval_asset_paths={'e5_Flat.index':'/data/search-r1/assets/wiki/e5_Flat.index',
                              'wiki-18.jsonl':'/data/search-r1/assets/wiki/wiki-18.jsonl'},
        retrieval_encoder='/data/search-r1/assets/e5-base-v2',
        retriever_gpu_lib='/data/search-r1/retriever-gpu-lib',
        python_executable='/data/search-g1/env/bin/python',
        python_headers='/data/search-r1/bootstrap-tools/python-dev/usr/include/python3.10:/data/search-r1/bootstrap-tools/python-dev/usr/include')
    baseline_relative='repo/reproduction/evidence/g1-historical-official-dev512-baseline.json'
    baseline_path=base/baseline_relative
    baseline=json.loads(baseline_path.read_text())
    baseline.update(source_artifact_relative=baseline_relative,source_artifact_sha256=digest(baseline_path))
    result.update(pretraining_baseline_policy='reuse_historical',historical_development_baseline=baseline,
        vllm_gpu_memory_utilization=0.25,
        runtime_memory_adaptation='BF16 policy forward/rollout; FP32 master weights/Adam/RoPE and selected-token probability arithmetic; FP64 gradient norm; TF32 flags not overridden; cache fraction0.25 retained; effective batches unchanged')
    if initial_probes_source_base is not None:
        # Selection metadata only; the pipeline must verify the actual immutable
        # source probe artifacts and receipts before it can adopt them.
        result['initial_probe_adoption_source_base']=str(initial_probes_source_base)
    if restore_receipt:
        from search_g1.qprobe_checkpoint import bind_restore
        result['actor_micro_batch_per_rank']=restore_parent['actor_micro_batch_per_rank']
        result['budget']=copy.deepcopy(restore_parent['budget'])
        for key in ('precision_profile','behavior_probability_policy','infra_rollout'):
            if key in restore_parent:result[key]=copy.deepcopy(restore_parent[key])
        bind_restore(result,restore_receipt)
    assert_formal_run(result)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:
        json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    fd=os.open(target.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True)
    p.add_argument('--model',default='/data/search-g1/assets/qwen2.5-3b-instruct')
    p.add_argument('--official-manifest',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--initial-probes-source-base');p.add_argument('--restore-receipt');a=p.parse_args()
    print(json.dumps(prepare(a.base,a.model,a.official_manifest,a.source_commit,a.initial_probes_source_base,a.restore_receipt)))
