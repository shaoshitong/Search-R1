"""Isolated pinned-old evaluation at any admitted checkpoint/split. No training.

The original raw-token search controller, greedy sampler, and Eco eval scorer
are reused. A separate ordinal journal makes already scored batches resumable.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid


from reproduction.legacy_evaluation.runtime import validate_binding,legacy_config,ROLE
from reproduction.legacy_evaluation.contract import PROTOCOL_PATH,digest,protocol,verify_loaded_workers,verify_result
from legacy_eval_control import activate,setup_worker


def resolved_document(document, run_id):
    """Unwrap the original launch proof, retaining its science/run binding."""
    if 'resolved_config' not in document:
        return document
    if document.get('run_id') != protocol()['protocol_authoritative_source']['historical_run']:
        raise ValueError('Resolved launch config belongs to a different experiment')
    config = document['resolved_config']
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != document.get('resolved_config_sha256'):
        raise ValueError('Resolved launch config canonical SHA changed')
    return config


def reuse_complete_result(directory,journal_binding,ids,sources,rows,binding):
    """Resume a fully measured journal using its original actual worker proof."""
    from reproduction.legacy_evaluation.journal import aggregate_final
    target=Path(directory)/'legacy-result.json'
    if not target.is_file() or target.is_symlink():
        raise ValueError('LEGACY_COMPLETE_JOURNAL_NEEDS_RETAINED_ACTUAL_RESULT_PROOF')
    result=json.loads(target.read_text());verify_result(result,binding['split'],binding['checkpoint_step'],ids,sources)
    measured=aggregate_final(journal_binding,ids,sources,rows)
    if any(result.get(k)!=measured[k] for k in ('run_id','checkpoint_manifest_sha256','journal_rows_sha256','weighted_em','weighted_f1')) or result.get('actual_gpu_execution_verified') is not True or result.get('canonical_legacy_evaluation') is not True:
        raise ValueError('LEGACY_EXISTING_RESULT_NOT_ACTUALLY_VERIFIED')
    records=[p for p in Path(directory).glob('loaded-workers-*.json') if not p.is_symlink() and digest(p)==result.get('loaded_workers_sha256')]
    if len(records)!=1:raise ValueError('LEGACY_EXISTING_LOADED_PROOF_MISSING')
    proofs=json.loads(records[0].read_text());verify_loaded_workers(proofs)
    if any(p.get('actual_vllm_version')!='0.6.3' or p.get('actual_vllm_config_dtype')!='torch.float32' or p.get('actual_engine_max_model_len')!=4596 for p in proofs):
        raise ValueError('LEGACY_EXISTING_ENGINE_RUNTIME_MISMATCH')
    if any(p['checkpoint']!=binding['checkpoint'] or p['checkpoint_manifest_sha256']!=binding['checkpoint_manifest_sha256'] or p['generation_config_sha256']!=binding['generation_config_sha256'] for p in proofs):
        raise ValueError('LEGACY_EXISTING_LOADED_CHECKPOINT_MISMATCH')
    return result


def run_evaluation(config,binding,source_base,directory):
    activate(binding['legacy_source_root'])
    import numpy as np
    import ray
    import torch
    import transformers,vllm
    if vllm.__version__!='0.6.3':raise ValueError('LEGACY_REQUIRES_VLLM_0_6_3')
    from torch.utils.data import DataLoader, Subset
    from verl import DataProto
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.main_ppo_eco import EcoRewardManager
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
    from reproduction.common8.validation_identity import expected, bind_validation_dataframe
    from reproduction.common8.calibration_evaluation import rows_from_batch
    from reproduction.common8.behavior_runtime import attach_generation_identity, cast_token_fields
    from reproduction.common8.generation_behavior import _run as original_loop
    from reproduction.legacy_evaluation.journal import load_prefix, append_batch, aggregate_final

    tokenizer = hf_tokenizer(binding['checkpoint'])
    official=hf_tokenizer(binding['official_model'])
    for field in ('chat_template','special_tokens_map','all_special_ids','padding_side','truncation_side'):
        if getattr(tokenizer,field)!=getattr(official,field):
            raise ValueError('LEGACY_TOKENIZER_SEMANTICS_CHANGED:'+field)
    if tokenizer.get_vocab()!=official.get_vocab() or json.loads(tokenizer.backend_tokenizer.to_str())!=json.loads(official.backend_tokenizer.to_str()):
        raise ValueError('LEGACY_TOKENIZER_BACKEND_OR_VOCAB_CHANGED')
    del official
    record, frame, ids = expected(Path(source_base), binding['split'])
    assert record['sha256'] == binding['validation_sha256']
    sources = frame['data_source'].tolist()
    dataset = RLHFDataset(parquet_files=str(Path(source_base)/'data'/record['path']),
        tokenizer=tokenizer, prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length, filter_prompts=True,
        return_raw_chat=config.data.get('return_raw_chat', False), truncation='error')
    dataset.dataframe = bind_validation_dataframe(dataset.dataframe, Path(source_base), binding['split'])
    journal_binding = binding['journal_binding']
    rows = load_prefix(directory, journal_binding, ids, sources)
    assert len(rows) <= len(ids) == protocol()['datasets'][binding['split']]['qa_count']
    if len(rows)==len(ids):
        return reuse_complete_result(directory,journal_binding,ids,sources,rows,binding)
    loader = DataLoader(Subset(dataset, range(len(rows), len(ids))),
        batch_size=config.data.val_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    pools = ResourcePoolManager({'final_rollout': [8]}, {Role.ActorRollout: 'final_rollout'})
    pools.create_resource_pool()
    from legacy_eval_control.worker import make_worker_class
    worker_cls = ray.remote(make_worker_class())
    # A single rollout role needs no colocated WorkerDict. Its zero-argument
    # base constructor is incompatible with ActorRolloutRefWorker subclasses.
    workers = RayWorkerGroup(resource_pool=pools.get_resource_pool(Role.ActorRollout),
        ray_cls_with_init=RayClassWithInitArgs(cls=worker_cls,
            config=config.actor_rollout_ref, role='actor_rollout'))
    proofs = workers.init_model()
    verify_loaded_workers(proofs)
    assert all(p['actual_vllm_version']=='0.6.3' and p['actual_vllm_config_dtype']=='torch.float32' and p['actual_engine_max_model_len']==4596 for p in proofs)
    assert len(proofs) == 8 and sorted(p['rank'] for p in proofs) == list(range(8))
    assert len({p['cuda_uuid'] for p in proofs}) == 8
    assert all(p['world_size'] == 8 and not p['optimizer_constructed'] and p['optimizer_steps'] == 0
               and p['checkpoint'] == binding['checkpoint'] for p in proofs)
    _write_once(Path(directory)/('loaded-workers-'+os.environ['SLURM_JOB_ID']+'.json'), proofs)
    scorer = EcoRewardManager(tokenizer=tokenizer, num_examine=1, reward_mode='eval')
    if binding.get('probe_version'):
        scorer.expected_probe_version = binding['probe_version']
    generation = GenerationConfig(max_turns=config.max_turns,
        max_start_length=config.data.max_start_length, max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length, max_obs_length=config.data.max_obs_length,
        num_gpus=8, no_think_rl=config.algorithm.no_think_rl,
        search_url=config.retriever.url, topk=config.retriever.topk)
    manager = LLMGenerationManager(tokenizer=tokenizer, actor_rollout_wg=workers,
                                  config=generation, is_validation=True)
    scorer.reset_metrics()
    _write_once(Path(directory)/('runtime-versions-'+os.environ['SLURM_JOB_ID']+'.json'),
        dict(vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__,
             historical_torch_transformers_versions_not_independently_bound=True,
             actual_engine_schedulers=[{k:p[k] for k in ('rank','actual_engine_max_num_seqs','actual_engine_max_num_batched_tokens')} for p in proofs]))
    import time
    for batch_dict in loader:
        batch_started = time.monotonic()
        offset = len(rows)
        batch = DataProto.from_single_dict(batch_dict)
        row_ids = [f"{binding['run_id']}-legacy{binding['checkpoint_step']}-ordinal-{offset+i}" for i in range(len(batch))]
        identity = dict(run_id=binding['run_id'], source_sha256=binding['legacy_source_sha256'],
            slurm_job_id=os.environ['SLURM_JOB_ID'], logical_update=binding['checkpoint_step'],
            policy_snapshot_id=binding['run_id']+':'+str(binding['checkpoint_step'])+':validation-'+uuid.uuid4().hex,
            parent_group_ids=row_ids, trajectory_ids=row_ids, validation=True)
        inputs = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        inputs.meta_info = dict(eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, recompute_log_prob=False, do_sample=False, validate=True)
        attach_generation_identity(inputs, identity, do_sample=False)
        manager.timing_raw = {}
        captured = []
        try:
            output = original_loop(manager, inputs,
                inputs.batch['input_ids'][:, -generation.max_start_length:].clone(), captured)
        except Exception as error:
            # Keep the original controller exception: its training-profile
            # failure writer would demand a fresh-training run manifest here.
            from reproduction.common8.behavior_runtime import _evidence_value
            private, _ = _evidence_value(dict(role='final_only_failed_generation',
                checkpoint_step=binding['checkpoint_step'], actual_training_outer=False, occurrence_offset=offset,
                identity=identity, actual_sampling_records=captured,
                failure_type=type(error).__name__))
            try:
                _write_once(Path(directory)/('generation-failure-'+uuid.uuid4().hex+'.json'), private)
            except Exception as receipt_error:
                print('FINAL_RESUME_FAILURE_RECEIPT_ERROR='+type(receipt_error).__name__, flush=True)
            raise
        batch = batch.union(output)
        new_rows = [dict(row, ordinal=offset+i) for i, row in enumerate(
            rows_from_batch(batch, tokenizer, allowed_sources=record['source_counts']))]
        cast_token_fields(batch)
        em = scorer(batch).sum(-1).cpu().tolist()
        assert em == [r['em'] for r in new_rows]
        assert [r['question_id'] for r in new_rows] == ids[offset:offset+len(new_rows)]
        append_batch(directory, journal_binding, ids, sources, new_rows)
        rows = load_prefix(directory, journal_binding, ids, sources)
        print('SEARCH_G1_LEGACY_EVALUATION_PROGRESS='+json.dumps(dict(qa_count=len(rows), target=len(ids),
            checkpoint_step=binding['checkpoint_step'], optimizer_steps=0, batch_seconds=time.monotonic()-batch_started,
            batch_questions=len(new_rows), completed_unix=time.time())), flush=True)
    result=aggregate_final(journal_binding,ids,sources,rows)
    result.update(actual_gpu_execution_verified=True,canonical_legacy_evaluation=True,
        loaded_workers_sha256=digest(Path(directory)/('loaded-workers-'+os.environ['SLURM_JOB_ID']+'.json')),
        control_source_commit=binding['control_source_commit'],legacy_source_commit=binding['legacy_source_commit'])
    verify_result(result,binding['split'],binding['checkpoint_step'],ids,sources)
    return result


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def finalize_result(path, value):
    path = Path(path)
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != value:
            raise ValueError('Existing final result differs from complete journal')
        return
    _write_once(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-base', required=True)
    parser.add_argument('--journal', required=True)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    binding_path = Path(args.binding).resolve()
    binding = json.loads(binding_path.read_text())
    validate_binding(binding, args.config)
    original = resolved_document(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True),
                                 binding['run_id'])
    config = legacy_config(original,binding['checkpoint'],binding['retrieval_url'])
    activate(binding['legacy_source_root'])
    import ray
    os.environ['SEARCH_G1_PROFILE'] = 'legacy_evaluation'
    os.environ['SEARCH_G1_LEGACY_BINDING'] = str(binding_path)
    runtime_env = {'env_vars': {key: value for key, value in os.environ.items()
        if key.startswith(('SEARCH_G1_', 'ECO_')) or key in ('SLURM_JOB_ID', 'RAY_TMPDIR')}}
    assert not ray.is_initialized()
    runtime_env['env_vars']['PYTHONPATH']=str(binding['legacy_source_root'])+':'+str(Path(__file__).resolve().parents[2])
    runtime_env['env_vars']['SEARCH_G1_LEGACY_SOURCE_ROOT']=str(binding['legacy_source_root'])
    runtime_env['worker_process_setup_hook']=setup_worker
    ray.init(address='local',num_gpus=8,_temp_dir=os.environ['RAY_TMPDIR'],runtime_env=runtime_env)
    try:
        task = ray.remote(run_evaluation)
        result = ray.get(task.remote(config, binding, args.source_base, args.journal))
        finalize_result(Path(args.journal)/'legacy-result.json', result)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
