"""Final-only CP200 evaluation. No fit, optimizer construction, or training restore.

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


def validate_binding(binding, config_path):
    """Reject mismatched artifacts before any Ray or CUDA initialization."""
    if binding.get('role') != 'final_only_cp200_resume' or binding.get('checkpoint_step') != 200:
        raise ValueError('Only the completed experiment CP200 may enter final-only evaluation')
    required = ('checkpoint', 'checkpoint_manifest', 'checkpoint_manifest_sha256',
                'generation_config_sha256', 'resolved_config_sha256', 'validation_sha256',
                'run_id', 'source_commit', 'source_sha256', 'journal_binding')
    if any(name not in binding for name in required):
        raise ValueError('Incomplete final checkpoint/dataset binding')
    checkpoint = Path(binding['checkpoint'])
    if not checkpoint.is_absolute() or checkpoint.resolve() != checkpoint or not checkpoint.is_dir():
        raise ValueError('Final checkpoint must be an existing absolute remote directory')
    pairs = ((config_path, 'resolved_config_sha256'),
             (binding['checkpoint_manifest'], 'checkpoint_manifest_sha256'),
             (checkpoint/'generation_config.json', 'generation_config_sha256'))
    for path, key in pairs:
        if file_sha(path) != binding[key]:
            raise ValueError('Final-only bound artifact changed: ' + key)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolved_document(document, run_id):
    """Unwrap the original launch proof, retaining its science/run binding."""
    if 'resolved_config' not in document:
        return document
    if document.get('run_id') != run_id:
        raise ValueError('Resolved launch config belongs to a different experiment')
    config = document['resolved_config']
    canonical = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != document.get('resolved_config_sha256'):
        raise ValueError('Resolved launch config canonical SHA changed')
    return config


def final_config(original, checkpoint, batch_size=256, cache=0.35):
    """Only change the checkpoint, evaluation batch, and cache allocation."""
    from omegaconf import OmegaConf, open_dict
    config = OmegaConf.create(copy.deepcopy(original))
    OmegaConf.set_struct(config, False)
    checkpoint = str(Path(checkpoint).resolve())
    config.actor_rollout_ref.model.path = checkpoint
    config.actor_rollout_ref.model.pop('actor_checkpoint_path', None)
    config.trainer.pop('full_state_restore_receipt', None)
    config.trainer.pop('weight_continuation_step', None)
    config.actor_rollout_ref.rollout.gpu_memory_utilization = cache
    config.data.val_batch_size = batch_size
    assert batch_size == 256 and cache == 0.35
    assert config.trainer.n_gpus_per_node == 8 and config.trainer.nnodes == 1
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.actor_rollout_ref.rollout.temperature == 1.0
    assert config.max_turns == 4 and config.retriever.topk == 3
    assert config.do_search and config.algorithm.adv_estimator == 'grpo'
    # Rollout-only has no optimizer, reference, or train dataloader.
    return config


def make_worker_class():
    import torch
    import torch.distributed as dist
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch

    class FinalRolloutWorker(ActorRolloutRefWorker):
        def __init__(self, config, role):
            assert os.environ['SEARCH_G1_PROFILE'] == 'final_resume'
            assert role == 'actor_rollout'
            super().__init__(config, role='rollout')

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            super().init_model()
            assert not self._is_actor and self._is_rollout
            assert self.actor_optimizer is None and self.actor_lr_scheduler is None
            # The same capture protocol is used without its old CP5 restoration
            # admission. This record binds the actual final-only HF loader path.
            from transformers import GenerationConfig
            from reproduction.common8.sampling_capture import generation_source, loaded_generation
            from verl.third_party.vllm.vllm_v_0_6_3.llm_engine_sp import _load_generation_config_dict
            run = json.loads(Path(os.environ['SEARCH_G1_FINAL_BINDING']).read_text())
            engine = self.rollout.inference_engine.llm_engine
            provenance_run = dict(profile='common8', run_id=run['run_id'],
                source_commit=run['source_commit'], source_sha256=run['source_sha256'],
                initial_model=run['checkpoint'],
                initial_model_sha256={'generation_config.json': run['generation_config_sha256']},
                execution_backend='slurm', slurm_job_id=os.environ['SLURM_JOB_ID'])
            original, source = generation_source(engine.model_config, _load_generation_config_dict,
                                                  run=provenance_run)
            engine._common8_generation_provenance = loaded_generation(source,
                engine.generation_config_fields, GenerationConfig.from_dict(original).to_diff_dict())
            from reproduction.common8.native_allocation import cuda_uuids
            uuids = cuda_uuids(count=torch.cuda.device_count())
            return dict(rank=dist.get_rank(), world_size=dist.get_world_size(),
                cuda_uuid=uuids[torch.cuda.current_device()], checkpoint=run['checkpoint'],
                checkpoint_manifest_sha256=run['checkpoint_manifest_sha256'],
                generation_config_sha256=run['generation_config_sha256'],
                optimizer_constructed=False, optimizer_steps=0, role='rollout', actual_model_loaded=True)

        @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
        def generate_sequences(self, prompts):
            assert prompts.meta_info.get('do_sample') is False
            assert prompts.meta_info.get('recompute_log_prob') is False
            before = os.environ['SEARCH_G1_PROFILE']
            os.environ['SEARCH_G1_PROFILE'] = 'common8'
            try:
                return super().generate_sequences(prompts)
            finally:
                os.environ['SEARCH_G1_PROFILE'] = before

    return FinalRolloutWorker


def run_final(config, binding, source_base, directory):
    import numpy as np
    import ray
    import torch
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
    from reproduction.common8.final_resume_journal import load_prefix, append_batch, aggregate_final

    tokenizer = hf_tokenizer(binding['checkpoint'])
    record, frame, ids = expected(Path(source_base), 'final')
    assert record['sha256'] == binding['validation_sha256']
    sources = frame['data_source'].tolist()
    dataset = RLHFDataset(parquet_files=str(Path(source_base)/'data'/record['path']),
        tokenizer=tokenizer, prompt_key=config.data.prompt_key,
        max_prompt_length=config.data.max_prompt_length, filter_prompts=True,
        return_raw_chat=config.data.get('return_raw_chat', False), truncation='error')
    dataset.dataframe = bind_validation_dataframe(dataset.dataframe, Path(source_base), 'final')
    journal_binding = binding['journal_binding']
    rows = load_prefix(directory, journal_binding, ids, sources)
    assert len(rows) <= len(ids) == 51713
    if len(rows) == len(ids):
        return aggregate_final(journal_binding, ids, sources, rows)
    loader = DataLoader(Subset(dataset, range(len(rows), len(ids))),
        batch_size=config.data.val_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    pools = ResourcePoolManager({'final_rollout': [8]}, {Role.ActorRollout: 'final_rollout'})
    pools.create_resource_pool()
    worker_cls = ray.remote(make_worker_class())
    # A single rollout role needs no colocated WorkerDict. Its zero-argument
    # base constructor is incompatible with ActorRolloutRefWorker subclasses.
    workers = RayWorkerGroup(resource_pool=pools.get_resource_pool(Role.ActorRollout),
        ray_cls_with_init=RayClassWithInitArgs(cls=worker_cls,
            config=config.actor_rollout_ref, role='actor_rollout'))
    proofs = workers.init_model()
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
    import time
    for batch_dict in loader:
        batch_started = time.monotonic()
        offset = len(rows)
        batch = DataProto.from_single_dict(batch_dict)
        row_ids = [f"{binding['run_id']}-final200-ordinal-{offset+i}" for i in range(len(batch))]
        identity = dict(run_id=binding['run_id'], source_sha256=binding['source_sha256'],
            slurm_job_id=os.environ['SLURM_JOB_ID'], logical_update=200,
            policy_snapshot_id=binding['run_id']+':200:final-'+uuid.uuid4().hex,
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
                checkpoint_step=200, actual_training_outer=False, occurrence_offset=offset,
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
        print('SEARCH_G1_FINAL_RESUME_PROGRESS='+json.dumps(dict(qa_count=len(rows), target=51713,
            checkpoint_step=200, optimizer_steps=0, batch_seconds=time.monotonic()-batch_started,
            batch_questions=len(new_rows), completed_unix=time.time())), flush=True)
    return aggregate_final(journal_binding, ids, sources, rows)


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
    import ray
    binding_path = Path(args.binding).resolve()
    binding = json.loads(binding_path.read_text())
    validate_binding(binding, args.config)
    original = resolved_document(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True),
                                 binding['run_id'])
    config = final_config(original, binding['checkpoint'])
    os.environ['SEARCH_G1_PROFILE'] = 'final_resume'
    os.environ['SEARCH_G1_FINAL_BINDING'] = str(binding_path)
    runtime_env = {'env_vars': {key: value for key, value in os.environ.items()
        if key.startswith(('SEARCH_G1_', 'ECO_')) or key in ('SLURM_JOB_ID', 'RAY_TMPDIR')}}
    assert not ray.is_initialized()
    ray.init(address='local', num_gpus=8, _temp_dir=os.environ['RAY_TMPDIR'], runtime_env=runtime_env)
    try:
        task = ray.remote(run_final)
        result = ray.get(task.remote(config, binding, args.source_base, args.journal))
        finalize_result(Path(args.journal)/'final-result.json', result)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
