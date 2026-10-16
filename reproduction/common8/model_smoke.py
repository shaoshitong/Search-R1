"""Actual eight-rank Qwen/FSDP/reference/vLLM smoke, with zero optimizer writes.

Run only inside the full native allocation, first alone and later with the owned
retriever/probe resident. This does not perform an agent benchmark or prove a
complete 2560-trajectory training memory peak.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from search_g1.process_jit_cache import configure
configure()
import torch

from reproduction.common8.configuration import REPO, resolved
from reproduction.common8.stage_retrieval import binding, memory_bytes
from reproduction.common8.validation_identity import expected
from reproduction.common8.verify_data import digest
from reproduction.common8.precision_evidence import PrecisionObservation, verify_precision

LOGPROB_ATOL = 0.05  # Original unchanged backend parity bound, not a training hyperparameter.


def compare(old, reference, mask, raw_tokens, raw_logprobs, responses):
    old, reference, mask = old.cpu(), reference.cpu(), mask.cpu()
    raw_tokens, raw_logprobs, responses = raw_tokens.cpu(), raw_logprobs.cpu(), responses.cpu()
    assert old.shape == reference.shape == mask.shape == responses.shape
    assert raw_tokens.shape == raw_logprobs.shape
    assert raw_tokens.shape[0] == mask.shape[0] and 0 < raw_tokens.shape[1] <= mask.shape[1]
    assert ((mask == 0) | (mask == 1)).all() and (mask.sum(-1) > 0).all()
    valid = mask.bool()
    assert torch.equal(valid, torch.arange(mask.shape[1]).unsqueeze(0) < mask.sum(-1).unsqueeze(1)), 'Non-prefix response mask'
    width = raw_tokens.shape[1]
    assert not valid[:, width:].any(), 'Worker invented tokens absent from actual engine output'
    raw_valid = valid[:, :width]
    assert torch.equal(raw_tokens[raw_valid], responses[:, :width][raw_valid])
    for value in (old[valid], reference[valid], raw_logprobs[raw_valid]):
        assert torch.isfinite(value).all(), 'Nonfinite valid-token log probability'
    actor_ref = (old[valid].float() - reference[valid].float()).abs()
    actor_rollout = (old[:, :width][raw_valid].float() - raw_logprobs[raw_valid].float()).abs()
    result = dict(actor_reference_max_abs=float(actor_ref.max()),
                  actor_vllm_max_abs=float(actor_rollout.max()), tolerance=LOGPROB_ATOL,
                  valid_tokens_per_question=mask.sum(-1).tolist())
    assert result['actor_reference_max_abs'] <= LOGPROB_ATOL, result
    assert result['actor_vllm_max_abs'] <= LOGPROB_ATOL, result
    return result


def aggregate(proofs, run_id, job_id, phase, ids, uuids, run=None):
    size=len(uuids)
    assert size in (4,8) and len(proofs) == size
    per_rank=64//size
    all_ids = []
    config_hashes = set()
    for rank, proof in enumerate(proofs):
        assert proof['accepted'] is True and proof['rank'] == rank
        assert (proof['run_id'], proof['slurm_job_id'], proof['phase']) == (run_id, job_id, phase)
        assert proof['cuda_uuid'] == uuids[rank]
        assert proof['actual_optimizer_writes'] == 0 and proof['sync_calls'] == 1
        assert proof['separate_actor_logprob_entrypoint'] is True
        assert 0 <= proof['internal_separate_logprob_max_abs'] <= LOGPROB_ATOL
        assert proof['sync_load_format'] == 'dtensor'
        verify_precision(proof['runtime_precision'],run)
        assert len(proof['question_ids']) == per_rank
        assert proof['parity']['tolerance'] == LOGPROB_ATOL
        assert len(proof['parity']['valid_tokens_per_question']) == per_rank
        assert all(n > 0 for n in proof['parity']['valid_tokens_per_question'])
        assert all(0 <= proof['parity'][k] <= LOGPROB_ATOL for k in
                   ('actor_reference_max_abs', 'actor_vllm_max_abs'))
        all_ids.extend(proof['question_ids']); config_hashes.add(proof['resolved_config_sha256'])
    assert all_ids == ids and len(ids) == len(set(ids)) == 64
    assert len(config_hashes) == 1
    return list(config_hashes)[0]


def coexistence(base, run, job):
    """Require this run's actual resident services; a phase name is insufficient."""
    import requests
    from reproduction.a800.owned_processes import same, identity
    from reproduction.a800.probe_admission import calibrated_version
    retrieval = json.loads((base / 'logs/retrieval-admission.json').read_text())
    assert retrieval['accepted'] is True
    assert retrieval['run_id'] == run['run_id'] and retrieval['slurm_job_id'] == job['JobId']
    ledger = json.loads((base / 'logs/process-ledger.json').read_text())
    assert ledger['run_id'] == run['run_id']
    for name in ('retriever', 'probe-server'):
        owned = ledger['jobs'][name]
        assert owned['state'] != 'STOPPED' and same(owned['leader'])
        assert identity(owned['leader']['pid'])['state'] != 'Z'
    live = json.loads((base / 'artifacts/live/manifest.json').read_text())
    version = calibrated_version(run, live)
    session = requests.Session(); session.trust_env = False
    response = session.get('http://127.0.0.1:18012/health', timeout=60)
    response.raise_for_status(); health = response.json()
    assert health['ok'] is True and health['version'] == version
    assert health['round'] == 0 and health['encoder'] == run['initial_model']
    return dict(probe_version=version, retrieval_admission_sha256=digest(base / 'logs/retrieval-admission.json'))


def main(base, phase):
    import datetime
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from search_g1.numerics import require_finite

    base = Path(base).resolve()
    run, job, allocation = binding(base)
    assert memory_bytes(job['TRES']) >= 512 * 2**30, 'Use the full native allocation for actual model work'
    assert os.environ['SEARCH_G1_PROFILE'] == 'common8' and 'WG_PREFIX' not in os.environ
    from reproduction.common8.topology import world_size
    size=world_size(run);per_rank=64//size
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['RANK']) == rank and int(os.environ['WORLD_SIZE']) == size
    assert torch.cuda.device_count() == size
    from reproduction.common8.native_allocation import cuda_uuids
    assert os.environ['CUDA_VISIBLE_DEVICES'].split(',') == allocation['visible_devices']
    assert cuda_uuids(size) == allocation['actual_cuda_runtime_uuids']
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=600))
    precision = None
    try:
        def collective(ok, message):
            value = torch.tensor(int(bool(ok)), device='cuda', dtype=torch.int32)
            dist.all_reduce(value, op=dist.ReduceOp.MIN)
            if not value.item(): raise RuntimeError(message)

        run2, config, _ = resolved(base)
        assert run2 == run
        services = coexistence(base, run, job) if phase == 'coexistence' else None
        for name, value in run['source_sha256'].items():
            assert digest(REPO / name) == value, 'Source changed after fresh run binding: ' + name
        for name, value in run['initial_model_sha256'].items():
            # Only rank0 reads the large original files, then all ranks reject together.
            ok = digest(Path(run['initial_model']) / name) == value if rank == 0 else True
            collective(ok, 'Original official model file changed: ' + name)
        record, _, ids = expected(base, 'stage')
        assert record['sha256'] == run['input_sha256']['stage.parquet']
        config_json = json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True, separators=(',', ':'))
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        root = base / 'logs' / ('model-smoke-' + phase)
        collective(not root.exists(), 'This model-smoke phase already has evidence; inspect it, do not overwrite')
        if rank == 0: root.mkdir()
        dist.barrier()

        torch.manual_seed(42)
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        actor = ActorRolloutRefWorker(config.actor_rollout_ref, role='actor_rollout')
        actor.init_model()
        reference = ActorRolloutRefWorker(config.actor_rollout_ref, role='ref')
        reference.init_model()
        # Installed after init/profiling; only actual synced model forwards are observed.
        precision = PrecisionObservation(actor, reference)
        for role, module in (('actor', actor.actor_module_fsdp), ('reference', reference.ref_module_fsdp)):
            require_finite({n: p.detach() for n, p in module.named_parameters()},
                           stage='common8-smoke-initial-' + role, process_group=module.process_group)
        dataset = RLHFDataset(str(base / 'data/stage.parquet'), actor.tokenizer,
                             max_prompt_length=config.data.max_prompt_length, truncation='error',
                             cache_dir=str(base / 'cache/model-smoke-data'))
        assert len(dataset) == 64
        indices = list(range(rank * per_rank, (rank + 1) * per_rank))
        batch = collate_fn([dataset[i] for i in indices])
        prompts = DataProto.from_dict(tensors={k: batch[k] for k in ('input_ids', 'attention_mask', 'position_ids')},
                                     meta_info={'do_sample': False, 'recompute_log_prob': True,
                                                'common8_model_smoke': True})
        engine = actor.rollout.inference_engine
        capture, sync_calls = [], []
        original_generate, original_sync = engine.generate, engine.sync_model_weights
        def generate(*args, **kwargs):
            output = original_generate(*args, **kwargs)
            capture.append((output[0].detach().cpu().clone(), output[1].detach().cpu().clone()))
            return output
        def sync(*args, **kwargs):
            sync_calls.append(kwargs.get('load_format'))
            return original_sync(*args, **kwargs)
        # Observation hooks on this disposable smoke worker; no active trainer is modified.
        engine.generate, engine.sync_model_weights = generate, sync
        try:
            output = actor.generate_sequences(prompts)
        finally:
            engine.generate, engine.sync_model_weights = original_generate, original_sync
        # Exercise the actual agent trainer's separate worker recomputation too.
        internal = output.pop(batch_keys=['old_log_probs']).batch['old_log_probs']
        independent = actor.compute_log_prob(output)
        # Direct calls mutate the input DataProto's device; Ray serialization
        # would isolate that mutation in the production controller.
        output = output.to('cpu').union(independent)
        width = output.batch['responses'].shape[-1]
        valid = output.batch['attention_mask'][:, -width:].bool()
        api_error = (internal[valid].float() - output.batch['old_log_probs'][valid].float()).abs()
        collective(api_error.numel() > 0 and torch.isfinite(api_error).all()
                   and api_error.max().item() <= LOGPROB_ATOL,
                   'Internal and actual separate actor logprob entrypoints disagree')
        ref = reference.compute_ref_log_prob(output)
        output = output.to('cpu')
        collective(len(capture) == len(sync_calls) == 1, 'Did not observe exactly one actual rollout and weight sync')
        response_width = output.batch['responses'].shape[-1]
        parity, error = None, None
        try:
            parity = compare(output.batch['old_log_probs'], ref.batch['ref_log_prob'],
                             output.batch['attention_mask'][:, -response_width:], *capture[0], output.batch['responses'])
        except AssertionError as failure:
            error = str(failure)
            print('COMMON8_MODEL_PARITY_REJECTED=' + json.dumps(dict(rank=rank, error=error)), flush=True)
        collective(error is None, 'Actual initial actor/reference/vLLM parity rejected on a rank')
        collective(sync_calls == ['dtensor'], 'Unexpected actual rollout weight load format')
        collective(not actor.actor_optimizer.state and getattr(actor.actor, '_optimizer_writes', 0) == 0,
                   'A model smoke must not write optimizer state')
        runtime_precision = precision.snapshot()
        precision_error = None
        try:
            verify_precision(runtime_precision,run)
        except (AssertionError, KeyError) as failure:
            precision_error = repr(failure)
        collective(precision_error is None, 'Actual common8 FP32 runtime precision rejected: ' + str(precision_error))
        if phase == 'coexistence':
            assert coexistence(base, run, job) == services, 'Resident services changed across the smoke'
        proof = dict(accepted=True, run_id=run['run_id'], slurm_job_id=job['JobId'], phase=phase,
                     rank=rank, cuda_uuid=allocation['actual_cuda_runtime_uuids'][rank],
                     question_ids=ids[rank*per_rank:(rank+1)*per_rank], stage_sha256=record['sha256'],
                     resolved_config_sha256=config_hash, actual_optimizer_writes=0,
                     actual_grpo_updates=0, sync_calls=len(sync_calls), sync_load_format=sync_calls[0],
                     separate_actor_logprob_entrypoint=True,
                     internal_separate_logprob_max_abs=float(api_error.max()),
                     runtime_precision=runtime_precision,
                     parity=parity, resident_services_before_generation=services,
                     initial_parameters_finite=True, checked_unix=time.time(),
                     scope='64-question single-action greedy actual worker smoke; not agent EM or full training peak')
        with (root / f'rank-{rank}.json').open('x') as f:
            json.dump(proof, f, indent=2); f.flush(); os.fsync(f.fileno())
        dist.barrier()
        if rank == 0:
            proofs = [json.loads((root / f'rank-{r}.json').read_text()) for r in range(size)]
            aggregate(proofs, run['run_id'], job['JobId'], phase, ids, allocation['actual_cuda_runtime_uuids'],run)
            final = dict(accepted=True, run_id=run['run_id'], slurm_job_id=job['JobId'], phase=phase,
                         ranks=proofs, rank_file_sha256={str(r): digest(root / f'rank-{r}.json') for r in range(size)},
                         scope=proof['scope'], actual_grpo_updates=0)
            with (root / 'complete.json').open('x') as f:
                json.dump(final, f, indent=2); f.flush(); os.fsync(f.fileno())
            print('COMMON8_MODEL_SMOKE_ACCEPTED=' + phase, flush=True)
    finally:
        if precision is not None:
            precision.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=('initial', 'coexistence'), required=True)
    main(os.environ['SEARCH_G1_BASE'], p.parse_args().phase)
