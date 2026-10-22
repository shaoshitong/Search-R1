"""Independent native4 real candidate generation from one immutable HF snapshot.

Run A and B as separate fresh processes in independent Slurm allocations or
sequentially in one diagnostic allocation, with the same checkpoint, five
candidate files, seed and geometry, changing only --retrieval-url. This executes the
original asynchronous search/answer trajectories, not a fixed token replay.
The default one chunk is a startup experiment; --chunks 5 completes the actual
2560-question candidate pool. There is no actor model, Adam or training update.

Resident inference-only engines omit FSDP weight movement and phase KV release;
the reported wall time is therefore a candidate-pipeline A/B measurement, not
a claim about complete training-outer speed. Stochastic asynchronous scheduling
can change the generated workload even from identical parents and policy.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
# Ray 2.40.0 source: src/ray/common/ray_config_def.h lines360/364,
# 594/598/866. ray_config.h lines67-74 defines the RAY_{name} overrides.
# Disable both distinct idle-worker prestart paths. Bound RPC pools rather
# than letting each daemon/core-worker size them from the whole node's CPUs.
RAY_SYSTEM_CONFIG = dict(enable_worker_prestart=False, prestart_worker_first_driver=False,
    num_server_call_thread=2, gcs_server_rpc_server_thread_num=2,
    gcs_server_rpc_client_thread_num=2)
PROCESS_THREAD_ENV = dict(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    **{'RAY_' + name: ('true' if value is True else 'false' if value is False else str(value))
       for name, value in RAY_SYSTEM_CONFIG.items()})
# Apply before importing Ray/Torch/Numpy, and also pass these exact values to
# workers below. Actor torch/OMP threads and CPU reservations remain unchanged.
os.environ.update(PROCESS_THREAD_ENV)
NATIVE_SOURCES = (
    'search_g1/async_rollout_runtime.py', 'search_r1/llm_agent/generation.py',
    'reproduction/common8/generation_behavior.py',
    'reproduction/common8/sampling_capture.py', 'search_g1/mixed_runtime.py',
    'search_g1/mixed_selection.py', 'search_g1/vllm_flash3.py',
    'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
    'reproduction/easy2560_joint4/native_ray_resources.py',
    'reproduction/operations/easy7b_20261021/benchmark_candidate_native4.py',
)


def json_value(value):
    """Serialize numpy-backed diagnostic values without changing source data."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError('Unsupported diagnostic JSON value: ' + type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False, default=json_value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_import_paths(paths=None):
    """Keep fresh source first and the driver's actual dependency overlays.

    This environment's Python launcher augments sys.path at startup. Passing
    only REPO as Ray PYTHONPATH discards those locations when default_worker.py
    execs the real interpreter. Preserve their current order, without importing
    or probing anything else.
    """
    return list(dict.fromkeys(value for value in
        [str(REPO), *(sys.path if paths is None else paths)] if isinstance(value, str) and value))


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False, default=json_value)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(root, stage, **values):
    record = dict(stage=stage, checked_unix=time.time(), **values)
    publish(Path(root) / 'progress.json', record)
    print(json.dumps(record, sort_keys=True, allow_nan=False, default=json_value), flush=True)


def bind_generation_capture(engine, tokenizer, checkpoint, run):
    """Apply the repository's native before/after bookkeeping to a standard LLM.

    vLLM's original group constructor remains the caller. No sampling value,
    generated token, logits, scheduler operation or model forward is changed.
    The resident standalone LLM supplies the same RequestOutput API consumed by
    vLLMRollout's original ContinuousRequestService and raw probability reader.
    """
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from vllm.engine.llm_engine import _load_generation_config_dict
    from reproduction.common8.sampling_capture import generation_source, loaded_generation
    document, binding = generation_source(SimpleNamespace(model=str(checkpoint)),
        _load_generation_config_dict, run=run)
    engine._common8_generation_provenance = loaded_generation(binding,
        engine.generation_config_fields, GenerationConfig.from_dict(document).to_diff_dict())
    # GroupCapture needs only these two scalar attributes on the tokenizer
    # group. Preserve that group's original request-tokenizer methods.
    engine.tokenizer.eos_token_id = tokenizer.eos_token_id
    engine.tokenizer.pad_token_id = tokenizer.pad_token_id
    original = engine._create_sequence_group_with_sampling

    def create_group(request_id, seq, caller, *args, **kwargs):
        capture = getattr(engine, '_common8_stop_capture', None)
        if capture is not None:
            capture.before(request_id, seq, caller)
        group = original(request_id, seq, caller, *args, **kwargs)
        if capture is not None:
            capture.after(request_id, seq, caller, group)
        return group

    engine._create_sequence_group_with_sampling = create_group


class CandidateWorker:
    """One Ray-owned GPU, original vLLMRollout phase methods, immutable policy."""
    def __init__(self, rank, options):
        self.rank, self.options = rank, options
        self.ready_receipt = None

    def initialize(self):
        import torch
        import ray
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from reproduction.easy2560_joint4.flash3_runtime import binding
        from search_g1.vllm_flash3 import install_vllm_flash3
        from verl.workers.rollout.vllm_rollout import vLLMRollout
        options = self.options
        torch.set_num_threads(options['worker_cpus'])
        torch.cuda.set_device(0)
        tokenizer = AutoTokenizer.from_pretrained(options['checkpoint'], local_files_only=True,
            trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        audit = install_vllm_flash3(dict(infra_flash_attention_3=binding(options['source_base'])))
        began = time.monotonic()
        llm = LLM(model=options['checkpoint'], tokenizer=options['checkpoint'],
            tensor_parallel_size=1, dtype='bfloat16', enforce_eager=True,
            trust_remote_code=False, max_model_len=4096,
            max_num_seqs=64, max_num_batched_tokens=8192, block_size=256,
            num_gpu_blocks_override=options['kv_tokens'] // 256,
            gpu_memory_utilization=options['gpu_memory_utilization'], swap_space=0,
            enable_prefix_caching=False, disable_log_stats=True,
            seed=options['seed'] + self.rank)
        bind_generation_capture(llm.llm_engine, tokenizer, options['checkpoint'], options['run'])
        # Reuse all actual phase request/capture/token transport methods. Only
        # constructor ownership differs: no FSDP module or hybrid sharding manager.
        rollout = vLLMRollout.__new__(vLLMRollout)
        rollout.config = OmegaConf.create(dict(n=1, tensor_model_parallel_size=1,
            response_length=500, free_cache_engine=False))
        rollout.inference_engine = llm
        rollout.pad_token_id, rollout.eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        rollout.sampling_params = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
            max_tokens=500, logprobs=1, ignore_eos=False, detokenize=False)
        rollout.generation_scheduler_limits = dict(max_num_seqs=64, max_num_batched_tokens=8192)
        rollout._search_g1_flash3_functions = audit
        rollout._generation_phase = rollout._generation_phase_setup_handle = None
        rollout._generation_cache_setup_started = False
        self.rollout = rollout
        self.ready_receipt = dict(rank=self.rank, pid=os.getpid(), hostname=socket.gethostname(),
            ray_gpu_ids=ray.get_gpu_ids(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            gpu_name=torch.cuda.get_device_name(), initialization_seconds=time.monotonic() - began,
            checkpoint=options['checkpoint'], seed=options['seed'] + self.rank,
            actual_max_num_seqs=llm.llm_engine.scheduler_config.max_num_seqs,
            actual_max_num_batched_tokens=llm.llm_engine.scheduler_config.max_num_batched_tokens,
            actual_max_model_len=llm.llm_engine.model_config.max_model_len,
            actual_block_size=llm.llm_engine.cache_config.block_size,
            actual_num_gpu_blocks=llm.llm_engine.cache_config.num_gpu_blocks,
            actual_engine_dtype=str(llm.llm_engine.model_config.dtype),
            generation_config=llm.llm_engine._common8_generation_provenance,
            actor_model_loaded=False, optimizer_loaded=False, training_modified=False)
        return self.ready_receipt

    def begin_generation_phase(self, snapshot):
        self.actual_work = dict(generation_requests=0, raw_generated_tokens=0, prompt_tokens=0)
        self.rollout.begin_generation_phase(snapshot)
        return {key: snapshot[key] for key in ('phase_id', 'policy_snapshot_id')}

    def actor_rollout_advance_generation_phase(self, handle, requests, max_steps=8):
        self.actual_work['generation_requests'] += len(requests)
        self.actual_work['prompt_tokens'] += sum(len(row['prompt_token_ids']) for row in requests)
        result = self.rollout.advance_generation_phase(handle, requests, max_steps=max_steps)
        for completed in result['completed']:
            record = completed['output'].non_tensor_batch['sampling_behavior'][0]
            self.actual_work['raw_generated_tokens'] += record['raw_generated_token_count']
        return result

    def end_generation_phase(self, handle, abort=False):
        return dict(self.rollout.end_generation_phase(handle, abort=abort), rank=self.rank,
            weight_syncs=0, immutable_hf_snapshot_loaded_once=True, actual_work=dict(self.actual_work))


class CandidateWorkerGroup:
    """The narrow interface expected by the original RayPhaseService."""
    def __init__(self, workers):
        self._workers = workers

    def begin_generation_phase(self, snapshot):
        import ray
        return ray.get([worker.begin_generation_phase.remote(snapshot) for worker in self._workers])

    def end_generation_phase(self, handle, abort=False):
        import ray
        return ray.get([worker.end_generation_phase.remote(handle, abort=abort) for worker in self._workers])


def parent_identity(parent, offset, snapshot, run):
    group_ids = [str(offset + index) + '-' + digest(str(value))
        for index, value in enumerate(parent.non_tensor_batch['index'])]
    return dict(run_id=run['run_id'], source_sha256=run['source_sha256'],
        slurm_job_id=run['slurm_job_id'], logical_update=1, validation=False,
        policy_snapshot_id=snapshot, parent_group_ids=group_ids,
        trajectory_ids=[run['run_id'] + '-' + group + '-sample-' + str(sample)
            for group in group_ids for sample in range(5)])


def parent_fingerprint(parent):
    """Hash actual initial prompts/targets without including them in reports."""
    tensor = hashlib.sha256()
    for name in ('input_ids', 'attention_mask', 'position_ids'):
        value = parent.batch[name].detach().cpu().contiguous()
        tensor.update(name.encode()); tensor.update(str(tuple(value.shape)).encode())
        tensor.update(str(value.dtype).encode()); tensor.update(value.numpy().tobytes())
    return dict(parent_count=len(parent), prompt_tensors_sha256=tensor.hexdigest(),
        stable_ids_sha256=digest([str(value) for value in parent.non_tensor_batch['candidate_stable_id']]),
        targets_sha256=digest([value['ground_truth'] for value in parent.non_tensor_batch['reward_model']]))


def selection_summary(selector, chunks):
    ordered = [selector.results[index] for index in range(len(selector.results))]
    counts = {category: sum(item.category == category for item in ordered)
        for category in ('mixed', 'all_correct', 'all_wrong')}
    result = dict(candidate_groups=len(ordered), candidate_categories=counts,
        candidate_mixed_fraction=counts['mixed'] / len(ordered),
        training_em=sum(sum(row.correctness) for row in ordered) / (5 * len(ordered)),
        selection_performed=False, selected_groups=None, selected_mixed_fraction=None)
    if chunks == 5:
        from search_g1.mixed_selection import InsufficientEligibleGroups
        try:
            members = selector.select()
        except InsufficientEligibleGroups:
            result['selection_status'] = 'INSUFFICIENT_ELIGIBLE_GROUPS'
        else:
            mixed = sum(row.category == 'mixed' for row in members)
            result.update(selection_status='COMPLETE', selection_performed=True,
                selected_groups=len(members), selected_mixed=mixed,
                selected_mixed_fraction=mixed / len(members),
                selected_ordinals=[row.ordinal for row in members])
    else:
        result['selection_status'] = 'STARTUP_CHUNKS_ONLY_NOT_FULL_2560_SELECTION'
    return result


def run(args):
    from reproduction.easy2560_joint4.native_ray_resources import raise_self_soft
    nproc = raise_self_soft()
    import ray
    import torch
    from transformers import AutoTokenizer
    from verl import DataProto
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
    from search_g1.async_rollout_runtime import _run as native_async_run
    from reproduction.common8.behavior_runtime import attach_generation_identity
    from search_g1.mixed_runtime import strict_vectors
    from search_g1.mixed_selection import MixedFirstSelector
    import_paths = worker_import_paths()
    worker_pythonpath = os.pathsep.join(import_paths)
    # Ray system subprocesses and the four explicit actor runtime environments
    # must both inherit the same already-working dependency search paths.
    os.environ['PYTHONPATH'] = worker_pythonpath
    source_base, checkpoint, root = map(lambda value: Path(value).resolve(),
        (args.source_base, args.checkpoint, args.root))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'raw').mkdir()
    source_run = json.loads((source_base / 'logs/run-manifest.json').read_text())
    source_map = {name: sha(REPO / name) for name in NATIVE_SOURCES}
    run_binding = dict(profile='common8', run_id=uuid.uuid4().hex,
        source_commit=source_run['source_commit'], source_sha256=source_map,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), initial_model=str(checkpoint),
        initial_model_sha256={'generation_config.json': sha(checkpoint / 'generation_config.json')})
    if not (run_binding['slurm_job_id'] and run_binding['slurm_job_id'].isdecimal()):
        raise ValueError('Run this four-GPU experiment in its own Slurm allocation')
    snapshot = run_binding['run_id'] + ':immutable-hf-' + digest(str(checkpoint))
    options = dict(source_base=str(source_base), checkpoint=str(checkpoint), run=run_binding,
        worker_cpus=args.worker_cpus, seed=args.seed, kv_tokens=args.kv_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization)
    launch = dict(role='independent_native4_real_candidate_pipeline',
        label=args.label, options=options, candidate_files=args.candidate_files,
        requested_chunks=args.chunks, retrieval_url=args.retrieval_url,
        runtime_source_sha256=source_map, benchmark_source_commit_is_source_base_parent=True,
        ray_system_config=RAY_SYSTEM_CONFIG, ray_process_thread_env=PROCESS_THREAD_ENV,
        native_nproc=nproc,
        import_path_receipt=dict(driver_executable=sys.executable, driver_prefix=sys.prefix,
            driver_ray_module=ray.__file__, driver_torch_module=torch.__file__,
            worker_pythonpath=import_paths),
        training_modified=False, optimizer_loaded=False, optimizer_writes=0,
        full_training_outer_speedup_proven=False,
        timing_scope='Real async multi-turn generation, HTTP retrieval, original strict-EM and Easy selection; '
            'resident standalone engines, no FSDP weight transfer or KV offload/reload',
        stochastic_causality_limit='Same parents and immutable policy do not guarantee the same sampled '
            'tokens, retrieval queries or workload under asynchronous scheduling; compare actual work too')
    publish(root / 'launch.json', launch)
    event(root, 'INITIALIZING_FOUR_TP1_FA3_ENGINES', requested_chunks=args.chunks)
    # Ray owns only GPUs granted to this separate Slurm allocation. Never join
    # an existing Ray cluster or use production workers.
    env = dict(VLLM_ATTENTION_BACKEND='FLASH_ATTN', VLLM_USE_V1='0',
        VLLM_NO_USAGE_STATS='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.worker_cpus),
        PYTHONPATH=worker_pythonpath, RAY_DEDUP_LOGS='0', **PROCESS_THREAD_ENV)
    # Ray's UNIX sockets have a short platform path limit. This new per-job
    # remote /tmp directory is ephemeral; every durable report/raw tensor stays
    # under the explicit experiment root. Do not use a long root as socket path.
    ray_temp = '/tmp/e7bc-' + run_binding['slurm_job_id'] + '-' + run_binding['run_id'][:6]
    env.update(TRITON_CACHE_DIR=str(root / 'cache/triton'), XDG_CACHE_HOME=str(root / 'cache'))
    ray.init(num_gpus=4, num_cpus=args.worker_cpus * 4 + 2,
        _temp_dir=ray_temp, include_dashboard=False,
        _system_config=RAY_SYSTEM_CONFIG,
        runtime_env={'env_vars': env})
    workers = []
    try:
        worker_class = ray.remote(num_gpus=1, num_cpus=args.worker_cpus)(CandidateWorker)
        workers = [worker_class.remote(rank, options) for rank in range(4)]
        ready = ray.get([worker.initialize.remote() for worker in workers])
        publish(root / 'workers.json', ready)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
            trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        manager = LLMGenerationManager(tokenizer, CandidateWorkerGroup(workers), GenerationConfig(
            max_turns=4, max_start_length=2048, max_prompt_length=4096,
            max_response_length=500, max_obs_length=500, num_gpus=4,
            search_url=args.retrieval_url, topk=3, async_phase_enabled=True))
        selector = MixedFirstSelector(target=512, limit=args.chunks * 512,
            fixed_pool=True)
        event(root, 'FOUR_TP1_FA3_ENGINES_READY', workers=4, requested_chunks=args.chunks)
        rows, phase_wall, started = [], 0.0, time.monotonic()
        for chunk, filename in enumerate(args.candidate_files[:args.chunks]):
            # Read only the trusted owned production artifact. Ignore old output
            # and worker RNG: both A/B jobs regenerate full trajectories from the
            # same explicit checkpoint and original parent tensors.
            load_start = time.monotonic()
            saved = torch.load(filename, map_location='cpu', weights_only=False)
            parent, previous_identity = saved['parents'], saved['identity']
            if len(parent) != 512:
                raise ValueError('Candidate file must contain exactly 512 original parent rows')
            parent = DataProto(batch=parent.batch.clone(), non_tensor_batch=copy.deepcopy(parent.non_tensor_batch),
                meta_info=copy.deepcopy(parent.meta_info))
            del saved
            input_proof = parent_fingerprint(parent)
            identity = parent_identity(parent, chunk * 512, snapshot, run_binding)
            for local in range(512):
                ordinal = selector.reserve(parent.non_tensor_batch['candidate_stable_id'][local],
                    parent.non_tensor_batch['candidate_original_question'][local])
                if ordinal != chunk * 512 + local:
                    raise ValueError('Saved full candidate pool has duplicate or reordered questions')
            repeated = parent.repeat(repeat_times=5, interleave=True)
            gen = repeated.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            attach_generation_identity(gen, identity, do_sample=True)
            gen.meta_info.update(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            load_seconds = time.monotonic() - load_start
            event(root, 'REAL_CANDIDATE_CHUNK_STARTED', chunk=chunk,
                completed_questions=chunk * 512, source_candidate_file=filename, **input_proof)
            began = time.monotonic()
            # This is the exact async run body used by LLMGenerationManager.
            # Bypass only its production failure-journal wrapper, which requires
            # a training manifest/attempt and must never write into production.
            captured = []
            try:
                output = native_async_run(manager, gen,
                    gen.batch['input_ids'][:, -manager.config.max_start_length:].clone().long(), captured)
            except BaseException:
                torch.save(dict(identity=identity, partial_actual_sampling_records=captured),
                    root / 'raw' / ('chunk-%d-failed-partial.pt' % chunk))
                raise
            generation_seconds = time.monotonic() - began
            del captured
            phase_wall += generation_seconds
            score_start = time.monotonic()
            vectors = strict_vectors(parent, output, tokenizer)
            for local, vector in enumerate(vectors):
                selector.complete(chunk * 512 + local,
                    identity['trajectory_ids'][local * 5:(local + 1) * 5], vector)
            score_seconds = time.monotonic() - score_start
            phase = output.meta_info['async_rollout']
            work = [record['actual_work'] for record in phase['phase_metrics']]
            report = dict(chunk=chunk, source_candidate_file=filename,
                source_original_policy_snapshot_id=previous_identity['policy_snapshot_id'],
                input=input_proof, generation_seconds=generation_seconds,
                load_parent_seconds=load_seconds, strict_score_seconds=score_seconds,
                questions=512, trajectories=len(output),
                actual_generation_requests=phase['generation_requests'],
                actual_raw_generated_tokens=sum(row['raw_generated_tokens'] for row in work),
                actual_prompt_tokens=sum(row['prompt_tokens'] for row in work),
                retained_model_tokens=int(output.batch['sampling_mask'].sum()),
                actual_retrieval_calls=sum(output.meta_info['actual_retrieval_stats']),
                retrieval_count_scope='Logical executed search actions; HTTP retries and proxy cache hits '
                    'are measured by the independent proxy/service reports',
                early_final_trajectories=sum(output.meta_info['early_final_turn_stats']),
                strict_correctness_sha256=digest(vectors),
                mixed_groups=sum(0 < sum(row) < 5 for row in vectors),
                correct_trajectories=sum(map(sum, vectors)), phase=phase)
            torch.save(dict(parents=parent, output=output, identity=identity,
                strict_correctness=vectors, ordinals=list(range(chunk * 512, (chunk + 1) * 512))),
                root / 'raw' / ('chunk-%d.pt' % chunk))
            publish(root / ('chunk-%d.json' % chunk), report)
            rows.append(report)
            event(root, 'REAL_CANDIDATE_CHUNK_COMPLETE', chunk=chunk,
                completed_questions=(chunk + 1) * 512, generation_seconds=generation_seconds,
                actual_raw_generated_tokens=report['actual_raw_generated_tokens'],
                actual_retrieval_calls=report['actual_retrieval_calls'], mixed_groups=report['mixed_groups'])
            del output, gen, repeated, parent
        result = dict(launch, status='REAL_CANDIDATE_PIPELINE_COMPLETED',
            actual_world_size=4, completed_chunks=len(rows),
            completed_questions=512 * len(rows), completed_trajectories=2560 * len(rows),
            input_fingerprints=[row['input'] for row in rows],
            input_fingerprints_sha256=digest([row['input'] for row in rows]),
            pipeline_seconds_including_input_io_score_and_output_save=time.monotonic() - started,
            generation_phase_wall_seconds=phase_wall,
            actual_raw_generated_tokens=sum(row['actual_raw_generated_tokens'] for row in rows),
            actual_generation_requests=sum(row['actual_generation_requests'] for row in rows),
            actual_prompt_tokens=sum(row['actual_prompt_tokens'] for row in rows),
            actual_retrieval_calls=sum(row['actual_retrieval_calls'] for row in rows),
            selection=selection_summary(selector, args.chunks),
            chunk_reports=['chunk-%d.json' % chunk for chunk in range(len(rows))],
            actual_fa3_prefill_and_decode=all(
                record['flash3_execution']['calls']['prefill'] > 0 and
                record['flash3_execution']['calls']['decode'] > 0
                for record in rows[-1]['phase']['phase_metrics']))
        publish(root / 'result.json', result)
        event(root, 'REAL_CANDIDATE_PIPELINE_COMPLETED', completed_chunks=len(rows),
            completed_questions=512 * len(rows), generation_phase_wall_seconds=phase_wall,
            actual_generation_requests=result['actual_generation_requests'],
            actual_retrieval_calls=result['actual_retrieval_calls'])
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
        ray.shutdown()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-base', required=True, help='Existing read-only compiled FA3 binding and source run')
    p.add_argument('--checkpoint', required=True, help='Explicit immutable owned HF checkpoint snapshot')
    p.add_argument('--candidate-files', required=True, nargs=5, help='Ordered original 0/512/1024/1536/2048.pt files')
    p.add_argument('--retrieval-url', required=True)
    p.add_argument('--root', '--output', dest='root', required=True, help='Fresh remote experiment output directory')
    p.add_argument('--label', default='candidate-native4')
    p.add_argument('--chunks', type=int, default=1, choices=range(1, 6))
    p.add_argument('--seed', type=int, default=317)
    p.add_argument('--worker-cpus', type=int, default=8)
    p.add_argument('--gpu-memory-utilization', type=float, default=.65)
    p.add_argument('--kv-tokens', type=int, default=131072)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.worker_cpus < 1 or args.kv_tokens < 4096 or args.kv_tokens % 256:
        p.error('Use positive worker CPU count and KV tokens >=4096 divisible by256')
    if not 0 < args.gpu_memory_utilization < 1:
        p.error('GPU memory utilization must be in (0,1)')
    resolved_root = Path(args.root).resolve()
    if resolved_root.is_relative_to('/Users') or resolved_root.is_relative_to('/private'):
        p.error('Large raw rollout output must remain on remote/shared storage')
    try:
        run(args)
    except BaseException as error:
        if Path(args.root).is_dir():
            publish(Path(args.root) / 'failure.json', dict(status='FAILED', error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), training_modified=False, optimizer_writes=0))
        raise


if __name__ == '__main__':
    main()
