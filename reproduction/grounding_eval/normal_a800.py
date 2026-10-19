"""NQ-only normal supplement using the historical Common8 controller unchanged.

Execute this file, not a trainer entry. The runtime source/checkpoint are read-only;
full trajectories and all Ray/HF scratch stay in the explicit remote output root.
One rollout-only native worker constructs no optimizer/reference/critic.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def gpu_lease(gpu_uuid):
    # Same OFD lock namespace used by the current A800 native actor controller.
    root = Path('/data/shaoshitong/.search-g1-gpu-uuid-locks')
    root.mkdir(exist_ok=True)
    if not re.fullmatch(r'GPU-[0-9a-f-]+', gpu_uuid):
        raise ValueError('Invalid physical GPU UUID')
    fd = os.open(root / (gpu_uuid + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.fcntl(fd, fcntl.F_OFD_SETLK,
                    struct.pack('hhqqi4x', fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                             '--format=csv,noheader,nounits'], text=True, timeout=15)
        rows = list(csv.reader(io.StringIO(inventory)))
        match = [row for row in rows if row[1].strip() == gpu_uuid]
        if len(match) != 1 or int(match[0][2]) > 512 or int(match[0][3]) != 0:
            raise RuntimeError('GPU is not an idle candidate at leased admission')
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader,nounits'], text=True, timeout=15)
        if any(row and row[0].strip() == gpu_uuid for row in csv.reader(io.StringIO(processes))):
            raise RuntimeError('GPU has an existing compute process')
        return fd, int(match[0][0])
    except BaseException:
        os.close(fd)
        raise


def make_worker_class(binding):
    import torch
    import torch.distributed as dist
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.base.decorator import register, Dispatch

    class NormalRolloutWorker(ActorRolloutRefWorker):
        def __init__(self, config, role):
            assert os.environ.get('SEARCH_G1_PROFILE') == 'grounding_normal'
            assert role == 'actor_rollout'
            super().__init__(config, role='rollout')

        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def init_model(self):
            # Execution-only scheduling limits; token/sampling protocol unchanged.
            import importlib
            module = importlib.import_module('verl.workers.rollout.vllm_rollout.vllm_rollout')
            native_llm = module.LLM
            def bounded_llm(*args, **kwargs):
                kwargs['max_num_batched_tokens'] = 16384
                kwargs['max_num_seqs'] = 256
                return native_llm(*args, **kwargs)
            module.LLM = bounded_llm
            try:
                super().init_model()
            finally:
                module.LLM = native_llm
            assert self._is_rollout and not self._is_actor and not self._is_ref
            assert self.actor_optimizer is None and self.actor_lr_scheduler is None
            from transformers import GenerationConfig
            from reproduction.common8.sampling_capture import generation_source, loaded_generation
            from verl.third_party.vllm.vllm_v_0_6_3.llm_engine_sp import _load_generation_config_dict
            engine = self.rollout.inference_engine.llm_engine
            run = dict(profile='common8', run_id=binding['source_run_id'],
                       source_commit=binding['source_commit'], source_sha256=binding['source_sha256'],
                       initial_model=binding['checkpoint'], initial_model_sha256={
                           'generation_config.json': binding['generation_config_sha256']},
                       execution_backend='ssh_local', slurm_job_id=None,
                       execution_id=binding['execution_id'])
            original, proof = generation_source(engine.model_config, _load_generation_config_dict, run=run)
            engine._common8_generation_provenance = loaded_generation(
                proof, engine.generation_config_fields, GenerationConfig.from_dict(original).to_diff_dict())
            return dict(rank=dist.get_rank(), world_size=dist.get_world_size(),
                        checkpoint=binding['checkpoint'], actual_model_loaded=True,
                        optimizer_constructed=False, optimizer_steps=0,
                        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        model_dtype=str(next(self.actor_module.parameters()).dtype),
                        rollout_dtype=str(self.config.rollout.dtype),
                        device_name=torch.cuda.get_device_name(torch.cuda.current_device()))

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

    return NormalRolloutWorker


def extract_rows(batch, source_rows, ordinals, binding, tokenizer):
    from search_g1.signals import decode_segments
    from search_r1.qa_protocol import extract_final_answer
    from verl.utils.reward_score import qa_em
    result = []
    for index, (source, ordinal) in enumerate(zip(source_rows, ordinals)):
        item = batch[index]
        plen = item.batch['prompts'].numel()
        width = int(item.batch['attention_mask'][plen:].sum())
        tokens = item.batch['responses'][:width]
        mask = item.batch['info_mask'][plen:plen + width]
        text, ranges = decode_segments(tokenizer, tokens, mask)
        spans = [text[a:b] for a, b in ranges]
        turns = item.non_tensor_batch['behavior_turns']
        seen, observations = [], []
        for turn in turns:
            ids = list(turn['observation_token_ids'])
            if ids:
                actual = tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
                # The original 500-token crop can cut the closing wrapper.
                # Remove only real wrapper text; preserve title/order/truncation.
                body = actual
                if '<information>' in body:
                    body = body.split('<information>', 1)[1]
                if '</information>' in body:
                    body = body.split('</information>', 1)[0]
                seen.append(body)
                observations.append(dict(token_ids=ids, actual_text=actual))
        calls = int(batch.meta_info['actual_retrieval_stats'][index])
        if calls != len(seen):
            raise ValueError('Actual retrieval and observed document blocks disagree')
        content = source['prompt'][0]['content']
        markers = list(re.finditer(r'Question:\s*', content))
        if not markers:
            raise ValueError('Unrecognized historical question prompt')
        question = content[markers[-1].end():].strip()
        gold = source['reward_model']['ground_truth']['target']
        if hasattr(gold, 'tolist'):
            gold = gold.tolist()
        gold = [gold] if isinstance(gold, str) else list(gold)
        answer = extract_final_answer(spans)
        initial_tokens = int(item.batch['attention_mask'][:plen].sum())
        per_turn = []
        for turn in turns:
            per_turn.append(dict(
                turn_index=int(turn['turn_index']),
                model_input_tokens=len(turn['actual_request_token_ids']),
                model_output_tokens_raw=int(turn['raw_generated_token_count']),
                model_output_tokens_kept=int(turn['kept_token_count']),
                retrieved_observation_tokens=len(turn['observation_token_ids']),
                correction_tokens=len(turn['correction_token_ids']),
                cached_input_tokens=None,
                cached_input_tokens_reason='Native Common8 sampling provenance does not expose a per-request cache-hit count',
                finish_reason=turn['finish_reason'], stop_reason=turn['stop_reason']))
        row = dict(qid=str(source['extra_info']['index']), data_source='nq',
                   evaluation_ordinal=int(ordinal), question=question, gold=gold,
                   model_spans=spans, seen_docs=seen, actual_retrieval_calls=calls,
                   model_generated_tokens=int(mask.sum()),
                   raw_generated_tokens=sum(t['model_output_tokens_raw'] for t in per_turn),
                   observation_tokens=int((1 - mask).sum()),
                   retrieved_observation_tokens=sum(t['retrieved_observation_tokens'] for t in per_turn),
                   correction_tokens=sum(t['correction_tokens'] for t in per_turn),
                   initial_prompt_tokens=initial_tokens,
                   trajectory_response_tokens=width,
                   trajectory_total_tokens=initial_tokens + width,
                   cumulative_model_input_tokens=sum(t['model_input_tokens'] for t in per_turn),
                   cached_input_tokens=None,
                   per_turn_token_usage=per_turn,
                   original_answer=answer,
                   original_em=int(answer is not None and qa_em.em_check(answer, gold)),
                   model_search_markers=sum(len(re.findall(r'<search>.*?</search>', x, re.S)) for x in spans),
                   source_run_id=binding['source_run_id'], source_shard_sha256=binding['dataset_sha256'],
                   checkpoint_id=binding['checkpoint_identity_sha256'],
                   normal_config_sha256=binding['normal_config_sha256'],
                   execution_batch_questions=len(source_rows),
                   actual_observations=observations,
                   behavior_turns=list(turns), response_token_ids=tokens.tolist(),
                   response_policy_mask=mask.tolist(),
                   prompt_token_ids=item.batch['prompts'][item.batch['prompts'] != tokenizer.pad_token_id].tolist(),
                   completed_at=time.time())
        # Hash the complete actual source record, not a score-only projection.
        row['source_record_hash'] = hashlib.sha256(canonical(row)).hexdigest()
        result.append(row)
    return result


def run(args):
    base, output = Path(args.source_base).resolve(), Path(args.output).resolve()
    if not str(output).startswith('/data/shaoshitong/grounding-eval-20261019/'):
        raise ValueError('A800 output must be this task isolated remote directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'records').mkdir(exist_ok=True)
    lock = (output / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime = base / 'repo'
    sys.path.insert(0, str(runtime))
    run_manifest = json.loads((base / 'logs/run-manifest.json').read_text())
    source_map = run_manifest['source_sha256']
    for name in ('reproduction/common8/generation_behavior.py', 'search_r1/llm_agent/generation.py',
                 'search_r1/qa_protocol.py', 'search_g1/signals.py',
                 'verl/workers/fsdp_workers.py',
                 'verl/workers/rollout/vllm_rollout/vllm_rollout.py',
                 'reproduction/common8/behavior_trace.py',
                 'reproduction/common8/sampling_capture.py'):
        if sha(runtime / name) != source_map[name]:
            raise ValueError('Historical runtime source differs: ' + name)
    checkpoint = base / 'checkpoints/search-g1-common8/actor/global_step_200'
    complete = json.loads((checkpoint / 'complete.json').read_text())
    if complete['run_id'] != run_manifest['run_id'] or complete['step'] != 200:
        raise ValueError('Checkpoint belongs to a different run/step')
    # Current whole-file hashes come from the independent read-only inventory.
    # Require an externally produced full-file proof before any CUDA allocation.
    verified = json.loads(Path(args.checkpoint_hashes).read_text())
    if 'inventory' in verified:
        verified = verified['inventory']
    matches = [row for row in verified['assets'] if row['checkpoint'] == str(checkpoint)]
    if len(matches) != 1 or matches[0].get('all_manifest_weights_hash_verified') is not True:
        raise ValueError('Missing successful current full checkpoint SHA inventory')
    hashes = {row['name']: row['sha256'] for row in matches[0]['files']}
    for name, expected in complete['files_sha256'].items():
        if hashes.get(name) != expected:
            raise ValueError('Checkpoint hash inventory does not match published checkpoint')
    for row in matches[0]['files']:
        stat = (checkpoint / row['name']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
            raise ValueError('Checkpoint changed since independent SHA inventory')
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    data = base / 'data/final.parquet'
    dataset_sha = sha(data)
    if dataset_sha != '55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1':
        raise ValueError('Wrong immutable final dataset')
    source_rows = pq.read_table(data).to_pylist()
    nq = [(i, row) for i, row in enumerate(source_rows) if row['data_source'] == 'nq']
    assert len(source_rows) == 51713 and len(nq) == 3610
    original = json.loads((base / 'logs/resolved-training-config.json').read_text())['resolved_config']
    normal = dict(record_schema_version=2, token_accounting='actual_native_per_call_raw_and_kept_v1',
                  controller='historical-common8-generation-behavior-unchanged', greedy=True,
                  max_start_length=2048, max_prompt_length=4096, max_response_length=500,
                  max_obs_length=500, max_turns=4, final_no_search_turn=True, topk=3,
                  retriever=args.retriever, dtype='bfloat16', tensor_parallel_size=1,
                  source_generation_sha256=source_map['reproduction/common8/generation_behavior.py'])
    stable = dict(method=args.method, source_run_id=run_manifest['run_id'],
                  source_commit=run_manifest['source_commit'], source_sha256=source_map,
                  checkpoint=str(checkpoint), checkpoint_identity_sha256=hashlib.sha256(canonical(hashes)).hexdigest(),
                  generation_config_sha256=hashes['generation_config.json'], dataset_sha256=dataset_sha,
                  normal_config=normal, normal_config_sha256=hashlib.sha256(canonical(normal)).hexdigest())
    binding_path = output / 'binding.json'
    if binding_path.exists():
        binding = json.loads(binding_path.read_text())
        if any(binding.get(k) != value for k, value in stable.items()):
            raise ValueError('Existing normal output binding differs')
    else:
        binding = dict(stable, execution_id=uuid.uuid4().hex)
        atomic(binding_path, binding)
    pending = []
    selected = nq if args.limit is None else nq[:args.limit]
    for ordinal, row in selected:
        dest = output / 'records' / ('%08d.json' % ordinal)
        if dest.exists():
            previous = json.loads(dest.read_text())
            assert previous['evaluation_ordinal'] == ordinal
            assert previous['checkpoint_id'] == binding['checkpoint_identity_sha256']
            assert previous['normal_config_sha256'] == binding['normal_config_sha256']
        else:
            pending.append((ordinal, row))
    if not pending:
        atomic(output / 'normal-complete.json', dict(target=len(selected), completed=len(selected), checked_at=time.time()))
        return
    gpu_fd, physical_index = gpu_lease(args.gpu_uuid)
    # vLLM 0.6.3 NVML mapping parses a numeric visible ordinal; UUID lease
    # above still authenticates exactly which physical ordinal this is.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(physical_index), SEARCH_G1_PROFILE='grounding_normal',
                      PYTHONDONTWRITEBYTECODE='1', HF_HOME=str(output / 'cache/hf'),
                      TRANSFORMERS_CACHE=str(output / 'cache/hf'),
                      RAY_TMPDIR='/tmp/gne-' + binding['execution_id'][:12], TOKENIZERS_PARALLELISM='false')
    os.environ.pop('SLURM_JOB_ID', None)
    os.environ['PYTHONPATH'] = str(runtime)
    atomic(output / 'worker-identity.json', dict(pid=os.getpid(), command=sys.argv,
           physical_index=physical_index, gpu_uuid=args.gpu_uuid, started_at=time.time()))
    import ray
    import torch
    from torch.utils.data import DataLoader, Subset
    from verl import DataProto
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
    from reproduction.common8.generation_behavior import _run as original_loop
    from reproduction.common8.behavior_runtime import attach_generation_identity
    ray.init(num_gpus=1, num_cpus=8, include_dashboard=False,
             _temp_dir=os.environ['RAY_TMPDIR'], runtime_env={'env_vars': {
                 key: os.environ[key] for key in ('SEARCH_G1_PROFILE', 'PYTHONPATH',
                       'HF_HOME', 'TRANSFORMERS_CACHE', 'PYTHONDONTWRITEBYTECODE')}})
    config = OmegaConf.create(original['actor_rollout_ref'])
    OmegaConf.set_struct(config, False)
    config.model.path = str(checkpoint)
    config.model.pop('actor_checkpoint_path', None)
    config.rollout.tensor_model_parallel_size = 1
    config.rollout.gpu_memory_utilization = args.cache
    config.rollout.load_format = 'dummy_hf'
    config.rollout.n = 1
    tokenizer = hf_tokenizer(str(checkpoint))
    dataset = RLHFDataset(parquet_files=str(data), tokenizer=tokenizer, cache_dir=str(output / 'data-cache'),
                         prompt_key='prompt', max_prompt_length=4096, filter_prompts=True,
                         return_raw_chat=False, truncation='error')
    assert len(dataset) == len(source_rows)
    ordinals = [x[0] for x in pending]
    loader = DataLoader(Subset(dataset, ordinals), batch_size=args.batch_size, shuffle=False,
                        drop_last=False, collate_fn=collate_fn)
    pools = ResourcePoolManager({'grounding_normal': [1]}, {Role.ActorRollout: 'grounding_normal'})
    pools.create_resource_pool()
    workers = RayWorkerGroup(resource_pool=pools.get_resource_pool(Role.ActorRollout),
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(make_worker_class(binding)),
                                              config=config, role='actor_rollout'))
    proofs = workers.init_model()
    assert len(proofs) == 1 and proofs[0]['actual_model_loaded'] and not proofs[0]['optimizer_constructed']
    atomic(output / ('loaded-worker-' + uuid.uuid4().hex + '.json'), proofs)
    generation = GenerationConfig(max_turns=4, max_start_length=2048, max_prompt_length=4096,
        max_response_length=500, max_obs_length=500, num_gpus=1, no_think_rl=False,
        search_url=args.retriever, topk=3)
    manager = LLMGenerationManager(tokenizer, workers, generation, is_validation=True)
    offset = 0
    try:
        for batch_dict in loader:
            start = time.monotonic()
            batch = DataProto.from_single_dict(batch_dict)
            batch_ordinals = ordinals[offset:offset + len(batch)]
            ids = ['%s-nq-%d' % (binding['source_run_id'], x) for x in batch_ordinals]
            identity = dict(run_id=binding['source_run_id'], source_sha256=source_map,
                slurm_job_id=None, logical_update=200, policy_snapshot_id=binding['source_run_id'] + ':200:nq-' + uuid.uuid4().hex,
                parent_group_ids=ids, trajectory_ids=ids, validation=True)
            inputs = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            inputs.meta_info = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                    recompute_log_prob=False, do_sample=False, validate=True)
            attach_generation_identity(inputs, identity, do_sample=False)
            manager.timing_raw = {}
            captured = []
            generated = original_loop(manager, inputs, inputs.batch['input_ids'][:, -2048:].clone(), captured)
            batch = batch.union(generated)
            new_rows = extract_rows(batch, [source_rows[x] for x in batch_ordinals], batch_ordinals, binding, tokenizer)
            for written, row in enumerate(new_rows, 1):
                atomic(output / 'records' / ('%08d.json' % row['evaluation_ordinal']), row)
                if written % 16 == 0:
                    atomic(output / 'progress.json', dict(
                        completed=len(selected) - len(pending) + offset + written,
                        target=len(selected), completed_at=time.time(), errors=0,
                        phase='normal_generated_records_publication'))
            offset += len(new_rows)
            progress = dict(completed=len(selected) - len(pending) + offset, target=len(selected),
                            newly_completed=len(new_rows), seconds=time.monotonic() - start,
                            completed_at=time.time(), errors=0)
            atomic(output / 'progress.json', progress)
            print('GROUNDING_NORMAL_PROGRESS=' + json.dumps(progress), flush=True)
        atomic(output / 'normal-complete.json', dict(target=len(selected), completed=len(selected), checked_at=time.time()))
    finally:
        ray.shutdown()
        # Retain this task's diagnostic logs remotely before returning.
        import shutil
        log_root = Path(os.environ['RAY_TMPDIR']) / 'session_latest/logs'
        try:
            if log_root.exists():
                shutil.copytree(log_root, output / ('ray-diagnostic-logs-' + uuid.uuid4().hex),
                                symlinks=False, ignore_dangling_symlinks=True)
        except OSError as error:
            atomic(output / ('diagnostic-copy-error-' + uuid.uuid4().hex + '.json'),
                   dict(error=repr(error), checked_at=time.time()))
        os.close(gpu_fd)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-base', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--checkpoint-hashes', required=True)
    parser.add_argument('--gpu-uuid', default='GPU-c63282c6-1d05-224e-234b-2a8c794b1f04')
    parser.add_argument('--retriever', default='http://127.0.0.1:19000/retrieve')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cache', type=float, default=.60)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 256 and .1 <= args.cache <= .70
    assert args.limit is None or 1 <= args.limit <= 3610
    run(args)


if __name__ == '__main__':
    main()
