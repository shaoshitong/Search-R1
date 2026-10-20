"""Native bounded shared-VRAM TP2 sampler; only its own process groups stop.

The launcher supplies private mount/network namespaces and native unit evidence.
Existing foreign GPU contexts are explicitly permitted and recorded. The staged
pool worker supplies unchanged input claiming, phase receipts and batch seals.
"""
import argparse
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

GIB = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pool(path):
    spec = importlib.util.spec_from_file_location('shared_vram_pool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indices(value, size):
    items = [int(x) for x in value.split(',')]
    sizes = (size,) if isinstance(size, int) else size
    require(len(items) == len(set(items)) and len(items) in sizes and min(items) >= 0,
            'Invalid physical GPU list')
    return items


def foreign_contexts(uuids):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    return [dict(gpu_uuid=row[0].strip(), pid=int(row[1]), used_memory_mib=row[2].strip(),
                 process_name=row[3].strip()) for row in csv.reader(io.StringIO(output))
            if row and row[0].strip() in uuids and int(row[1]) != os.getpid()]


def actual_uuids(expected):
    from reproduction.common8.native_allocation import cuda_uuids
    actual = cuda_uuids(len(expected))
    require(actual == expected, 'Actual CUDA runtime UUID order differs from admitted numeric CVD')
    return actual


def admit(pool, args):
    require(socket.gethostname().split('.')[0] == 'h100-5', 'This shared-VRAM allocation is only on h100-5')
    require(os.stat('/proc/self/ns/net').st_ino != os.stat('/proc/1/ns/net').st_ino
            and os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino,
            'Private mount/network namespaces are required')
    require(args.allocation_evidence, 'Native bounded-unit evidence is required')
    evidence = pool.read(args.allocation_evidence)
    require(evidence['mode'] == 'NATIVE_SHARED_VRAM' and evidence['node'] == 'h100-5'
            and int(evidence['main_pid']) == os.getpid(), 'Native evidence must bind this actual controller PID/host')
    properties = dict(re.findall(r'^(\w+)=(.*)$', evidence['unit_show'], re.MULTILINE))
    group = evidence['cgroup']
    require(properties.get('MainPID') == str(os.getpid()) and properties.get('ActiveState') == 'active'
            and properties.get('ControlGroup') == group and group.endswith('/' + evidence['unit']),
            'Native unit evidence does not bind its active MainPID/cgroup')
    require(any(line.split(':', 2)[-1] == group for line in Path('/proc/self/cgroup').read_text().splitlines()),
            'Controller is outside its own evidenced native cgroup')
    model, retrieval = indices(args.model_gpus, (2, 4)), indices(args.retrieval_gpus, (4, 5))
    all_indices = model + retrieval
    require(len(set(all_indices)) == len(all_indices), 'All model and retrieval GPUs must be distinct')
    inventory = {row['index']: row for row in pool.nvml().values()}
    require(all(index in inventory for index in all_indices), 'Requested physical GPU is unavailable')
    expected = [inventory[index]['uuid'] for index in all_indices]
    locks = []
    try:
        for uuid in sorted(expected):
            fd = pool.try_lock(Path(args.base) / 'pool/gpu-locks' / (uuid + '.lock'))
            require(fd is not None, 'Another cooperating own sampler holds GPU UUID ' + uuid)
            locks.append(fd)
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(map(str, all_indices)))
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), '--cuda-probe',
                    '--expected-uuids', ','.join(expected)], env=env, text=True, timeout=45)
        runtime = json.loads(output)['actual_cuda_runtime_uuids']
        require(runtime == expected, 'Full root-controller device mapping differs')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 28831))
        binding = dict(job_id='shared5-' + str(os.getpid()), node='h100-5',
            allocation_kind='HUMAN_AUTHORIZED_NATIVE_FREE_VRAM_CO_SHARE', gpu_exclusive_allocation=False,
            native_unit=evidence['unit'], native_cgroup=group, controller_pid=os.getpid(),
            model_indices=model, model_cuda_uuids=expected[:len(model)], retrieval_indices=retrieval,
            retrieval_cuda_uuids=expected[len(model):], actual_cuda_runtime_uuids=runtime,
            model_engine_count=len(model) // 2,
            model_cuda_visible_devices=','.join(map(str, model)),
            retrieval_cuda_visible_devices=','.join(map(str, retrieval)),
            original_cuda_visible_devices=evidence.get('original_cuda_visible_devices', os.environ.get('CUDA_VISIBLE_DEVICES')),
            initial_gpu_memory=[inventory[index] for index in all_indices], foreign_initial_contexts=foreign_contexts(set(expected)),
            foreign_contexts_allowed=True, foreign_processes_signalled=False,
            allocation_evidence=str(args.allocation_evidence), allocation_evidence_sha256=pool.sha(args.allocation_evidence),
            native_resource_limits=evidence['resource_limits'], captured_epoch=time.time())
        return binding, locks
    except BaseException:
        for fd in locks:
            os.close(fd)
        raise


def multi_sampler(pool, binding, args):
    numeric = binding['model_cuda_visible_devices']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == numeric, 'TP2 sampler CVD differs from controller')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # The root controller already validated every numeric device in one probe.
    # Each sampler receives only its own admitted TP2 pair and acquires no locks.
    require(len(binding['model_indices']) == len(binding['model_cuda_uuids']) == 2,
            'Each sampler must receive exactly its own admitted TP2 pair')
    from reproduction.operations.qwen7b_resample import worker
    s = worker.s
    ordinal = int(os.environ.get('Q7B_SHARED_RETRY_ORDINAL', '0'))
    launch = Path(args.launch_dir)

    def device(rank):
        if 'device_binding' in s._CACHED:
            return s._CACHED['device_binding']
        import torch
        require(rank == 0 and torch.cuda.device_count() == 2
                and os.environ['CUDA_VISIBLE_DEVICES'] == numeric, 'TP2 logical rank/device count differs')
        result = dict(physical_gpu_index=binding['model_indices'][0], gpu_uuid=binding['model_cuda_uuids'][0],
            physical_gpu_indices=binding['model_indices'], gpu_uuids=binding['model_cuda_uuids'],
            tensor_parallel_size=2, shared_vram=True, normalized_cuda_visible_devices=numeric,
            originally_assigned_device=binding['original_cuda_visible_devices'], native_unit=binding['native_unit'])
        s._CACHED['device_binding'] = result
        return result

    def fullprefill(config, model, seed, free, total):
        if 'llm' not in s._CACHED:
            import torch
            from vllm import LLM
            # The installed 0.6.3 EngineArgs explicitly accepts an executor
            # class; its MP worker factory imports this sibling in each rank.
            from shared_vram_vllm_adapter import SharedVramExecutor
            memory = pool.nvml()
            sample = [memory[uuid] for uuid in binding['model_cuda_uuids']]
            reserve = int(1.5 * GIB)
            utilization = min((row['free_bytes'] - reserve) / row['total_bytes'] for row in sample) * .85 ** ordinal
            require(0 < utilization < 1, 'Observed free VRAM cannot accommodate the requested 1.5GiB reserve')
            limits = dict(max_num_seqs=max(1, args.model_max_seqs // (2 ** ordinal)),
                          max_num_batched_tokens=max(s.BUDGET['total_tokens'], args.model_batch_tokens // (2 ** ordinal)))
            block_cap = max(256, args.model_kv_block_cap // (2 ** ordinal))
            os.environ['Q7B_SHARED_KV_BLOCK_CAP'] = str(block_cap)
            os.environ['Q7B_SHARED_KV_RESERVE_BYTES'] = str(reserve)
            os.environ['Q7B_SHARED_KV_PROFILE_DIR'] = str(launch / 'kv-profile')
            initialization = dict(tensor_parallel_size=2, actual_gpu_memory=sample,
                gpu_memory_utilization=utilization, reserve_bytes_per_gpu=reserve,
                scheduler_limits=limits, retry_ordinal=ordinal, max_model_len=s.BUDGET['total_tokens'],
                dtype=s.COMPUTE_DTYPE, eager=True, disable_custom_all_reduce=True,
                executor_class='shared_vram_vllm_adapter.SharedVramExecutor', kv_block_size=16,
                kv_block_cap=block_cap, real_profile_run=True, global_free_delta_used=False,
                first_new_sampling_request_started=False, epoch=time.time())
            pool.publish(launch / 'model-initialization.json', initialization)
            try:
                llm = LLM(model=model, tokenizer=model, trust_remote_code=False, dtype=s.COMPUTE_DTYPE,
                    tensor_parallel_size=2, max_model_len=s.BUDGET['total_tokens'], seed=seed,
                    gpu_memory_utilization=utilization, swap_space=0, enforce_eager=True,
                    enable_chunked_prefill=False, enable_prefix_caching=False, disable_custom_all_reduce=True,
                    distributed_executor_backend=SharedVramExecutor, block_size=16,
                    disable_async_output_proc=True, disable_log_stats=True, **limits)
            except torch.cuda.OutOfMemoryError as error:
                # Only this constructor site grants fresh-process retry. Runtime
                # token-generation/retrieval failures never produce this marker.
                pool.publish(launch / 'typed-model-initialization-oom.json', dict(initialization,
                    status='ACTUAL_TYPED_MODEL_INITIALIZATION_OOM', error_type=type(error).__name__,
                    error=str(error), actual_typed_cuda_oom=True, before_first_new_sampling_request=True))
                raise
            history = Path(args.job_dir) / 'typed-initialization-oom-history.json'
            attempts = pool.read(history) if history.exists() else []
            s._CACHED['llm'] = llm, utilization, limits, attempts
            pool.publish(launch / 'model-initialized.json', dict(initialization, actual_model_initialized=True,
                actual_compute_dtype=str(llm.llm_engine.model_config.dtype),
                actual_num_gpu_blocks=llm.llm_engine.cache_config.num_gpu_blocks,
                actual_kv_block_size=llm.llm_engine.cache_config.block_size,
                actual_rank_profile_proofs={str(path): pool.sha(path) for path in sorted((launch / 'kv-profile').glob('rank-*.json'))},
                epoch=time.time()))
        return s._CACHED['llm']

    s._configure_one_visible_gpu, s._build_llm = device, fullprefill
    return worker


def engine_binding(binding, index):
    start = 2 * index
    pair = binding['model_indices'][start:start + 2]
    uuids = binding['model_cuda_uuids'][start:start + 2]
    require(len(pair) == len(uuids) == 2, 'Missing admitted TP2 engine pair')
    multiple = len(binding['model_indices']) == 4
    return dict(binding, job_id=binding['job_id'] + ('-engine-%02d' % index if multiple else ''),
        parent_job_id=binding['job_id'], engine_index=index, model_engine_count=1,
        model_indices=pair, model_cuda_uuids=uuids, model_cuda_visible_devices=','.join(map(str, pair)),
        root_controller_actual_cuda_runtime_uuids=binding['actual_cuda_runtime_uuids'],
        actual_cuda_runtime_uuids=uuids, shared_retriever_owned_by_parent=True)


def engine_counts(pool, engine):
    directory = engine['jobdir']
    result = engine['result']
    progress = result if result is not None else pool.read(directory / 'progress.json') if (directory / 'progress.json').exists() else {}
    counts = {key: progress.get(key, 0) for key in ('completed_own_batches', 'questions', 'trajectories')}
    require(all(type(value) is int and value >= 0 for value in counts.values()), 'Invalid engine receipt counts')
    require(counts['trajectories'] == counts['questions'] * 5, 'Engine receipt G5 counts differ')
    return counts


def controller(pool, args):
    base = Path(args.base)
    config = pool.configuration(base)
    marker = Path(args.handoff_marker or str(base / 'preparation/pool-handoff.json'))
    handoff = pool.handoff(base, marker)
    binding, locks = admit(pool, args)
    attempt = 'attempt-%d-%d' % (time.time_ns(), os.getpid())
    jobdir = base / 'pool/jobs' / ('job-' + binding['job_id']) / attempt
    jobdir.mkdir(parents=True)
    pool.publish(jobdir / 'allocation.json', binding)
    pool.publish(jobdir / 'handoff.json', handoff)
    sources = {str(path): pool.sha(path) for path in
        (Path(__file__), Path(args.pool_worker), Path(args.retriever_script), Path(__file__).with_name('shared_vram_vllm_adapter.py'))}
    pool.publish(jobdir / 'sources.json', sources)
    evidence = pool.read(args.allocation_evidence)
    end = evidence['captured_epoch'] + evidence['resource_limits']['runtime_max_seconds']
    deadline = min(time.time() + args.run_seconds, end - 45)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', VLLM_HOST_IP='127.0.0.1',
        VLLM_ATTENTION_BACKEND='XFORMERS', VLLM_WORKER_MULTIPROC_METHOD='spawn',
        PYTHONDONTWRITEBYTECODE='1')
    children, handles, engines = [], [], []
    multiple = len(binding['model_indices']) == 4
    for index in range(len(binding['model_indices']) // 2):
        child_binding = engine_binding(binding, index)
        directory = base / 'pool/jobs' / ('job-' + child_binding['job_id']) / attempt if multiple else jobdir
        if multiple:
            directory.mkdir(parents=True)
            pool.publish(directory / 'allocation.json', child_binding)
            pool.publish(directory / 'handoff.json', handoff)
            pool.publish(directory / 'sources.json', sources)
        engines.append(dict(index=index, binding=child_binding, jobdir=directory,
            ordinal=0, history=[], process=None, launch=None, result=None, returncode=None))
    pool.publish(jobdir / 'engines.json', dict(engine_count=len(engines),
        engines=[dict(engine_index=x['index'], directory=str(x['jobdir']),
            model_indices=x['binding']['model_indices'], model_cuda_uuids=x['binding']['model_cuda_uuids']) for x in engines],
        shared_retriever_owned_by_parent=True, epoch=time.time()))

    def publish_children(last_reaped=None):
        pool.publish(jobdir / 'children.json', dict(own_process_groups=[p.pid for p in children],
            engine_processes={str(x['index']): x['process'].pid for x in engines if x['process'] is not None},
            last_reaped_sampler_pid=last_reaped, foreign_processes_signalled=False, epoch=time.time()))

    def spawn(name, command, cuda, directory, extra_env=None):
        child_env = dict(env, CUDA_VISIBLE_DEVICES=cuda, Q7B_POOL_INTERNAL=str(directory))
        child_env.update(extra_env or {})
        stream = (directory / (name + '.log')).open('ab')
        handles.append(stream)
        process = subprocess.Popen([sys.executable, '-u', *command], env=child_env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        publish_children()
        return process

    def launch_engine(engine):
        directory = engine['jobdir']
        launch = directory / 'sampler-launches' / ('launch-%02d' % engine['ordinal'])
        launch.mkdir(parents=True)
        engine['launch'] = launch
        engine['process'] = spawn('sampler', [str(Path(__file__).resolve()), '--sampler-loop', '--base', str(base),
            '--pool-worker', args.pool_worker, '--handoff-marker', str(marker), '--job-dir', str(directory),
            '--launch-dir', str(launch), '--deadline', str(deadline),
            '--model-max-seqs', str(args.model_max_seqs), '--model-batch-tokens', str(args.model_batch_tokens),
            '--model-kv-block-cap', str(args.model_kv_block_cap),
            '--batch-reserve-seconds', str(args.batch_reserve_seconds)],
            engine['binding']['model_cuda_visible_devices'], directory,
            dict(Q7B_SHARED_RETRY_ORDINAL=str(engine['ordinal'])))
        if multiple:
            pool.publish(directory / 'children.json', dict(own_process_groups=[engine['process'].pid],
                parent_controller_pid=os.getpid(), foreign_processes_signalled=False, epoch=time.time()))
        publish_children()

    last_progress = None

    def aggregate(status):
        nonlocal last_progress
        summaries = [dict(engine_index=x['index'], directory=str(x['jobdir']),
            model_indices=x['binding']['model_indices'],
            state='COMPLETE' if x['result'] is not None else 'SAMPLING' if x['process'] is not None else 'STARTING',
            **engine_counts(pool, x)) for x in engines]
        result = dict(status=status, run_id=pool.RUN_ID, job_id=binding['job_id'], engine_count=len(engines),
            completed_own_batches=sum(x['completed_own_batches'] for x in summaries),
            questions=sum(x['questions'] for x in summaries), trajectories=sum(x['trajectories'] for x in summaries),
            engines=summaries, aggregate_of_engine_receipts=True, global_goal_complete=False,
            foreign_processes_signalled=False, epoch=time.time())
        fingerprint = json.dumps(summaries, sort_keys=True)
        if multiple and fingerprint != last_progress:
            pool.publish(jobdir / 'progress.json', result)
            last_progress = fingerprint
        return result

    try:
        encoder = args.encoder or str(Path(config['base']).parent / 'search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2')
        retriever = spawn('retriever', [args.retriever_script, '--index', config['retrieval_index'],
            '--corpus', config['retrieval_corpus'], '--encoder', encoder, '--data-cache', config['retrieval_data_cache'],
            '--cache', str(jobdir / 'retrieval'), '--run-id', pool.RUN_ID, '--allocation-file', str(jobdir / 'allocation.json'),
            '--weights', args.retrieval_weights, '--logical-shards', str(args.retrieval_shards),
            '--port', '28831'], binding['retrieval_cuda_visible_devices'], jobdir)
        ready = None
        import urllib.request
        while ready is None:
            require(retriever.poll() is None, 'Own shared retriever initialization failed; see retriever.log')
            require(time.time() < deadline - args.batch_reserve_seconds, 'Retriever initialization exhausted batch time')
            try:
                with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=2) as response:
                    value = json.load(response)
                if value.get('ready'):
                    require(value['pid'] == retriever.pid and value['run_id'] == pool.RUN_ID
                        and value['gpu_shards'] == len(binding['retrieval_indices'])
                        and value['logical_flat_shards'] == args.retrieval_shards
                        and value['encoder_dtype'] == value['index_dtype'] == 'float16'
                        and value['actual_cuda_uuids'] == binding['retrieval_cuda_uuids']
                        and value['original_global_ids'] and value['endpoint'] == pool.URL,
                        'Own original full FP16 retriever binding differs')
                    ready = value
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
        pool.publish(jobdir / 'retriever-ready.json', ready)
        for engine in engines:
            if multiple:
                pool.publish(engine['jobdir'] / 'retriever-ready.json', dict(ready, shared_parent_directory=str(jobdir)))
            launch_engine(engine)
        while any(engine['result'] is None for engine in engines):
            require(retriever.poll() is None, 'Own shared retriever exited during actual sampling')
            require(time.time() < end - 10, 'Own native runtime deadline reached; preserve partial batch')
            for engine in engines:
                sampler = engine['process']
                if sampler is None or sampler.poll() is None:
                    continue
                pool.stop_group(sampler)
                children.remove(sampler)  # Never replay a reaped PGID after long-running work.
                engine['process'], engine['returncode'] = None, sampler.returncode
                if multiple:
                    pool.publish(engine['jobdir'] / 'children.json', dict(own_process_groups=[],
                        last_reaped_sampler_pid=sampler.pid, own_sampler_reaped=True,
                        foreign_processes_signalled=False, epoch=time.time()))
                publish_children(sampler.pid)
                if sampler.returncode == 0:
                    result = pool.read(engine['jobdir'] / 'sampler-exit.json')
                    require(result['run_id'] == pool.RUN_ID and result['job_id'] == engine['binding']['job_id'],
                            'Engine terminal receipt identity differs')
                    result.update(controller_status='OWN_SHARED_VRAM_ENGINE_FINISHED', tensor_parallel_size=2,
                        engine_index=engine['index'], shared_retriever_pid=retriever.pid,
                        shared_retriever_owned_by_parent=True, own_sampler_reaped=True,
                        sampler_returncode=0, global_goal_complete=False, foreign_processes_signalled=False,
                        epoch=time.time())
                    engine['result'] = result
                    if multiple:
                        pool.publish(engine['jobdir'] / 'complete.json', result)
                    continue
                oom = engine['launch'] / 'typed-model-initialization-oom.json'
                counts = engine_counts(pool, engine)
                require(engine['ordinal'] < 2 and oom.exists() and counts['completed_own_batches'] == 0,
                        'Engine %d actually failed with exit %d; no eligible local typed constructor OOM retry' %
                        (engine['index'], sampler.returncode))
                fact = pool.read(oom)
                require(fact['actual_typed_cuda_oom'] and fact['before_first_new_sampling_request'],
                        'Invalid engine constructor OOM marker')
                engine['history'].append(fact)
                pool.publish(engine['jobdir'] / 'typed-initialization-oom-history.json', engine['history'])
                engine['ordinal'] += 1
                launch_engine(engine)
            aggregate('OWN_SHARED_VRAM_ENGINES_ADVANCING')
            if any(engine['result'] is None for engine in engines):
                time.sleep(1)
        result = aggregate('OWN_SHARED_VRAM_ENGINES_FINISHED') if multiple else dict(engines[0]['result'])
        result.update(controller_status='OWN_SHARED_VRAM_JOB_FINISHED', tensor_parallel_size=2,
            model_engine_count=len(engines), retrieval_physical_gpus=len(binding['retrieval_indices']),
            global_goal_complete=False, foreign_processes_signalled=False)
    except BaseException as error:
        failure = dict(status='OWN_SHARED_VRAM_JOB_INTERRUPTED_OR_FAILED',
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            global_goal_complete=False, foreign_processes_signalled=False, epoch=time.time())
        pool.publish(jobdir / 'failure.json', failure)
        for engine in engines:
            if multiple and engine['result'] is None:
                pool.publish(engine['jobdir'] / 'failure.json', dict(failure,
                    status='OWN_SHARED_VRAM_ENGINE_INTERRUPTED_OR_FAILED', engine_index=engine['index'],
                    sampler_returncode=engine['returncode'], **engine_counts(pool, engine)))
        raise
    finally:
        for process in reversed(children):
            pool.stop_group(process)
        for stream in handles:
            stream.close()
        for fd in locks:
            os.close(fd)
    result.update(sampler_returncodes=[engine['returncode'] for engine in engines],
        retriever_returncode=retriever.returncode, own_children_reaped=True, epoch=time.time())
    if not multiple:
        result['sampler_returncode'] = engines[0]['returncode']
    pool.publish(jobdir / 'complete.json', result)
    print(json.dumps(dict(result, job_directory=str(jobdir))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base')
    parser.add_argument('--pool-worker', default=str(Path(__file__).with_name('pool_worker.py')))
    parser.add_argument('--retriever-script', default=str(Path(__file__).with_name('shared_vram_retriever.py')))
    parser.add_argument('--handoff-marker')
    parser.add_argument('--allocation-evidence', default=os.environ.get('Q7B_SHARED_VRAM_ALLOCATION_EVIDENCE'))
    parser.add_argument('--model-gpus', '--model-devices', default='3,4')
    parser.add_argument('--retrieval-gpus', '--retrieval-devices', default='0,2,5,6,7')
    parser.add_argument('--retrieval-weights', default='17,22,22,33,34')
    parser.add_argument('--retrieval-shards', type=int, default=128)
    parser.add_argument('--model-max-seqs', type=int, default=8)
    parser.add_argument('--model-batch-tokens', type=int, default=4096)
    parser.add_argument('--model-kv-block-cap', type=int, default=1024)
    parser.add_argument('--encoder')
    parser.add_argument('--run-seconds', type=int, default=28500)
    parser.add_argument('--batch-reserve-seconds', type=int, default=300)
    parser.add_argument('--sampler-loop', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job-dir', help=argparse.SUPPRESS)
    parser.add_argument('--launch-dir', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--cuda-probe', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--expected-uuids', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cuda_probe:
        print(json.dumps(dict(actual_cuda_runtime_uuids=actual_uuids(args.expected_uuids.split(',')))))
        return
    require(sys.platform == 'linux' and args.base, 'Shared-VRAM execution requires remote Linux and --base')
    require(args.run_seconds > 0 and args.batch_reserve_seconds > 0, 'Positive run duration/batch reserve required')
    require(args.model_max_seqs > 0 and args.model_batch_tokens >= 4096
            and args.model_kv_block_cap >= 256 and args.retrieval_shards in (64, 128), 'Invalid execution batch/cache/index limits')
    pool = load_pool(args.pool_worker)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    if args.sampler_loop:
        pool.single_gpu_sampler = lambda binding: multi_sampler(pool, binding, args)
        try:
            pool.sample_loop(args)
        except BaseException as error:
            pool.publish(Path(args.launch_dir) / 'failure.json', dict(status='OWN_SHARED_SAMPLER_PARTIAL_OR_FAILED',
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                global_goal_complete=False, epoch=time.time()))
            raise
    else:
        controller(pool, args)


if __name__ == '__main__':
    main()
