"""One Slurm GPU: private retriever plus frozen official G5 sampling on NAS10.

The Slurm launcher must provide private mount/network namespaces, working
loopback, the original canonical asset paths, and one CUDA_VISIBLE_DEVICES.
Only the shared handoff marker authorizes this pool to follow the unlocked r6.
This module never submits jobs, changes r6, deletes history, or declares the
global dataset audited/complete. All process groups it terminates are its own.
"""

import argparse
import csv
import datetime
import fcntl
import hashlib
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
import urllib.request


RUN_ID = '70188b8e50fd4b45bc420ce58014f46e'
SEED = 20261021
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
URL = 'http://127.0.0.1:28831/retrieve'
GIB = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def handoff(base, marker):
    value = read(marker)
    if value.get('status') == 'R6_NATIVE_PIPELINE_FAILED_NODE_DOWN':
        require(value['run_id'] == RUN_ID and value['seed'] == SEED,
                'Withdrawn-node recovery belongs to another run')
        receipt = Path(value['r6_terminal_receipt'])
        require(sha(receipt) == value['r6_terminal_receipt_sha256'],
                'Withdrawn-node terminal proof SHA differs')
        stopped = read(receipt)
        require(stopped['status'] == 'R6_NATIVE_WORKERS_TERMINAL_NODE_WITHDRAWN'
                and stopped['run_id'] == RUN_ID and stopped['seed'] == SEED
                and stopped['source_node'] == 'h100-11' and stopped['source_attempt'] == 'r6',
                'Source failure identity differs')
        require(stopped['actual_worker_returncodes'] == [-9] * 8
                and len(set(stopped['actual_worker_pids'])) == 8,
                'All original source workers must have actual terminal returncodes')
        failure = stopped['original_pipeline_failure']
        require(failure['error_type'] == 'RuntimeError' and failure['error'] ==
                'Sampling worker terminal returncodes: [-9, -9, -9, -9, -9, -9, -9, -9]',
                'Original source pipeline failure differs')
        observations = stopped['observations']
        require(len(observations) == 2 and stopped['observations_elapsed_seconds'] >= 60
                and observations[-1]['observed_epoch'] - observations[0]['observed_epoch'] >= 60
                and stopped['stable_terminal_files'] and stopped['stable_closed_batch_fingerprint'],
                'Source terminal records and sealed bank were not stable across two observations')
        require(all(x['source_node_down'] and x['source_node_jobs'] == []
                    and 'DOWN' in x['source_node_state'] for x in observations),
                'Source node withdrawal was not confirmed twice')
        require(observations[0]['closed_batch_fingerprint_sha256'] ==
                observations[1]['closed_batch_fingerprint_sha256']
                and observations[0]['terminal_files'] == observations[1]['terminal_files'],
                'Terminal source proof contains conflicting observations')
        require(stopped['source_native_unit_status'] == 'UNAVAILABLE_SOURCE_HOST_WITHDRAWN'
                and stopped['source_gpu_cleanup_status'] == 'UNAVAILABLE_SOURCE_HOST_WITHDRAWN'
                and stopped['r6_cgroup_status'] == 'UNAVAILABLE_SOURCE_HOST_WITHDRAWN'
                and stopped['unsealed_partials_retained'] and not stopped['global_goal_complete'],
                'Withdrawn source must retain the explicit unknown native cleanup status')
        return dict(marker=str(marker), marker_sha256=sha(marker),
                    r6_terminal_receipt=str(receipt),
                    r6_terminal_receipt_sha256=value['r6_terminal_receipt_sha256'],
                    source_native_cleanup_confirmed=False)
    require(value['status'] == 'R6_STOP_CONFIRMED_POOL_HANDOFF'
            and value['run_id'] == RUN_ID and value['seed'] == SEED,
            'Missing explicit same-run r6-to-pool handoff')
    receipt = Path(value['r6_stop_receipt'])
    require(sha(receipt) == value['r6_stop_receipt_sha256'], 'r6 stop receipt SHA differs')
    stopped = read(receipt)
    require(stopped['status'] == 'USER_NODE_RETURN_SAMPLING_STOP_SEALED'
            and stopped['stop_returncode'] == 0 and stopped['remaining_original_births'] == {},
            'r6 stop was not confirmed')
    groups = [x['r6_cgroup_remaining_pids'] for x in (value, stopped) if 'r6_cgroup_remaining_pids' in x]
    require(groups and all(x == [] for x in groups), 'r6 cgroup was not confirmed empty')
    service = dict(re.findall(r'^(\w+)=(.*)$', stopped['service'], re.MULTILINE))
    require(service.get('MainPID') == '0' and service.get('ActiveState') in ('inactive', 'failed'),
            'r6 service lacks terminal evidence')
    intent = read(receipt.parent / 'intent.json')
    require(intent['run_id'] == RUN_ID, 'r6 stop intent belongs to a different run')
    return dict(marker=str(marker), marker_sha256=sha(marker),
                r6_stop_receipt=str(receipt), r6_stop_receipt_sha256=value['r6_stop_receipt_sha256'])


def configuration(base):
    config = read(base / 'config.json')
    for key, expected in dict(run_id=RUN_ID, seed=SEED, model_revision=REVISION,
                              warmup_only=True, actor_hidden_native=False,
                              actor_optimizer_updates=0, train_questions=169028,
                              train_trajectories=845140, search_url=URL).items():
        require(config.get(key) == expected, 'Root config differs: ' + key)
    require(Path(config['base']).samefile(base), 'Canonical root BASE alias is unavailable')
    return config


def nvml():
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total', '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True, timeout=15)
    return {row[1].strip(): dict(index=int(row[0]), uuid=row[1].strip(),
            free_bytes=int(row[2]) * 1024 ** 2, total_bytes=int(row[3]) * 1024 ** 2)
            for row in csv.reader(io.StringIO(output))}


def applications(uuid):
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    return sorted(int(row[1]) for row in csv.reader(io.StringIO(output)) if row and row[0].strip() == uuid)


def try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        # POSIX process locks are shared over NFS and are not inherited by a
        # subsequently forked vLLM worker. A persistent engine cannot keep a
        # finished batch locked after the sampling parent closes this fd.
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def admit(base, evidence_path):
    job = os.environ.get('SLURM_JOB_ID', '')
    node = socket.gethostname().split('.')[0]
    require(job.isdigit() and node in ('h100-5', 'h100-10'), 'Requires this task\'s single-GPU Slurm allocation on h100-5 or h100-10')
    visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    require(len(visible) == 1, 'Exactly one Slurm CUDA_VISIBLE_DEVICES is required')
    require(os.stat('/proc/self/ns/net').st_ino != os.stat('/proc/1/ns/net').st_ino
            and os.stat('/proc/self/ns/mnt').st_ino != os.stat('/proc/1/ns/mnt').st_ino,
            'Private network and mount namespaces are required')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 28831))
    require(evidence_path, 'Allocation evidence must be captured before entering the private network namespace')
    evidence = read(evidence_path)
    require(str(evidence['job_id']) == job and evidence['node'] == node,
            'Outer Slurm allocation evidence belongs to a different job/node')
    detail = evidence['scontrol_job_detail']
    require(re.search(r'\bJobId=' + re.escape(job) + r'\b', detail), 'Outer allocation detail JobId differs')
    require(re.search(r'\bJobState=RUNNING\b', detail), 'Slurm allocation is not RUNNING')
    allocation = next((line for line in detail.splitlines() if re.search(r'\bNodes=' + re.escape(node) + r'\s', line)), '')
    allocated = re.search(r'\bGRES=gpu:(?:[A-Za-z0-9_-]+:)?1\(IDX:(\d+)\)', allocation)
    require(allocated is not None, 'Slurm must bind exactly one physical GPU index to this node')
    from reproduction.common8.native_allocation import cuda_uuids
    uuid = cuda_uuids(1)[0]
    first = nvml()[uuid]
    require(first['index'] == int(allocated.group(1)), 'Actual CUDA UUID differs from Slurm GPU allocation')
    if visible[0].startswith('GPU-'):
        require(visible[0] == uuid, 'CVD UUID differs from actual CUDA UUID')
    require(first['total_bytes'] >= 78 * GIB, 'This pool requires an 80GiB-class GPU')
    lock = try_lock(base / 'pool/gpu-locks' / (uuid + '.lock'))
    require(lock is not None, 'Another cooperating pool job owns this GPU UUID')
    try:
        samples = []
        for _ in range(2):
            actual = nvml()[uuid]
            foreign = [pid for pid in applications(uuid) if pid != os.getpid()]
            require(not foreign, 'Foreign compute application already uses assigned GPU: ' + repr(foreign))
            require(actual['free_bytes'] >= 54 * GIB, 'Assigned GPU has less than 54GiB genuinely free before initialization')
            samples.append(actual)
            if len(samples) == 1:
                time.sleep(2)
        end = re.search(r'\bEndTime=(\S+)', detail)
        require(end is not None, 'Slurm EndTime is unavailable')
        end_epoch = datetime.datetime.fromisoformat(end.group(1)).timestamp()
        return dict(job_id=job, node=node, gpu_uuid=uuid, physical_gpu_index=first['index'],
                    originally_assigned_device=visible[0], actual_cuda_runtime_uuids=[uuid],
                    before_initialization_samples=samples, slurm_end_epoch=end_epoch,
                    outer_allocation_evidence=str(evidence_path),
                    outer_allocation_evidence_sha256=sha(evidence_path),
                    outer_allocation_captured_epoch=evidence['captured_epoch']), lock
    except BaseException:
        os.close(lock)
        raise


def completed(receipt, inp, config):
    if not receipt.exists():
        return None
    value = read(receipt)
    require(value['run_id'] == config['run_id'] and value['input_sha256'] == sha(inp)
            and value['model_revision'] == REVISION and value['batch'] == inp.stem,
            'Existing complete batch input/run/revision binding differs: ' + str(receipt))
    return value


def single_gpu_sampler(binding):
    # vLLM 0.6.3 parses CVD tokens with int(). Do this before importing torch
    # (including through the frozen sampling modules), in this fresh process.
    visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
    require(len(visible) == 1, 'Sampler requires one admitted CUDA device before normalization')
    require(nvml()[binding['gpu_uuid']]['index'] == binding['physical_gpu_index'],
            'Admitted NVML physical index/UUID mapping changed')
    if visible[0].startswith('GPU-'):
        require(visible[0] == binding['gpu_uuid'], 'Sampler UUID CVD differs from admitted UUID')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(binding['physical_gpu_index'])
    from reproduction.common8.native_allocation import cuda_uuids
    require(cuda_uuids(1) == [binding['gpu_uuid']], 'Normalized numeric CVD maps to a different actual CUDA UUID')
    # Import the original science only inside the dedicated sampler process.
    from reproduction.operations.qwen7b_resample import worker
    s = worker.s

    def device(rank):
        require(rank == 0 and os.environ['CUDA_VISIBLE_DEVICES'] == str(binding['physical_gpu_index']),
                'TP1 sampler lost its single Slurm-visible device')
        from reproduction.common8.native_allocation import cuda_uuids
        import torch
        require(torch.cuda.device_count() == 1 and cuda_uuids(1) == [binding['gpu_uuid']],
                'Sampler runtime CUDA UUID differs from admitted Slurm device')
        return dict(physical_gpu_index=binding['physical_gpu_index'], gpu_uuid=binding['gpu_uuid'],
                    originally_assigned_device=binding['originally_assigned_device'],
                    normalized_cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    slurm_job_id=binding['job_id'])

    def fullprefill(config, model, seed, free, total):
        if 'llm' not in s._CACHED:
            import gc
            import torch
            from vllm import LLM
            allowed = {os.getpid(), int(os.environ['Q7B_POOL_CONTROLLER_PID']), int(os.environ['Q7B_POOL_RETRIEVER_PID'])}
            require(set(applications(binding['gpu_uuid'])) <= allowed,
                    'Foreign compute process appeared before generator initialization')
            utilization = min(float(config.get('rollout_utilization_cap', .90)), (free - 3 * GIB) / total)
            limits = dict(max_num_seqs=int(config.get('rollout_max_num_seqs', 512)),
                          max_num_batched_tokens=int(config.get('rollout_max_batched_tokens', 32768)))
            attempts = []
            while True:
                try:
                    llm = LLM(model=model, tokenizer=model, trust_remote_code=False, dtype=s.COMPUTE_DTYPE,
                        tensor_parallel_size=1, max_model_len=s.BUDGET['total_tokens'], seed=seed,
                        gpu_memory_utilization=utilization, swap_space=0, enforce_eager=True,
                        enable_chunked_prefill=False, enable_prefix_caching=False,
                        disable_async_output_proc=True, disable_log_stats=True, **limits)
                    break
                except torch.cuda.OutOfMemoryError:
                    if len(attempts) >= 2:
                        raise
                    attempts.append(dict(actual_typed_cuda_oom=True, utilization=utilization, limits=dict(limits)))
                    gc.collect()
                    torch.cuda.empty_cache()
                    utilization *= .85
                    limits['max_num_seqs'] = max(32, limits['max_num_seqs'] // 2)
                    limits['max_num_batched_tokens'] = max(4096, limits['max_num_batched_tokens'] // 2)
            s._CACHED['llm'] = llm, utilization, limits, attempts
        return s._CACHED['llm']

    s._configure_one_visible_gpu = device
    s._build_llm = fullprefill
    return worker


def batch_config(worker, base, inp, batch, config):
    canonical_dest = Path(config['base']) / 'sampling' / inp.parent.name / 'batches' / inp.stem
    canonical_dest.mkdir(parents=True, exist_ok=True)
    outer = canonical_dest / 'outers/outer-000001'
    outer.mkdir(parents=True, exist_ok=True)
    questions = outer / 'questions.json'
    wanted = dict(outer=1, rows=batch['rows'])
    if questions.exists():
        require(read(questions) == wanted, 'Retained sampling questions differ from input')
    else:
        worker.atomic(questions, wanted)
    path = canonical_dest / 'config.json'
    if path.exists():
        original = path.read_bytes()
        own = json.loads(original)
        for key, expected in dict(run_id=RUN_ID, seed=SEED, model_revision=REVISION,
                                  actor_version='official7b-' + REVISION, world_size=1,
                                  actor_model_dir=config['model_dir'], warmup_only=True,
                                  actor_hidden_native=False, actor_optimizer_updates=0,
                                  prefix_policy_tokens=20, search_url=URL,
                                  sampling_attempt_id=config.get('sampling_attempt_id')).items():
            require(own.get(key) == expected, 'Retained batch config differs: ' + key)
        require(Path(own['base']).samefile(canonical_dest), 'Retained batch config BASE differs')
        for phase in ('prefix', 'full'):
            receipt = outer / phase / 'rank-0-complete.json'
            if receipt.exists():
                require(read(receipt)['config_sha256'] == hashlib.sha256(original).hexdigest(),
                        'Retained phase receipt configSHA differs; config was preserved')
        require(path.read_bytes() == original, 'Existing batch config changed while claiming')
    else:
        require(not any(outer.glob('*/rank-0-complete.json')), 'Sealed phase has lost its original config')
        own = dict(config, base=str(canonical_dest), world_size=1, actor_model_dir=config['model_dir'],
                   actor_version='official7b-' + REVISION, actor_hidden_native=False,
                   prefix_policy_tokens=20, rollout_memory_utilization='auto', rollout_utilization_cap=.90,
                   rollout_free_reserve_gib=3, rollout_max_num_seqs=512, rollout_max_batched_tokens=32768,
                   rollout_enforce_eager=True, rollout_retry_initialization_oom=True)
        worker.atomic(path, own)
    return canonical_dest, path


def sample_loop(args):
    base, jobdir = Path(args.base), Path(args.job_dir)
    require(os.environ.get('Q7B_POOL_INTERNAL') == str(jobdir), 'Internal sampler requires its own controller')
    binding = read(jobdir / 'allocation.json')
    config = configuration(base)
    worker = single_gpu_sampler(binding)
    inputs = sorted((base / 'inputs').glob('shard-*/batch-*.json'))
    require(len(inputs) == 1322, 'Canonical 1322-input partition is incomplete')
    accepted = questions = trajectories = busy = seen_complete = 0
    longest = 0.
    status = 'POOL_INPUT_SCAN_FINISHED'
    for inp in inputs:
        handoff(base, Path(args.handoff_marker))
        reserve = max(args.batch_reserve_seconds, longest * 1.5 + 60)
        if args.deadline - time.time() < reserve:
            status = 'POOL_BATCH_BOUNDARY_TIME_LIMIT'
            break
        dest = base / 'sampling' / inp.parent.name / 'batches' / inp.stem
        receipt = dest / 'complete.json'
        if completed(receipt, inp, config) is not None:
            seen_complete += 1
            continue
        lock = try_lock(base / 'pool/batch-locks' / inp.parent.name / (inp.stem + '.lock'))
        if lock is None:
            busy += 1
            continue
        try:
            # r6 does not implement these locks; the shared handoff remains mandatory.
            handoff(base, Path(args.handoff_marker))
            if completed(receipt, inp, config) is not None:
                seen_complete += 1
                continue
            started = time.monotonic()
            batch = read(inp)
            require(batch['split'] == 'train' and batch['model_revision'] == REVISION
                    and batch['rows'] and all(row['split'] == 'train' for row in batch['rows']),
                    'Input batch differs from canonical train partition')
            digest = sha(inp)
            dest, config_path = batch_config(worker, base, inp, batch, config)
            prefix = worker.phase(config_path, 'prefix')
            full = worker.phase(config_path, 'continue')
            require(prefix['questions'] == full['questions'] == len(batch['rows'])
                    and prefix['trajectories'] == full['trajectories'] == 5 * len(batch['rows']),
                    'Actual phase counts differ from the whole G5 input batch')
            worker.atomic(dest / 'complete.json', dict(run_id=RUN_ID, input_sha256=digest,
                batch=inp.stem, questions=len(batch['rows']), trajectories=5 * len(batch['rows']),
                prefix=prefix, full=full, hidden_cached=False, model_revision=REVISION,
                actor_optimizer_updates=0, epoch=time.time()))
            completed(dest / 'complete.json', inp, config)
            elapsed = time.monotonic() - started
            longest = max(longest, elapsed)
            accepted += 1
            questions += len(batch['rows'])
            trajectories += 5 * len(batch['rows'])
            progress = dict(status='POOL_OWN_SEALED_BATCH_ADVANCED', run_id=RUN_ID,
                job_id=binding['job_id'], completed_own_batches=accepted, questions=questions,
                trajectories=trajectories, latest_input=str(inp), latest_batch_seconds=elapsed,
                longest_batch_seconds=longest, deadline_epoch=args.deadline,
                global_goal_complete=False, epoch=time.time())
            publish(jobdir / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        finally:
            os.close(lock)
    result = dict(status=status, run_id=RUN_ID, job_id=binding['job_id'],
        completed_own_batches=accepted, questions=questions, trajectories=trajectories,
        observed_existing_complete_batches=seen_complete, observed_locked_batches=busy,
        global_goal_complete=False, global_audit_performed=False, epoch=time.time())
    if not accepted and busy and status == 'POOL_INPUT_SCAN_FINISHED':
        result['status'] = 'POOL_REMAINING_BATCHES_LOCKED_BY_OTHER_JOBS'
    publish(jobdir / 'sampler-exit.json', result)
    print(json.dumps(result), flush=True)


def stop_group(process):
    # Every child was created with start_new_session=True, giving it its own PGID.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    # A child process can exit before its own worker descendants. The original
    # unique session/PGID remains ours until the last member exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def controller(args):
    base = Path(args.base)
    config = configuration(base)
    marker = Path(args.handoff_marker or str(base / 'preparation/pool-handoff.json'))
    proof = handoff(base, marker)
    args.handoff_marker = str(marker)
    binding, gpu_lock = admit(base, args.allocation_evidence)
    jobdir = base / 'pool/jobs' / ('job-' + binding['job_id']) / ('attempt-%d-%d' % (time.time_ns(), os.getpid()))
    jobdir.mkdir(parents=True)
    publish(jobdir / 'allocation.json', binding)
    publish(jobdir / 'handoff.json', proof)
    deadline = min(time.time() + args.run_seconds, binding['slurm_end_epoch'] - 45)
    children, handles = [], []
    env = dict(os.environ, VLLM_HOST_IP='127.0.0.1', Q7B_POOL_INTERNAL=str(jobdir),
               Q7B_POOL_CONTROLLER_PID=str(os.getpid()), PYTHONDONTWRITEBYTECODE='1',
               CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(binding['physical_gpu_index']))
    encoder = args.encoder or str(Path(config['base']).parent / 'search-g1-prefix5-h10011-runtime-assets-20261018/common8/assets/e5-base-v2')

    def spawn(name, arguments, file_direct=False):
        stream = (jobdir / (name + '.log')).open('ab')
        handles.append(stream)
        command = [sys.executable, '-u'] + ([] if file_direct else ['-m']) + arguments
        process = subprocess.Popen(command,
            env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        publish(jobdir / 'children.json', dict(pids=[p.pid for p in children],
                own_process_groups=[p.pid for p in children], epoch=time.time()))
        return process

    try:
        require(deadline - time.time() > args.batch_reserve_seconds, 'Insufficient remaining Slurm time to start a batch')
        retriever = spawn('retriever', ['reproduction.operations.qwen7b_warmup.retriever',
            '--index', config['retrieval_index'], '--corpus', config['retrieval_corpus'],
            '--encoder', encoder, '--data-cache', config['retrieval_data_cache'],
            '--cache', str(jobdir / 'retrieval'), '--run-id', RUN_ID, '--port', '28831', '--batch-queries', '512'])
        ready = None
        while ready is None:
            require(retriever.poll() is None, 'Own retriever exited during initialization; see job retriever.log')
            require(deadline - time.time() > args.batch_reserve_seconds, 'Retriever initialization consumed safe batch time')
            try:
                with urllib.request.urlopen('http://127.0.0.1:28831/health', timeout=2) as response:
                    value = json.load(response)
                if value.get('ready'):
                    require(value['run_id'] == RUN_ID and value['pid'] == retriever.pid
                            and value['endpoint'] == URL and value['encoder_dtype'] == 'float16'
                            and value['index_dtype'] == 'float16' and value['gpu_shards'] == 1,
                            'Ready service differs from own original FP16 retriever')
                    ready = value
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
        publish(jobdir / 'retriever-ready.json', ready)
        env['Q7B_POOL_RETRIEVER_PID'] = str(retriever.pid)
        sampler = spawn('sampler', [str(Path(__file__).resolve()), '--sample-loop', '--base', str(base),
            '--handoff-marker', str(marker), '--job-dir', str(jobdir),
            '--deadline', str(deadline), '--batch-reserve-seconds', str(args.batch_reserve_seconds)], file_direct=True)
        while sampler.poll() is None:
            require(retriever.poll() is None, 'Own retriever exited while sampler was active')
            require(time.time() < binding['slurm_end_epoch'] - 10, 'Slurm shutdown deadline reached; preserve partial data')
            time.sleep(1)
        require(sampler.returncode == 0, 'Own sampler exited %d; see sampler.log and failure.json' % sampler.returncode)
        result = read(jobdir / 'sampler-exit.json')
        result.update(controller_status='OWN_POOL_JOB_FINISHED', global_goal_complete=False)
    except BaseException as error:
        publish(jobdir / 'failure.json', dict(status='OWN_POOL_JOB_INTERRUPTED_OR_FAILED',
            run_id=RUN_ID, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), global_goal_complete=False, epoch=time.time()))
        raise
    finally:
        for process in reversed(children):
            stop_group(process)
        for stream in handles:
            stream.close()
        os.close(gpu_lock)
    result.update(retriever_stopped_by_owner=True, sampler_returncode=sampler.returncode,
                  retriever_returncode=retriever.returncode, epoch=time.time())
    publish(jobdir / 'complete.json', result)
    print(json.dumps(dict(result, job_directory=str(jobdir))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--handoff-marker', help='Default BASE/preparation/pool-handoff.json; mandatory confirmed r6 terminal evidence')
    parser.add_argument('--encoder', help='NAS10 canonical retained E5 directory; original r6 path is the default')
    parser.add_argument('--allocation-evidence', default=os.environ.get('Q7B_POOL_ALLOCATION_EVIDENCE'),
                        help='JSON captured outside private netns: job_id, node, scontrol_job_detail, captured_epoch')
    parser.add_argument('--run-seconds', type=int, default=1200, help='Own job lifetime, normally 900-1800 seconds; also bounded by actual Slurm EndTime')
    parser.add_argument('--batch-reserve-seconds', type=int, default=300, help='Minimum remaining time before claiming a batch; also uses 1.5x observed longest batch plus 60 seconds')
    parser.add_argument('--sample-loop', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--job-dir', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(sys.platform == 'linux', 'Pool jobs run only on the remote Linux Slurm nodes')
    require(args.run_seconds > 0 and args.batch_reserve_seconds > 0, 'Positive run duration and batch reserve required')

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    if args.sample_loop:
        try:
            sample_loop(args)
        except BaseException as error:
            if args.job_dir:
                publish(Path(args.job_dir) / 'sampler-failure.json', dict(status='OWN_SAMPLER_PARTIAL_OR_FAILED',
                    error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                    global_goal_complete=False, epoch=time.time()))
            raise
    else:
        controller(args)


if __name__ == '__main__':
    main()
