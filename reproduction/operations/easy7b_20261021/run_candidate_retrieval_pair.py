"""Sequential real native4 candidate A/B inside one diagnostic Slurm allocation.

Each item gets a new CPU proxy, cold completed-payload LRU and new candidate
driver/engines. Order: legacy one chunk, async one chunk, legacy five chunks,
async five chunks. This controller never starts/stops a training or GPU service
job; cleanup is confined to the proxy/benchmark children it creates itself.
With --startup-only, stop after the two512-parent startup runs and leave full=None.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).resolve().parent


def publish(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(root, stage, **values):
    record = dict(stage=stage, checked_unix=time.time(), **values)
    publish(root / 'progress.json', record)
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def health(url, timeout=5):
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(url, method='GET'), timeout=timeout) as response:
        return json.loads(response.read())


def health_observation(url):
    """Preserve diagnostic read failures without replacing a child exit code."""
    try:
        return dict(ok=True, checked_unix=time.time(), response=health(url))
    except Exception as error:
        return dict(ok=False, checked_unix=time.time(), error_type=type(error).__name__, error=str(error))


def service_urls(endpoint):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.path != '/retrieve':
        raise ValueError('Service ready document must name its /retrieve endpoint')
    return {mode: urlunsplit((parsed.scheme, parsed.netloc, route, '', ''))
        for mode, route in (('legacy', '/retrieve-original'), ('async', '/retrieve'), ('health', '/health'))}


def wait_service(args, root):
    # retrieval_service publishes this after asset load, before uvicorn binds.
    ready_path = Path(args.service_base) / 'logs/service-ready.json'
    deadline = time.monotonic() + args.service_ready_timeout
    last_error = None
    event(root, 'WAITING_PRIVATE_SERVICE_READY', service_ready=str(ready_path),
        timeout_seconds=args.service_ready_timeout)
    while time.monotonic() < deadline:
        try:
            ready = json.loads(ready_path.read_text())
            urls = service_urls(ready['endpoint'])
            observed = health(urls['health'])
            if ready.get('ready') is True and observed.get('ok') is True:
                publish(root / 'service-ready-observed.json', dict(ready_path=str(ready_path),
                    ready=ready, health=observed, checked_unix=time.time()))
                return ready_path, ready, urls
            last_error = 'Ready flag or actual HTTP health is not ready'
        except (OSError, ValueError, KeyError) as error:
            last_error = type(error).__name__ + ': ' + str(error)
        time.sleep(1)
    raise RuntimeError('Private service did not become ready within bounded wait: ' + str(last_error))


def wait_proxy(process, url, mode, endpoint, timeout=45):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError('Owned ' + mode + ' proxy exited during startup: ' + str(exit_code))
        try:
            observed = health(url, timeout=2)
            if observed.get('ok') is True and observed.get('mode') == mode and observed.get('endpoint') == endpoint:
                return observed
            last_error = 'Unexpected private proxy mode/endpoint'
        except (OSError, ValueError) as error:
            last_error = type(error).__name__ + ': ' + str(error)
        time.sleep(.5)
    raise RuntimeError('Owned proxy did not become ready: ' + str(last_error))


def stop_owned(process):
    """Use this exact Popen handle only; no names, ports, jobs or PID searches."""
    if process is None:
        return None
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)
    return process.returncode


def numeric_delta(before, after):
    left, right = before.get('response', {}), after.get('response', {})
    left, right = left.get('statistics', {}), right.get('statistics', {})
    return {key: right[key] - left[key] for key in left.keys() & right.keys()
        if type(left[key]) in (int, float) and type(right[key]) in (int, float)}


def one(args, root, ready_path, urls, *, mode, chunks, name, search_only_early_stop=False):
    directory = root / name
    directory.mkdir()
    port = 18011 if mode == 'legacy' else 18012
    proxy_url = 'http://127.0.0.1:' + str(port) + '/retrieve'
    proxy_health = 'http://127.0.0.1:' + str(port) + '/health'
    proxy_command = [args.python, '-B', str(HERE / 'candidate_proxy_benchmark.py'),
        '--endpoint', urls[mode], '--service-ready', str(ready_path),
        '--port', str(port), '--mode', mode]
    candidate_command = [args.python, '-B', str(HERE / 'benchmark_candidate_native4.py'),
        '--source-base', args.source_base, '--checkpoint', args.checkpoint,
        '--candidate-files', *args.candidate_files, '--retrieval-url', proxy_url,
        '--root', str(directory / 'candidate'), '--label', name, '--chunks', str(chunks),
        '--seed', str(args.seed), '--worker-cpus', str(args.worker_cpus),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--kv-tokens', str(args.kv_tokens)]
    if search_only_early_stop:
        candidate_command.append('--search-only-early-stop')
    proxy_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', PYTHONPATH=str(REPO))
    candidate_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(REPO))
    record = dict(name=name, mode=mode, chunks=chunks, search_only_early_stop=search_only_early_stop,
        proxy_command=proxy_command,
        candidate_command=candidate_command, started_unix=time.time(),
        service_health_before=health_observation(urls['health']),
        cold_proxy_cache=True, fresh_driver_and_engines=True, production_modified=False)
    publish(directory / 'launch.json', record)
    event(root, 'STARTING_COLD_PROXY', name=name, mode=mode, chunks=chunks, endpoint=urls[mode])
    proxy = candidate = None
    try:
        with (directory / 'proxy.log').open('x') as proxy_log:
            proxy = subprocess.Popen(proxy_command, env=proxy_environment, cwd=str(REPO),
                stdin=subprocess.DEVNULL, stdout=proxy_log, stderr=subprocess.STDOUT)
            record.update(proxy_pid=proxy.pid,
                proxy_health_before=wait_proxy(proxy, proxy_health, mode, urls[mode]))
            publish(directory / 'launch.json', record)
            event(root, 'REAL_CANDIDATE_CHILD_RUNNING', name=name, mode=mode, chunks=chunks,
                proxy_pid=proxy.pid, candidate_output=str(directory / 'candidate'))
            began = time.monotonic()
            with (directory / 'candidate.log').open('x') as candidate_log:
                candidate = subprocess.Popen(candidate_command, env=candidate_environment,
                    cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=candidate_log,
                    stderr=subprocess.STDOUT)
                record['candidate_pid'] = candidate.pid
                publish(directory / 'launch.json', record)
                record['candidate_exit_code'] = candidate.wait()
            record['candidate_process_wall_seconds'] = time.monotonic() - began
            record['proxy_health_after'] = health_observation(proxy_health)
            record['service_health_after'] = health_observation(urls['health'])
            record['service_numeric_statistics_delta'] = numeric_delta(
                record['service_health_before'], record['service_health_after'])
            record['service_delta_scope'] = ('Only counters exposed by service health; original-route '
                'fallback counters may not be included. Proxy health records actual backend requests.')
            result_path = directory / 'candidate/result.json'
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                record['candidate_result'] = str(result_path)
                record['candidate_result_summary'] = {key: result.get(key) for key in (
                    'status', 'completed_chunks', 'completed_questions', 'completed_trajectories',
                    'input_fingerprints_sha256', 'generation_phase_wall_seconds',
                    'pipeline_seconds_including_input_io_score_and_output_save',
                    'actual_raw_generated_tokens', 'actual_generation_requests', 'actual_prompt_tokens',
                    'actual_retrieval_calls', 'selection', 'actual_fa3_prefill_and_decode')}
            if record['candidate_exit_code']:
                raise RuntimeError(name + ' candidate child failed: ' + str(record['candidate_exit_code']))
            if not result_path.is_file():
                raise RuntimeError(name + ' exited without a real candidate result')
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if candidate is not None and candidate.poll() is None:
            record['owned_candidate_cleanup_exit_code'] = stop_owned(candidate)
        record['owned_proxy_cleanup_exit_code'] = stop_owned(proxy)
        record['finished_unix'] = time.time()
        publish(directory / 'execution.json', record)
    event(root, 'REAL_CANDIDATE_CHILD_COMPLETE', name=name, mode=mode, chunks=chunks,
        **record['candidate_result_summary'])
    return record


def comparison(legacy, asynchronous):
    a, b = legacy['candidate_result_summary'], asynchronous['candidate_result_summary']
    work_keys = ('actual_raw_generated_tokens', 'actual_generation_requests',
        'actual_prompt_tokens', 'actual_retrieval_calls')
    return dict(legacy_result=legacy['candidate_result'], async_result=asynchronous['candidate_result'],
        same_input_fingerprints=a['input_fingerprints_sha256'] == b['input_fingerprints_sha256'],
        legacy_generation_seconds=a['generation_phase_wall_seconds'],
        async_generation_seconds=b['generation_phase_wall_seconds'],
        observed_generation_wall_ratio=a['generation_phase_wall_seconds'] / b['generation_phase_wall_seconds'],
        legacy_pipeline_seconds=a['pipeline_seconds_including_input_io_score_and_output_save'],
        async_pipeline_seconds=b['pipeline_seconds_including_input_io_score_and_output_save'],
        legacy_actual_work={key: a[key] for key in work_keys},
        async_actual_work={key: b[key] for key in work_keys},
        actual_work_counts_equal=all(a[key] == b[key] for key in work_keys),
        legacy_selection=a['selection'], async_selection=b['selection'],
        legacy_proxy=legacy['proxy_health_after'], async_proxy=asynchronous['proxy_health_after'],
        causal_wall_speedup_proven=False,
        scope='Observed real candidate-pipeline A/B with fixed parents/policy and cold proxies. '
            'Asynchronous stochastic trajectories may perform different work; no full training-outer '
            'or final-quality equivalence is established by this timing.')


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    job = os.environ.get('SLURM_JOB_ID')
    schedule = [(1, 'legacy', 'startup-legacy'), (1, 'async', 'startup-async')]
    if not args.startup_only:
        schedule += [(5, 'legacy', 'full-legacy'), (5, 'async', 'full-async')]
    publish(root / 'launch.json', dict(arguments=vars(args), slurm_job_id=job,
        slurm_gpus_on_node=os.environ.get('SLURM_GPUS_ON_NODE'),
        slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
        started_unix=time.time(), production_modified=False,
        startup_only=args.startup_only, order=[name for _, _, name in schedule]))
    if not (job and job.isdecimal()):
        raise ValueError('Use this runner only inside its independent four-GPU Slurm allocation')
    ready_path, ready, urls = wait_service(args, root)
    completed = []
    for chunks, mode, name in schedule:
        completed.append(one(args, root, ready_path, urls, mode=mode, chunks=chunks, name=name))
        publish(root / 'completed.json', dict(completed=[row['name'] for row in completed],
            checked_unix=time.time()))
    result = dict(status=('STARTUP_PAIR_COMPLETED' if args.startup_only else
        'NATIVE4_REAL_CANDIDATE_RETRIEVAL_PAIR_COMPLETED'), startup_only=args.startup_only,
        slurm_job_id=job, checkpoint=args.checkpoint, candidate_files=args.candidate_files,
        service_ready=str(ready_path), service_endpoint=ready['endpoint'],
        retrieval_asset_versions={key: ready[key] for key in
            ('encoder_manifest_sha256', 'index_sha256', 'corpus_sha256')},
        completed_native_ray_workers=4, slurm_gpus_on_node=os.environ.get('SLURM_GPUS_ON_NODE'),
        optimizer_loaded=False, optimizer_writes=0,
        production_modified=False, startup=comparison(completed[0], completed[1]),
        full=(None if args.startup_only else comparison(completed[2], completed[3])),
        finished_unix=time.time())
    publish(root / 'result.json', result)
    measured = result['startup'] if args.startup_only else result['full']
    prefix = 'startup_' if args.startup_only else 'full_'
    event(root, result['status'], completed_children=len(completed), startup_only=args.startup_only,
        comparison_scope='startup_512_questions' if args.startup_only else 'full_2560_questions',
        **{prefix + 'observed_generation_wall_ratio': measured['observed_generation_wall_ratio'],
           prefix + 'actual_work_counts_equal': measured['actual_work_counts_equal']})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-base', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--candidate-files', nargs=5, required=True)
    p.add_argument('--service-base', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--startup-only', action='store_true',
        help='Only fresh legacy/async 512-parent startup runs; no full2560 comparison')
    p.add_argument('--service-ready-timeout', type=float, default=300)
    p.add_argument('--seed', type=int, default=317)
    p.add_argument('--worker-cpus', type=int, default=8)
    p.add_argument('--gpu-memory-utilization', type=float, default=.65)
    p.add_argument('--kv-tokens', type=int, default=131072)
    return p


def main():
    p = parser(); args = p.parse_args()
    if not 0 < args.service_ready_timeout <= 1800:
        p.error('Use a bounded service-ready wait between 0 and1800seconds')
    root = Path(args.root).resolve()
    if root.is_relative_to('/Users') or root.is_relative_to('/private'):
        p.error('Candidate raw data and experiment output must remain remote')
    try:
        run(args)
    except BaseException as error:
        if root.is_dir():
            publish(root / 'failure.json', dict(status='FAILED', error_type=type(error).__name__,
                error=str(error), traceback=traceback.format_exc(), training_modified=False))
        raise


if __name__ == '__main__':
    main()
