"""Finite CPU-only synthetic Q48/G20 benchmark; never a learning experiment.

Each implementation runs in a fresh subprocess with the same synthetic input,
two pinned CPUs/threads, 3GiB data limit and independently watched 3GiB RSS.
Shared-library virtual mappings are reported rather than called resident RAM.
No model, checkpoint, feature, trajectory or evaluation data is opened.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

MEMORY = 3 * 1024 ** 3
MODULE = 'reproduction.operations.qwen7b_warmup.benchmark_exact_uniform_g5_dp'


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def vm(pid='self'):
    result = {}
    try:
        for row in Path('/proc/%s/status' % pid).read_text().splitlines():
            if row.startswith(('VmRSS:', 'VmPeak:', 'VmSize:', 'VmData:')):
                key, value, _ = row.split()
                result[key[:-1] + '_bytes'] = int(value) * 1024
    except FileNotFoundError:
        pass
    return result


def limits():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        raise ValueError('Two permitted CPUs required for the declared benchmark')
    os.sched_setaffinity(0, allowed[:2])
    resource.setrlimit(resource.RLIMIT_DATA, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 900))


def worker(base, mode, chunk):
    limits()
    import torch
    from .exact_uniform_g5_loss import exact_uniform_g5_loss
    from .exact_uniform_g5_dp_loss import exact_uniform_g5_dp_loss
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only benchmark must not initialize CUDA')
    q, g, seed = 48, 20, 9137
    generator = torch.Generator().manual_seed(seed)
    z = (torch.randn(q, g, generator=generator, dtype=torch.float32) * 2).requires_grad_(True)
    labels = torch.randint(0, 2, (q, g), generator=generator).tolist()
    groups = [dict(qid='synthetic-%03d' % i,
        trajectory_ids=['synthetic-%03d:t%02d' % (i, j) for j in range(g)], labels=labels[i]) for i in range(q)]
    name = '%s-chunk%d' % (mode, chunk)
    started, cpu_started = time.monotonic(), time.process_time()
    atomic(base / (name + '-progress.json'), dict(status='SYNTHETIC_LOSS_STARTED',
        epoch=time.time(), q=q, source_G=g, combination_chunk=chunk, pid=os.getpid(), **vm()))
    last = [0]
    def progress(row):
        if row['combinations_done'] - last[0] >= 2048 or row['combinations_done'] == 15504:
            last[0] = row['combinations_done']
            atomic(base / (name + '-progress.json'), dict(status='SYNTHETIC_DP_CHUNKS_ADVANCING',
                epoch=time.time(), elapsed_seconds=time.monotonic() - started, pid=os.getpid(), **row, **vm()))
    fn = exact_uniform_g5_loss if mode == 'old32' else exact_uniform_g5_dp_loss
    kwargs = dict(combination_chunk=chunk)
    if mode == 'dp':
        kwargs['progress_callback'] = progress
    result = fn(z, groups, **kwargs)
    gradient = torch.autograd.grad(result['loss'], z)[0]
    elapsed, cpu = time.monotonic() - started, time.process_time() - cpu_started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > MEMORY or not torch.isfinite(gradient).all():
        raise ValueError('CPU memory guard or finite-gradient qualification failed')
    answer = dict(status='SYNTHETIC_OBJECTIVE_AND_GRADIENT_COMPLETE', mode=mode,
        combination_chunk=chunk, questions=q, source_G=g, combinations_per_question=15504,
        virtual_views=q * 15504, seed=seed, dtype=str(z.dtype), CPU_affinity=sorted(os.sched_getaffinity(0)),
        torch_threads=torch.get_num_threads(), elapsed_seconds=elapsed, process_CPU_seconds=cpu,
        peak_RSS_bytes=peak, memory_cap_bytes=MEMORY,
        memory_guard='RLIMIT_DATA3GiB_and_parent_RSS_watchdog; shared-library_VMS_not_RSS',
        **vm(), loss=float(result['loss'].detach()),
        components={k: float(v) for k, v in result['components'].items()},
        gradient=gradient.tolist(), finite_gradient=True,
        gradient_sha256=hashlib.sha256(gradient.detach().contiguous().numpy().tobytes()).hexdigest(),
        maximum_resident_G5_rows=result['maximum_resident_G5_rows'],
        maximum_resident_pattern_rows=result['maximum_resident_pattern_rows'],
        retained_loss_gradient_elements=result['retained_loss_gradient_elements'],
        confusion_integer_all_views=result['confusion_integer_all_views'],
        actor_updates=0, GPU_requested=0, CUDA_initialized=torch.cuda.is_initialized(),
        synthetic_only_not_learning_evidence=True, epoch=time.time())
    atomic(base / (name + '.json'), answer)
    atomic(base / (name + '-progress.json'), {k: v for k, v in answer.items() if k != 'gradient'})


def run(base):
    base.mkdir(parents=True, exist_ok=True)
    if not base.is_absolute() or not str(base).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('Independent shared-storage benchmark output required')
    with (base / 'benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / 'complete.json').exists():
            return json.loads((base / 'complete.json').read_text())
        if (base / 'intent.json').exists():
            raise ValueError('Prior outcome must be inspected; no blind repeated benchmark')
        started = time.monotonic()
        atomic(base / 'intent.json', dict(pid=os.getpid(), epoch=time.time(), CPU_limit=2,
            memory_cap_bytes=MEMORY, wall_budget_seconds=1560, synthetic_only=True, GPU_requested=0))
        all_results = []
        for mode, chunk in (('old32', 64), ('dp', 64), ('dp', 256)):
            if time.monotonic() - started >= 1500:
                raise TimeoutError('Finite benchmark wall budget exhausted')
            name = '%s-chunk%d' % (mode, chunk)
            args = [sys.executable, '-u', '-m', MODULE, '--base', str(base), '--worker', mode,
                    '--combination-chunk', str(chunk)]
            with (base / (name + '.log')).open('ab') as log:
                child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                atomic(base / 'controller.json', dict(status='CPU_SYNTHETIC_CHILD_RUNNING',
                    child_pid=child.pid, mode=mode, combination_chunk=chunk, epoch=time.time(),
                    completed_variants=len(all_results), GPU_requested=0))
                child_started, observed_peak = time.monotonic(), 0
                while child.poll() is None:
                    observed_peak = max(observed_peak, vm(child.pid).get('VmRSS_bytes', 0))
                    if observed_peak > MEMORY or time.monotonic() - child_started > 360:
                        child.kill()
                        child.wait()
                        raise RuntimeError('Owned synthetic child exceeded RSS3GiB or360s bound')
                    time.sleep(.1)
            if child.returncode:
                raise RuntimeError(name + ' failed: ' + (base / (name + '.log')).read_text()[-2500:])
            result = json.loads((base / (name + '.json')).read_text())
            result['parent_observed_peak_RSS_bytes'] = observed_peak
            all_results.append(result)
            atomic(base / 'progress.json', dict(status='SYNTHETIC_VARIANT_COMPLETE',
                completed_variants=len(all_results), latest={k: v for k, v in result.items() if k != 'gradient'},
                epoch=time.time(), GPU_requested=0))
        old = all_results[0]
        for result in all_results[1:]:
            differences = [abs(a - b) for aa, bb in zip(old['gradient'], result['gradient']) for a, b in zip(aa, bb)]
            result['vs_old32'] = dict(loss_absolute_difference=abs(old['loss'] - result['loss']),
                maximum_gradient_absolute_difference=max(differences),
                time_speedup=old['elapsed_seconds'] / result['elapsed_seconds'],
                peak_RSS_ratio=result['peak_RSS_bytes'] / old['peak_RSS_bytes'])
            if (abs(old['loss'] - result['loss']) > 2e-5 or max(differences) > 2e-6
                    or old['confusion_integer_all_views'] != result['confusion_integer_all_views']):
                raise ValueError('Actual CPU synthetic old32/DP equivalence failed')
        result = dict(status='ACTUAL_CPU2_Q48_G20_ALL15504_DP_BENCHMARK_COMPLETE',
            variants=[{k: v for k, v in r.items() if k != 'gradient'} for r in all_results],
            elapsed_seconds=time.monotonic() - started, wall_budget_seconds=1560,
            synthetic_only_not_learning_evidence=True, model_tensors_opened=False,
            feature_tensors_opened=False, GPU_requested=0, actor_updates=0, epoch=time.time())
        atomic(base / 'complete.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--worker', choices=('old32', 'dp'))
    parser.add_argument('--combination-chunk', type=int, default=256)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.base, args.worker, args.combination_chunk)
        else:
            print(json.dumps(run(args.base)), flush=True)
    except BaseException as error:
        atomic(args.base / ('failure-%s.json' % (args.worker or 'controller')),
            dict(error_type=type(error).__name__, error=str(error), epoch=time.time()))
        raise
