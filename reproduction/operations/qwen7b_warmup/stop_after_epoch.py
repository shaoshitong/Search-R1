"""Honor a newer human epoch boundary without modifying a running trainer.

Runs on the login host. A short CPU-only step in the existing allocation
signals only the four verified Python ranks, whose existing SIGUSR1 handler
saves Adam/EMA/RNG and returns normally. No new allocation or broad scancel.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


MODULE = 'reproduction.operations.qwen7b_warmup.train'


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-%d' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def authorized(base, request_sha):
    path = base / 'control/human-epoch4-selection-request.json'
    assert sha(path) == request_sha, 'Human request changed; do not signal'
    request = read(path)
    assert request['enabled'] and request['stop_after_epoch'] == 4
    assert request['job_id'] == 740358 and request['source'] == 'latest_direct_human_message'
    return request


def ranks(base, job_id):
    found = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            args = directory.joinpath('cmdline').read_bytes().split(b'\0')
            # The torchrun launcher also mentions MODULE; require its direct -m.
            if args[2:4] != [b'-m', MODULE.encode()]:
                continue
            if b'--base' not in args or args[args.index(b'--base') + 1] != str(base).encode():
                continue
            env = dict(item.split(b'=', 1) for item in directory.joinpath('environ').read_bytes().split(b'\0') if b'=' in item)
            if env.get(b'SLURM_JOB_ID') != str(job_id).encode() or env.get(b'WORLD_SIZE') != b'4':
                continue
            rank = int(env[b'RANK'])
            assert int(env[b'LOCAL_RANK']) == rank
            tick = directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
            found.append(dict(pid=int(directory.name), rank=rank, start_ticks=tick, uid=os.getuid()))
        except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
            continue
    assert len(found) == 4 and sorted(item['rank'] for item in found) == [0, 1, 2, 3], found
    return sorted(found, key=lambda item: item['rank'])


def node_action(base, request_sha, send):
    request = authorized(base, request_sha)
    assert os.environ.get('SLURM_JOB_ID') == str(request['job_id'])
    found = ranks(base, request['job_id'])
    if send:
        assert (base / 'control/epoch4-stop-checkpoint-verified.json').exists()
        verified = read(base / 'control/epoch4-stop-checkpoint-verified.json')
        receipt = read(base / 'probe/epoch-0004.json')
        assert verified['request_sha256'] == request_sha and verified['checkpoint_sha256'] == receipt['sha256']
        assert receipt['completed_epoch'] and receipt['epoch'] == 4 and receipt['steps'] == 18292
        atomic(base / 'control/epoch4-stop-signal-intent.json', dict(ranks=found, request_sha256=request_sha, time=time.time()))
        for item in found:
            authorized(base, request_sha)
            directory = Path('/proc') / str(item['pid'])
            assert directory.stat().st_uid == item['uid']
            assert directory.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] == item['start_ticks']
            os.kill(item['pid'], signal.SIGUSR1)
    print('OWNED_RANKS_JSON=' + json.dumps(dict(ranks=found, signalled=send, time=time.time())), flush=True)


def step(base, request_sha, send):
    command = ['srun', '--jobid=740358', '--overlap', '--exact', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=1', '--mem=512M', '--gres=gpu:0', '--kill-on-bad-exit=0',
               'python3', str(Path(__file__).resolve()), '--base', str(base), '--request-sha', request_sha,
               '--node-signal' if send else '--node-inspect']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.rsplit('OWNED_RANKS_JSON=', 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--request-sha', required=True)
    parser.add_argument('--node-inspect', action='store_true')
    parser.add_argument('--node-signal', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.node_inspect or args.node_signal:
        node_action(args.base, args.request_sha, args.node_signal)
        return
    base = args.base
    request = authorized(base, args.request_sha)
    if args.inspect:
        print(json.dumps(step(base, args.request_sha, False)))
        return
    control = base / 'control'
    lock = (control / 'epoch4-stop.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic(control / 'epoch4-stop-controller.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
           request_sha256=args.request_sha, job_id=request['job_id'], started=time.time(), script_sha256=sha(__file__)))
    last_queue_check, queue = 0., 'RUNNING'
    while True:
        authorized(base, args.request_sha)
        if time.monotonic() - last_queue_check >= 30:
            queue = subprocess.check_output(['squeue', '-h', '-j', '740358', '-o', '%T'], universal_newlines=True).strip()
            last_queue_check = time.monotonic()
            progress = read(base / 'probe/progress.json')
            atomic(control / 'epoch4-stop-progress.json', dict(status='WAITING_FOR_COMPLETE_EPOCH4', progress=progress, time=time.time()))
        receipt_path = base / 'probe/epoch-0004.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt['epoch'] == 4 and receipt['steps'] == 18292 and receipt['completed_epoch']
            assert receipt['actor_updates'] == 0 and receipt['independent_cpu_reload'] and receipt['all_rank_optimizer_moments_match']
            path = Path(receipt['path'])
            assert path.resolve() == (base / 'probe/epoch-0004.pt').resolve()
            assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256']
            assert receipt['config_sha256'] == sha(base / 'config.json') and receipt['bank_sha256'] == sha(base / 'bank/complete.json')
            atomic(control / 'epoch4-stop-checkpoint-verified.json', dict(request_sha256=args.request_sha, checkpoint_sha256=receipt['sha256'], time=time.time()))
            queue = subprocess.check_output(['squeue', '-h', '-j', '740358', '-o', '%T'], universal_newlines=True).strip()
            if queue == 'RUNNING':
                result = step(base, args.request_sha, True)
                atomic(control / 'epoch4-stop-signalled.json', dict(result, checkpoint=receipt, request_sha256=args.request_sha))
            else:
                assert not queue, queue
            break
        assert queue == 'RUNNING', 'Job terminal before fourth complete checkpoint: ' + queue
        time.sleep(2)
    # Observation never sends another signal; preserve an uncertain outcome.
    while subprocess.check_output(['squeue', '-h', '-j', '740358', '-o', '%T'], universal_newlines=True).strip():
        time.sleep(2)
    accounting = subprocess.check_output(['sacct', '-n', '-P', '-j', '740358', '--format=JobIDRaw,State,ExitCode'], universal_newlines=True)
    atomic(control / 'epoch4-stop-complete.json', dict(status='USER_REQUESTED_FOUR_EPOCH_BOUNDARY_STOP_OBSERVED',
           accounting=accounting, latest=read(base / 'probe/latest.json'), epoch4=read(base / 'probe/epoch-0004.json'),
           request_sha256=args.request_sha, time=time.time(), original_twenty_epoch_goal_superseded=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--base' in os.sys.argv:
            base = Path(os.sys.argv[os.sys.argv.index('--base') + 1])
            atomic(base / 'control/epoch4-stop-failure.json', dict(error_type=type(error).__name__, error=str(error)[:2000], time=time.time()))
        raise
