"""Advance existing unseen sampling to a single four-checkpoint score job.

No Slurm success dependency is attached to independent experiments. This
controller waits for actual sealed input data and four real checkpoints,
checks their recovery receipts, then submits one GPU scorer once.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def queue(job):
    result = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%i|%T|%M|%R'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode:
        assert 'Invalid job id specified' in result.stderr, result.stderr
        rows = [line.split('|') for line in accounting(job).splitlines() if line.strip()]
        terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
        assert rows and all(row[1].split()[0] in terminal for row in rows), rows
        return ''
    return result.stdout


def accounting(job):
    return subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job), '--format=JobID,State,ExitCode'], universal_newlines=True)


def score_entrypoint(base, code, runtime, script):
    # Unique tokens cannot replace CODE inside QWEN7B_CODE_ROOT.
    entry = '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-score4-unseen
#SBATCH --partition=acd_ue
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/logs/score-%j.log
set -euo pipefail
exec 9>@@BASE@@/control/scoring.lock
flock -n 9
export QWEN7B_CODE_ROOT=@@CODE@@
srun --kill-on-bad-exit=1 bash @@RUNTIME@@ -u @@SCRIPT@@ --mode score --base @@BASE@@ --micro-groups 32
'''
    for token, value in [('@@BASE@@', str(base)), ('@@CODE@@', shlex.quote(str(code))),
                         ('@@RUNTIME@@', shlex.quote(str(runtime))), ('@@SCRIPT@@', shlex.quote(str(script)))]:
        entry = entry.replace(token, value)
    assert '@@' not in entry and entry.count('export QWEN7B_CODE_ROOT=') == 1
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--selection-script', type=Path, required=True)
    parser.add_argument('--resume-after-failed-score', type=int)
    args = parser.parse_args()
    base, script = args.base, args.selection_script
    cfg = read(base / 'config.json')
    training = Path(cfg['training_base'])
    control = base / 'control'
    lock = (control / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume_after_failed_score is not None:
        failed = args.resume_after_failed_score
        request_path = training / 'control/human-epoch4-selection-request.json'
        assert sha(request_path) == cfg['human_request_sha256'] and read(request_path)['enabled']
        previous = read(control / 'score-submission.json')
        assert previous['job_id'] == failed and not queue(failed)
        assert sha(base / 'score.sbatch') == previous['entrypoint_sha256']
        assert 'not a valid identifier' in (base / 'logs' / ('score-%d.log' % failed)).read_text()
        rows = [line.split('|') for line in accounting(failed).splitlines() if line.strip()]
        assert [str(failed), 'FAILED', '1:0'] in [row[:3] for row in rows], rows
        assert not list((base / 'results').glob('epoch-*.json')), 'Explicit reconciliation required for existing scores'
        archive = control / ('failed-score-%d' % failed)
        archive.mkdir(exist_ok=False)
        for name in ('score-submission.json', 'score-submission-intent.json', 'score-scheduling-adjustment.json',
                     'coordinator-failure.json', 'coordinator-controller.json', 'coordinator-progress.json'):
            path = control / name
            if path.exists():
                path.rename(archive / name)
        (base / 'score.sbatch').rename(archive / 'score.sbatch')
        atomic(archive / 'recovery.json', dict(failed_job=failed, accounting=rows,
               no_prediction_outputs=True, reason='Invalid exported variable in generated shell entrypoint', time=time.time()))
    publication = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                       script_sha256=sha(__file__), selection_script=str(script), selection_sha256=sha(script), time=time.time())
    atomic(control / 'coordinator-controller.json', publication)
    request = training / 'control/human-epoch4-selection-request.json'

    def authorize():
        assert sha(request) == cfg['human_request_sha256'], 'Newer human instruction requires reconciliation'
        assert read(request)['enabled'] and read(request)['stop_after_epoch'] == 4
        assert sha(base / 'config.json') == read(control / 'sampling-submission.json')['config_sha256']

    sampling = read(control / 'sampling-submission.json')['job_id']
    code = training / 'repo-postprocess-96df5a23f23f'
    runtime = code / 'reproduction/operations/qwen7b_warmup/runtime.sh'
    environment = dict(os.environ, QWEN7B_CODE_ROOT=str(code))
    terminal_missing_since = None
    while not (base / 'bank/complete.json').exists():
        authorize()
        shards = []
        for shard in range(4):
            root = base / 'sampling' / ('shard-%02d' % shard)
            item = dict(shard=shard, complete=(root / 'complete.json').exists())
            for name in ('progress.json', 'failure.json'):
                if (root / name).exists():
                    item[name] = read(root / name)
            shards.append(item)
        live = queue(sampling)
        atomic(control / 'coordinator-progress.json', dict(stage='WAITING_FOR_AUTHENTIC_UNSEEN_G5_BANK',
               shards=shards, queue=live, fit=read(training / 'probe/progress.json'), time=time.time()))
        if all(item['complete'] for item in shards):
            subprocess.check_call(['bash', str(runtime), '-u', str(script), '--mode', 'merge', '--base', str(base)], env=environment)
            break
        if not live:
            terminal_missing_since = terminal_missing_since or time.monotonic()
            assert time.monotonic() - terminal_missing_since < 90, 'Sampling terminal without full bank: ' + accounting(sampling)
        else:
            terminal_missing_since = None
        time.sleep(30)
    # Ensure sampler cleanup has released its own allocations before scoring.
    while queue(sampling):
        authorize()
        time.sleep(15)
    sample_accounting = accounting(sampling)
    array_rows = [line.strip().split('|') for line in sample_accounting.splitlines() if line.strip().startswith(str(sampling) + '_')]
    assert {int(row[0].split('_')[-1]) for row in array_rows} == {0, 1, 2, 3}, sample_accounting
    assert all(row[1:3] == ['COMPLETED', '0:0'] for row in array_rows), sample_accounting
    terminal_missing_since = None
    while not (training / 'control/epoch4-stop-complete.json').exists():
        authorize()
        assert not (training / 'control/epoch4-stop-failure.json').exists(), 'Boundary stop controller needs repair'
        if not queue(740358):
            terminal_missing_since = terminal_missing_since or time.monotonic()
            assert time.monotonic() - terminal_missing_since < 120, 'Warmup terminal without requested stop receipt: ' + accounting(740358)
        atomic(control / 'coordinator-progress.json', dict(stage='BANK_READY_WAITING_FOR_COMPLETE_FOUR_EPOCH_STOP',
               bank=read(base / 'bank/complete.json'), fit=read(training / 'probe/progress.json'), time=time.time()))
        time.sleep(15)
    authorize()
    stop = read(training / 'control/epoch4-stop-complete.json')
    assert stop['request_sha256'] == cfg['human_request_sha256'] and stop['epoch4']['steps'] == 18292
    assert not queue(740358)
    auditors = training / 'auditors/fe0e65058ff3dba9deec831f9fc8dec18153caea'
    subprocess.check_call(['python3', str(auditors / 'audit_predictions.py'), '--base', str(training), '--epoch', '4'])
    subprocess.check_call(['bash', str(runtime), '-u', str(auditors / 'audit_checkpoint.py'), '--base', str(training),
                          '--receipt', str(training / 'probe/epoch-0004.json')], env=environment)
    entry = score_entrypoint(base, code, runtime, script)
    (base / 'score.sbatch').write_text(entry)
    subprocess.check_call(['bash', '-n', str(base / 'score.sbatch')])
    exported = next(line for line in entry.splitlines() if line.startswith('export QWEN7B_CODE_ROOT='))
    actual = subprocess.check_output(['bash', '-c', exported + '\nprintf "%s" "$QWEN7B_CODE_ROOT"'], universal_newlines=True)
    assert actual == str(code), 'Actual shell export differs from immutable code root'
    atomic(control / 'score-entrypoint-qualification.json', dict(entrypoint_sha256=sha(base / 'score.sbatch'),
           actual_code_root=actual, shell_syntax_passed=True, actual_export_passed=True, time=time.time()))
    submission = control / 'score-submission.json'
    if submission.exists():
        scored = read(submission)
    else:
        name = 'q7b-score4-unseen-' + cfg['run_id'][:8]
        intent = control / 'score-submission-intent.json'
        assert not intent.exists(), 'Unknown score submission must be reconciled; refuse duplicate'
        assert not subprocess.check_output(['squeue', '-h', '-u', 'mou176', '-n', name, '-o', '%i'], universal_newlines=True).strip()
        atomic(intent, dict(name=name, time=time.time(), script_sha256=sha(script)))
        result = subprocess.run(['sbatch', '--parsable', '--job-name=' + name, str(base / 'score.sbatch')],
                                cwd=str(base), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=20)
        assert result.returncode == 0, result.stderr
        scored = dict(job_id=int(result.stdout.strip().split(';')[0]), name=name, script_sha256=sha(script),
                      entrypoint_sha256=sha(base / 'score.sbatch'), config_sha256=sha(base / 'config.json'), time=time.time())
        atomic(submission, scored)
    job = scored['job_id']
    while True:
        authorize()
        live = queue(job)
        progress = read(base / 'results/progress.json') if (base / 'results/progress.json').exists() else None
        atomic(control / 'coordinator-progress.json', dict(stage='FOUR_CHECKPOINT_GPU_SCORING', queue=live,
               score_job=job, progress=progress, time=time.time()))
        if not live:
            terminal = accounting(job)
            for _ in range(12):
                rows = [line.strip().split('|')[:3] for line in terminal.splitlines()]
                if [str(job), 'COMPLETED', '0:0'] in rows:
                    break
                assert not any(row[0] == str(job) and row[1] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL') for row in rows), terminal
                time.sleep(5)
                terminal = accounting(job)
            else:
                raise RuntimeError('Score accounting outcome not yet confirmed: ' + terminal)
            break
        time.sleep(15)
    comparison = read(base / 'results/comparison.json')
    assert comparison['status'] == 'FOUR_EMA_UNSEEN2048_TOP512_TRUE_MIXED_COMPARISON_VERIFIED'
    atomic(control / 'pipeline-produced.json', dict(status='FOUR_EPOCH_AND_UNSEEN_SELECTION_OUTPUTS_READY_FOR_FINAL_GOAL_ACCEPTANCE',
           recommended_epoch=comparison['recommended_epoch'], sampling_accounting=sample_accounting,
           score_accounting=terminal, time=time.time(), original_goal_not_marked_complete_here=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import sys
        if '--base' in sys.argv:
            base = Path(sys.argv[sys.argv.index('--base') + 1])
            atomic(base / 'control/coordinator-failure.json', dict(error_type=type(error).__name__, error=str(error)[-2000:], time=time.time()))
        raise
