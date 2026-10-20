"""Finite read-only single-job observer; no launch, rollback or source edits."""
import json
from pathlib import Path
import subprocess
import sys
import time

CONTROL = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
REPO = CONTROL / 'work/qwen7b-probe-warmup-20261019'
OUT = Path(__file__).parent / 'actual-training-observer'
OUT.mkdir(exist_ok=True)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    previous, started = None, time.time()
    for ordinal in range(170):
        try:
            done = subprocess.run([sys.executable, '-m',
                'reproduction.operations.qwen7b_warmup.deploy_exact_uniform_g20_dp_train',
                '--control-root', str(CONTROL), '--action', 'status'], cwd=REPO,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if done.returncode:
                raise RuntimeError(done.stdout[-1200:] + done.stderr[-1200:])
            raw = json.loads(done.stdout.splitlines()[-1])
            if raw['job_id'] != '743485':
                raise ValueError('Unexpected job identity; no mutation attempted')
            summary = {key: raw[key] for key in ('epoch', 'job_id', 'squeue', 'sacct',
                'controller', 'progress.json', 'next-update-replay.json', 'latest.json',
                'failure.json', 'interrupted.json', 'gpu-memory.json') if key in raw}
            if 'complete.json' in raw:
                complete = raw['complete.json']
                summary['complete.json'] = {key: complete[key] for key in
                    ('status', 'cursor', 'steps', 'metrics', 'checkpoint', 'actor_updates',
                     'optimizer_reset', 'config_sha256', 'plan_sha256')}
            atomic(OUT / 'latest-good.json', summary)
            with (OUT / 'observations.jsonl').open('a') as stream:
                stream.write(json.dumps(summary) + '\n')
            cursor = summary.get('progress.json', {}).get('cursor')
            if cursor != previous or ordinal % 5 == 0:
                print(json.dumps(summary), flush=True)
                previous = cursor
            if 'failure.json' in summary or 'interrupted.json' in summary:
                atomic(OUT / 'attention.json', dict(status='ACTUAL_SINGLE_G20_DP_ERROR_OR_INTERRUPTION_NO_AUTOMATIC_RETRY', actual=summary))
                return 2
            if 'complete.json' in summary:
                atomic(OUT / 'complete-observation.json', dict(status='ACTUAL_136_AND_NATIVE_VALIDATION_COMPLETE_PENDING_INDEPENDENT_METRIC_AUDIT', actual=summary))
                return 0
            if not summary.get('squeue') and summary.get('sacct'):
                atomic(OUT / 'attention.json', dict(status='OWNED_JOB_LEFT_QUEUE_WITHOUT_COMPLETE_NO_BLIND_RETRY', actual=summary))
                return 3
        except BaseException as error:
            failure = dict(epoch=time.time(), status='READ_ONLY_OBSERVATION_ERROR_LAST_GOOD_PRESERVED',
                error_type=type(error).__name__, error=str(error)[-2500:])
            atomic(OUT / 'last-read-error.json', failure)
            print(json.dumps(failure), flush=True)
        if ordinal < 169:
            time.sleep(45)
    atomic(OUT / 'finite-window-ended.json', dict(status='FINITE_OBSERVER_ENDED_NO_JOB_MUTATION',
        elapsed_seconds=time.time() - started))
    return 4


if __name__ == '__main__':
    raise SystemExit(main())
