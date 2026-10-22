"""Once-only CPU audit for the terminal owned746125 recovery; no training."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from reproduction.easy2560_joint4.control import own_base, publish
from reproduction.operations.easy7b_20261021.preserve_failed746125 import preserve

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022'


def submit(base, operator_source, attempt=1):
    base = own_base(base)
    if str(base) != BASE:
        raise ValueError('Explicit owned CP34 recovery base required')
    operator_source = Path(operator_source).resolve()
    source = json.loads((operator_source/'source-publication.json').read_text())
    script = operator_source/'reproduction/operations/easy7b_20261021/source34_audit.sh'
    if not script.is_file() or source['fresh_source_all_files_sha_verified'] is not True:
        raise ValueError('Published operator source required')
    failure = preserve()
    prep = base/'preparation'; prep.mkdir(parents=True, exist_ok=True)
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError('Explicit supported submission attempt required')
    suffix = '' if attempt == 1 else '-r2'
    claim = prep/('source34-audit-submit-claim'+suffix+'.json')
    result = prep/('source34-audit-submission'+suffix+'.json')
    if claim.exists():
        if not result.is_file():
            raise ValueError('Existing audit claim has uncertain result; inspect Slurm before any retry')
        return json.loads(result.read_text())
    if attempt == 2:
        previous = json.loads((prep/'source34-audit-submission.json').read_text())
        if previous.get('returncode') == 0 or previous.get('job') is not None:
            raise ValueError('Prior audit was not an explicit failed submission; inspect Slurm')
        queue = subprocess.run(['squeue', '-u', 'jxu957', '-h', '-o', '%j'],
                               capture_output=True, text=True, timeout=30)
        if queue.returncode != 0 or 'easy7b-source34-cpu-audit' in queue.stdout.splitlines():
            raise ValueError('Unable to exclude existing source34 audit')
    if any((base/name).exists() for name in ('job.id', 'logs/run-manifest.json')):
        raise ValueError('Recovery already has training identity; inspect existing audit')
    log = str(prep/'source34-audit-%j.log')
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
            '--account=jxu957', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
            '--mem=128G', '--time=00:30:00', '--gres=gpu:1',
            '--job-name=easy7b-source34-cpu-audit', '--export=NONE',
            '--output='+log, '--error='+log,
            str(script), str(base), str(operator_source)]
    publish(claim, dict(base=str(base), operator_source=str(operator_source),
        operator_commit=source['source_commit'], argv=argv, created_unix=time.time(),
        source_job='746125', source_job_terminal=failure['status'], submission_attempt=attempt,
        prior_failed_submission_retained=attempt == 2, training_submitted=False))
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    text = completed.stdout.strip()
    job = text.split(';')[0] if completed.returncode == 0 else None
    if job is not None and not job.isdigit():
        job = None
    value = dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr,
        job=job, checked_unix=time.time(), base=str(base), source_step=34,
        actual_full_state_audit_complete=False, training_submitted=False,
        operator_commit=source['source_commit'])
    publish(result, value)
    if completed.returncode != 0 or job is None:
        raise ValueError('Audit submission failed or unknown; retained claim/result require inspection')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--operator-source', required=True)
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(json.dumps(submit(args.base, args.operator_source, args.attempt), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
