"""Once-only CPU audit for the terminal owned746125 recovery; no training."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from reproduction.easy2560_joint4.control import own_base, publish
from reproduction.operations.easy7b_20261021.preserve_failed746125 import preserve

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022'


def submit(base, operator_source):
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
    claim = prep/'source34-audit-submit-claim.json'
    result = prep/'source34-audit-submission.json'
    if claim.exists():
        if not result.is_file():
            raise ValueError('Existing audit claim has uncertain result; inspect Slurm before any retry')
        return json.loads(result.read_text())
    if any((base/name).exists() for name in ('job.id', 'logs/run-manifest.json')):
        raise ValueError('Recovery already has training identity; inspect existing audit')
    log = str(prep/'source34-audit-%j.log')
    argv = ['sbatch', '--parsable', '--output='+log, '--error='+log,
            str(script), str(base), str(operator_source)]
    publish(claim, dict(base=str(base), operator_source=str(operator_source),
        operator_commit=source['source_commit'], argv=argv, created_unix=time.time(),
        source_job='746125', source_job_terminal=failure['status'], training_submitted=False))
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
    args = parser.parse_args()
    print(json.dumps(submit(args.base, args.operator_source), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
