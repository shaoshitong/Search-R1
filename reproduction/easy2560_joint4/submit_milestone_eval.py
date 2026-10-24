"""Once-only separate two-H100 Slurm submission; no training controls."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = args.root
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    claim, receipt = root / 'submit-claim.json', root / 'submit-receipt.json'
    if claim.exists() or (root / 'job.id').exists():
        print(json.dumps(dict(status='EXISTING_ATTEMPT_NO_REPLAY',
            receipt=json.loads(receipt.read_text()) if receipt.exists() else None)), flush=True)
        return
    source = root / 'source' / args.commit
    command = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
        '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:2', '--cpus-per-task=24',
        '--mem=256G', '--time=12:00:00', '--exclude=ACD1-13', '--export=NONE',
        '--job-name=easy7b-eval100-150-51713', '--chdir=' + str(source),
        '--output=' + str(root / 'logs/eval-%j.log'), '--error=' + str(root / 'logs/eval-%j.log'),
        '--wrap=exec ' + shlex.join(['bash', str(source / 'reproduction/easy2560_joint4/milestone_eval.sh'),
                                  str(root), args.commit])]
    with claim.open('x') as stream:
        json.dump(dict(command=command, commit=args.commit, steps=[100, 150],
            world_size=2, training_modified=False, created_unix=time.time()), stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    record = dict(command=command, commit=args.commit, returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr, training_modified=False,
        checked_unix=time.time(), automatic_retry=False)
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission outcome; inspect existing claim before retry')
        record['job'] = job
        with (root / 'job.id').open('x') as stream:
            stream.write(job + '\n'); stream.flush(); os.fsync(stream.fileno())
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
