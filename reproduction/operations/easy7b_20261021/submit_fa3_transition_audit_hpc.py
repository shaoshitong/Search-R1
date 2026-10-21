"""Allocate CPU RAM for the full-state source read, then submit normal training.

The CPU job waits for the explicitly authorized old producer to terminate. It
uses the already published transition entry; it does not cancel any job.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp12-20261022'
TRAINING_COMMIT = 'ebb6844db6a1ae739a06d5cc31d3510e7b4779b1'


def main():
    base = Path(BASE)
    sys.path.insert(0, str(base/'source'/TRAINING_COMMIT))
    from reproduction.easy2560_joint4.control import publish
    prepared = json.loads((base/'preparation/fa3-transition-prepared.json').read_text())
    if prepared['commit'] != TRAINING_COMMIT or prepared['status'] != 'READY_FOR_AUTHORIZED_BOUNDARY_TRANSITION':
        raise ValueError('Prepared published transition differs')
    claim = base/'preparation/fa3-transition-cpu-submit-claim.json'
    if claim.exists():
        print(json.dumps(dict(status='EXISTING_CPU_AUDIT_SUBMISSION_INSPECT_ACTUAL',
                              claim=json.loads(claim.read_text())))); return
    command = [str(base/'env/bin/python'), '-B', '-m',
               'reproduction.operations.easy7b_20261021.fa3_transition', 'submit',
               '--base', BASE, '--commit', TRAINING_COMMIT]
    argv = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
            '--account=jxu957', '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
            '--mem=128G', '--time=00:30:00', '--job-name=easy7b-fa3-state-read',
            '--dependency=afterany:745371', '--export=NONE', '--chdir='+str(base/'repo'),
            '--output='+str(base/'logs/fa3-transition-source-audit-%j.log'),
            '--error='+str(base/'logs/fa3-transition-source-audit-%j.log'),
            '--wrap='+shlex.join(command)]
    publish(claim, dict(command=argv, training_commit=TRAINING_COMMIT,
        source_job='745371', cpu_only=True, gpu_allocations=0, created_unix=time.time()))
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    value = dict(command=argv, returncode=result.returncode, stdout=result.stdout,
                 stderr=result.stderr, checked_unix=time.time())
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal(): raise ValueError('Inspect unknown sbatch outcome')
        value['job'] = job
    publish(base/'preparation/fa3-transition-cpu-submission.json', value)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
