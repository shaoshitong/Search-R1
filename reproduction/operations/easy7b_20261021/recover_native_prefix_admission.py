"""Continue an actually failed admission from its already audited full source.

This is a once-only operator in a new target directory. It does not change the
training runtime, repeat the 91 GB source audit, or retry an unknown sbatch.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.easy2560_joint4 import control


def recover(failed_base, target_base, commit, excluded_node='ACD1-13'):
    failed, target = control.own_base(failed_base), control.own_base(target_base)
    job = (failed/'job.id').read_text().strip()
    queue = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i|%u|%T'],
                           capture_output=True, text=True, timeout=15)
    accounting = subprocess.run(['sacct', '-n', '-X', '-j', job,
        '--format=JobIDRaw,User,State,ExitCode,NodeList', '-P'], capture_output=True, text=True, timeout=15)
    rows = [line.split('|') for line in accounting.stdout.splitlines() if line.startswith(job+'|')]
    if (os.getuid() != 14090 or queue.returncode or queue.stdout.strip() or accounting.returncode or
            len(rows) != 1 or rows[0][1:3] != ['jxu957', 'FAILED'] or rows[0][4] != excluded_node):
        raise ValueError('Actual own admission failure must be terminal on the excluded node')
    run = json.loads((failed/'logs/run-manifest.json').read_text())
    if run['source_commit'] != commit or run['full_state_restore']['completed_outer'] != 54:
        raise ValueError('Failed source differs from the qualified CP54 continuation')
    if any(path.stat().st_size for path in (failed/'logs').glob('optimizer-rank-*.jsonl')):
        raise ValueError('Actual optimizer evidence needs separate latest-boundary recovery')
    receipt = failed/'preparation/fa3-full-state-source-receipt.json'
    submitted = json.loads((failed/'preparation/fa3-transition-submitted.json').read_text())
    if control.sha(receipt) != submitted['restore_receipt_sha256']:
        raise ValueError('Previously completed CPU source receipt differs')
    if target.exists():
        raise ValueError('Recovery target already exists; inspect its claims instead of replaying')
    target.mkdir()
    preserved = target/'preparation/failed-job-'+job
    preserved.mkdir(parents=True)
    files = {}
    names = ['logs/run-manifest.json', 'logs/training-'+job+'.log', 'logs/native-gpu-inventory.xml',
        'preparation/training-submit-claim.json', 'preparation/training-submission.json',
        'preparation/fa3-transition-submitted.json', 'preparation/lean-transition-retrieval-exit.json']
    for name in names:
        source = failed/name
        if not source.is_file(): continue
        destination = preserved/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[name] = dict(bytes=source.stat().st_size, sha256=control.sha(source))
        if control.sha(destination) != files[name]['sha256']:
            raise ValueError('Failure evidence copy differs')
    control.publish(preserved/'preserved.json', dict(source_job=job, source_base=str(failed),
        source_queue=queue.stdout, source_accounting=accounting.stdout, files=files,
        optimizer_updates_observed=0, original_files_retained=True, checked_unix=time.time()))
    source = failed/'source'/commit
    proof = json.loads((source/'source-publication.json').read_text())
    destination = target/'source'/commit
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)
    for name, record in proof['files'].items():
        path = destination/name
        if path.stat().st_size != record['bytes'] or control.sha(path) != record['sha256']:
            raise ValueError('Immutable recovery source differs')
    # Only this submission-process function is wrapped. Live source and all
    # training forwards, numerical operations and runtime files are unchanged.
    original = control.launch_command
    def launch_command(*args, **kwargs):
        argv = original(*args, **kwargs)
        return argv[:1]+['--exclude='+excluded_node]+argv[1:]
    control.launch_command = launch_command
    try:
        result = control.submit(target, commit, actor_micro=8, predecessor=str(failed),
            restore_receipt=str(receipt), infra_profile='fa3-lean-v1',
            external_retriever_receipt=str(failed/'preparation/external-retriever-receipt.json'))
    finally:
        control.launch_command = original
    value = dict(status='FAILED_ADMISSION_NATIVE4_CONTINUATION_SUBMITTED', failed_job=job,
        job=result['job'], target_base=str(target), source_commit=commit,
        restored_completed_outer=54, restore_receipt=str(receipt),
        restore_receipt_sha256=control.sha(receipt), fresh_cpu_audit_repeated=False,
        excluded_node=excluded_node, submission=result, update201_allowed=False,
        actual_cuda_restore_accepted=False, checked_unix=time.time())
    control.publish(target/'preparation/admission-recovery-submitted.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-base', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--exclude-node', default='ACD1-13')
    args = parser.parse_args()
    print(json.dumps(recover(args.failed_base, args.base, args.commit, args.exclude_node)), flush=True)


if __name__ == '__main__':
    main()
