"""Preserve the terminal pre-manifest 730912 bootstrap failure, without tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730226-20261018')
JOB = '730912'
NODE = 'ACD1-26'
COMMIT = 'd91575dbca221f66c467dccafa8508d9fb703688'
ERROR = 'EASY2560_RESTORE_SOURCE_FILE_IDENTITY_CHANGED'
FILES = ('job.id', 'preparation/training-submit-claim.json',
         'preparation/training-submission.json', 'preparation/run-source.json',
         'logs/training-730912.log')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def command(argv):
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=30)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def observe(base, execute=command):
    base = Path(base)
    if base != BASE: raise ValueError('Only the bound failed730912 base is permitted')
    claim = read(base / FILES[1]); submitted = read(base / FILES[2]); source = read(base / FILES[3])
    if ((base / 'job.id').read_text().strip() != JOB or submitted.get('job') != JOB or
            submitted.get('returncode') != 0 or
            any(value.get('source_commit') != COMMIT for value in (claim, submitted, source)) or
            claim.get('actor_micro_batch_per_rank') != 16 or
            not claim.get('restore_receipt') or
            submitted.get('restore_receipt_sha256') != claim.get('restore_receipt_sha256')):
        raise ValueError('Failed730912 submission/source binding differs')
    accounting = execute(['sacct', '-j', JOB, '--format=JobIDRaw,State,ExitCode,NodeList,Start,End',
                          '--noheader', '--parsable2'])
    rows = {r[0]: r[1:] for r in (line.strip().split('|') for line in accounting['stdout'].splitlines())
            if len(r) >= 6}
    expected = {JOB: ('FAILED', '1:0'), JOB + '.batch': ('FAILED', '1:0'),
                JOB + '.extern': ('COMPLETED', '0:0')}
    if accounting['returncode'] or any(tuple(rows.get(job, ())[:2]) != state or
            rows[job][2] != NODE or rows[job][4] in ('', 'Unknown', 'None')
            for job, state in expected.items()):
        raise ValueError('730912 top/batch must be failed and extern closed')
    logs = base / 'logs'
    if ((logs / 'run-manifest.json').exists() or (logs / 'native-allocation-admission.json').exists() or
            (logs / 'search-g1-common8-training.log').exists() or
            any(logs.glob('full-state-restored-*.json')) or
            any(logs.glob('optimizer-rank-*.jsonl')) or any(logs.glob('scheduler-rank-*.jsonl'))):
        raise ValueError('730912 is not the observed pre-manifest/no-CUDA bootstrap failure')
    trainer = base / FILES[4]
    if ERROR not in trainer.read_text():
        raise ValueError('The actual730912 bootstrap error differs')
    return dict(checked_unix=time.time(), accounting=accounting, source_job=JOB,
                source_commit=COMMIT, run_manifest_exists=False, cuda_training_evidence_present=False,
                failure_kind='cross_node_source_metadata_identity_rejected_before_run_manifest',
                actual_cuda_restore_accepted=False, actual_training_updates=0)


def preserve(base=BASE, execute=command):
    base = Path(base)
    if base != BASE: raise ValueError('Only the bound failed730912 base is permitted')
    target = base / ('failed-job-' + JOB); receipt = target / 'failure-evidence.json'
    if target.exists():
        if not receipt.is_file(): raise ValueError('Partial preservation exists; inspect without overwriting')
        value = read(receipt)
        if value.get('source_job') != JOB or value.get('source_commit') != COMMIT:
            raise ValueError('Existing preservation identity differs')
        for item in value['files']:
            if sha(target / item['relative_path']) != item['sha256']:
                raise ValueError('Existing preservation SHA differs')
        return value
    observation = observe(base, execute)
    target.mkdir(mode=0o770)
    publish(target / 'claim.json', dict(source_job=JOB, source_commit=COMMIT,
        pid=os.getpid(), operator_sha256=sha(__file__), created_unix=time.time()))
    publish(target / 'terminal-observation.json', observation)
    records = []
    for name in FILES:
        source = base / name; copied = target / name; before = source.stat()
        copied.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as original, copied.open('xb') as output:
            shutil.copyfileobj(original, output, 1024 * 1024); output.flush(); os.fsync(output.fileno())
        after = source.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError('Original failed730912 evidence changed during copy')
        digest = sha(source)
        if sha(copied) != digest: raise ValueError('Preserved copy SHA differs')
        records.append(dict(relative_path=name, bytes=after.st_size, sha256=digest))
    value = dict(status='FAILED730912_PREMANIFEST_EVIDENCE_PRESERVED', source_job=JOB,
        source_commit=COMMIT, source_base=str(base), files=records,
        terminal_observation=observation, checked_unix=time.time(), tensors_copied=False,
        env_or_repo_modified=False, original_evidence_modified=False, jobs_submitted=0,
        actual_cuda_restore_accepted=False)
    publish(receipt, value)
    fd = os.open(target, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args(); result = preserve(args.base)
    print(json.dumps(dict(status=result['status'], root=str(args.base / ('failed-job-' + JOB)),
        files=len(result['files']), bytes=sum(row['bytes'] for row in result['files']),
        receipt_sha256=sha(args.base / ('failed-job-' + JOB) / 'failure-evidence.json'),
        actual_cuda_restore_accepted=False, jobs_submitted=0)))


if __name__ == '__main__': main()
