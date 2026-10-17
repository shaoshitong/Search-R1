"""Preserve only the terminal 730226 logs and CP5 JSON evidence, never tensors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018')
JOB = '730226'
NODE = 'ACD1-61'
RUN = 'dfe9c94084bb48b2a9bb7b60f9d882da'
COMMIT = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
ERROR = 'EvidenceError: Missing actual common8 monitor tensor hashes'
CP = Path('checkpoints/search-g1-common8/actor/global_step_5')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def document(path):
    return json.loads(Path(path).read_text())


def command(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=30)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def observe(base, execute=command):
    base = Path(base)
    if base != BASE:
        raise ValueError('Only the explicitly bound 730226 base is permitted')
    run = document(base / 'logs/run-manifest.json')
    if ((base / 'job.id').read_text().strip() != JOB or
            run['run_id'] != RUN or run['source_commit'] != COMMIT or
            run.get('training_world_size') != 4):
        raise ValueError('Failed-job source identity differs')
    accounting = execute(['sacct', '-j', JOB, '--format=JobIDRaw,State,ExitCode,NodeList,End',
                          '--noheader', '--parsable2'])
    queue = execute(['squeue', '-h', '-j', JOB, '--format=%i|%T|%N'])
    rows = {r[0]: r[1:] for r in (s.strip().split('|') for s in accounting['stdout'].splitlines()) if len(r) >= 5}
    expected = {JOB: ('FAILED', '1:0'), JOB + '.batch': ('FAILED', '1:0'),
                JOB + '.extern': ('COMPLETED', '0:0')}
    if (accounting['returncode'] or queue['returncode'] or queue['stdout'].strip() or
            any(tuple(rows.get(k, ())[:2]) != v or rows[k][2] != NODE for k, v in expected.items())):
        raise ValueError('730226 is not the confirmed terminal failed and released job')
    cleanup = document(base / 'logs/cleanup-status.json')
    if cleanup.get('cleaned') is not True or cleanup.get('run_id') != RUN:
        raise ValueError('Owned runtime cleanup has not completed')
    complete = document(base / CP / 'complete.json')
    state = document(base / CP / 'training_state/manifest.json')
    required = {'training_state/rank-%d.pt' % r for r in range(4)} | {'training_state/driver.pt'}
    if (complete.get('run_id') != RUN or complete.get('step') != 5 or
            complete.get('finite_all_saved_tensors') is not True or
            complete.get('actual_optimizer_writes_per_rank') != 100 or
            not required <= set(complete['files_sha256']) or
            any(not (base / CP / name).is_file() for name in required) or
            state.get('completed_outer') != 5 or state.get('question_probe') is not None):
        raise ValueError('The real native4 checkpoint5 evidence is incomplete')
    updates = [json.loads(r) for r in (base / 'logs/actual-updates.jsonl').read_bytes().split(b'\n')[:-1] if r]
    if ([r.get('step') for r in updates] != [1, 2, 3, 4, 5] or
            any(r.get('run_id') != RUN or r.get('actual_optimizer_writes') != 20 for r in updates)):
        raise ValueError('Expected exactly the five actually completed outers')
    trainer = base / 'logs/search-g1-common8-training.log'
    with trainer.open('rb') as f:
        f.seek(max(0, trainer.stat().st_size - 65536))
        if ERROR not in f.read().decode(errors='replace'):
            raise ValueError('The observed post-five monitor schema failure differs')
    # This is an additional observation, not a request to stop or allocate work.
    node_absence = execute(['scontrol', 'listpids', JOB, NODE])
    return dict(checked_unix=time.time(), accounting=accounting, queue=queue,
                node_step_observation=node_absence, cleanup=cleanup,
                checkpoint=str(base / CP), checkpoint_complete_sha256=sha(base / CP / 'complete.json'),
                completed_outer=5, actual_optimizer_writes_per_rank=100,
                actual_cuda_restore_accepted=False)


def preserve(base=BASE, execute=command):
    base = Path(base)
    if base != BASE:
        raise ValueError('Only the explicitly bound 730226 base is permitted')
    target = base / ('failed-job-' + JOB)
    receipt = target / 'failure-evidence.json'
    if target.exists():
        if not receipt.is_file():
            raise ValueError('Existing partial preservation must be inspected; no retry overwrite')
        value = document(receipt)
        if value.get('source_run_id') != RUN or value.get('source_commit') != COMMIT or value.get('source_job') != JOB:
            raise ValueError('Existing preservation identity differs')
        for item in value['files']:
            if sha(target / item['relative_path']) != item['sha256']:
                raise ValueError('Existing preserved copy SHA differs')
        return value
    observation = observe(base, execute)
    target.mkdir(mode=0o770)  # Atomic once-only claim; partial copies are never overwritten.
    publish(target / 'claim.json', dict(source_job=JOB, source_run_id=RUN, source_commit=COMMIT,
            process_pid=os.getpid(), checked_unix=time.time(), operator_source_sha256=sha(__file__)))
    publish(target / 'observation.json', observation)
    paths = [Path('logs') / name for name in (
        'run-manifest.json', 'native-allocation-admission.json', 'cleanup-status.json',
        'pipeline-status.json', 'actual-updates.jsonl', 'actual-attempts.jsonl',
        'resolved-training-config.json', 'training-730226.log', 'trainer.log',
        'search-g1-common8-training.log', 'policy-safety-calibration.json',
        'formal-inline-development-after.json')]
    paths += [Path('logs') / ('%s-rank-%d.jsonl' % (kind, rank))
              for kind in ('optimizer', 'scheduler') for rank in range(4)]
    paths += [Path('preparation/run-source.json'), CP / 'complete.json', CP / 'training_state/manifest.json']
    records = []
    for rel in paths:
        src = base / rel
        if not src.is_file():
            continue
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = src.stat()
        with src.open('rb') as source, dst.open('xb') as copied:
            shutil.copyfileobj(source, copied, 1024 * 1024)
            copied.flush()
            os.fsync(copied.fileno())
        after = src.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError('Failed source evidence changed while copying')
        digest = sha(src)
        if sha(dst) != digest:
            raise ValueError('Preserved copy SHA differs')
        records.append(dict(relative_path=str(rel), source=str(src), copy=str(dst),
                            bytes=after.st_size, sha256=digest))
    value = dict(status='FAILED730226_IMMUTABLE_EVIDENCE_PRESERVED', source_job=JOB,
        source_base=str(base), source_run_id=RUN, source_commit=COMMIT,
        failure_kind='post_outer5_monitor_identity_schema_rejection', completed_outer=5,
        last_retained_write_count_per_rank=100, checkpoint_available=True,
        checkpoint=str(base / CP), actual_cuda_restore_accepted=False,
        callback_status='fixed512 produced; first-five acceptance rejected; outer6 never started',
        files=records, observation=observation, checked_unix=time.time(),
        source_runtime_mutations_performed=False, tensors_copied=False, jobs_submitted=False)
    publish(receipt, value)
    fd = os.open(target, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    value = preserve(args.base)
    print(json.dumps(dict(status=value['status'], root=str(args.base / ('failed-job-' + JOB)),
          files=len(value['files']), bytes=sum(x['bytes'] for x in value['files']),
          receipt_sha256=sha(args.base / ('failed-job-' + JOB) / 'failure-evidence.json'),
          checkpoint_available=True, actual_cuda_restore_accepted=False)))


if __name__ == '__main__':
    main()
