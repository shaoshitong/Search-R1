"""Bounded, CPU-only observations of the existing native4 job.

Queue state is reported separately from actual completed-outer evidence.
This operator never submits, allocates, imports torch, or changes the run.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from reproduction.prefix20_joint4.control import own_base


def document(path, limit=4*2**20):
    path = Path(path)
    if not path.is_file():
        return None
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('PREFIX20_OBSERVATION_DOCUMENT_UNSAFE_OR_OVERSIZE')
    return json.loads(path.read_text())


def tail(path, limit=12000):
    path = Path(path)
    if not path.is_file():
        return None
    if path.is_symlink():
        raise ValueError('PREFIX20_OBSERVATION_LOG_SYMLINK')
    stat = path.stat()
    with path.open('rb') as stream:
        stream.seek(max(0, stat.st_size-limit))
        raw = stream.read(limit)
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                tail=raw.decode(errors='replace'))


def listing(path, limit=25):
    path = Path(path)
    if not path.is_dir():
        return []
    if path.is_symlink():
        raise ValueError('PREFIX20_OBSERVATION_DIRECTORY_SYMLINK')
    values = sorted(path.iterdir(), key=lambda item:item.stat().st_mtime_ns)[-limit:]
    return [dict(name=item.name, bytes=item.stat().st_size,
                 mtime_ns=item.stat().st_mtime_ns, directory=item.is_dir())
            for item in values]


def query(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=25)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def observe(base, training_commit):
    base = own_base(base)
    if not re.fullmatch('[0-9a-f]{40}', training_commit) or os.getuid() != 14090:
        raise ValueError('PREFIX20_OBSERVATION_JXU_AND_FULL_TRAINING_COMMIT_REQUIRED')
    job = (base/'job.id').read_text().strip()
    submitted = document(base/'preparation/training-submission.json')
    if (not job.isdecimal() or submitted is None or submitted.get('job') != job or
            submitted.get('source_commit') != training_commit or submitted.get('returncode') != 0):
        raise ValueError('PREFIX20_OBSERVATION_SUBMITTED_IDENTITY_DIFFERS')
    source = document(base/'preparation/run-source.json')
    if source is None or source.get('source_commit') != training_commit:
        raise ValueError('PREFIX20_OBSERVATION_RUNTIME_SOURCE_BINDING_DIFFERS')
    manifest = document(base/'logs/run-manifest.json')
    if manifest is not None and (manifest.get('source_commit') != training_commit or
            manifest.get('training_world_size') != 4 or manifest.get('data_selection_mode') != 'prefix20_probe'):
        raise ValueError('PREFIX20_OBSERVATION_RUN_IDENTITY_DIFFERS')
    docs = {name:document(base/'logs'/name) for name in (
        'prefix20-configuration-prepared.json', 'native-allocation-admission.json',
        'pipeline-status.json', 'pipeline-error.json', 'training-return.json', 'final-identity.json')}
    docs['run_manifest'] = None if manifest is None else {key:manifest.get(key) for key in (
        'run_id', 'source_commit', 'training_world_size', 'completed_updates', 'data_selection_mode')}
    directory = base/'artifacts/question-probe/completed'
    completed = sorted(directory.glob('outer-*.json')) if directory.is_dir() else []
    if any(not re.fullmatch('outer-[0-9]{4}\\.json', path.name) for path in completed):
        raise ValueError('PREFIX20_OBSERVATION_OUTER_RECEIPT_NAME_DIFFERS')
    latest = document(completed[-1]) if completed else None
    return dict(checked_unix=time.time(), hostname=socket.gethostname(), uid=os.getuid(),
        base=str(base), job=job, training_source_commit=training_commit,
        scontrol=query(['scontrol','show','job','-d',job]),
        queue=query(['squeue','-h','-j',job,'-o','%i|%T|%P|%u|%C|%m|%R|%b|%q']),
        accounting=query(['sacct','-nP','-j',job,'--format=JobID,State,ExitCode,Start,End,NodeList']),
        docs=docs, logs=listing(base/'logs'), probe=listing(base/'artifacts/question-probe'),
        completed_outer_receipts=len(completed), latest_outer=latest,
        training=tail(base/('logs/training-'+job+'.log')),
        actor=tail(base/'logs/search-g1-common8-training.log'),
        ledger=document(base/'logs/process-ledger.json'),
        scope='read-only actual observations; submission or RUNNING alone is not iteration acceptance')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--training-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(observe(args.base, args.training_commit)))


if __name__ == '__main__':
    main()
