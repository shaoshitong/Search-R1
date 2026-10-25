"""Remote-only preparation and independent 4+2 GPU submission.

The caller transports published source into ``BASE/source/COMMIT`` first.  No
weights, environments, wheels or rollout data pass through this command's host.
Submitting is explicit; preparing this directory never allocates a GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from .infra_profile import BASE, ENV_SOURCE, MODEL, WIKI, RETRIEVAL_PORT, own_base

DATA_NAMES = ('manifest.json', 'train.parquet', 'validation.parquet',
              'stage.parquet', 'final.parquet', 'exclusions-audit.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    """Atomic small receipt, separate from expensive model checkpoints."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def link(target, source):
    source = Path(source).resolve(strict=True); target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError('Existing asset link differs: ' + str(target))
    elif target.exists():
        raise ValueError('Refuse to replace an existing asset: ' + str(target))
    else:
        target.symlink_to(source)


def prepare_environment(base, predecessor=ENV_SOURCE):
    base = own_base(base); old = Path(predecessor)
    for name in ('logs', 'preparation', 'data', 'assets', 'env/bin',
                 'env/lib/python3.10', 'cache/hf', 'cache/triton',
                 'cache/torch-extensions', 'cache/inductor', 'tmp', 'artifacts',
                 'retrieval/logs', 'retrieval/cache', 'retrieval/tmp'):
        (base / name).mkdir(parents=True, exist_ok=True)
    for name in DATA_NAMES:
        link(base / 'data' / name, old / 'data' / name)
    for name in ('env/lib/python3.10/site-packages', 'assets/e5-base-v2',
                 'retriever-gpu-lib', 'bootstrap-tools/python-dev/usr/include/python3.10'):
        link(base / name, old / name)
    # Existing entry is a wrapper with sklearn/FAISS overlays. Rebind only its
    # repository; libraries and the underlying interpreter remain unchanged.
    wrapper = (old / 'env/bin/python').read_text().replace(str(old / 'repo'), str(base / 'repo'))
    target = base / 'env/bin/python'
    if target.exists():
        if target.is_symlink() or target.read_text() != wrapper:
            raise ValueError('Existing Python wrapper differs')
    else:
        target.write_text(wrapper); target.chmod(0o755)
    for name in ('probe-cpu-runtime.json',):
        if (old / 'preparation' / name).is_file() and not (base / 'preparation' / name).exists():
            shutil.copy2(old / 'preparation' / name, base / 'preparation' / name)
    receipt = dict(base=str(base), predecessor=str(old), model=MODEL,
        python_wrapper_sha256=sha(target), assets_read_only=True,
        downloads=0, packages_installed=0, shared_environment_modified=False)
    publish(base / 'preparation/environment-reuse.json', receipt)
    return receipt


def prepare_flash3(base):
    """Reuse compiled bytes by a remote hardlink, or copy on another filesystem."""
    from reproduction.easy2560_joint4.flash3_runtime import SOURCES, binding
    base = own_base(base); root = base / 'preparation/flash3-runtime'
    if (root / 'ready.json').exists():
        return binding(base)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, (source, expected, size) in SOURCES.items():
        source = Path(source)
        if source.stat().st_size != size or sha(source) != expected:
            raise ValueError('Compiled FA3 source SHA differs: ' + role)
        target = root / source.name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if target.stat().st_size != size or sha(target) != expected:
            raise ValueError('Compiled FA3 binding differs: ' + role)
        files[role] = dict(name=target.name, bytes=size, sha256=expected)
    publish(root / 'ready.json', dict(status='PINNED_PRIVATE_FA3_RUNTIME_SHA_PASS',
        files=files, shared_environment_modified=False, compilation=False))
    return binding(base)


def bind_source(base, commit):
    """One immutable runtime copy, preserving source of any existing run."""
    base = own_base(base)
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Require the full published source commit')
    source = base / 'source' / commit
    proof = json.loads((source / 'source-publication.json').read_text())
    if proof['source_commit'] != commit or not proof['fresh_source_all_files_sha_verified']:
        raise ValueError('Published source receipt is incomplete')
    repo = base / 'repo'
    receipt = base / 'preparation/run-source.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['source_commit'] != commit:
            raise ValueError('Existing run uses another source commit; use a new base')
        return previous
    repo.mkdir(parents=True, exist_ok=False)
    for name, record in proof['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source member')
        origin = source / name
        if origin.stat().st_size != record['bytes'] or sha(origin) != record['sha256']:
            raise ValueError('Published source differs: ' + name)
        target = repo / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
    value = dict(source_commit=commit, runtime_repo=str(repo), files=proof['files'])
    publish(receipt, value)
    return value


def prepare(base, commit, predecessor=ENV_SOURCE):
    env = prepare_environment(base, predecessor)
    source = bind_source(base, commit)
    fa3 = prepare_flash3(base)
    return dict(environment=env, source=source, flash3=fa3, training_submitted=False)


def launch_command(base, commit, role='train', *, partition='acd_u',
                   qos='formal-user', account='jxu957', time_limit='7-00:00:00',
                   restore_receipt=None, external_retriever_receipt=None, exclude_nodes=None):
    base = own_base(base)
    if role not in ('train', 'retrieval'):
        raise ValueError('Unknown allocation role')
    train = role == 'train'
    gpu, cpu, memory = (4, 48, '512G') if train else (2, 24, '384G')
    logs = base / 'logs' if train else base / 'retrieval/logs'
    entry = base / 'repo/reproduction/dynrpo7b' / ('train.sh' if train else 'retrieval.sh')
    command = ['sbatch', '--parsable', '--partition=' + partition, '--qos=' + qos,
        '--account=' + account, '--nodes=1', '--ntasks=1', '--gres=gpu:' + str(gpu),
        '--cpus-per-task=' + str(cpu), '--mem=' + memory, '--time=' + time_limit,
        '--job-name=dynrpo7b-' + role, '--export=NONE', '--chdir=' + str(base / 'repo'),
        '--output=' + str(logs / (role + '-%j.log')),
        '--error=' + str(logs / (role + '-%j.log')), str(entry), str(base), commit]
    if exclude_nodes:
        command.insert(2, '--exclude='+','.join(exclude_nodes))
    if train:
        command += [str(restore_receipt or ''), str(external_retriever_receipt or '')]
    # Deliberately no afterok dependency on another experiment.
    return command


def submit(base, commit, role='train', **kwargs):
    """Run only on HPC3; ambiguous submission outcomes retain their claim."""
    base = own_base(base)
    claim = base / 'preparation' / (role + '-submit-claim.json')
    receipt = base / 'preparation' / (role + '-submission.json')
    job_path = base / ('job.id' if role == 'train' else 'retrieval/job.id')
    if claim.exists() or job_path.exists():
        raise RuntimeError('Existing ' + role + ' submission: query its real job before any retry')
    if role == 'train' and not kwargs.get('external_retriever_receipt'):
        raise ValueError('Train requires this run\'s independent ready retrieval receipt')
    command = launch_command(base, commit, role, **kwargs)
    with claim.open('x') as f:
        json.dump(dict(command=command, source_commit=commit, created_unix=time.time()), f)
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    value = dict(source_commit=commit, role=role, command=command,
        returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        checked_unix=time.time())
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal():
            raise RuntimeError('Unknown submission result; inspect retained claim')
        value['job'] = job
        with job_path.open('x') as f:
            f.write(job + '\n')
    publish(receipt, value)
    if result.returncode:
        raise RuntimeError('Slurm rejected submission; see ' + str(receipt))
    return value


def publish_retriever(base):
    from reproduction.operations.easy7b_20261021.publish_retrieval_external_receipt import publish_receipt
    base = own_base(base)
    return publish_receipt(base / 'retrieval', base / 'preparation/external-retriever.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'retrieval', 'retriever-ready'))
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--commit')
    parser.add_argument('--predecessor', default=ENV_SOURCE)
    parser.add_argument('--partition', default='acd_u')
    parser.add_argument('--qos', default='formal-user')
    parser.add_argument('--account', default='jxu957')
    parser.add_argument('--time-limit', default='7-00:00:00')
    parser.add_argument('--restore-receipt')
    parser.add_argument('--external-retriever-receipt')
    args = parser.parse_args()
    if args.mode == 'retriever-ready':
        result = publish_retriever(args.base)
    elif not args.commit:
        parser.error('--commit is required')
    elif args.mode == 'prepare':
        result = prepare(args.base, args.commit, args.predecessor)
    else:
        result = submit(args.base, args.commit, args.mode, partition=args.partition,
            qos=args.qos, account=args.account, time_limit=args.time_limit,
            restore_receipt=args.restore_receipt,
            external_retriever_receipt=args.external_retriever_receipt)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
