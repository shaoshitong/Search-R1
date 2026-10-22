"""Small once-only launcher; existing assets are reused without new downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021'
PREDECESSOR = '/data/user/jxu957/experiments/qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017'
from .model7b import MODEL
DATA_NAMES = ('manifest.json', 'train.parquet', 'validation.parquet', 'stage.parquet',
              'final.parquet', 'exclusions-audit.json')
MICROS = (32, 16, 8, 4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial-' + str(time.time_ns()))
    try:
        with temporary.open('x') as out:
            json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
            out.write('\n'); out.flush(); os.fsync(out.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if temporary.exists(): temporary.unlink()


def own_base(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or not re.fullmatch(
            r'/data/user/jxu957/experiments/easy2560-corollout-joint4-[A-Za-z0-9._-]+', str(path)):
        raise ValueError('EASY2560_DEDICATED_REMOTE_BASE_REQUIRED')
    return path


def link(target, source):
    source = Path(source).resolve(); target = Path(target)
    if not source.exists(): raise FileNotFoundError(str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source: raise ValueError('EASY2560_EXISTING_ASSET_LINK_DIFFERS')
    elif target.exists():
        raise ValueError('EASY2560_EXISTING_ASSET_IS_NOT_LINK')
    else: target.symlink_to(source)


def prepare_environment(base, predecessor=PREDECESSOR):
    """Reuse the known Python/site/header/FAISS runtime, without refitting it."""
    base = own_base(base); old = Path(predecessor)
    if old == base or not (old / 'env/bin/python').is_file():
        raise ValueError('EASY2560_EXISTING_REMOTE_RUNTIME_REQUIRED')
    for name in ('logs', 'preparation', 'data', 'tmp', 'cache/hf', 'env/bin',
                 'env/lib/python3.10', 'artifacts', 'assets'):
        (base / name).mkdir(parents=True, exist_ok=True)
    for name in DATA_NAMES: link(base / 'data' / name, old / 'data' / name)
    for name in ('env/lib/python3.10/site-packages', 'assets/e5-base-v2',
                 'retriever-gpu-lib', 'bootstrap-tools/python-dev/usr/include/python3.10'):
        link(base / name, old / name)
    # The wrapper supplies the immutable CPU sklearn and FAISS overlays too.
    original = (old / 'env/bin/python').read_text()
    wrapper = original.replace(str(old / 'repo'), str(base / 'repo'))
    if str(old / 'repo') in wrapper:
        raise ValueError('EASY2560_RUNTIME_REPO_REBIND_FAILED')
    python = base / 'env/bin/python'
    if python.exists():
        if python.is_symlink() or python.read_text() != wrapper:
            raise ValueError('EASY2560_EXISTING_RUNTIME_WRAPPER_DIFFERS')
    else:
        with python.open('x') as f: f.write(wrapper); f.flush(); os.fsync(f.fileno())
        python.chmod(0o755)
    for name in ('probe-cpu-runtime.json',):
        source = old / 'preparation' / name
        target = base / 'preparation' / name
        if source.is_file() and not target.exists(): shutil.copy2(source, target)
    proof = dict(base=str(base), predecessor=str(old), data_read_only_links=True,
                 python_wrapper_sha256=sha(python), model=MODEL,
                 downloads=0, teacher_warmup_updates=0, credentials_persisted=False)
    receipt = base / 'preparation/environment-reuse.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != proof: raise ValueError('EASY2560_REUSED_ENVIRONMENT_CHANGED')
    else: publish(receipt, proof)
    return proof


def bind_repo(base, commit):
    base = own_base(base)
    if not re.fullmatch('[0-9a-f]{40}', commit): raise ValueError('EASY2560_FULL_COMMIT_REQUIRED')
    source = base / 'source' / commit
    proof = json.loads((source / 'source-publication.json').read_text())
    if proof['source_commit'] != commit or not proof['fresh_source_all_files_sha_verified']:
        raise ValueError('EASY2560_SOURCE_PUBLICATION_REQUIRED')
    repo = base / 'repo'; receipt = base / 'preparation/run-source.json'
    if repo.is_symlink(): raise ValueError('EASY2560_RUNTIME_REPO_MUST_BE_REAL_COPY')
    if not repo.exists():
        staging = base / ('repo.partial-' + commit); staging.mkdir()
        for name, record in proof['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('EASY2560_UNSAFE_SOURCE_MEMBER')
            origin = source / name
            if origin.is_symlink() or origin.stat().st_size != record['bytes'] or sha(origin) != record['sha256']:
                raise ValueError('EASY2560_DEPLOYED_SOURCE_CHANGED')
            target = staging / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        staging.rename(repo)
        publish(receipt, dict(source_commit=commit, files=proof['files'], runtime_repo=str(repo)))
    bound = json.loads(receipt.read_text())
    if bound['source_commit'] != commit or bound['files'] != proof['files']:
        raise ValueError('EASY2560_EXISTING_RUNTIME_BOUND_TO_OTHER_COMMIT')
    for name, record in proof['files'].items():
        path = repo / name
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('EASY2560_RUNTIME_SOURCE_DIFFERS_FROM_PUBLISHED_COMMIT')
    return bound


def launch_command(base, commit, actor_micro=8, restore_receipt=None, infra_profile='baseline',
                   external_retriever_receipt=None):
    from search_g1.easy7b_infra import profile_flags
    profile_flags(infra_profile)
    base = own_base(base)
    if actor_micro not in MICROS: raise ValueError('EASY2560_UNSUPPORTED_PHYSICAL_MICRO')
    command = ['sbatch', '--parsable', '--partition=emergency_acd', '--qos=formal-user',
            '--account=jxu957', '--nodes=1', '--ntasks=1', '--gres=gpu:4',
            '--cpus-per-task=48', '--mem=512G', '--time=7-00:00:00',
            '--job-name=easy2560-7b-corollout200', '--export=NONE',
            '--chdir=' + str(base / 'repo'),
            '--output=' + str(base / 'logs/training-%j.log'),
            '--error=' + str(base / 'logs/training-%j.log'),
            str(base / 'source' / commit / 'reproduction/easy2560_joint4/train.sh'),
            str(base), commit, str(actor_micro), str(restore_receipt) if restore_receipt else '']
    if infra_profile != 'baseline' or external_retriever_receipt is not None:
        command.append(infra_profile)
    if external_retriever_receipt is not None:
        command.append(str(external_retriever_receipt))
    return command


def restore_binding(receipt):
    """Pin the explicit small recovery receipt before the single submission."""
    if receipt is None:
        return dict(restore_receipt=None, restore_receipt_sha256=None)
    path = Path(receipt)
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
        raise ValueError('EASY2560_CANONICAL_RESTORE_RECEIPT_REQUIRED')
    return dict(restore_receipt=str(path), restore_receipt_sha256=sha(path))


def bootstrap(base, commit, predecessor=PREDECESSOR):
    environment = prepare_environment(base, predecessor)
    source = bind_repo(base, commit)
    return dict(environment=environment, source=source, training_submitted=False)


def submit(base, commit, actor_micro=8, predecessor=PREDECESSOR, restore_receipt=None, infra_profile='baseline',
           external_retriever_receipt=None):
    base = own_base(base); bootstrap(base, commit, predecessor)
    claim = base / 'preparation/training-submit-claim.json'
    receipt = base / 'preparation/training-submission.json'
    if any(p.exists() for p in (claim, receipt, base / 'job.id', base / 'logs/run-manifest.json')):
        raise ValueError('EASY2560_SUBMISSION_ALREADY_ATTEMPTED_INSPECT_ACTUAL_JOB')
    if infra_profile in ('fa3-exact-cache-v1', 'fa3-lean-v1'):
        from .flash3_runtime import prepare as prepare_flash3_runtime
        prepare_flash3_runtime(base)
    restoration = restore_binding(restore_receipt)
    from .retrieval_proxy import external_binding
    external = external_binding(external_retriever_receipt)
    command = launch_command(base, commit, actor_micro, restoration['restore_receipt'], infra_profile,
                             external_retriever_receipt)
    publish(claim, dict(source_commit=commit, command=command,
                        actor_micro_batch_per_rank=actor_micro, logical_micro_batch_per_rank=4,
                        infra_profile=infra_profile,
                        external_retriever=external,
                        **restoration,
                        personal_storage_budget_bytes=2 * 10**12, created_unix=time.time()))
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    value = dict(source_commit=commit, command=command, returncode=result.returncode,
                 stdout=result.stdout, stderr=result.stderr, checked_unix=time.time(),
                 actor_micro_batch_per_rank=actor_micro, training_world_size=4, **restoration)
    value['infra_profile'] = infra_profile
    value['external_retriever'] = external
    if result.returncode == 0:
        job = result.stdout.strip().split(';')[0]
        if not job.isdecimal(): raise ValueError('EASY2560_UNKNOWN_SUBMISSION_RESULT_INSPECT_CLAIM')
        value['job'] = job
        with (base / 'job.id').open('x') as f: f.write(job + '\n'); f.flush(); os.fsync(f.fileno())
    publish(receipt, value)
    if result.returncode: raise RuntimeError('EASY2560_SUBMISSION_FAILED_INSPECT_SAVED_RESULT')
    return value


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('bootstrap', 'submit'))
    p.add_argument('--base', default=BASE); p.add_argument('--commit', required=True)
    p.add_argument('--predecessor', default=PREDECESSOR)
    p.add_argument('--restore-receipt')
    p.add_argument('--external-retriever-receipt')
    from search_g1.easy7b_infra import PROFILES
    p.add_argument('--infra-profile', choices=PROFILES, default='baseline')
    p.add_argument('--actor-micro', type=int, choices=MICROS, default=8); a = p.parse_args()
    value = (bootstrap(a.base, a.commit, a.predecessor) if a.mode == 'bootstrap' else
             submit(a.base, a.commit, a.actor_micro, a.predecessor, a.restore_receipt, a.infra_profile,
                    a.external_retriever_receipt))
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
