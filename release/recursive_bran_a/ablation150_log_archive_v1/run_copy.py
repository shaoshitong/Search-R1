"""One CPU archival child with durable owner/start/actual-wait provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PREFIX = 'release/recursive_bran_a/'
RUNTIME = tuple(PREFIX + package + '/' + name for package, names in (
    ('ablation150_log_archive_v1', ('__init__.py', 'archive.py', 'run_copy.py')),
    ('ablation150_archive_v1', ('__init__.py', 'archive.py')),
    ('ablation150_closed_storage_v2', ('__init__.py', 'execute.py')),
    ('ablation150_storage_v1', ('__init__.py', 'inspect.py')),
) for name in names)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(path, root):
    path, root = Path(path), Path(root)
    require(path.is_absolute() and path.is_relative_to(root) and '..' not in path.parts and
            all(not item.is_symlink() for item in (path, *path.parents)), 'unsafe deployment/control path')
    return path


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, uid=path.stat().st_uid, tick=int(fields[19]), state=fields[0])
    except FileNotFoundError:
        return dict(pid=pid, identity_unavailable=True)


def run(args):
    deployment = inside(args.deployment, args.deployment)
    source = inside(deployment/'source', deployment)
    require(Path(__file__).resolve() == source/PREFIX/'ablation150_log_archive_v1/run_copy.py',
            'wrapper is not this independent deployment source')
    record = json.loads((deployment/'deployment.json').read_bytes())
    require(record.get('commit') == args.implementation_commit and record.get('source') == str(source) and
            record.get('all_manifest_sha_equal') is True, 'deployment implementation/source not accepted')
    manifest = inside(args.runtime_manifest, deployment)
    require(sha(manifest) == args.runtime_manifest_sha256, 'runtime manifest SHA changed')
    values = json.loads(manifest.read_bytes())
    require(values.get('commit') == args.implementation_commit and
            set(RUNTIME).issubset(values.get('files', {})), 'runtime manifest misses exact dependency closure')
    runtime = {}
    for name in RUNTIME:
        path = inside(source/name, source)
        runtime[name] = sha(path)
        require(runtime[name] == values['files'][name], 'deployed runtime SHA differs: ' + name)
    control = inside(args.control, deployment)
    require(control.parent == deployment/'control', 'control must be a new immediate deployment/control child')
    control.parent.mkdir(exist_ok=True)
    require(not control.exists(), 'control/owner already exists; use a new independently authorized control directory')
    control.mkdir(exist_ok=False)
    # Import only after every actual runtime byte is bound to the published
    # deployment. This dependency is stdlib-only; no model/tensor imports.
    from .archive import closed
    binding = dict(protocol='ablation150_log_supplement_child_wait_v1', time=time.time(),
        owner=process(os.getpid()), parent_pid=os.getppid(), source_commit=args.source_commit,
        implementation_commit=args.implementation_commit, deployment=str(deployment),
        deployment_sha256=sha(deployment/'deployment.json'), runtime_files=runtime,
        runtime_manifest=str(manifest), runtime_manifest_sha256=args.runtime_manifest_sha256,
        canonical_complete=args.canonical_complete, canonical_complete_sha256=args.canonical_complete_sha256,
        storage_ready=args.storage_ready, storage_ready_sha256=args.storage_ready_sha256,
        tree_summary_root=args.tree_summary_root, training_action_executed=False)
    closed.durable_json(control/'owner.json', binding, exclusive=True)
    command = [sys.executable, '-B', '-u', '-m',
        'release.recursive_bran_a.ablation150_log_archive_v1.archive', 'copy']
    for name in ('source_commit', 'implementation_commit', 'canonical_complete',
                 'canonical_complete_sha256', 'storage_ready', 'storage_ready_sha256', 'tree_summary_root'):
        command.extend(['--'+name.replace('_', '-'), str(getattr(args, name))])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
    try:
        child = subprocess.Popen(command, cwd=source, env=env)
    except OSError as failure:
        closed.durable_json(control/'launch-error.json', dict(time=time.time(), error=repr(failure),
            actual_child_started=False, actual_child_wait_rc=None, training_action_executed=False), exclusive=True)
        raise
    started = dict(time=time.time(), child=process(child.pid), command=command,
        command_sha256=hashlib.sha256(json.dumps(command, separators=(',', ':')).encode()).hexdigest(),
        owner_pid=os.getpid(), actual_child_started=True, actual_child_wait_rc=None)
    error = None
    try:
        closed.durable_json(control/'child-start.json', started, exclusive=True)
        print(json.dumps(dict(event='SUPPLEMENT_CHILD_STARTED', **started), sort_keys=True), flush=True)
    except (OSError, ValueError) as failure:
        error = repr(failure)
    rc = child.wait()  # This exact child wait is the only source of its exit code.
    result = dict(time=time.time(), owner=binding['owner'], child=started['child'],
        command_sha256=started['command_sha256'], actual_child_wait_rc=rc,
        start_receipt_error=error, training_action_executed=False,
        status='CHILD_WAIT0_COPY_RETURNED' if rc == 0 and error is None else 'SUPPLEMENT_HOLD_INCOMPLETE')
    closed.durable_json(control/'actual-exit.json', result, exclusive=True)
    print(json.dumps(result, sort_keys=True), flush=True)
    return rc if error is None else 2


def main():
    parser = argparse.ArgumentParser()
    for name in ('deployment', 'control', 'runtime-manifest', 'runtime-manifest-sha256',
                 'source-commit', 'implementation-commit', 'canonical-complete',
                 'canonical-complete-sha256', 'storage-ready', 'storage-ready-sha256', 'tree-summary-root'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    args.deployment, args.control, args.runtime_manifest = map(Path, (args.deployment, args.control, args.runtime_manifest))
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(status='SUPPLEMENT_HOLD_INCOMPLETE', error=str(error),
            training_action_executed=False), sort_keys=True), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
