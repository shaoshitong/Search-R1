"""Launch published CPU tree/storage services independently of training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
ARCHIVE = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/recursive-ablation150-archive-20261014')


def proc(pid):
    try:
        path = Path('/proc', str(pid))
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, uid=path.stat().st_uid, tick=int(fields[19]), state=fields[0])
    except FileNotFoundError:
        return None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--service', choices=['tree', 'storage'], required=True)
    parser.add_argument('--mode', choices=['once', 'start', 'watch', 'status'], required=True)
    parser.add_argument('--storage-ready', type=Path)
    parser.add_argument('--storage-ready-sha256')
    args = parser.parse_args()
    publication = read(args.publication)
    source = Path(publication['source'])
    assert source.parent == BASE and source.name == 'source-analysis-'+publication['source_commit'][:8]
    assert Path(__file__).resolve().is_relative_to(source)
    for name, expected in publication['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == expected
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    control = BASE/'analysis'/f"{args.service}-control-{publication['source_commit'][:8]}"
    control.mkdir(exist_ok=True)
    owner_path, exit_path = control/'owner.json', control/'actual-exit.json'
    if args.service == 'tree':
        output = BASE/'analysis'/('tree-summary-'+publication['source_commit'][:8])
        state_path = output/'tree-summary-state.json'
        command = [sys.executable, '-B', '-u', '-m',
            'release.recursive_bran_a.ablation150_tree_summary_v1.collect',
            '--base', str(BASE), '--output', str(output), '--interval', '180']
    else:
        assert args.storage_ready and args.storage_ready_sha256
        assert hashlib.sha256(args.storage_ready.read_bytes()).hexdigest() == args.storage_ready_sha256
        state_path = ARCHIVE/'maintenance-current.json'
        command = [sys.executable, '-B', '-u', '-m',
            'release.recursive_bran_a.ablation150_archive_v1.archive', 'maintain',
            '--source-commit', publication['scientific_source_commit'],
            '--implementation-commit', publication['source_commit'],
            '--storage-ready', str(args.storage_ready), '--storage-ready-sha256', args.storage_ready_sha256,
            '--interval', '300', '--prune-superseded-after-low-space', '--reclaim-after-accepted']
    if args.mode == 'status':
        owner = read(owner_path)
        print(json.dumps(dict(time=time.time(), service=args.service, owner=owner,
            process=proc(owner['pid']) if owner else None, state=read(state_path),
            actual_exit=read(exit_path)), allow_nan=False), flush=True)
        return 0
    if args.mode == 'once':
        return subprocess.run(command+['--once'], env=env, cwd=source).returncode
    if args.mode == 'watch':
        rc = subprocess.run(command, env=env, cwd=source).returncode
        exit_path.write_text(json.dumps(dict(time=time.time(), rc=rc, service=args.service,
            owner=proc(os.getpid()), source_commit=publication['source_commit'])))
        return rc
    if owner_path.exists():
        owner = read(owner_path)
        current = proc(owner['pid'])
        assert current and (current['uid'], current['tick']) == (owner['uid'], owner['tick']), \
            'Existing auxiliary owner is terminal; inspect its actual exit before independent recovery'
        print(json.dumps(dict(already_running=True, owner=owner)), flush=True)
        return 0
    flags = ['--publication', str(args.publication), '--service', args.service, '--mode', 'watch']
    if args.service == 'storage':
        flags += ['--storage-ready', str(args.storage_ready), '--storage-ready-sha256', args.storage_ready_sha256]
    with (control/'service.log').open('ab', buffering=0) as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), *flags],
            env=env, cwd=source, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    owner = dict(proc(child.pid), time=time.time(), service=args.service, output=str(control),
        source_commit=publication['source_commit'], publication=str(args.publication),
        training_action='NONE', GPU_action='NONE')
    owner_path.write_text(json.dumps(owner, indent=2))
    print(json.dumps(dict(started_not_progress_accepted=True, owner=owner)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
