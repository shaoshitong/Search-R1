"""Read-only observation of the explicit Python 3.8 callback recovery.

Load the published ordinary observer privately and route only that instance to
the separate recovery root. The original failed controller remains untouched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re


HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
OLD_ROOT = HK/'post-callback-boundary-control/job-745371/future-50-200'
ROOT = OLD_ROOT/'recovery-python38-r1'
OBSERVER = 'observe_post_callback_future_hk.py'
RUNNER = 'archive_post_callback_future_hk.py'


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK_FUTURE_RECOVERY_OBSERVE_' + message)


def regular(path, maximum=128*1024):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file() and
         0 < path.stat().st_size <= maximum, 'SMALL_CANONICAL_FILE_REQUIRED')
    return path


def document(path):
    return json.loads(regular(path, 16*1024**2).read_bytes())


def tools_path(launch):
    commit = launch.get('launcher_commit', '')
    need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit) is not None,
         'PUBLISHED_LAUNCHER_COMMIT_REQUIRED')
    tools = HK/'post-callback-future-recovery-tools'/commit
    need(launch.get('tools') == str(tools), 'PUBLISHED_TOOLS_PATH_DIFFERS')
    return tools


def load_published(launch, name):
    tools = tools_path(launch); proof = launch.get('public_code', {}).get(name, {})
    path = regular(tools/name)
    need(proof.get('path') == str(path) and proof.get('bytes') == path.stat().st_size and
         proof.get('sha256') == hashlib.sha256(path.read_bytes()).hexdigest(), 'PUBLISHED_SOURCE_DIFFERS')
    spec = importlib.util.spec_from_file_location('_read_only_recovered_' + name.replace('.', '_'), str(path))
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def observe(verify_files=False):
    launch_path = ROOT/'launch-receipt.json'
    if not launch_path.exists():
        # A failed launch may have only its immutable public-code claim.
        launch_path = ROOT/'launch-claim.json'
    need(launch_path.exists(), 'RECOVERY_LAUNCH_RECORD_REQUIRED')
    launch = document(launch_path)
    ordinary = load_published(launch, OBSERVER)
    ordinary.ROOT = ROOT
    ordinary.load_runner = lambda record: load_published(record, RUNNER)
    value = ordinary.observe(verify_files)
    value['recovery_root'] = str(ROOT); value['original_failed_root'] = str(OLD_ROOT)
    value['recovery_records'] = {name:document(ROOT/name) for name in (
        'recovery-accept-result.json', 'recovery-result.json', 'companion-start-receipt.json',
        'recovery-prerequisites.json') if (ROOT/name).is_file()}
    value['original_failed_controller_modified'] = False
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files', action='store_true',
        help='Fresh transferred dependency SHA plus accepted-core mapping; no encoder payload rehash')
    args = parser.parse_args()
    print(json.dumps(observe(args.verify_files), sort_keys=True), flush=True)


if __name__ == '__main__': main()
