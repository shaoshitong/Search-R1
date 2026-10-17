"""Deploy only a clean, pushed Git source archive through the existing bridge."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from .control import BASE, PREDECESSOR, own_base
from reproduction.prefix20_joint4.deploy_source import REMOTE, IDENTITY


def published_payload(repo, commit, base):
    """Use exact Git bytes, so unrelated in-progress files cannot be deployed."""
    raw = subprocess.check_output(['git', '-C', str(repo), 'archive', '--format=tar.gz', commit])
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        for item in archive:
            if item.isdir(): continue
            name = Path(item.name)
            if not item.isfile() or name.is_absolute() or '..' in name.parts or item.name in manifest:
                raise ValueError('EASY2560_GIT_SOURCE_MEMBER_UNSAFE')
            data = archive.extractfile(item).read()
            manifest[item.name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return dict(base=str(own_base(base)), commit=commit, manifest=manifest,
                tar_gz=base64.b64encode(raw).decode())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bridge-source', required=True)
    p.add_argument('--control-work', required=True); p.add_argument('--output', required=True)
    p.add_argument('--base', default=BASE); p.add_argument('--commit'); a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', a.commit or 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip()
    if not branch or (a.commit is None and subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True)):
        raise ValueError('EASY2560_CLEAN_COMMITTED_SOURCE_REQUIRED')
    remote = subprocess.check_output(['git', '-C', str(repo), 'ls-remote',
                                     'git@github.com:shaoshitong/Search-R1.git', 'refs/heads/' + branch], text=True)
    if not remote or remote.split()[0] != commit: raise ValueError('EASY2560_LATEST_SOURCE_NOT_PUSHED')
    payload = json.dumps(published_payload(repo, commit, a.base))
    spec = importlib.util.spec_from_file_location('bound_account_bridge', a.bridge_source)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    result = bridge.run('jxu957', str(Path(PREDECESSOR) / 'env/bin/python') + ' -B -c ' + shlex.quote(REMOTE),
                        payload, a.control_work, login_host='10.120.48.31',
                        observation_timeout=90, expected_host_identity=IDENTITY)
    proof = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                 source_commit=commit, branch=branch, checked_unix=time.time())
    Path(a.output).write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof)); raise SystemExit(result.returncode)


if __name__ == '__main__': main()
