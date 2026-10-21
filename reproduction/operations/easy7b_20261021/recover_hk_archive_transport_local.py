"""Deploy published controller code to HKUST3 and recover its failed CPU reader once."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('deploy', 'launch'), required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    repo = directory.parents[2]; workspace = repo.parents[1]
    def git(*argv): return subprocess.check_output(['git', '-C', str(repo), *argv])
    assert not git('status', '--porcelain').strip(), 'Publish before deploying controller repair'
    commit = git('rev-parse', 'HEAD').decode().strip()
    assert git('ls-remote', 'origin', 'refs/heads/codex/easy2560-qwen7b-20261021').decode().split()[0] == commit
    transport = module('existing_bound_bridge', workspace/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py')
    hk = '/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017'
    if args.mode == 'deploy':
        deploy = module('existing_source_deployer', directory/'deploy_immutable_source.py')
        names = [name for name in git('ls-files', '-z').decode().split('\0') if name]
        manifest = {name:dict(bytes=(repo/name).stat().st_size,
                    sha256=hashlib.sha256((repo/name).read_bytes()).hexdigest()) for name in names}
        payload = json.dumps(dict(base=hk, commit=commit, manifest=manifest,
                    tar_gz=base64.b64encode(git('archive', '--format=tar.gz', 'HEAD')).decode()))
        code = "import os,socket\nassert os.getuid()==1002 and socket.gethostname()=='xielab3'\n" + deploy.REMOTE
    else:
        secret = Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
        line = next(line for line in secret.splitlines() if line.startswith('账号 jxu957'))
        password = re.split(r'密码\s*[:：]?\s*', line, 1)[1].strip().strip('`')
        secret = line = None
        payload = json.dumps(dict(controller_commit=commit, password=password)); password = None
        code = (directory/'launch_hk_archive_transport_recovery.py').read_text()
    result = transport._campus_long(shlex.join(['python3', '-B', '-c', code]), payload, 55)
    payload = None
    receipt = workspace/'outputs/easy2560-qwen7b-20261021'/('hk-archive-transport-'+args.mode+'-'+str(int(time.time()))+'.json')
    receipt.write_text(json.dumps(dict(returncode=result.returncode, stdout=result.stdout,
                           stderr=result.stderr, controller_commit=commit), indent=2)+'\n')
    print(receipt.name, 'returncode', result.returncode)
    print(result.stdout if result.returncode == 0 else result.stderr[-2000:])
    raise SystemExit(bool(result.returncode))


if __name__ == '__main__': main()
