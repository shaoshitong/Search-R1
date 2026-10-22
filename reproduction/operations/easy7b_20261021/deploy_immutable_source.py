"""Deploy one pushed Git archive snapshot; working-tree drafts are never read."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import time

from reproduction.operations.easy7b_20261021.watch_fa3_transition_local import (
    DEFAULT_LOGIN_HOST, resolve_login_route)

REMOTE = r'''
import base64,hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['base'])/'source'/p['commit']
if root.exists():
 proof=json.loads((root/'source-publication.json').read_text())
 assert proof['source_commit']==p['commit'] and proof['files']==p['manifest']
else:
 stage=root.with_name('.partial-'+p['commit']);stage.mkdir(parents=True,exist_ok=False);seen=set()
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['tar_gz'])),mode='r:gz') as bundle:
  for member in bundle:
   if member.isdir():continue
   relative=Path(member.name)
   assert member.isfile() and not relative.is_absolute() and '..' not in relative.parts
   assert member.name in p['manifest'] and member.name not in seen
   seen.add(member.name);data=bundle.extractfile(member).read();record=p['manifest'][member.name]
   assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
   target=stage/member.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);target.chmod(member.mode&0o777)
 assert seen==set(p['manifest'])
 proof=dict(source_commit=p['commit'],files=p['manifest'],fresh_source_all_files_sha_verified=True,checked_unix=time.time())
 with (stage/'source-publication.json').open('x') as stream:json.dump(proof,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 stage.rename(root)
assert all((root/name).stat().st_size==record['bytes'] and hashlib.sha256((root/name).read_bytes()).hexdigest()==record['sha256'] for name,record in p['manifest'].items())
print(json.dumps(dict(source_commit=p['commit'],source_root=str(root),verified_files=len(p['manifest']),verified_bytes=sum(r['bytes']for r in p['manifest'].values()),fresh_source_all_files_sha_verified=True,checked_unix=time.time())))
'''


def archive_manifest(archive):
    """Hash the exact regular-file bytes that REMOTE will extract with tar modes."""
    manifest = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        for member in bundle:
            if member.isdir(): continue
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts or
                    not relative.parts or str(relative) != member.name or member.name in manifest):
                raise ValueError('Git archive contains an unsafe, duplicate or nonregular member')
            data = bundle.extractfile(member).read()
            if len(data) != member.size:
                raise ValueError('Git archive member is incomplete')
            manifest[member.name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--branch', help='Published branch; defaults to the current checkout branch')
    parser.add_argument('--login-host', default=DEFAULT_LOGIN_HOST)
    parser.add_argument('--login-identity-json',
                        help='Explicit hostname/boot_id/pid_namespace JSON; required for a different login host')
    args = parser.parse_args(argv)
    try:
        login_host, login_identity = resolve_login_route(args.login_host, args.login_identity_json)
    except ValueError as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[3]
    def git(*command):
        return subprocess.check_output(['git', '-C', str(repo), *command])
    commit = git('rev-parse', 'HEAD').decode().strip()
    branch = args.branch or git('branch', '--show-current').decode().strip()
    if not branch:
        raise ValueError('An explicit published branch is required from a detached checkout')
    assert git('ls-remote', 'origin', 'refs/heads/'+branch).decode().split()[0] == commit
    from reproduction.easy2560_joint4.control import own_base
    base = own_base(args.base)
    # Pin the observed HEAD SHA rather than resolving HEAD a second time while
    # another agent may commit. This one archive supplies both bytes and modes.
    archive = git('archive', '--format=tar.gz', commit)
    manifest = archive_manifest(archive)
    payload = json.dumps(dict(base=str(base), commit=commit, manifest=manifest,
                              tar_gz=base64.b64encode(archive).decode()))
    cwd = repo.parents[1]
    bridge_path = cwd/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
    spec = importlib.util.spec_from_file_location('bound_bridge', bridge_path)
    bridge = importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    command = shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python', '-B', '-c', REMOTE])
    observed = bridge.run('jxu957', command, payload, str(cwd/'work'),
        login_host=login_host, observation_timeout=55, expected_host_identity=login_identity)
    result = dict(returncode=observed.returncode, stdout=observed.stdout,
                  stderr=observed.stderr, checked_unix=time.time(),
                  login_host=login_host, login_identity=login_identity)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(bool(observed.returncode))


if __name__ == '__main__':
    main()
