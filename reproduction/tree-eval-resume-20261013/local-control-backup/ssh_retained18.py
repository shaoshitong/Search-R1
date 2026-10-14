"""Existing management-key SSH route to the retained Wuhan H100 nodes.

This client does not read/copy a private key, change SSH configuration, or
choose a training allocation. Model/data transfers must stay inside H100.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess

CONFIG = '/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json'
TARGETS = {'h100-10': '10.33.0.18', 'h100-8': '10.33.0.8',
           'h100-5': '10.33.0.5', 'h100-11': '10.33.4.14'}
MGMT_CONTROL_PARENT = Path('/tmp').resolve() / 'tree-mgmt-3fw9a1ab'


def _management_control_options():
    value = os.environ.get('TREE_MGMT_CONTROL_PATH')
    if value is None:
        return []
    # OpenSSH expands percent/dollar tokens; reject them and reserve room for
    # its temporary socket suffix within the macOS Unix socket path limit.
    if (not value or any(ord(c) < 33 for c in value) or '%' in value or
            '$' in value or not os.path.isabs(value) or len(os.fsencode(value)) > 80):
        raise ValueError('TREE_MGMT_CONTROL_PATH requires a literal absolute path of at most 80 bytes')
    path = Path(value)
    if value != str(path) or path.name != 'control':
        raise ValueError('TREE_MGMT_CONTROL_PATH requires this task socket filename: control')
    parent = path.parent
    real_parent = parent.resolve(strict=True)
    info = parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) != 0o700 or real_parent != MGMT_CONTROL_PARENT):
        raise ValueError('ControlPath parent must be this task owned mode-700 /tmp/tree-mgmt-3fw9a1ab directory')
    if path.is_symlink() or (path.exists() and
            (not stat.S_ISSOCK(path.lstat().st_mode) or path.lstat().st_uid != os.getuid())):
        raise ValueError('ControlPath may only name an absent path or an owned Unix socket')
    return ['-o', 'ControlPath='+value, '-o', 'ControlMaster=auto',
            '-o', 'ControlPersist=600']


def route(command, target='h100-10'):
    if target not in TARGETS:
        raise ValueError('Only the retained same-cluster target list is permitted')
    source = '''import json,shlex,subprocess,sys
from pathlib import Path
cfg=json.loads(Path(CONFIG).read_text())
target=TARGET
expected_ip=EXPECTED_IP
if cfg['targets'][target]!=expected_ip or cfg['gateway_ip']!='10.33.0.18' or cfg['user']!='root' or cfg['management_identity_file']!='/home/lichen/.ssh/id_ed25519':
    raise RuntimeError('Existing verified management route identity changed')
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1']
args=prefix.copy()
if expected_ip!=cfg['gateway_ip']:
    args+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',cfg['user']+'@'+cfg['gateway_ip']])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:
    args+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
sys.exit(subprocess.call(args+[cfg['user']+'@'+expected_ip,COMMAND]))
'''
    for name, value in [('CONFIG', CONFIG), ('TARGET', target),
                        ('EXPECTED_IP', TARGETS[target]), ('COMMAND', command)]:
        source = source.replace(name, repr(value))
    # Honor the user's working mgmt-host SSH configuration. A forced legacy
    # SOCKS route can fail even while ordinary `ssh mgmt-host` succeeds.
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o',
            'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o',
            'ServerAliveCountMax=3', '-o', 'KexAlgorithms=curve25519-sha256',
            '-o', 'IPQoS=none'] + _management_control_options() + [
            'mgmt-host', 'python3 -c '+shlex.quote(source)]


def run(command, target='h100-10', timeout=60):
    return subprocess.run(route(command, target), capture_output=True,
                          text=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=sorted(TARGETS), default='h100-10')
    parser.add_argument('command')
    args = parser.parse_args()
    raise SystemExit(subprocess.call(route(args.command, args.target)))


if __name__ == '__main__':
    main()
