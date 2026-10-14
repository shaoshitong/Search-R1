"""Existing management-key SSH route to the retained Wuhan H100 nodes.

This client does not read/copy a private key, change SSH configuration, or
choose a training allocation. Model/data transfers must stay inside H100.
"""
import argparse
import json
import shlex
import subprocess

CONFIG = '/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json'
TARGETS = {'h100-10': '10.33.0.18', 'h100-8': '10.33.0.8',
           'h100-5': '10.33.0.5', 'h100-11': '10.33.4.14'}


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
            '-o', 'IPQoS=none', 'mgmt-host', 'python3 -c '+shlex.quote(source)]


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
