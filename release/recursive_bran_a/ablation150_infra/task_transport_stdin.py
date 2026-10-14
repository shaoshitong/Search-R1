"""Existing trusted H100 route, with code on stdin rather than a large argv."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--management-socket', required=True)
    parser.add_argument('--target', choices=['h100-10', 'h100-11'], required=True)
    parser.add_argument('--python-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=45)
    args = parser.parse_args()
    assert 1 <= args.timeout <= 600
    source = Path(args.python_file).read_text()
    remote = '''import datetime,json,shlex,subprocess
from pathlib import Path
cfg=json.loads(Path('/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json').read_text())
payload=json.loads(PAYLOAD)
target=payload['target'];source=payload['source'];timeout=payload['timeout']
ip=cfg['targets'][target];gateway=cfg['gateway_ip'];user=cfg['user']
prefix=['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',cfg['management_identity_file'],
 '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
 '-o','UpdateHostKeys=no','-o','ConnectTimeout=8','-o','ConnectionAttempts=1',
 '-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
command=prefix.copy()
if ip!=gateway:command+=['-o','ProxyCommand='+shlex.join(prefix+['-W','%h:%p',user+'@'+gateway])]
keys=cfg.get('trusted_target_host_keys',{}).get(target)
if keys:command+=['-o','KnownHostsCommand='+shlex.join(['/bin/echo','\\n'.join(keys)])]
program='env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python3 -B -'
r=subprocess.run(command+[user+'@'+ip,program],input=source,capture_output=True,text=True,timeout=timeout)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
'''.replace('PAYLOAD', repr(json.dumps(dict(target=args.target, source=source, timeout=args.timeout))))
    command = ['ssh', '-S', args.management_socket, '-o', 'ControlMaster=no', '-T',
        '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3', '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15',
        '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none', 'mgmt-host',
        'env -u LD_LIBRARY_PATH python3 -']
    result = subprocess.run(command, input=remote, capture_output=True, text=True, timeout=args.timeout+25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(output)+'.transport.stderr').write_text(result.stderr)
    Path(str(output)+'.transport.stdout').write_text(result.stdout)
    if result.returncode:
        return result.returncode
    record = json.loads(result.stdout)
    Path(str(output)+'.record.json').write_text(json.dumps(record, indent=2))
    output.write_text(record['stdout'])
    print(json.dumps(dict(output=str(output), observed_utc=record['observed_utc'],
                         exit_code=record['exit_code'], stderr=record['stderr'])))
    return record['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
