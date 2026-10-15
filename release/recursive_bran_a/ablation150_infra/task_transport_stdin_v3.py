"""Version3 observation transport with explicit timeout receipts and optional required socket."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--management-socket', required=True)
    parser.add_argument('--target', choices=['h100-10', 'h100-11'], required=True)
    parser.add_argument('--python-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--script-args-json', default='[]')
    parser.add_argument('--require-socket', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.timeout <= 600
    if args.require_socket and args.management_socket == 'none':
        parser.error('--require-socket needs an explicit task socket')
    script_args = json.loads(args.script_args_json)
    assert isinstance(script_args, list) and len(script_args) <= 32
    assert all(isinstance(v, str) and len(v) <= 4096 and '\x00' not in v for v in script_args)
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
if payload['script_args']:program+=' '+shlex.join(payload['script_args'])
r=subprocess.run(command+[user+'@'+ip,program],input=source,capture_output=True,text=True,timeout=timeout)
print(json.dumps(dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)),flush=True)
'''.replace('PAYLOAD', repr(json.dumps(dict(target=args.target, source=source, timeout=args.timeout, script_args=script_args))))
    command = ['ssh', '-S', args.management_socket, '-o', 'ControlMaster=no', '-T',
        '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3', '-o', 'BatchMode=yes',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=15',
        '-o', 'ProxyCommand=' + ('false' if args.require_socket else 'none'),
        '-o', 'ProxyJump=none', 'mgmt-host',
        'env -u LD_LIBRARY_PATH python3 -']
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    actual_outer = None
    timed_out = False
    try:
        result = subprocess.run(command, input=remote, capture_output=True,
                                text=True, timeout=args.timeout + 25)
        actual_outer = result.returncode
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        def decoded(value):
            return value.decode('utf-8', 'replace') if isinstance(value, bytes) else (value or '')
        stdout, stderr = decoded(exc.stdout), decoded(exc.stderr)
    Path(str(output) + '.transport.stderr').write_text(stderr)
    Path(str(output) + '.transport.stdout').write_text(stdout)
    observation = dict(started=started, finished=time.time(),
        outer_command_actual_exit=actual_outer, outer_timed_out=timed_out,
        timeout_limit_seconds=args.timeout + 25,
        socket_required=args.require_socket, cold_fallback_allowed=not args.require_socket,
        management_socket=args.management_socket,
        program_sha256=hashlib.sha256(source.encode()).hexdigest(),
        remote_training_state='UNKNOWN',
        scope='Readonly observation result is not a production training exit')
    def finish(status, wrapper_exit, **extra):
        observation.update(status=status, wrapper_exit_code=wrapper_exit, **extra)
        receipt = Path(str(output) + '.observation.json')
        receipt.write_text(json.dumps(observation, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(output=str(output), observation=str(receipt), status=status,
            wrapper_exit_code=wrapper_exit, outer_command_actual_exit=actual_outer,
            reader_actual_exit=observation.get('reader_actual_exit'))))
        return wrapper_exit
    if timed_out:
        return finish('TRANSPORT_TIMEOUT', 124)
    if actual_outer:
        return finish('MANAGER_COMMAND_FAILED', actual_outer)
    try:
        record = json.loads(stdout)
        assert isinstance(record, dict)
        assert isinstance(record['exit_code'], int) and not isinstance(record['exit_code'], bool)
        assert isinstance(record['stdout'], str) and isinstance(record['stderr'], str)
        assert isinstance(record['observed_utc'], str)
    except (ValueError, KeyError, AssertionError, TypeError) as exc:
        return finish('INVALID_MANAGER_ENVELOPE', 125, parse_error=str(exc))
    Path(str(output) + '.record.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['exit_code']:
        return finish('REMOTE_READER_FAILED', record['exit_code'],
                      reader_actual_exit=record['exit_code'], observed_utc=record['observed_utc'])
    output.write_text(record['stdout'])
    return finish('COMPLETE', 0, reader_actual_exit=0, observed_utc=record['observed_utc'])



if __name__ == '__main__':
    raise SystemExit(main())
