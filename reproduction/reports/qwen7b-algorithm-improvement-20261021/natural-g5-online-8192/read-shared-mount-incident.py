import json
from pathlib import Path
import socket
import subprocess
import time

out = {'time': time.time(), 'hostname': socket.gethostname(), 'paths': {}, 'commands': {}, 'final4096_read': False}
for name in ('/data', '/data/user', '/data/user/jxu957', '/data/modulefiles', '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'):
    path = Path(name)
    try:
        stat = path.stat()
        out['paths'][name] = {'exists': True, 'device': stat.st_dev, 'inode': stat.st_ino, 'uid': stat.st_uid}
    except OSError as error:
        out['paths'][name] = {'exists': False, 'error': str(error)}
for command in (['findmnt', '-T', '/data', '-o', 'TARGET,SOURCE,FSTYPE,OPTIONS'],
                ['squeue', '-h', '-j', '749459,749460', '-o', '%i|%T|%N|%R'],
                ['sacct', '-n', '-P', '-j', '749459,749460', '-o', 'JobIDRaw,State,ExitCode,Elapsed,NodeList']):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=10)
        out['commands'][' '.join(command)] = {'code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as error:
        out['commands'][' '.join(command)] = {'error': str(error)}
print(json.dumps(out))
