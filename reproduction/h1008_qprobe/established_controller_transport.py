"""Opt-in local reuse of the user's established direct controller route.

No global SSH/network/auth changes, no borrowed sockets or credentials copied.
The eight-GPU trainer never invokes this Mac-side transport.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess


CONTROLLER = Path('/Users/shaoshitong/Documents/remote_codex_control/control.py')


def remote(script, evidence, *, target='h100-8', control_path=None):
    addresses = {'h100-8': '10.33.0.8', 'h100-10': '10.33.0.18'}
    if target not in addresses:
        raise ValueError('Use only the current trainer or independent backup node')
    spec = importlib.util.spec_from_file_location('qprobe_established_local_controller', CONTROLLER)
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    route = controller.H100_ACCESS
    if route.get('gateway_ip') != '10.33.0.18' or route.get('targets', {}).get(target) != addresses[target]:
        raise ValueError('Do not silently fall back to retired16 or a different target')
    args = controller.route(target, '/usr/bin/python3 -')
    if control_path:
        socket = Path(control_path); parent = socket.parent
        if (not parent.name.startswith('qprobe3-mgmt-') or parent.stat().st_uid != os.getuid()
                or stat.S_IMODE(parent.stat().st_mode) != 0o700
                or socket.lstat().st_uid != os.getuid() or not stat.S_ISSOCK(socket.lstat().st_mode)):
            raise ValueError('Use the explicitly owned private task master only')
        subprocess.run(['ssh', '-S', str(socket), '-O', 'check', 'mgmt-host'],
                       capture_output=True, check=True, timeout=10)
        position = args.index('mgmt-host')
        args[position:position] = ['-S', str(socket), '-o', 'ControlMaster=no']
    record = dict(route='established control.py direct mgmt-host -> retained18 -> '+target,
                  controller=str(CONTROLLER), controller_sha256=hashlib.sha256(CONTROLLER.read_bytes()).hexdigest(),
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  task_control_path=str(control_path) if control_path else None,
                  global_network_or_auth_changed=False, copied_credentials=False)
    try:
        result = subprocess.run(args, input=script.encode(), capture_output=True, timeout=240)
        record.update(returncode=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())
    except subprocess.TimeoutExpired as error:
        record.update(error='BOUNDED_ESTABLISHED_ROUTE_TIMEOUT', returncode=None,
                      stdout=(error.stdout or b'').decode(), stderr=(error.stderr or b'').decode())
    with Path(evidence).open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    if record['returncode'] != 0:
        raise RuntimeError(record.get('error', record['stderr']))
    return record['stdout']
