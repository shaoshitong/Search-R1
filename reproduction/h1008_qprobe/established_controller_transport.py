"""Explicit task-local reuse of the verified user controller, without SSH edits."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

CONTROLLER = Path('/Users/shaoshitong/Documents/remote_codex_control/control.py')
CONTROLLER_SHA256 = '06764cd49f1e29d86867f31e94394b94a598d935aed971da0557396a6668be66'
MODE = 'established_user_control_py_direct'


def validate_connection(connection):
    if (connection.get('transport_mode') != MODE or connection.get('controller') != str(CONTROLLER)
            or connection.get('control_path') is not None or connection.get('borrowed_socket', False)
            or connection.get('compact_handshake', False) or connection.get('compat_handshake', False)
            or connection.get('controller_sha256', CONTROLLER_SHA256) != CONTROLLER_SHA256):
        raise ValueError('Established controller must be explicit, pinned and socket-free')


def _controller():
    raw = CONTROLLER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONTROLLER_SHA256:
        raise ValueError('Established controller source changed; no alternate route permitted')
    spec = importlib.util.spec_from_file_location('g1_established_task_controller', CONTROLLER)
    module = importlib.util.module_from_spec(spec)
    # Execute exactly the local static source whose SHA was checked, not a second read.
    exec(compile(raw, str(CONTROLLER), 'exec'), module.__dict__)
    route = module.H100_ACCESS
    if (route.get('gateway_ip') != '10.33.0.18'
            or route.get('targets', {}).get('h100-10') != '10.33.0.18'):
        raise ValueError('Established route must retain this task h100-10 / retained18 binding')
    return module


def remote(script, evidence, *, target='h100-10'):
    if target != 'h100-10':
        raise ValueError('This opt-in adapter only controls this task h100-10 / retained18')
    controller = _controller()
    args = controller.route(target, '/usr/bin/python3 -')
    record = dict(transport_mode=MODE, controller=str(CONTROLLER), controller_sha256=CONTROLLER_SHA256,
        target=target, worker_command='/usr/bin/python3 -',
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), control_path=None,
        global_network_or_auth_changed=False, copied_credentials=False, borrowed_socket=False)
    # Reserve evidence before an external action. An existing receipt cannot cause a replay.
    with Path(evidence).open('x') as stream:
        try:
            result = subprocess.run(args, input=script.encode(), capture_output=True, timeout=240)
            record.update(returncode=result.returncode, stdout=result.stdout.decode(), stderr=result.stderr.decode())
        except subprocess.TimeoutExpired as error:
            record.update(error='BOUNDED_ESTABLISHED_ROUTE_TIMEOUT', returncode=None,
                stdout=(error.stdout or b'').decode(), stderr=(error.stderr or b'').decode())
        except OSError as error:
            record.update(error=type(error).__name__, returncode=None, stdout='', stderr=str(error))
        json.dump(record, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    if record['returncode'] != 0:
        raise RuntimeError(record.get('error', record['stderr']))
    return record['stdout']
