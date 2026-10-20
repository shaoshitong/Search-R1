"""Explicit task-local management SOCKS option; frozen H100 runtime is untouched.

The default remains the existing pinned controller. Opting in changes only this
Python process's outer mgmt-host SSH command, using an already running local
SOCKS service. It starts no proxy/VPN and edits no shared controller or SSH file.
Set B_REPLICA_MGMT_SOCKS5_7892=1 for this task's shared_observe/follow_shared
or read_ip14_live entry; unset it to retain the original direct transport.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

FLAG = 'B_REPLICA_MGMT_SOCKS5_7892'
MODE = 'explicit_existing_local_socks5_7892'


def enabled():
    value = os.environ.get(FLAG, '')
    if value not in ('', '0', '1'):
        raise ValueError('Management observation opt-in must be0 or1')
    return value == '1'


def management_argv(args):
    """Preserve identity, endpoint, KEX and en0; never reuse a control socket."""
    if not enabled():
        return list(args)
    if not args or args[0] not in ('ssh', '/usr/bin/ssh'):
        raise ValueError('Require the pinned outer SSH argv')
    kept, index = [args[0]], 1
    while index < len(args):
        if args[index] == '-o' and index + 1 < len(args):
            option = args[index + 1]
            if option in ('ProxyCommand=none', 'ProxyJump=none'):
                index += 2
                continue
            if option.startswith(('ProxyCommand=', 'ProxyJump=')):
                raise ValueError('Do not combine another explicit management proxy')
        if args[index] == '-S':
            raise ValueError('Do not borrow a management control socket')
        kept.append(args[index])
        index += 1
    return kept[:1] + [
        '-o', 'ProxyCommand=nc -X 5 -x 127.0.0.1:7892 %h %p',
        '-o', 'ProxyJump=none', '-o', 'StrictHostKeyChecking=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'ControlMaster=no',
        '-o', 'ControlPath=none',
    ] + kept[1:]


def management_ssh(controller):
    return management_argv(controller.ssh('mgmt-host'))


def remote(script, evidence, control_path=None, compat_handshake=False,
           target='h100-8', compact_handshake=False, *, established_controller=False):
    from reproduction.h1008_qprobe.management_transport import remote as direct
    if not enabled():
        return direct(script, evidence, control_path, compat_handshake, target,
                      compact_handshake, established_controller=established_controller)
    if (not established_controller or control_path is not None
            or compat_handshake or compact_handshake):
        raise ValueError('SOCKS opt-in retains the pinned socket-free controller only')
    if target not in ('h100-5', 'h100-10', 'h100-11'):
        raise ValueError('Use the current three permitted H100 resources')
    from reproduction.h1008_qprobe.established_controller_transport import (
        _controller, CONTROLLER, CONTROLLER_SHA256)
    controller = _controller()
    original_ssh = controller.ssh

    def outer_only(host):
        args = original_ssh(host)
        return management_argv(args) if host == 'mgmt-host' else args

    # This module instance exists only in the calling task's Python process.
    controller.ssh = outer_only
    args = controller.route(target, '/usr/bin/python3 -')
    record = dict(transport_mode=MODE, controller=str(CONTROLLER),
        controller_sha256=CONTROLLER_SHA256, target=target,
        script_sha256=hashlib.sha256(script.encode()).hexdigest(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        control_path=None, borrowed_socket=False, copied_credentials=False,
        global_network_or_auth_changed=False, proxy_or_vpn_started=False)
    with Path(evidence).open('x') as stream:
        try:
            result = subprocess.run(args, input=script.encode(), capture_output=True,
                                    timeout=240)
            record.update(returncode=result.returncode,
                stdout=result.stdout.decode(), stderr=result.stderr.decode())
        except subprocess.TimeoutExpired as error:
            record.update(returncode=None, error='BOUNDED_TASK_SOCKS_ROUTE_TIMEOUT',
                stdout=(error.stdout or b'').decode(), stderr=(error.stderr or b'').decode())
        except OSError as error:
            record.update(returncode=None, error=type(error).__name__,
                          stdout='', stderr=str(error))
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    if record['returncode'] != 0:
        raise RuntimeError(record.get('error', record['stderr']))
    return record['stdout']
