"""Optional task-owned authenticated SSH master; no network/auth reconfiguration."""
import os
from pathlib import Path
import stat
import subprocess


def apply(argv):
    value = os.environ.get('Q7B_MGMT_CONTROL_PATH')
    if not value:
        return argv
    path = Path(value)
    allowed = ('/tmp/q7b-mgmt-70188b8e50fd', '/tmp/q7b-mgmt-source-70188b8e50fd')
    if (str(path.parent) not in allowed or path.name != 'socket'
            or path.is_symlink() or path.parent.is_symlink()):
        raise ValueError('Only this sampling run authenticated-master socket is accepted')
    parent, socket = path.parent.stat(), path.stat()
    if (parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) != 0o700
            or socket.st_uid != os.getuid() or not stat.S_ISSOCK(socket.st_mode)):
        raise ValueError('Task master is not an owned private Unix socket')
    if not argv or argv[0] != 'ssh' or 'mgmt-host' not in argv:
        raise ValueError('Task master must preserve the existing direct management route')
    check = subprocess.run(['ssh', '-S', str(path), '-O', 'check', 'mgmt-host'],
        stdin=subprocess.DEVNULL, text=True, capture_output=True, timeout=5)
    if check.returncode:
        raise RuntimeError('Task authenticated master unavailable; reconcile before remote mutation')
    # Existing host key, identity and inner retained18 command are unchanged.
    return [argv[0], '-S', str(path), *argv[1:]]
