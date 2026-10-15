"""Task-only IPQoS=none SSH candidate; never operates a remote job."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def save(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--attempts', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.attempts <= 100:
        parser.error('--attempts must be between 1 and 100')
    state = args.state.resolve()
    logs = args.logs.resolve()
    state.parent.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = dict(pid=os.getpid(), uid=os.getuid(), started=time.time())
    journal = logs / 'actual-ssh-exits.jsonl'
    for index in range(1, args.attempts + 1):
        root = Path(tempfile.mkdtemp(prefix='rba-f-', dir='/tmp'))
        root.chmod(0o700)
        socket = root / 'control.sock'
        log = logs / f'ssh-{time.time_ns()}-{index:03d}.log'
        log.touch(mode=0o600, exist_ok=False)
        command = [
            'ssh', '-M', '-N', '-S', str(socket), '-v', '-E', str(log),
            '-o', 'ControlMaster=yes', '-o', 'ControlPersist=no',
            '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=12',
            '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none',
            '-o', 'IPQoS=none',
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UpdateHostKeys=no', '-o', 'ConnectTimeout=8', 'mgmt-host',
        ]
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        record = dict(owner=owner, attempt=index, time=time.time(),
                      socket=str(socket), ssh_pid=child.pid, ssh_log=str(log),
                      phase='FOREGROUND_SSH_STARTED_NOT_YET_ACCEPTED',
                      hostname_accepted=False, node_accepted=False,
                      actual_ssh_wait_rc=None, remote_job_action='NONE',
                      global_configuration_changed=False)
        save(state, record)
        print(json.dumps(record), flush=True)
        rc = child.wait()
        record.update(time=time.time(), phase='ACTUAL_SSH_EXIT',
                      actual_ssh_wait_rc=rc)
        save(state, record)
        with journal.open('a') as stream:
            os.chmod(journal, 0o600)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)
        try:
            root.rmdir()  # Only an empty directory created by this attempt.
        except OSError:
            pass
        if rc == 0:
            return 0  # SSH exit is never evidence of a training exit.
        if index < args.attempts:
            time.sleep(3)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
