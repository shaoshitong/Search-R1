"""Run on HKUST3; stdin carries a RAM-only password and verified public keys.

Dataset bytes move solely HKUST3 -> HPC3 SFTP. No password is written to a
file, argument, environment variable, receipt or log. Public host keys persist.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time

SHA = '10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'
SIZE = 1073443
SOURCE = Path('/backup/shaoshitong/qwen7b-probe-heldout2048-20261020/source/nq-dev-bcafb8dd07d4.jsonl')
ROOT = '/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/selection-2048-20261020'
DESTINATION = ROOT + '/source/' + SOURCE.name


def main():
    import paramiko
    payload = json.load(sys.stdin)
    assert payload['account'] == 'mou176' and payload['host'] == '10.120.81.30'
    assert SOURCE.stat().st_size == SIZE and hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SHA
    hosts = SOURCE.parent / 'verified-hpc3-public-known-hosts'
    hosts.write_text(payload['known_hosts'])
    os.chmod(str(hosts), 0o600)
    client = paramiko.SSHClient()
    client.load_host_keys(str(hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    # The established control connection pins this HostKeyAlias, while routing
    # its TCP socket directly to the campus IP. Preserve the same trust binding.
    sock = socket.create_connection((payload['host'], 22), timeout=15)
    client.connect('hpc3login.hpc.hkust-gz.edu.cn', sock=sock,
                   username=payload['account'], password=payload.pop('password'),
                   look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=15, banner_timeout=15)
    try:
        sftp = client.open_sftp()
        for path in (ROOT, ROOT + '/source'):
            try:
                sftp.stat(path)
            except IOError as error:
                if getattr(error, 'errno', None) != 2:
                    raise
                sftp.mkdir(path)
        try:
            sftp.stat(DESTINATION)
            exists = True
        except IOError as error:
            if getattr(error, 'errno', None) != 2:
                raise
            exists = False
        if not exists:
            temporary = DESTINATION + '.part-%d-%d' % (int(time.time()), os.getpid())
            sftp.put(str(SOURCE), temporary, confirm=True)
            check_path = temporary
        else:
            check_path = DESTINATION
        code = 'import pathlib,hashlib,json,os;p=pathlib.Path(%r);d=p.read_bytes();assert len(d)==%d and hashlib.sha256(d).hexdigest()==%r;print(json.dumps(dict(path=str(p),bytes=len(d),sha256=hashlib.sha256(d).hexdigest(),uid=os.getuid())))' % (check_path, SIZE, SHA)
        _, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(code), timeout=30)
        result = stdout.read().decode()
        assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()[-1000:]
        verified = json.loads(result)
        assert verified['uid'] == 21930
        if not exists:
            sftp.rename(temporary, DESTINATION)
        receipt = dict(status='REMOTE_ONLY_HKUST3_TO_HPC3_PINNED_NQ_DEV_SHA_VERIFIED',
                       source=str(SOURCE), destination=DESTINATION, bytes=SIZE, sha256=SHA,
                       dataset_revision='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c',
                       mac_data_relay=False, account='mou176', epoch_time=time.time())
        path = ROOT + '/source/complete.json'
        with sftp.open(path + '.tmp', 'w') as handle:
            handle.write(json.dumps(receipt, indent=2))
        if not exists:
            sftp.rename(path + '.tmp', path)
        else:
            # POSIX rename replaces only this task's small receipt atomically.
            sftp.posix_rename(path + '.tmp', path)
        sftp.close()
        print(json.dumps(receipt), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
