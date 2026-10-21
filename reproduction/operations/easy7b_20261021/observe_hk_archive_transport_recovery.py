"""Observe the unique replacement CPU controller; never start or stop it."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time

assert socket.gethostname() == 'xielab3' and os.getuid() == 1002
parent = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/archive-controller/job-745371')
root = parent/'recovery-transport-r1-20261021'
receipt = json.loads((root/'launch-receipt.json').read_text())
assert receipt['source_job'] == '745371' and receipt['training_commit'] == 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
assert receipt['base'] == '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
assert receipt['requested_steps'] == [5, 50, 100, 150, 200]
birth = receipt['birth']; actual = None; same = False
try:
    process = Path('/proc')/str(birth['pid'])
    fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
    actual = dict(pid=birth['pid'], start_ticks=int(fields[19]), state=fields[0],
                  uid=process.stat().st_uid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  cwd=os.readlink(str(process/'cwd')))
    same = all(actual[key] == birth[key] for key in ('pid', 'start_ticks', 'uid', 'boot_id')) and actual['state'] != 'Z'
except (FileNotFoundError, ProcessLookupError):
    pass
records = {name:json.loads((root/name).read_text()) for name in
           ('claim.json', 'progress.json', 'failure.json', 'result.json') if (root/name).is_file()}
preserved = all(hashlib.sha256((parent/name).read_bytes()).hexdigest() == row['sha256']
                for name, row in receipt['parent_failure_preserved'].items())
log = root/'controller.stdout.log'
with log.open('rb') as stream:
    stream.seek(max(0, log.stat().st_size-1800)); tail = stream.read().decode(errors='replace')
print(json.dumps(dict(checked_unix=time.time(), root=str(root), controller_commit=receipt['controller_commit'],
                     training_commit=receipt['training_commit'], birth=birth, actual=actual, same_birth=same,
                     parent_failure_unchanged=preserved, records=records,
                     log_bytes=log.stat().st_size, log_tail=tail,
                     controllers_started=0, jobs_submitted=0, training_modified=False)))
