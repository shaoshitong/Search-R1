"""Read the independent CP5 supplement only; never start, stop, or modify it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time


HK = Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017')
JOB = '745371'
RUN = '232f20e112ac4a978c5aba52a0367195'
BASE = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021'
TRAINING = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'
MODULE_COMMIT = 'b87a0c3d7a7ca2d9f6751a99179897d8d8aca56d'
ROOT = HK/'post-callback-boundary-control'/('job-'+JOB)/'global_step_5'
CORE = HK/'checkpoints'/RUN/'global_step_5'
DESTINATION = CORE/'post-callback-boundary'
NAMES = {'rolling-full-manifest.json', 'boundary/manifest.json', 'boundary/reward-probe-manifest.json',
         'boundary/driver.pt'} | {'boundary/rank-%d.pt' % rank for rank in range(4)}


def need(condition, message):
    if not condition: raise ValueError('POST_CALLBACK5_OBSERVE_' + message)


def regular(path, maximum=8*1024**2):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file() and
         path.stat().st_size <= maximum, 'SMALL_CANONICAL_FILE_REQUIRED')
    return path


def read_json(path):
    return json.loads(regular(path).read_bytes())


def digest(path, maximum=128*1024**2):
    path = regular(path, maximum); before = path.stat(); value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): value.update(block)
    after = path.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns) ==
         (after.st_ino,after.st_size,after.st_mtime_ns), 'FILE_CHANGED_DURING_OBSERVATION')
    return value.hexdigest()


def observe(verify_files=False):
    need(socket.gethostname() == 'xielab3' and os.getuid() == 1002, 'HKUST3_OWNER_REQUIRED')
    records = {}
    for name in ('launch-claim.json', 'launch-receipt.json', 'launch-failure.json', 'claim.json',
                 'hk-tool.json', 'hpc-tool.json', 'source-snapshot-result.json', 'transfer.json',
                 'result.json', 'failure.json'):
        path = ROOT/name
        if path.exists(): records[name] = read_json(path)
    for name in ('launch-claim.json','launch-receipt.json','claim.json','result.json'):
        value = records.get(name)
        if value is not None:
            need(value.get('base') == BASE and value.get('source_job') == JOB and value.get('run_id') == RUN and
                 value.get('step') == 5 and value.get('training_commit') == TRAINING and
                 value.get('module_commit') == MODULE_COMMIT, 'FOREIGN_CONTROL_IDENTITY')
    launch = records.get('launch-receipt.json', {})
    birth = launch.get('birth') or records.get('claim.json', {}).get('birth')
    actual = None; same = False
    if birth:
        try:
            process = Path('/proc')/str(birth['pid'])
            fields = (process/'stat').read_text().rsplit(')',1)[1].split()
            actual = dict(pid=birth['pid'], start_ticks=int(fields[19]), state=fields[0],
                uid=process.stat().st_uid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                hostname=socket.gethostname(), sid=int(fields[3]), cwd=os.readlink(str(process/'cwd')))
            same = all(actual[key] == birth[key] for key in ('pid','start_ticks','uid','boot_id','hostname')) and actual['state'] != 'Z'
            if 'sid' in birth: same = same and actual['sid'] == birth['sid']
        except (FileNotFoundError, ProcessLookupError):
            pass
    archived = None; checks = None
    if (DESTINATION/'boundary-archive-receipt.json').exists():
        archived = read_json(DESTINATION/'boundary-archive-receipt.json')
        binding = archived.get('binding', {})
        need(archived.get('status') == 'EASY2560_POST_CALLBACK_BOUNDARY_ARCHIVE_SHA_FSYNC_PASS' and
             all(binding.get(k) == v for k,v in dict(base=BASE,run_id=RUN,source_commit=TRAINING,step=5,
                core_archive=str(CORE)).items()) and archived.get('file_count') == 8 and
             set(archived.get('files',{})) == NAMES and archived.get('actual_cuda_restore_accepted') is False and
             archived.get('phase',{}).get('sampling_rng_stage') == 'post_QA5', 'ARCHIVE_RECEIPT_IDENTITY_DIFFERS')
        if verify_files:
            checks = {}
            for name,row in archived['files'].items():
                path = regular(DESTINATION/name, 128*1024**2)
                checks[name] = path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
            need(all(checks.values()), 'ARCHIVED_EIGHT_FILE_SHA_DIFFERS')
    log = ROOT/'runner.stdout.log'; tail = ''; log_bytes = 0
    if log.exists():
        regular(log, 16*1024**2); log_bytes = log.stat().st_size
        with log.open('rb') as stream:
            stream.seek(max(0,log_bytes-2400)); tail = stream.read().decode(errors='replace')
    failed = 'failure.json' in records or 'launch-failure.json' in records
    result = records.get('result.json', {})
    complete = (result.get('status') == 'EASY7B_CP5_POST_CALLBACK_ARCHIVE_ONCE_COMPLETE' and
                result.get('emit_returncode') == 0 and archived is not None)
    state = ('COMPLETE' if complete else 'FAILED' if failed else 'RUNNING' if same else
             'TERMINAL_OR_UNKNOWN' if records else 'NOT_LAUNCHED')
    return dict(status='POST_CALLBACK5_READ_ONLY_OBSERVATION', state=state, root=str(ROOT),
        source_job=JOB, run_id=RUN, step=5, birth=birth, actual=actual, same_birth=same, records=records,
        supplementary_archive_receipt=archived, eight_file_sha_readback=checks,
        full_payload_sha_checked=bool(verify_files and archived is not None),
        staging_children=[str(p) for p in CORE.glob('.post-callback-boundary.partial-*')],
        log_bytes=log_bytes, log_tail=tail, controllers_started=0, jobs_submitted=0,
        training_modified=False, deletion_performed=False, checked_unix=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-files', action='store_true')
    args = parser.parse_args()
    print(json.dumps(observe(args.verify_files), sort_keys=True), flush=True)


if __name__ == '__main__': main()
