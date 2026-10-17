"""Narrow metadata-only eligibility for correcting the known CP50 preflight bug.

The v1 reader SHA below prints HF/Torch stages before opening any saved tensor.
Its missing resume scheduler field is checked before source/split/native reads.
A single final failure line plus the pinned source/error/terminal receipts proves
this specific attempt stopped there; generic failed audits are never replayable.
"""
import hashlib
import json
from pathlib import Path

BUG_READER_SHA256 = '03e2eefffbd85caeaa1a8c67012d9949d9aa01b34cc5946aed62b94d68eed162'
BUG_HELPER_COMMIT = '24af78b189f27acda74c869aa0240c42d7c05fa8'
FAILED_STATUS = 'PREFIX_SELECTION200_CPU_MILESTONE_AUDIT_FAILED_TRAINING_UNCHANGED'


def proof_name(outer, version='v1'):
    if outer not in (5,50,100,150,200) or version not in ('v1','v2'):
        raise ValueError('Unsupported once-only audit identity')
    return 'joint-milestone-%04d-independent-full-sha-cpu-once-%s' % (outer, version)


def verify_predecessor(base, config):
    """Read small immutable evidence only; preserve every byte of the failed v1."""
    base = Path(base)
    if (config.get('audit_version') != 'v2' or config['outer'] != 50
            or config['base'] != str(base)
            or config.get('predecessor_control_name') != 'acceptance50-cpu-control-v1'
            or config['control_name'] != 'acceptance50-cpu-control-v2'
            or config['unit'] != 'prefix-diversity-4a1cc2d5-cal1-checkpoint50-cpu-v2'):
        raise ValueError('v2 is only the explicit isolated CP50 metadata correction')
    files = {}
    def raw(path):
        path = Path(path)
        if base not in path.parents or path.resolve(strict=True) != path:
            raise ValueError('Correction evidence must remain in this canonical run')
        before = path.stat()
        if before.st_size > 1024**2:
            raise ValueError('Correction preflight may read only small evidence')
        data = path.read_bytes(); after = path.stat()
        signature = lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        if signature(before) != signature(after):raise ValueError('Correction evidence changed')
        files[str(path)] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        return data
    def read(path):return json.loads(raw(path))
    previous = base/config['predecessor_control_name']
    path = base/'preparation'/(proof_name(50)+'.json')
    prior = read(path); expected = config['predecessor_failed_proof_sha256']
    if files[str(path)]['sha256'] != expected:raise ValueError('Failed v1 proof differs')
    intent = read(path.with_name(path.stem+'-intent.json'))
    deploy = read(previous/'deploy-receipt.json'); exit_ = read(previous/'reader-exit.json')
    start = read(previous/'reader-start.json')
    log = raw(previous/'reader.log').decode(); lines = log.splitlines()
    if len(lines) != 1:raise ValueError('Prior audit already entered a payload stage or has ambiguous log')
    final = json.loads(lines[0]); old = deploy['config']
    launch = base/'preparation/launch-config.json'; actual = read(launch)
    joint = base/'outers/outer-000050/complete.json'; read(joint)
    if not (prior['status'] == FAILED_STATUS and prior['error_type'] == 'KeyError'
            and prior['error'] == "'scheduler_completed_outer'"
            and prior['reader_source_sha256'] == BUG_READER_SHA256
            and prior['independent_full_file_sha_performed'] is False
            and prior['actual_cpu_deserialization_performed'] is False
            and prior['elapsed_seconds'] < 5
            and final['status'] == FAILED_STATUS and final['error'] == prior['error']
            and final['proof']['sha256'] == expected
            and deploy['helper_commit'] == exit_['helper_commit'] == start['helper_commit'] == BUG_HELPER_COMMIT
            and deploy['files']['reproduction/prefix_diversity200/checkpoint_cpu_once.py']['sha256'] == BUG_READER_SHA256
            and exit_['status'] == 'CPU_ACCEPTANCE_FAILED_NO_REPLAY'
            and exit_['subprocess_exit'] == exit_['exit_code'] == 1
            and exit_['proof_sha256'] == expected and exit_['error'] == prior['error']
            and exit_['elapsed_seconds'] < 5 and exit_['max_rss_kib'] < 1024**2
            and prior['run_id'] == intent['run_id'] == old['run_id'] == actual['run_id'] == config['run_id']
            and prior['outer'] == intent['outer'] == old['outer'] == 50
            and intent['config_sha256'] == old['runtime_config_sha256'] == config['runtime_config_sha256'] == files[str(launch)]['sha256']
            and intent['source_commit'] == old['runtime_source_commit'] == config['runtime_source_commit'] == actual['source_commit']
            and prior['joint_complete_sha256'] == intent['joint_complete_sha256'] == files[str(joint)]['sha256']):
        raise ValueError('Prior failure is not the pinned scheduler preflight with zero payload reads')
    # Check the original birth, without treating a recycled PID as the prior reader.
    stat_path = Path('/proc')/str(start['pid'])/'stat'
    if stat_path.exists():
        current = stat_path.read_text().split(') ',1)[1].split()
        original = start['proc_stat'].split(') ',1)[1].split()
        if current[19] == original[19]:raise ValueError('Prior reader birth is still present')
    return dict(status='PINNED_V1_METADATA_FAILURE_BEFORE_ANY_LARGE_PAYLOAD_VERIFIED',
        failed_proof_sha256=expected, predecessor_helper_commit=BUG_HELPER_COMMIT,
        predecessor_reader_sha256=BUG_READER_SHA256, original_v1_preserved=True,
        prior_large_payload_read=False, files=files)
