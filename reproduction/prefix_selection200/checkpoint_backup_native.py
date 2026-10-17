"""Independent joint-state copy on h100-10; all large bytes stay remote.

The source is the existing read-only h100-8 NFS export. The destination must be
h100-10 local ext4. Copy and independent destination SHA reads are limited to
20MiB/s, with128GiB ordinary free space retained. This never loads a model,
uses CUDA, submits a job, changes training, or overwrites an existing intent.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time

MOUNT = Path('/mnt/shared/workspace/training_data_node_8')
PARENT = Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-independent-backup-20261016')
RATE = 20 * 1024**2
RESERVE = 128 * 1024**3
CPU_PASS = 'NATIVE_ACTOR_PERIODIC_PROBE_JOINT_FULL_SHA_CPU_PASS'


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition: raise ValueError(message)


def run_identity(value):
    need(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value) is not None,
         'An explicit complete logical run identity is required')
    return value


def checkpoint_namespace(run_id, outer):
    run_id = run_identity(run_id)
    need(type(outer) is int and outer in (5,50,100,150,200), 'Unexpected checkpoint milestone')
    return PARENT / (run_id[:12]+'-outer-%04d-once-v1'%outer)


def record(root, name, value):
    path = root / name; temporary = path.with_name(name + '.tmp')
    with temporary.open('w') as output:
        json.dump(value, output, sort_keys=True, allow_nan=False)
        output.flush(); os.fsync(output.fileno())
    os.replace(temporary, path)
    fd = os.open(root, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def exported(original):
    original = Path(original)
    need(original.is_absolute() and '..' not in original.parts and Path('/mnt/data') in original.parents,
         'Backup source must be an explicit file on the h100-8 data export')
    return MOUNT / original.relative_to('/mnt/data')


OFFICIAL_STAGE = Path('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3/backup-dependencies/official-model-once-v1')


def exported_dependency(original, config):
    """Resolve a /dataset official asset only through its actual full-SHA stage.

    Preserve its original path in the restore mapping. Never manufacture a
    matching NFS path or treat an arbitrary same-named export as the model.
    """
    original = Path(original)
    if Path('/mnt/data') in original.parents:
        return original, exported(original)
    official = Path(config['official_model_dir'])
    need(official in original.parents and original.parent == official,
         'Only pinned official-model files may use a /dataset export alias')
    name = original.name
    expected = config['model_files_sha256'].get(name)
    need(expected is not None, 'Unrecognized official dependency file')
    result = read(exported(OFFICIAL_STAGE/'RESULT.json'))
    intent = read(exported(OFFICIAL_STAGE/'INTENT.json'))
    need(result['status'] == 'SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS'
         and result['run_id'] == intent['run_id']
         and Path(result['source_model']) == official
         and intent['files_sha256'] == config['model_files_sha256'],
         'Existing official export stage does not bind this exact model')
    matching = [row for row in result['files'] if Path(row['source_path']) == original]
    need(len(matching) == 1, 'Original official file is absent or duplicated in staged proof')
    row = matching[0]; alias = Path(row['staged_path'])
    need(OFFICIAL_STAGE/'model' in alias.parents and '..' not in alias.parts
         and all(row[key] == expected for key in
                 ('expected_sha256','source_full_read_sha256','staged_full_read_sha256')),
         'The actual official export alias differs from its full-read proof')
    return alias, exported(alias)


def signature(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def full_read(path, target=None, callback=None, rate=RATE):
    before = signature(path); total = 0; digest = hashlib.sha256()
    began = last = time.monotonic()
    with Path(path).open('rb') as source:
        while True:
            block = source.read(1024**2)
            if not block: break
            if target is not None: target.write(block)
            digest.update(block); total += len(block)
            time.sleep(max(0, total / rate - (time.monotonic() - began)))
            if callback and time.monotonic() - last >= 10:
                callback(total); last = time.monotonic()
    need(signature(path) == before and total == before[2], 'Source/target changed during its actual full read')
    if callback: callback(total)
    return total, digest.hexdigest()


def collect_hashes(value, result):
    if isinstance(value, dict):
        if all(k in value for k in ('path', 'bytes', 'sha256')):
            key = str(value['path']); item = dict(bytes=value['bytes'], sha256=value['sha256'])
            need(key not in result or result[key] == item, 'Conflicting independent file proofs')
            result[key] = item
        for item in value.values(): collect_hashes(item, result)
    elif isinstance(value, list):
        for item in value: collect_hashes(item, result)


def inventory(intent):
    source = Path(intent['source_base']); outer = intent['outer']; run_id = run_identity(intent['run_id'])
    need(source.parent == Path('/mnt/data/shaoshitong') and source.name.startswith('search-g1-prefix-selection200-'),
         'Only this formal experiment can be copied')
    checkpoint_namespace(run_id, outer)
    proof_path = exported(intent['cpu_proof'])
    need(sha(proof_path) == intent['cpu_proof_sha256'], 'Actual independent CPU proof changed')
    proof = read(proof_path)
    need(proof['status'] == CPU_PASS and proof['run_id'] == run_id and proof['outer'] == outer
         and all(row.get('independent_full_sha') for row in proof['files']),
         'Only a fully read-back complete joint checkpoint is admitted')
    config = read(exported(source / 'preparation/launch-config.json'))
    need(config['base'] == str(source), 'Source config belongs to a different namespace')
    point = source / 'outers' / ('outer-%06d' % outer)
    joint_path = point / 'complete.json'; joint = read(exported(joint_path))
    need(config['run_id'] == joint['run_id'] == run_id and config['actor_hidden_native']
         and proof['actor_actual_adam_steps_per_rank']==[20*outer]*8
         and proof['probe_actual_adam_steps_per_rank']==[100*(outer//20)]*8
         and joint['outer'] == outer and joint['actor']['actual_adam_steps_per_rank'] == [20*outer]*8
         and joint['probe']['actual_adam_steps_per_rank'] == [100*(outer//20)]*8
         and joint['probe']['completed_outer']==outer//20*20
         and joint['actor']['scheduler_completed_outer'] == outer, 'Actual saved joint clocks differ')
    expected = {}; collect_hashes(proof['files'], expected)
    independently_read_source_paths = set(expected)
    collect_hashes(proof.get('grounding_lifecycle', {}).get('assets', []), expected)
    collect_hashes(joint, expected)
    collect_hashes(read(exported(point/'prefix/complete.json')), expected)
    scan = point/'prefix/scan'
    for name, producer in read(exported(scan/'scan-complete.json'))['files'].items():
        collect_hashes(dict(producer, path=str(scan/name)), expected)
    collect_hashes(dict(joint['probe'], path=joint['probe_snapshot']), expected)
    # Native hidden tensors are consumed derived caches, pruned after joint
    # commit. They are not checkpoint dependencies and must not race pruning.
    expected={k:v for k,v in expected.items() if not Path(k).name.endswith('-actor-hidden.npy')}
    originals = set(expected) | {str(joint_path), str(source/'actor/latest.json'),
        str(source/'probe/initialization-complete.json'), str(source/'probe/latest.json'),
        str(joint['probe_snapshot']),
        str(source/'preparation/launch-config.json'), str(source/'preparation/source-files.json'),
        str(source/'question-pool/binding.json'), str(joint['pool_snapshot']), str(intent['cpu_proof'])}
    # The immutable milestone probe is authoritative; source latest may already
    # have advanced. Never copy a mutable live latest tensor/receipt as milestone.
    originals.discard(str(source/'probe/latest.json'))
    originals.discard(str(source/'actor/latest.json'))
    for original in (Path(joint['actor']['state_path']), point):
        actual = exported(original)
        need(actual.is_dir(), 'Complete saved actor/outer directory is absent')
        for path in actual.rglob('*'):
            relative = path.relative_to(actual)
            if 'validation' in relative.parts or '__pycache__' in relative.parts: continue
            need(not path.is_symlink(), 'Do not follow a symlink in the immutable source bundle')
            if path.is_file() and not path.name.endswith('-actor-hidden.npy'):
                need(not path.name.endswith(('.partial', '.tmp')), 'Saved bundle has an unfinished file')
                originals.add(str(original / relative))
    assets = []
    for name in sorted(originals):
        original = Path(name); alias, actual = exported_dependency(original, config)
        need(actual.is_file() and not actual.is_symlink(), 'Required checkpoint file is absent: '+name)
        info = actual.stat(); independent = expected.get(name)
        if independent: need(info.st_size == independent['bytes'], 'Independent source file size changed')
        relative = (original.relative_to(source) if source in original.parents else
                    Path('external') / original.relative_to('/'))
        assets.append(dict(original_path=name, source_export_alias=str(alias), source_path=str(actual), relative_path=str(relative),
            bytes=info.st_size, expected_sha256=independent['sha256'] if independent else None,
            expected_sha_origin=('independent_source_CPU_full_file_readback' if name in independently_read_source_paths else
                'native_source_producer_receipt_not_independent_readback' if independent else
                'no_prior_digest_actual_copy_and_target_full_reads_required')))
    need(len({row['relative_path'] for row in assets}) == len(assets), 'Backup destination paths collide')
    return assets, proof


def run(root):
    root = Path(root)
    need(socket.gethostname() == 'h100-10' and root.parent == PARENT and root.resolve() == root,
         'Run this copy only on independent h100-10 local storage')
    intent = read(root/'INTENT.json'); run_id = run_identity(intent['run_id'])
    need(root == checkpoint_namespace(run_id, intent['outer']), 'Namespace differs from pinned run/milestone')
    need(sha(__file__) == intent['code_sha256'], 'Backup code changed after exclusive intent')
    need(not (root/'RESULT.json').exists() and not (root/'checkpoint.partial').exists(),
         'Existing result or partial requires reconciliation, never blind replay')
    source_mount = subprocess.check_output(['findmnt','-n','-T',str(exported(intent['source_base'])),'-o','SOURCE'],text=True)
    destination_fs = subprocess.check_output(['findmnt','-n','-T',str(root),'-o','FSTYPE'],text=True).strip()
    need('10.32.0.8:/mnt/data' in source_mount and destination_fs == 'ext4'
         and root.stat().st_dev != exported(intent['source_base']).stat().st_dev,
         'NFS visibility alone is not an independent physical backup')
    os.nice(19)
    result = dict(run_id=run_id, outer=intent['outer'], pid=os.getpid(), started_utc=utc(),
        source_base=intent['source_base'], source_cpu_proof_sha256=intent['cpu_proof_sha256'],
        code_sha256=intent['code_sha256'], code_commit=intent['code_commit'],
        semantic_reload_performed=False, training_modified=False, model_bytes_transit_Mac=False,
        all_experiment_backup_complete=False, bytes_per_second_limit=RATE,
        actor_hidden_periodic_probe=True,probe_actual_adam_per_rank=100*(intent['outer']//20),
        consumed_derived_native_hidden_excluded=True)
    try:
        files, proof = inventory(intent); total = sum(row['bytes'] for row in files)
        fs = os.statvfs(root); need(fs.f_bavail*fs.f_frsize >= total+RESERVE, 'Ordinary destination space is insufficient')
        record(root,'MANIFEST.json',dict(run_id=run_id,outer=intent['outer'],expected_files=len(files),expected_bytes=total,files=files))
        staging = root/'checkpoint.partial'; staging.mkdir(); copied=[]
        for row in files:
            fs=os.statvfs(root); need(fs.f_bavail*fs.f_frsize >= row['bytes']+RESERVE, 'Ordinary free reserve exhausted')
            source=Path(row['source_path']); target=staging/row['relative_path']; target.parent.mkdir(parents=True,exist_ok=True)
            def progress(count, phase='copy'):
                record(root,'PROGRESS.json',dict(utc=utc(),pid=os.getpid(),completed_files=len(copied),
                    completed_bytes=sum(p['bytes'] for p in copied),expected_files=len(files),expected_bytes=total,
                    current_file=row['relative_path'],current_phase=phase,current_file_bytes=count))
            with target.open('xb') as output:
                count,digest=full_read(source,output,progress); output.flush();os.fsync(output.fileno())
            need(count==row['bytes'] and (row['expected_sha256'] is None or digest==row['expected_sha256']),
                 'Actual source full SHA differs from its independent checkpoint proof')
            read_count,read_sha=full_read(target,callback=lambda n:progress(n,'independent_target_full_sha'))
            need((read_count,read_sha)==(count,digest),'Independent destination full SHA differs')
            copied.append(dict(row,source_full_sha256=digest,destination_full_sha256=read_sha));progress(count,'verified')
        need(sha(exported(intent['cpu_proof']))==intent['cpu_proof_sha256'],'Source proof changed during copy')
        record(staging,'RESTORE_MAPPING.json',dict(run_id=run_id,outer=intent['outer'],
            original_absolute_paths_preserved_in_receipts=True,
            mapping={row['original_path']:row['relative_path'] for row in copied},
            note='Restore helpers must resolve recorded source paths through this mapping; original receipts are unchanged.'))
        mapping_proof=dict(relative_path='RESTORE_MAPPING.json',bytes=(staging/'RESTORE_MAPPING.json').stat().st_size,
            sha256=sha(staging/'RESTORE_MAPPING.json'),actual_full_target_read=True)
        for directory in sorted([staging]+[p for p in staging.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
            fd=os.open(directory,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        final=root/'checkpoint';need(not final.exists(),'Never replace an existing final copy');os.rename(staging,final)
        result.update(status='INDEPENDENT_PREFIX200_JOINT_BACKUP_FULL_SHA_PASS',completed_utc=utc(),
            files=copied,bytes=total,destination=str(final),independent_destination_mount='h100-10 local ext4')
        result['generated_restore_mapping']=mapping_proof
        record(root,'RESULT.json',result);return result
    except BaseException as error:
        result.update(status='FAILED_PRESERVE_SOURCE_AND_PARTIAL',error=repr(error),completed_utc=utc())
        record(root,'RESULT.json',result);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--namespace',required=True)
    args=parser.parse_args();print(json.dumps(run(args.namespace),sort_keys=True))
