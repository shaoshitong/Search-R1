"""CPU-only fixed recovery dependencies; no model loading or training changes.

Reuse the already verified official-model staging alias, resolve canonical data
symlinks, and prove the existing E5 alias against the actual source. Large files
are copied and independently read only on h100-10, after the CP50 copier ends.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

try:
    from .checkpoint_backup_native import (MOUNT, PARENT, RUN, RESERVE, RATE,
        exported, full_read, need, read, record, sha, signature, utc)
except ImportError:  # Published standalone CPU control bundle on the remote hosts.
    from checkpoint_backup_native import (MOUNT, PARENT, RUN, RESERVE, RATE,
        exported, full_read, need, read, record, sha, signature, utc)

STAGE = Path('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3/backup-dependencies/official-model-once-v1')
E5_ALIAS = Path('/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/e5-base-v2')
NAME = RUN[:12] + '-fixed-recovery-assets-once-v1'
PASS = 'INDEPENDENT_FIXED_RECOVERY_ASSETS_FULL_SHA_PASS'


def safe_original(path):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts and
         any(Path(root) in p.parents for root in ('/mnt/data', '/dataset')),
         'Only explicit remote recovery assets are admitted')
    return p


def inventory(base):
    """Runs once on source8; an inventory is not an independent backup."""
    base = Path(base)
    need(socket.gethostname() == 'h100-8' and base.parent == Path('/mnt/data/shaoshitong')
         and base.name.startswith('search-g1-prefix-selection200-actorhidden20-'), 'Wrong source')
    config_path = base/'preparation/launch-config.json'
    c = read(config_path)
    need(c['run_id'] == RUN and c['actor_hidden_native'] and c['probe_learning_rate'] == 1e-5,
         'Use the current actor-hidden/LR authorization')
    rows = {}

    def add(original, alias, role, expected=None):
        original = safe_original(original); alias = Path(alias).resolve()
        need(Path('/mnt/data') in alias.parents and alias.is_file(), 'A real exported file alias is required')
        need(original.is_file() and original.stat().st_size == alias.stat().st_size, 'Alias size differs')
        key = str(original)
        value = dict(original_path=key, exported_source_path=str(alias),
            bytes=alias.stat().st_size, export_mtime_ns=alias.stat().st_mtime_ns,
            expected_sha256=expected, roles=[role])
        if key in rows:
            old = dict(rows[key]); old['roles'] = [role]
            need(old == value, 'Conflicting recovery alias')
            rows[key]['roles'].append(role)
        else:
            rows[key] = value

    stage = read(STAGE/'RESULT.json'); intent = read(STAGE/'INTENT.json')
    need(stage['status'] == 'SAME_HOST_OFFICIAL_MODEL_STAGE_FULL_SHA_PASS'
         and stage['run_id'] == intent['run_id']
         and Path(stage['source_model']).resolve() == Path(c['official_model_dir']).resolve()
         and intent['files_sha256'] == c['model_files_sha256'], 'Existing official staging identity differs')
    seen = set()
    for row in stage['files']:
        name = Path(row['source_path']).name
        need(name not in seen and name in c['model_files_sha256'], 'Duplicate/unknown official asset')
        expected = c['model_files_sha256'][name]
        need(all(row[k] == expected for k in ('expected_sha256', 'source_full_read_sha256', 'staged_full_read_sha256')),
             'Existing same-host staging proof differs from the official file identity')
        need(Path(row['staged_path']).resolve().is_relative_to(STAGE/'model'), 'Official staging path escapes')
        add(row['source_path'], row['staged_path'], 'official_actor_and_fixed_reference', expected)
        seen.add(name)
    need(seen == set(c['model_files_sha256']), 'Official dependency set is incomplete')
    for name in ('RESULT.json', 'INTENT.json', 'LAUNCH.json'):
        add(STAGE/name, STAGE/name, 'existing_same_host_staging_provenance', sha(STAGE/name))

    encoder = Path(c['retrieval_encoder']).resolve()
    encoder_files = sorted(p for p in encoder.rglob('*') if p.is_file())
    need(encoder_files, 'Retrieval encoder has no files')
    for original in encoder_files:
        alias = E5_ALIAS/original.relative_to(encoder)
        # The alias may belong to an older run; equality of directory names is
        # insufficient. Both complete files are actually read, with bounded IO.
        source_count, expected = full_read(original)
        alias_count, alias_sha = full_read(alias)
        need((source_count, expected) == (alias_count, alias_sha), 'E5 export alias bytes differ')
        add(original, alias, 'frozen_retrieval_encoder', expected)

    for key in ('canonical_train', 'validation_questions', 'final_questions',
                'source_question_pool', 'probe_source_checkpoint', 'retrieval_index', 'retrieval_corpus'):
        original = Path(c[key]); expected = c.get(key+'_sha256')
        if key == 'probe_source_checkpoint': expected = c['probe_source_sha256']
        add(original, original.resolve(), key, expected)
    manifest = Path(c['grounding_manifest']); m = read(manifest)
    need(sha(manifest) == c['grounding_manifest_sha256'], 'Grounding version manifest changed')
    add(manifest, manifest.resolve(), 'fixed_grounding_manifest', c['grounding_manifest_sha256'])
    for key in ('commit', 'boundary'):
        original = Path(m[key]); add(original, original.resolve(), 'fixed_grounding_'+key, sha(original))
    archive = m.get('restoration_origin', {}).get('source_archive')
    if archive: add(archive, Path(archive).resolve(), 'original_grounding_manifest_provenance', sha(archive))
    for path in (config_path, base/'preparation/source-files.json'):
        add(path, path, 'frozen_runtime_binding', sha(path))
    return dict(status='FIXED_RECOVERY_ASSETS_INVENTORY_ONLY', run_id=RUN, source_base=str(base),
        source_config_sha256=sha(config_path), files=sorted(rows.values(), key=lambda r:r['original_path']),
        bytes=sum(r['bytes'] for r in rows.values()), source_host='h100-8', created_utc=utc(),
        official_staging_original_run_id=stage['run_id'], official_staging_proof_sha256=sha(STAGE/'RESULT.json'),
        e5_source_and_export_alias_actually_full_read=True, independent_backup_performed=False,
        training_modified=False, model_bytes_transit_Mac=False,
        remaining=['independent source copy and destination full SHA', 'terminal run logs/trajectories/final outputs'])


def admitted_rows(plan):
    need(plan['status'] == 'FIXED_RECOVERY_ASSETS_INVENTORY_ONLY' and plan['run_id'] == RUN, 'Wrong inventory')
    need(plan['files'] and sum(r['bytes'] for r in plan['files']) == plan['bytes'], 'Inventory total differs')
    names = set()
    for row in plan['files']:
        original = safe_original(row['original_path']); source = exported(row['exported_source_path'])
        relative = Path('payload')/original.relative_to('/')
        need(str(relative) not in names and row['bytes'] >= 0, 'Duplicate/invalid recovery payload')
        expected = row['expected_sha256']
        need(expected is None or (len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)), 'Invalid expected SHA')
        names.add(str(relative))
        yield row, source, relative


def run(root):
    root = Path(root)
    need(socket.gethostname() == 'h100-10' and root == PARENT/NAME and root.resolve() == root, 'Wrong independent destination')
    intent = read(root/'INTENT.json')
    need(sha(__file__) == intent['worker_sha256']
         and sha(Path(__file__).with_name('checkpoint_backup_native.py')) == intent['common_sha256'], 'Published backup control changed')
    os.nice(19)
    dependency = PARENT/(RUN[:12]+'-outer-0050-once-v1')
    # Serialize with the existing copy. Waiting is explicit progress, never a
    # second CP50 launch or an increase in aggregate copy bandwidth.
    while not (dependency/'RESULT.json').exists():
        receipt = read(dependency/'RECEIPT.json'); proc = Path('/proc')/str(receipt['pid'])/'stat'
        state = proc.read_text().rsplit(')', 1)[1].split() if proc.exists() else []
        need(state and state[0] != 'Z' and int(state[19]) == receipt['start_ticks'], 'CP50 copier is absent; reconcile its result')
        record(root, 'PROGRESS.json', dict(phase='WAIT_EXISTING_CP50_COPY', dependency=str(dependency), utc=utc()))
        time.sleep(60)
    need(read(dependency/'RESULT.json')['status'] == 'INDEPENDENT_PREFIX200_JOINT_BACKUP_FULL_SHA_PASS', 'CP50 copy did not finish successfully')
    need(sha(exported(intent['inventory_path'])) == intent['inventory_sha256'], 'Pinned source inventory changed')
    plan = read(exported(intent['inventory_path'])); files = list(admitted_rows(plan))
    need(os.statvfs(root).f_bavail*os.statvfs(root).f_frsize >= plan['bytes']+RESERVE, 'Ordinary destination reserve exhausted')
    need(subprocess.check_output(['findmnt','-n','-T',str(root),'-o','FSTYPE'],text=True).strip() == 'ext4', 'Independent local ext4 required')
    staging = root/'checkpoint.partial'; staging.mkdir(); copied = []
    for row, source, relative in files:
        need(source.resolve().is_relative_to(MOUNT.resolve()) and source.stat().st_dev != root.stat().st_dev,
             'NFS source and physical destination must differ')
        need(source.stat().st_size == row['bytes'] and source.stat().st_mtime_ns == row['export_mtime_ns'], 'Pinned fixed source metadata changed')
        need(os.statvfs(root).f_bavail*os.statvfs(root).f_frsize >= row['bytes']+RESERVE, 'Ordinary destination reserve exhausted')
        target = staging/relative; target.parent.mkdir(parents=True, exist_ok=True)
        def progress(count, phase='source_copy'):
            record(root, 'PROGRESS.json', dict(utc=utc(), current_file=str(relative), current_phase=phase,
                current_file_bytes=count, completed_files=len(copied), expected_files=len(files),
                completed_bytes=sum(r['bytes'] for r in copied), expected_bytes=plan['bytes']))
        with target.open('xb') as output:
            count, digest = full_read(source, output, progress); output.flush(); os.fsync(output.fileno())
        need(count == row['bytes'] and (row['expected_sha256'] is None or digest == row['expected_sha256']), 'Actual source SHA differs')
        actual_count, actual_sha = full_read(target, callback=lambda n:progress(n,'independent_target_full_sha'))
        need((actual_count, actual_sha) == (count, digest), 'Independent destination SHA differs')
        copied.append(dict(row, relative_path=str(relative), source_full_sha256=digest, destination_full_sha256=actual_sha))
        progress(count, 'verified')
    need(sha(exported(intent['inventory_path'])) == intent['inventory_sha256'], 'Inventory changed during copy')
    record(staging, 'RESTORE_MAPPING.json', dict(run_id=RUN, mapping={r['original_path']:r['relative_path'] for r in copied}))
    for directory in sorted([staging]+[p for p in staging.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    need(not (root/'checkpoint').exists(), 'Never replace a previous final copy')
    os.rename(staging, root/'checkpoint')
    result=dict(status=PASS, run_id=RUN, completed_utc=utc(), files=copied,
        bytes=sum(r['bytes'] for r in copied), inventory_sha256=intent['inventory_sha256'],
        semantic_reload_performed=False, all_experiment_backup_complete=False,
        model_bytes_transit_Mac=False, training_modified=False,
        remaining=['terminal logs/trajectories/final outputs and later normal checkpoints'])
    record(root, 'RESULT.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory-base');p.add_argument('--root');a=p.parse_args()
    if a.inventory_base: print(json.dumps(inventory(a.inventory_base)))
    else:
        try: run(a.root)
        except BaseException as e:
            record(Path(a.root),'ERROR.json',dict(status='INCOMPLETE_PRESERVE_SOURCE_AND_PARTIAL',error=repr(e),utc=utc()))
            raise
