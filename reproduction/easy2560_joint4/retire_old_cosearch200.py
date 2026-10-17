"""Retire exactly25 old FAILED713379 HPC tensors; preserve the full HK33.

Standalone CPU/stdlib operator. No training, model reload, Slurm mutation,
watcher modification, marker creation or recursive deletion is implemented.
The two fresh33 SHA workers are independent and share an explicit operation ID.
Only original-owner MOU can unlink the fixed25, after both UID reader checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time


BASE = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011')
CP = BASE/'checkpoints/global_step_200'
ARCHIVE = Path('/backup/shaoshitong/cosearch-instruct200-common-em-20261011/checkpoints/global_step_200')
HPC_AUDIT = BASE/'preparation/cp200-tensor-retirement'
HK_AUDIT = ARCHIVE.parent.parent/'preparation/cp200-tensor-retirement'
SOURCE_JOB = '713379'
SOURCE_UID = 21930
OTHER_UID = 14090
HK_UID = 1002
HPC_HOST = 'ACD-Login-1'
HK_HOST = 'xielab3'
CANDIDATE_SHA = 'd29f22692fffb884a34fb9fc5841c882831c2d7b335e893822879c67b0e68b8c'
SOURCE_RECEIPT_SHA = '620b6fe6d9485ec8652903f49ee6e7bd6c09675932a702c7d6f735db77ce736d'
ARCHIVE_COPY_SHA = '77bd3f89a1bff32dae765b731791ba0fc088c043fd269e2c95eb3ff4d87ee41f'
HK_RECEIPT_SHA = 'e50fde1e701024a9664740bcb4a20d1d04cff108196434d0b7de72678be67fd8'
# Published d45 HK worker predates only the source-side preserved-file fix.
# Its HK SHA path is AST-checked unchanged; never accept this for source proof.
LEGACY_HK_OPERATOR_SHA = '8015dea69062d69cc06760f2177d335ac85f6e3a137736f7d2504a117a998927'
TOTAL_BYTES = 76555489172
TARGET_BYTES = 76555486380
ROLES = ('actor', 'reranker_actor_rollout')
TARGET_NAMES = tuple(sorted(['data.pt'] + [role+'/'+kind+'_world_size_4_rank_'+str(rank)+'.pt'
    for role in ROLES for kind in ('model', 'optim', 'extra_state') for rank in range(4)]))
RECOVERY_NAMES = tuple(sorted(role+'/recovery-state-rank-'+str(rank)+'.json'
                              for role in ROLES for rank in range(4)))
CORE_NAMES = tuple(sorted(TARGET_NAMES + RECOVERY_NAMES))
ROOT_METADATA = ('checkpoint-receipt-200.json', 'archive-verified-200.json',
                 'data-recovery-200.json', 'dependent-final-audit/state.json', 'controller-state.json')
CP_METADATA = ('full-sha-receipt.json', 'actor/fsdp_config.json', 'reranker_actor_rollout/fsdp_config.json')


def need(ok, message):
    if not ok: raise ValueError('COSEARCH200_RETIRE_' + message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(raw): return hashlib.sha256(raw).hexdigest()


def no_links(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and
         not any(p.is_symlink() for p in (path, *path.parents)), 'SYMLINK_OR_NONCANONICAL_PATH')
    return path


def identity(path):
    path = no_links(path); s = path.lstat()
    need(stat.S_ISREG(s.st_mode), 'REGULAR_FILE_REQUIRED')
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                allocated_bytes=s.st_blocks*512, uid=s.st_uid, links=s.st_nlink)


def small_raw(path):
    """Authenticate content, allowing a live writer to replace identical JSON."""
    path = no_links(path); before = identity(path)
    need(before['bytes'] <= 2*1024**2, 'SMALL_RECEIPT_REQUIRED')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        first = os.fstat(fd); raw = b''
        while True:
            part = os.read(fd, 1024**2)
            if not part: break
            raw += part
        last = os.fstat(fd)
        need((first.st_dev, first.st_ino, first.st_size, first.st_mtime_ns) ==
             (last.st_dev, last.st_ino, last.st_size, last.st_mtime_ns), 'RECEIPT_CHANGED_DURING_READ')
    finally: os.close(fd)
    return raw, digest(raw)


def small(path):
    raw, sha = small_raw(path)
    return json.loads(raw), sha


def fsync_dir(path):
    fd = os.open(no_links(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)


def write(path, value, replace=False):
    path = no_links(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial-'+str(os.getpid())) if replace else path
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f: f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    if replace: os.replace(temporary, path)
    fsync_dir(path.parent)


def birth(pid=None):
    pid = os.getpid() if pid is None else pid
    proc = Path('/proc')/str(pid); fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(host=socket.gethostname(), pid=pid, start_ticks=int(fields[19]), uid=proc.stat().st_uid,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        pid_namespace=os.readlink(proc/'ns/pid'))


def context(site, reader=False):
    if site == 'hk': ok = socket.gethostname() == HK_HOST and os.getuid() == HK_UID
    else: ok = socket.gethostname() == HPC_HOST and os.getuid() in (
        (SOURCE_UID, OTHER_UID) if reader else (SOURCE_UID,))
    need(ok, 'EXACT_' + site.upper() + '_OWNER_HOST_REQUIRED')


def audit_root(value, operation_id, site):
    need(isinstance(operation_id, str) and re.fullmatch('[0-9a-f]{32}', operation_id), 'EXPLICIT_OPERATION_ID_REQUIRED')
    context(site); root = no_links(value)
    need(root == (HK_AUDIT if site == 'hk' else HPC_AUDIT)/operation_id, 'OWNED_AUDIT_ROOT_REQUIRED')
    root.mkdir(parents=True, exist_ok=True)
    need(root.stat().st_uid == os.getuid(), 'AUDIT_ROOT_OWNER_DIFFERS')
    return root


def load_plan(path):
    candidate, sha = small(path); need(sha == CANDIDATE_SHA, 'FIXED_CANDIDATE_SHA_REQUIRED')
    need(candidate['base'] == str(BASE) and candidate['source_job'] == SOURCE_JOB and
        candidate['source_job_state'] == 'FAILED' and candidate['exit_code'] == '1:0' and
        candidate['checkpoint_step'] == 200, 'EXACT_FAILED200_SCOPE_REQUIRED')
    receipt = candidate['full_checkpoint_receipt']
    need(receipt['files'] == 33 and receipt['bytes'] == TOTAL_BYTES and
        receipt['source_receipt_sha256'] == SOURCE_RECEIPT_SHA and
        receipt['hk_receipt_copy_sha256'] == ARCHIVE_COPY_SHA and
        receipt['archive_directory'] == str(ARCHIVE), 'FIXED_CORE33_RECEIPT_BINDING_REQUIRED')
    target = candidate['conservative_tensor_only_candidate']; rows = target['source_paths'] + candidate['preserve_small_recovery_proofs']
    need(target['files'] == 25 and target['bytes'] == TARGET_BYTES and len(rows) == 33 and
        {r['relative_path'] for r in target['source_paths']} == set(TARGET_NAMES) and
        {r['relative_path'] for r in candidate['preserve_small_recovery_proofs']} == set(RECOVERY_NAMES),
        'EXACT25_TARGETS_AND8_RECOVERY_REQUIRED')
    core = {}
    for row in rows:
        name = row['relative_path']
        need(name not in core and row['source_path'] == str(CP/name) and row['archive_path'] == str(ARCHIVE/name) and
            row['source_uid'] == SOURCE_UID and type(row['bytes']) is int and row['bytes'] > 0 and
            re.fullmatch('[0-9a-f]{64}', row['historical_receipt_sha256']), 'EXACT_CORE_PATH_OR_SHA_DIFFERS')
        core[name] = dict(bytes=row['bytes'], sha256=row['historical_receipt_sha256'],
                          source_mtime_ns=row['source_mtime_ns'])
    need(tuple(sorted(core)) == CORE_NAMES and sum(r['bytes'] for r in core.values()) == TOTAL_BYTES and
         sum(core[n]['bytes'] for n in TARGET_NAMES) == TARGET_BYTES, 'CORE33_TOTAL_OR_NAMES_DIFFER')
    return dict(candidate_sha256=sha, base=str(BASE), source_job=SOURCE_JOB, checkpoint_step=200,
                archive=str(ARCHIVE), targets=list(TARGET_NAMES), core=core)


def receipt_map(value, plan, status):
    need(value['status'] == status and value['step'] == 200 and len(value['files']) == 33,
         'HISTORICAL_FULL33_RECEIPT_REQUIRED')
    rows = {}
    for row in value['files']:
        name = str(Path(row['path']).relative_to(CP))
        need(name in plan['core'] and name not in rows and
            (row['bytes'], row['sha256']) == (plan['core'][name]['bytes'], plan['core'][name]['sha256']),
            'HISTORICAL_CORE33_MAP_DIFFERS')
        rows[name] = dict(bytes=row['bytes'], sha256=row['sha256'])
    need(tuple(sorted(rows)) == CORE_NAMES, 'HISTORICAL_CORE33_NAMES_DIFFER')
    return rows


def source_receipts(plan):
    source, sha = small(BASE/'checkpoint-receipt-200.json'); need(sha == SOURCE_RECEIPT_SHA, 'SOURCE_RECEIPT_CONTENT_SHA_DIFFERS')
    archived, sha = small(BASE/'archive-verified-200.json'); need(sha == ARCHIVE_COPY_SHA, 'ARCHIVE_COPY_CONTENT_SHA_DIFFERS')
    first = receipt_map(source, plan, 'BOTH_ROLE_FULL_CHECKPOINT_FILES_SHA_PASS')
    need(receipt_map(archived, plan, 'HKUST3_ALL_CHECKPOINT_SHA_PASS') == first, 'HISTORY_MAPS_DIFFER')
    # These live metadata writers are retained. Their inode/mtime are deliberately
    # not pinned across a long SHA pass, and this operator never writes them.
    for name in ROOT_METADATA: need((BASE/name).is_file() and not (BASE/name).is_symlink(), 'RETAIN_ROOT_METADATA_REQUIRED')
    for name in CP_METADATA: need((CP/name).is_file() and not (CP/name).is_symlink(), 'RETAIN_CP_METADATA_REQUIRED')
    return dict(source_receipt_sha256=SOURCE_RECEIPT_SHA, archive_copy_sha256=ARCHIVE_COPY_SHA)


def terminal(run_command=subprocess.run):
    records = {}
    for label, argv in (
        ('queue', ['squeue', '--noheader', '--jobs', SOURCE_JOB, '--format=%i|%T']),
        ('accounting', ['sacct', '--noheader', '--parsable2', '--jobs', SOURCE_JOB,
                        '--format=JobIDRaw,State,ExitCode,End'])):
        r = run_command(argv, capture_output=True, text=True, timeout=30)
        records[label] = dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
    need(records['queue']['returncode'] in (0, 1) and not records['queue']['stdout'].strip(), 'OLD_JOB_STILL_QUEUED_OR_UNKNOWN')
    need(records['accounting']['returncode'] == 0, 'OLD_JOB_ACCOUNTING_UNKNOWN')
    rows = [line.strip().split('|') for line in records['accounting']['stdout'].splitlines() if line.strip()]
    main = [r for r in rows if r[0] == SOURCE_JOB]
    need(len(main) == 1 and len(main[0]) == 4 and main[0][1] == 'FAILED' and main[0][2] == '1:0' and
         main[0][3] not in ('', 'Unknown', 'None', 'N/A'), 'EXACT_FAILED_MAIN_JOB_REQUIRED')
    need(any(len(r) == 4 and r[:3] == [SOURCE_JOB+'.extern', 'COMPLETED', '0:0'] and
        r[3] not in ('', 'Unknown', 'None', 'N/A') for r in rows), 'OLD_EXTERN_NOT_CLOSED')
    return records


def progress(root, **values): write(root/'progress.json', dict(values, checked_unix=time.time()), replace=True)


def measured_sha(path, uid, root, phase, done, total, completed):
    before = identity(path); need(before['uid'] == uid and before['links'] == 1, 'PAYLOAD_OWNER_OR_HARDLINK_DIFFERS')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW); h = hashlib.sha256(); count = 0; last = time.monotonic()
    try:
        st = os.fstat(fd); need((st.st_dev, st.st_ino) == (before['device'], before['inode']), 'PAYLOAD_REPLACED_AT_OPEN')
        while True:
            block = os.read(fd, 8*1024**2)
            if not block: break
            h.update(block); count += len(block)
            if time.monotonic()-last >= 2:
                progress(root, phase=phase, hashed_bytes=done+count, total_bytes=total,
                         completed_files=completed, current_file=str(path), current_file_bytes=count)
                last = time.monotonic()
        st = os.fstat(fd)
        need((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink) ==
             (before['device'], before['inode'], before['bytes'], before['mtime_ns'], 1) and
             count == before['bytes'] and identity(path) == before, 'PAYLOAD_CHANGED_DURING_SHA')
    finally: os.close(fd)
    return dict(identity=before, sha256=h.hexdigest(), bytes=count)


def checkpoint_tree():
    """CP-only metadata walk; large preserved assets are never content-read."""
    no_links(CP); files = {}; directory_bytes = 0
    for directory, dirs, names in os.walk(CP, followlinks=False):
        directory = no_links(directory); directory_bytes += directory.stat().st_blocks*512
        for name in dirs: no_links(directory/name)
        for name in names:
            path = directory/name; rel = str(path.relative_to(CP)); record = identity(path)
            if rel not in TARGET_NAMES:
                if record['bytes'] <= 2*1024**2:
                    record['sha256'] = small_raw(path)[1]
                else:
                    record['content_sha256_read'] = False
            files[rel] = record
    return dict(files=files, allocated_bytes=directory_bytes+sum(r['allocated_bytes'] for r in files.values()))


def verify_kept(before, after):
    expected = {name:r for name,r in before['files'].items() if name not in TARGET_NAMES}
    actual = {name:r for name,r in after['files'].items() if name not in TARGET_NAMES}
    need(expected == actual, 'NONTARGET_METADATA_CHANGED_OR_MISSING')


def validate(plan, root, operation_id, site):
    context(site); done = 0; files = {}; started = time.time(); before_terminal = None
    if site == 'hk':
        receipt, sha = small(ARCHIVE/'hkust3-archive-receipt.json')
        need(sha == HK_RECEIPT_SHA, 'HK_COMPLETE_RECEIPT_SHA_DIFFERS')
        receipt_map(receipt, plan, 'HKUST3_ALL_CHECKPOINT_SHA_PASS')
        directory = ARCHIVE; uid = HK_UID; kept = None
    else:
        before_terminal = terminal(); source_receipts(plan); directory = CP; uid = SOURCE_UID; kept = checkpoint_tree()
    for name in CORE_NAMES:
        row = measured_sha(directory/name, uid, root, site.upper()+'_FRESH33_SHA', done, TOTAL_BYTES, len(files))
        need((row['bytes'], row['sha256']) == (plan['core'][name]['bytes'], plan['core'][name]['sha256']), 'FRESH_CORE33_SHA_DIFFERS')
        if site != 'hk': need(row['identity']['mtime_ns'] == plan['core'][name]['source_mtime_ns'], 'CANDIDATE_SOURCE_MTIME_DIFFERS')
        files[name] = row; done += row['bytes']
        progress(root, phase=site.upper()+'_FRESH33_SHA', hashed_bytes=done, total_bytes=TOTAL_BYTES, completed_files=len(files))
    if site != 'hk':
        source_receipts(plan); verify_kept(kept, checkpoint_tree()); after_terminal = terminal()
    else:
        _, sha = small(ARCHIVE/'hkust3-archive-receipt.json'); need(sha == HK_RECEIPT_SHA, 'HK_COMPLETE_RECEIPT_CHANGED')
        after_terminal = None
    return dict(status='ALL33_'+site.upper()+'_FRESH_SHA_PASS', operation_id=operation_id,
        candidate_sha256=CANDIDATE_SHA, files=files, hashed_bytes=done, started_unix=started, checked_unix=time.time(),
        process_identity=birth(), preserved_checkpoint_tree=kept, terminal_before=before_terminal,
        terminal_after=after_terminal, operator_source_sha256=digest(Path(__file__).read_bytes()))


def readers(plan, proc_root=Path('/proc')):
    context('hpc', reader=True); targets = {str(CP/name) for name in TARGET_NAMES}; found = []; inaccessible = []; scanned = 0
    for proc in proc_root.glob('[0-9]*'):
        try:
            if proc.stat().st_uid != os.getuid(): continue
            scanned += 1; matches = []
            for fd in (proc/'fd').iterdir():
                try: target = os.readlink(fd).removesuffix(' (deleted)')
                except FileNotFoundError: continue
                if target in targets: matches.append(target)
            for line in (proc/'maps').read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].removesuffix(' (deleted)') in targets: matches.append(parts[5])
            if matches: found.append(dict(pid=int(proc.name), matches=sorted(set(matches))))
        except (FileNotFoundError, ProcessLookupError): pass
        except (OSError, ValueError) as error: inaccessible.append(dict(pid=int(proc.name), error=str(error)))
    return dict(status='OWN_UID_READER_SCAN', operation_id=None, candidate_sha256=CANDIDATE_SHA,
        host=socket.gethostname(), uid=os.getuid(), process_identity=birth(), checked_unix=time.time(),
        scanned_own_processes=scanned, readers=found, inaccessible=inaccessible,
        coverage_scope='own UID processes visible in this pinned login host PID namespace; not a cluster-wide assertion')


def proof_bind(plan, proof, operation_id, site):
    need(proof['status'] == 'ALL33_'+site.upper()+'_FRESH_SHA_PASS' and
        proof['candidate_sha256'] == CANDIDATE_SHA and proof['operation_id'] == operation_id and
        tuple(sorted(proof['files'])) == CORE_NAMES and proof['hashed_bytes'] == TOTAL_BYTES, 'FRESH33_BOUND_PROOF_REQUIRED')
    actor = proof['process_identity']
    accepted_sha = {digest(Path(__file__).read_bytes())}
    if site == 'hk': accepted_sha.add(LEGACY_HK_OPERATOR_SHA)
    need(actor['host'] == (HK_HOST if site == 'hk' else HPC_HOST) and
         actor['uid'] == (HK_UID if site == 'hk' else SOURCE_UID) and
         proof['operator_source_sha256'] in accepted_sha, 'FRESH_PROOF_OWNER_HOST_DIFFERS')
    for name in CORE_NAMES:
        row = proof['files'][name]
        need((row['bytes'], row['sha256']) == (plan['core'][name]['bytes'], plan['core'][name]['sha256']) and
            row['identity']['links'] == 1 and row['identity']['bytes'] == row['bytes'] and
            row['identity']['uid'] == (HK_UID if site == 'hk' else SOURCE_UID), 'FRESH_PROOF_CORE_MAP_DIFFERS')


def reader_proofs(guards, operation_id, source_birth):
    for uid in (SOURCE_UID, OTHER_UID):
        g = guards[str(uid)]
        need(g['status'] == 'OWN_UID_READER_SCAN' and g['host'] == HPC_HOST and g['uid'] == uid and
            g['candidate_sha256'] == CANDIDATE_SHA and g['operation_id'] == operation_id and
            g['process_identity']['host'] == HPC_HOST and g['process_identity']['uid'] == uid and
            all(g['process_identity'][key] == source_birth[key] for key in ('boot_id', 'pid_namespace')) and
            not g['readers'] and not g['inaccessible'] and g['scanned_own_processes'] >= 1 and
            0 <= time.time()-g['checked_unix'] <= 300,
            'FRESH_BOTH_UID_READER_PROOFS_REQUIRED')


def journal(root, record):
    path = root/'actions.jsonl'; fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    try:
        raw = canonical(record); need(os.write(fd, raw) == len(raw), 'SHORT_DURABLE_JOURNAL_WRITE'); os.fsync(fd)
    finally: os.close(fd)
    fsync_dir(root)


def journal_summary(root):
    path = root/'actions.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    confirmed = [row for row in rows if row['stage'] == 'UNLINKED']
    need(len({r['relative_path'] for r in confirmed}) == len(confirmed), 'DUPLICATE_UNLINK_RECEIPT')
    return dict(confirmed_unlinked_files=[r['relative_path'] for r in confirmed],
        confirmed_unlinked_allocated_bytes=sum(r['before']['allocated_bytes'] for r in confirmed),
        confirmed_unlinked_logical_bytes=sum(r['before']['bytes'] for r in confirmed))


def execute(plan, source, archive, guards, root, operation_id):
    if (root/'deletion-claim.json').exists() or (root/'deletion-claim.json').is_symlink():
        raise FileExistsError('COSEARCH200_RETIRE_EXISTING_DELETION_CLAIM_NO_AUTOMATIC_RETRY')
    context('hpc'); proof_bind(plan, source, operation_id, 'source'); proof_bind(plan, archive, operation_id, 'hk')
    reader_proofs(guards, operation_id, source['process_identity']); source_receipts(plan); current_terminal = terminal()
    current_readers = readers(plan)
    need(not current_readers['readers'] and not current_readers['inaccessible'], 'ACTIVE_OR_UNKNOWN_OWN_UID_READER')
    before_tree = checkpoint_tree(); verify_kept(source['preserved_checkpoint_tree'], before_tree)
    need(tuple(sorted(before_tree['files'])) == tuple(sorted(source['preserved_checkpoint_tree']['files'])), 'CHECKPOINT_FILE_SET_CHANGED')
    # Authenticate all33 identities before the first unlink, including the eight
    # recovery JSON files which stay in place. Live ROOT metadata is SHA-only.
    for name in CORE_NAMES: need(identity(CP/name) == source['files'][name]['identity'], 'SOURCE_IDENTITY_CHANGED_BEFORE_UNLINK')
    claim = dict(operation_id=operation_id, candidate_sha256=CANDIDATE_SHA, source_job=SOURCE_JOB,
        targets=list(TARGET_NAMES), process_identity=birth(), terminal=current_terminal, reader_proofs=guards,
        created_unix=time.time(), authorization='human authorized old failed experiment checkpoints; exact25 only; full HK33 retained')
    write(root/'deletion-claim.json', claim)
    try:
        for name in TARGET_NAMES:
            path = CP/name; before = identity(path)
            need(before == source['files'][name]['identity'] and before['uid'] == SOURCE_UID and before['links'] == 1,
                 'TARGET_CHANGED_BEFORE_PREPARED')
            journal(root, dict(stage='PREPARED', relative_path=name, before=before, checked_unix=time.time()))
            parent = no_links(path.parent); fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                expected = parent.stat(); actual = os.fstat(fd)
                need((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'TARGET_PARENT_REPLACED')
                actual = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
                need(stat.S_ISREG(actual.st_mode) and (actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns,
                    actual.st_nlink, actual.st_uid, actual.st_blocks*512) ==
                    (before['device'], before['inode'], before['bytes'], before['mtime_ns'], 1, SOURCE_UID, before['allocated_bytes']),
                    'TARGET_REPLACED_BEFORE_UNLINK')
                os.unlink(path.name, dir_fd=fd); os.fsync(fd)
            finally: os.close(fd)
            need(not path.exists() and not path.is_symlink(), 'UNLINKED_TARGET_REAPPEARED')
            journal(root, dict(stage='UNLINKED', relative_path=name, before=before, checked_unix=time.time()))
            progress(root, phase='EXACT25_UNLINK', **journal_summary(root))
        after_tree = checkpoint_tree(); verify_kept(before_tree, after_tree)
        need(not set(TARGET_NAMES) & set(after_tree['files']), 'TARGET_STILL_PRESENT')
        result = dict(status='EXACT25_UNLINKED_PENDING_INDEPENDENT_POSTCHECKS', operation_id=operation_id,
            candidate_sha256=CANDIDATE_SHA, source_job=SOURCE_JOB, checkpoint_step=200, **journal_summary(root),
            checkpoint_tree_allocated_before_bytes=before_tree['allocated_bytes'],
            checkpoint_tree_allocated_after_bytes=after_tree['allocated_bytes'],
            observed_checkpoint_tree_allocated_delta=before_tree['allocated_bytes']-after_tree['allocated_bytes'],
            preserved_checkpoint_tree=after_tree, archive_proof=archive,
            all_hk33_retained_required=True, filesystem_quota_free_bytes_measured=False, checked_unix=time.time())
        write(root/'deletion-result.json', result); return result
    except BaseException as error:
        failure = dict(status='PARTIAL_OR_FAILED_NO_COMPLETION_CLAIM', operation_id=operation_id,
            candidate_sha256=CANDIDATE_SHA, **journal_summary(root), error_type=type(error).__name__,
            error=str(error), checked_unix=time.time(), no_automatic_retry=True,
            note='PREPARED without UNLINKED is not counted as confirmed freed space')
        write(root/'deletion-failure.json', failure); raise


def verify_after(plan, result, operation_id, site):
    context('hk' if site == 'hk' else 'hpc', reader=site != 'hk')
    need(result['status'] == 'EXACT25_UNLINKED_PENDING_INDEPENDENT_POSTCHECKS' and
        result['operation_id'] == operation_id and result['candidate_sha256'] == CANDIDATE_SHA and
        tuple(sorted(result['confirmed_unlinked_files'])) == TARGET_NAMES, 'BOUND_UNLINK_RESULT_REQUIRED')
    if site == 'hk':
        archive = result['archive_proof']; proof_bind(plan, archive, operation_id, 'hk')
        _, sha = small(ARCHIVE/'hkust3-archive-receipt.json'); need(sha == HK_RECEIPT_SHA, 'RETAIN_HK_COMPLETE_RECEIPT_REQUIRED')
        for name in CORE_NAMES: need(identity(ARCHIVE/name) == archive['files'][name]['identity'], 'RETAINED_HK33_IDENTITY_CHANGED')
        status = 'HK33_RETAINED_AFTER_UNLINK'
    else:
        source_receipts(plan); tree = checkpoint_tree(); verify_kept(result['preserved_checkpoint_tree'], tree)
        need(not set(TARGET_NAMES) & set(tree['files']) and tuple(sorted(tree['files'])) ==
             tuple(sorted(result['preserved_checkpoint_tree']['files'])), 'POST_UNLINK_TARGET_OR_METADATA_DIFFERS')
        status = 'HPC25_ABSENT_ALL_OTHER_CP_METADATA_RETAINED'
    return dict(status=status, operation_id=operation_id, candidate_sha256=CANDIDATE_SHA,
        deletion_result_sha256=digest(canonical(result)), process_identity=birth(), checked_unix=time.time(),
        verification_scope='independent post-unlink identity/existence and small metadata check; huge payload SHA not repeated')


def finalize(result, postchecks, root, operation_id):
    context('hpc'); needed = {'HK33_RETAINED_AFTER_UNLINK', 'HPC25_ABSENT_ALL_OTHER_CP_METADATA_RETAINED'}
    need(result['status'] == 'EXACT25_UNLINKED_PENDING_INDEPENDENT_POSTCHECKS' and
        result['operation_id'] == operation_id and result['candidate_sha256'] == CANDIDATE_SHA and
        tuple(sorted(result['confirmed_unlinked_files'])) == TARGET_NAMES, 'BOUND_UNLINK_RESULT_REQUIRED')
    need(len(postchecks) == 2 and {r['status'] for r in postchecks} == needed, 'TWO_INDEPENDENT_POSTCHECKS_REQUIRED')
    for row in postchecks:
        need(row['operation_id'] == operation_id and row['candidate_sha256'] == CANDIDATE_SHA and
            row['deletion_result_sha256'] == digest(canonical(result)) and row['checked_unix'] >= result['checked_unix'],
            'POSTCHECK_BOUND_RESULT_DIFFERS')
        actor = row['process_identity']; hk = row['status'].startswith('HK33')
        need(actor['host'] == (HK_HOST if hk else HPC_HOST) and actor['uid'] in
             ((HK_UID,) if hk else (SOURCE_UID, OTHER_UID)), 'POSTCHECK_OWNER_HOST_DIFFERS')
    done = dict(result, status='AUTHORIZED_EXACT25_HPC_PT_RETIRED_FULL_HK33_RETAINED',
                independent_postchecks=postchecks, finalized_unix=time.time())
    write(root/'complete.json', done); return done


def launch(args, plan):
    site = 'hk' if args.phase == 'hk' else 'hpc'; root = audit_root(args.root, args.operation_id, site)
    write(root/(args.phase+'-launch-claim.json'), dict(operation_id=args.operation_id,
        candidate_sha256=CANDIDATE_SHA, phase=args.phase, parent=birth(),
        operator_source_sha256=digest(Path(__file__).read_bytes()), checked_unix=time.time()))
    log = (root/(args.phase+'.log')).open('xb')
    argv = [sys.executable, '-B', __file__, 'work', '--phase', args.phase, '--root', str(root),
            '--candidate', args.candidate, '--operation-id', args.operation_id]
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(child=birth(process.pid), phase=args.phase, operation_id=args.operation_id)
    write(root/(args.phase+'-process.json'), record); return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('plan', 'launch', 'work', 'readers', 'apply', 'verify-source', 'verify-archive', 'finalize'))
    p.add_argument('--candidate', required=True); p.add_argument('--operation-id', required=True); p.add_argument('--root')
    p.add_argument('--phase', choices=('hk', 'source')); p.add_argument('--source-proof'); p.add_argument('--archive-proof')
    p.add_argument('--guards'); p.add_argument('--result'); p.add_argument('--postchecks')
    args = p.parse_args(); plan = load_plan(args.candidate)
    need(re.fullmatch('[0-9a-f]{32}', args.operation_id), 'EXPLICIT_OPERATION_ID_REQUIRED')
    if args.mode == 'readers': result = dict(readers(plan), operation_id=args.operation_id)
    elif args.mode.startswith('verify-'):
        result = verify_after(plan, small(args.result)[0], args.operation_id, 'hk' if args.mode == 'verify-archive' else 'source')
    elif args.mode == 'launch':
        need(args.phase is not None, 'SHA_PHASE_REQUIRED'); result = launch(args, plan)
    else:
        site = 'hk' if args.mode == 'work' and args.phase == 'hk' else 'hpc'
        root = audit_root(args.root, args.operation_id, site)
        if args.mode == 'plan': write(root/'plan.json', plan); result = plan
        elif args.mode == 'apply':
            lock = os.open(root/'apply.lock', os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = execute(plan, small(args.source_proof)[0], small(args.archive_proof)[0],
                                 small(args.guards)[0], root, args.operation_id)
            finally: os.close(lock)
        elif args.mode == 'finalize':
            result = finalize(small(args.result)[0], small(args.postchecks)[0], root, args.operation_id)
        else:
            need(args.phase is not None, 'SHA_PHASE_REQUIRED')
            lock = os.open(root/(args.phase+'.lock'), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                write(root/(args.phase+'-worker.json'), birth())
                try:
                    result = validate(plan, root, args.operation_id, args.phase)
                    write(root/(args.phase+'-validation.json'), result)
                    write(root/(args.phase+'-exit.json'), dict(returncode=0, checked_unix=time.time()))
                except BaseException as error:
                    write(root/(args.phase+'-exit.json'), dict(returncode=1, error_type=type(error).__name__,
                        error=str(error), checked_unix=time.time())); raise
            finally: os.close(lock)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__': main()
