"""Once final metadata/log/exit audit; reuse completed CPU proofs, never train."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

BASE = '/mnt/data/shaoshitong/search-g1-question20-groupjoint100-20261016-cee20bcd'
RUN = 'cee20bcd24194b79a9a8a88809bd5886'
BINDING = '791ab0fd04af07e4655f364462454ea9bcfecf1e8d21101b1a41313cc60eda3b'
TEST = 'a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'


def read(path):
    return json.loads(Path(path).read_text())


def stable_sha(path):
    path = Path(path)
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    assert signature(before) == signature(after) == signature(path.stat()), str(path)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=before.st_size,
                dev=before.st_dev, inode=before.st_ino, stable_original_inode=True)


def once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return dict(arguments=args, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def controller_terminal(base):
    """Recover the original completion when native memory/accounting are absent."""
    allocation = read(base / 'logs/native-allocation.json')
    fields = dict(item.split('=', 1) for item in allocation['native'].split() if '=' in item)
    assert fields['JobId'] == '3412' and fields['Command'] == str(base / 'preparation/native-launch.sh')
    assert fields['JobName'] == 'g1-q20-joint100-' + RUN[:12]
    started = datetime.datetime.fromisoformat(fields['StartTime']).timestamp()
    events, files = [], []
    pattern = re.compile(r'\bJobId[= ]3412\b')
    for path in (Path('/var/log/slurm/slurmctld.log.1'), Path('/var/log/slurm/slurmctld.log')):
        if not path.is_file():
            continue
        with path.open('rb') as handle:
            before = os.fstat(handle.fileno())
            while handle.tell() < before.st_size:
                offset = handle.tell()
                raw = handle.readline()
                text = raw.decode(errors='replace').rstrip('\n')
                if not pattern.search(text):
                    continue
                stamp = re.match(r'\[([^]]+)\]', text)
                if not stamp:
                    continue
                when = datetime.datetime.fromisoformat(stamp[1]).timestamp()
                if when >= started:
                    events.append(dict(path=str(path), offset=offset, bytes=len(raw),
                                       sha256=hashlib.sha256(raw).hexdigest(), unix=when, raw=text))
            after = os.fstat(handle.fileno())
        assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        files.append(dict(path=str(path), dev=before.st_dev, inode=before.st_ino,
                          initial_bytes=before.st_size, inspected_until_initial_eof=True))
    events.sort(key=lambda row: row['unix'])
    exits = [row for row in events if '_job_complete:' in row['raw'] and 'WEXITSTATUS' in row['raw']]
    done = [row for row in events if '_job_complete:' in row['raw'] and re.search(r'\bdone\b', row['raw'])]
    adverse = [row for row in events if re.search(r'REQUEST_KILL_JOB|CANCELLED|FAILED|WTERMSIG|WCOREDUMP|requeue', row['raw'], re.I)]
    verified = bool(exits and done and not adverse and
                    re.search(r'\bWEXITSTATUS[ =]+0\b', exits[-1]['raw']) and
                    done[-1]['unix'] >= exits[-1]['unix'])
    return dict(verified=verified, state='COMPLETED' if verified else None,
                exit_code='0:0' if verified else None, source='original_slurmctld_completion_records',
                allocation_file=stable_sha(base / 'logs/native-allocation.json'),
                original_running_native_record=allocation['native'], expected_start_unix=started,
                controller_files=files, matching_original_events=events,
                completion_basis='WEXITSTATUS0 plus subsequent _job_complete done; no cancellation/signal/failure events')


def terminal(base):
    """Use actual native records, never infer exit0 from Python completion."""
    receipt = read(base / 'preparation/submit-receipt.json')
    assert str(receipt['job_id']) == '3412'
    expected_name = 'g1-q20-joint100-' + RUN[:12]
    records = [command(['scontrol', 'show', 'job', '3412', '-o'])]
    if records[0]['returncode'] == 0:
        fields = dict(item.split('=', 1) for item in records[0]['stdout'].split() if '=' in item)
        assert fields['JobId'] == '3412' and fields['JobName'] == expected_name
        assert fields['Command'] == str(base / 'preparation/native-launch.sh')
        return dict(verified=fields.get('JobState') == 'COMPLETED' and fields.get('ExitCode') == '0:0',
                    state=fields.get('JobState'), exit_code=fields.get('ExitCode'),
                    source='actual_scontrol', records=records)
    records.append(command(['sacct', '-n', '-P', '-j', '3412',
                            '--format=JobIDRaw,JobName%80,State,ExitCode,NodeList']))
    if records[-1]['returncode'] == 0:
        rows = [line.split('|') for line in records[-1]['stdout'].splitlines() if line.strip()]
        matches = [row for row in rows if row[0] == '3412']
        if len(matches) == 1:
            row = matches[0]
            assert row[1] == expected_name and row[4] == 'h100-8'
            return dict(verified=row[2] == 'COMPLETED' and row[3] == '0:0', state=row[2],
                        exit_code=row[3], source='actual_sacct', records=records)
    controller = controller_terminal(base)
    controller['unavailable_native_accounting_records'] = records
    return controller


def identity(record):
    path = Path('/proc') / str(record['pid'])
    try:
        text = (path / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return (int(fields[19]), int(fields[3]), path.stat().st_uid) == (
            record['start_ticks'], record['sid'], record['uid'])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def source_retention(base, config):
    """Stat original retained inputs/compact shards; do not rehash trajectories."""
    path = base / 'preparation/source-inventory.json'
    assert stable_sha(path)['sha256'] == config['source_inventory_sha256']
    rows = [row for row in read(path)['sources'] if row['split'] == 'train']
    assert {row['epoch'] for row in rows} == set(range(1, 71)) and len(rows) == 70
    result = []
    for row in sorted(rows, key=lambda r: r['epoch']):
        files = []
        if 'rollout_complete' in row:
            small = Path(row['rollout_complete'])
            assert stable_sha(small)['sha256'] == row['sha256']
            complete = read(small)
            assert complete['questions'] == 5120 and complete['trajectories'] == 25600
            for shard in complete['shards']:
                actual = Path(shard['path']).stat()
                assert actual.st_size == shard['bytes']
                files.append(dict(path=shard['path'], bytes=actual.st_size,
                                  producer_sha256=shard['sha256'], contents_rehashed=False))
            question = Path(complete['questions_path'])
            assert question.is_file()
            files.append(dict(path=str(question), bytes=question.stat().st_size, contents_rehashed=False))
        else:
            source = Path(row['inputs'])
            complete = read(source / 'scan-complete.json')
            arrays = Path(complete.get('source_inputs_dir', source))
            for name in ('packed_context_ids.npy', 'packed_policy_ids.npy',
                         'packed_policy_positions.npy', 'policy_offsets.npy'):
                target = arrays / name
                assert target.is_file()
                files.append(dict(path=str(target), bytes=target.stat().st_size, contents_rehashed=False))
            assert (source / 'index.jsonl').is_file()
        result.append(dict(source_data_epoch=row['epoch'], descriptor=row, retained_files=files))
    return result


def audit(base):
    base = Path(base)
    assert str(base) == BASE and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    proof = base / 'preparation/final100-log-exit-retention-once-v1.json'
    if proof.exists():
        prior = read(proof)
        assert prior['run_id'] == RUN and prior['binding_sha256'] == BINDING
        return dict(status='EXISTING_FINAL100_PROOF_NO_REPLAY', proof=prior, proof_file=stable_sha(proof))
    config = read(base / 'preparation/launch-config.json')
    assert config['run_id'] == RUN and config['source_commit'] == '2fd5c19fa0d9c5d6c4eb514a7e9a308eb6bc6479'
    assert config['data_schedule'] == list(range(1, 71)) + list(range(1, 31))
    latest = read(base / 'train/latest.json')
    pipeline = read(base / 'logs/pipeline-complete.json')
    cleanup = read(base / 'logs/cleanup-status.json')
    assert latest['epoch'] == pipeline['epochs'] == 100
    assert latest['actual_adam_steps_per_rank'] == [10000] * 8 and latest['actual_learning_rate'] == 3e-6
    assert pipeline['latest'] == latest and pipeline['run_id'] == cleanup['run_id'] == RUN
    assert pipeline['new_rollouts'] == pipeline['actor_updates'] == 0 and not pipeline['old_head_state_imported']
    assert cleanup['cleaned'] and cleanup['owned_births_all_gone'] and not cleanup['remaining_owned_births']
    native = terminal(base)
    if not native['verified']:
        return dict(status='FINAL100_NATIVE_EXIT0_NOT_YET_VERIFIED_NO_AUDIT_INTENT',
                    run_id=RUN, latest=latest, native=native, training_modified=False)
    retained_sources = source_retention(base, config)
    intent = base / 'preparation/final100-log-exit-retention-once-v1-intent.json'
    if intent.exists():
        return dict(status='EXISTING_FINAL100_AUDIT_INTENT_NO_BLIND_REPLAY', intent=str(intent))
    once(intent, dict(run_id=RUN, started_unix=time.time(), native=native))
    epochs = []
    for epoch in range(1, 101):
        root = base / 'train/epochs' / ('epoch-%04d' % epoch)
        complete = read(root / 'complete.json')
        bundle_path = base / 'epochs' / ('epoch-%04d' % epoch) / 'data-bundle.json'
        bundle = read(bundle_path)
        assert complete['epoch'] == bundle['epoch'] == epoch
        assert complete['actual_adam_steps_per_rank'] == [100 * epoch] * 8
        assert complete['new_adam_steps_per_rank'] == 100 and complete['binding_sha256'] == BINDING
        assert bundle['source_data_epoch'] == config['data_schedule'][epoch - 1]
        assert bundle['test_manifest_sha256'] == TEST and bundle['new_rollouts'] == 0
        assert bundle['checkpoint'] == {key: complete[key] for key in bundle['checkpoint']}
        epochs.append(dict(epoch=epoch, complete=stable_sha(root / 'complete.json'),
                           data_bundle=stable_sha(bundle_path)))
    assert not (base / 'epochs/epoch-0101').exists() and not (base / 'train/epochs/epoch-0101').exists()
    cpu = []
    for epoch in range(10, 101, 10):
        path = base / 'preparation' / ('checkpoint%04d-independent-full-sha-cpu-once-v1.json' % epoch)
        saved = read(path)
        assert saved['status'] == 'JOINT100_CHECKPOINT_INDEPENDENT_FULL_SHA_CPU_PASS'
        assert saved['run_id'] == RUN and saved['binding_sha256'] == BINDING and saved['epoch'] == epoch
        assert saved['actual_adam_steps_per_rank'] == [100 * epoch] * 8
        actual = Path(saved['checkpoint_path']).stat()
        assert (actual.st_dev, actual.st_ino, actual.st_size) == (saved['dev'], saved['inode'], saved['checkpoint_bytes'])
        cpu.append(dict(epoch=epoch, proof=stable_sha(path), checkpoint_sha256=saved['checkpoint_sha256']))
    assert (base / 'train/latest.pt').stat().st_ino == (base / 'train/head-epoch-0100.pt').stat().st_ino
    ownership_path = base / 'logs/process-ledger.json'
    ownership = read(ownership_path)
    assert ownership['run_id'] == RUN
    births = [row for phase in ownership['jobs'].values() for row in phase['ledger'].values()]
    assert not [row for row in births if identity(row)]
    groups = list(Path('/sys/fs/cgroup').glob('**/job_3412')) + list(Path('/sys/fs/cgroup').glob('**/job3412'))
    assert not groups
    gpu = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    assert gpu['returncode'] == 0
    gpu_pids = set(gpu['stdout'].split())
    assert not [row for row in births if str(row['pid']) in gpu_pids and identity(row)]
    logs = sorted({p for p in (base / 'logs').rglob('*') if p.is_file()})
    logs += sorted((base / 'train/epochs').glob('epoch-*/rank-*-updates.jsonl'))
    hashed_logs = [stable_sha(path) for path in logs]
    last_rank = []
    for rank in range(8):
        path = base / 'train/epochs/epoch-0100' / ('rank-%d-updates.jsonl' % rank)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 100 and [row['epoch_step'] for row in rows] == list(range(1, 101))
        assert [row['actual_adam_steps'] for row in rows] == list(range(9901, 10001))
        assert sum(row['local_trajectories'] for row in rows) == 3200 and rows[-1]['actual_learning_rate'] == 3e-6
        last_rank.append(dict(rank=rank, actual_adam_steps=10000, actual_writes=100, trajectories=3200))
    final = dict(status='JOINT100_FINAL_NATIVE_EXIT0_LOG_SHA_RETENTION_PASS', run_id=RUN,
                 binding_sha256=BINDING, base=str(base), native=native, latest=latest,
                 all_100_complete_epochs=epochs, cpu_proofs_reused_not_reexecuted=cpu,
                 terminal_original_inode_logs=hashed_logs, owned_birth_count=len(births),
                 owned_births_present=0, native_cgroups_present=0, owned_gpu_births_present=0,
                 last_epoch_rank_ledgers=last_rank, source_70_batches=retained_sources,
                 epoch101_exists=False, checkpoint_sha_repeated=False, checkpoint_cpu_load_repeated=False,
                 source_trajectory_contents_rehashed=False, gpu_forward_performed=False,
                 training_modified=False, formal_actor_training_started=False, backup_performed=False,
                 large_payload_transferred_to_Mac=False, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    once(proof, final)
    return dict(status=final['status'], proof_file=stable_sha(proof), proof=final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--native-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(terminal(Path(args.base)) if args.native_only else audit(args.base),
                     sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
