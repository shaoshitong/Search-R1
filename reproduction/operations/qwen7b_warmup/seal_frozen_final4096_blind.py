"""Seal finished final-data production without reading labels or probe scores.

Only immutable JSON receipts, payload SHA streams, file sizes, lock ownership
and allocation terminal states are inspected. No EM JSONL row is decoded, no
class count is computed and no probe is loaded. Later candidate confirmation
must separately join/replay its frozen G5 and score the frozen candidate.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
JOBS = ('749134', '749136', '749137', '749138')
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def read(path):
    return json.loads(Path(path).read_bytes())


def proof(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def checked(ref):
    own = proof(ref['path'])
    assert own['bytes'] == ref['bytes'] and own['sha256'] == ref['sha256'], ref['path']
    return own


def main():
    prepared = read(BASE/'prepared.json')
    missing = [dict(event=e['key'], shard=i) for e in prepared['arrays'] for i in range(4)
               if not (Path(e['original_event_root'])/'sampling'/('shard-%02d' % i)/'complete.json').exists()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_ACTUAL_SHARD_COMPLETIONS', missing=missing)))
        return
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(JOBS),
        '--format=JobID,JobIDRaw,State,ExitCode,NodeList'], universal_newlines=True)
    allocations = [dict(zip(('array_task', 'raw_job_id', 'state', 'exit_code', 'node'), line.split('|')[:5]))
                   for line in raw.splitlines() if line.strip()]
    expected = {job+'_'+str(i) for job in JOBS for i in range(4)}
    finished = {a['array_task'] for a in allocations if a['state'] == 'COMPLETED' and a['exit_code'] == '0:0'}
    if finished != expected:
        print(json.dumps(dict(status='WAITING_FOR_REAL_ALLOCATION_CLEANUP_OR_ACCOUNTING', allocations=allocations)))
        return
    events = []
    for entry in prepared['arrays']:
        checked(entry['entry']); checked(entry['config'])
        event = Path(entry['original_event_root']); config = read(entry['config']['path'])
        assert config['run_id'] == entry['run_id'] and config['model_revision'] == REVISION
        assert config['fitting_authorized'] is False and config['actor_optimizer_updates'] == 0
        inputs = {item['path']:item for item in read(event/'inputs/manifest.json')['batches']}
        plan_path = event.parent/'planned-g10-and-fixed-g5.json'
        plan_ref = next(ref for ref in config['immutable_preparation_sources'] if ref['path'] == str(plan_path))
        checked(plan_ref); plan = read(plan_path)
        assert plan['labels_observed'] is False and plan['actual_responses_generated'] == 0
        expected_ids = {tid:str(group['qid']) for group in plan['planned_G10']
                        for tid in group['event_trajectory_ids'][config['event_index']]}
        assert len(expected_ids) == 10240 and len(plan['planned_G10']) == 2048
        seen_ids = set()
        shards = []
        for index in range(4):
            root = event/'sampling'/('shard-%02d' % index); generic = read(root/'complete.json')
            assert generic['status'] == 'ALL_ASSIGNED_CANONICAL_G5_COMPLETE'
            assert generic['run_id'] == entry['run_id'] and generic['model_revision'] == REVISION
            assert generic['shard'] == index and generic['questions'] == 512 and generic['trajectories'] == 2560
            own_inputs = [item for item in inputs.values() if item['shard'] == index]
            receipts = {str(root/'batches'/Path(item['path']).stem/'complete.json'):item for item in own_inputs}
            assert len(receipts) == 4 and set(generic['batches']) == set(receipts)
            batches = []
            for path, inp in sorted(receipts.items()):
                checked(inp); batch = read(path)
                assert batch['run_id'] == entry['run_id'] and batch['input_sha256'] == inp['sha256']
                assert batch['questions'] == inp['questions'] == 128 and batch['trajectories'] == 640
                assert batch['actor_optimizer_updates'] == 0
                prefix, full = batch['prefix'], batch['full']
                assert prefix['trajectories'] == full['trajectories'] == 640
                assert prefix['run_id'] == full['run_id'] == entry['run_id']
                assert prefix['actor_version'] == full['actor_version'] == 'official7b-'+REVISION
                assert prefix['sampling_attempt_id'] == full['sampling_attempt_id'] == config['sampling_attempt_id']
                # Hash byte streams only: strict_em values and answer text are never decoded.
                journals = {key:checked(full[key]) for key in ('full', 'own_em')}
                assert all(full[key]['count'] == 640 for key in journals)
                native = checked(prefix['actor_hidden'])
                native_index = checked(read(native['path'])['files']['index.jsonl'])
                batch_ids = set()
                with Path(native_index['path']).open() as stream:
                    for line in stream:
                        row = json.loads(line); tid = row['trajectory_id']
                        assert tid not in seen_ids and tid not in batch_ids
                        assert expected_ids.get(tid) == str(row['qid'])
                        batch_ids.add(tid)
                assert len(batch_ids) == 640
                seen_ids.update(batch_ids)
                batches.append(dict(complete=proof(path), questions=128, trajectories=640,
                    full_and_own_EM_byte_SHA_verified=journals, native_feature_manifest=native,
                    native_index=native_index, exact_precommitted_native_trajectory_ids=640))
            with (root/'job.lock').open('r+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            shards.append(dict(shard=index, complete=proof(root/'complete.json'),
                own_job_lock_free=True, questions=512, trajectories=2560, batches=batches))
        assert seen_ids == set(expected_ids)
        events.append(dict(event=entry['key'], run_id=entry['run_id'], config=entry['config'],
            precommitted_plan=plan_ref, exact_native_ID_and_qid_coverage=True,
            missing_or_extra_native_trajectories=0, shards=shards))
    result = dict(status='ALL_FROZEN_FINAL_G10_COLLECTION_SHARDS_COMPLETED_BLIND',
        prepared=proof(BASE/'prepared.json'), source_publication=prepared['source_publication'],
        collector_commit=prepared['collector_commit'], allocations=allocations, events=events,
        frozen_distinct_questions=4096, question_events=8192, real_full_trajectories=40960,
        precommitted_fixed_G5_trajectories=20480, completed_shards=16, completed_batches=64,
        complete_manifest_and_full_journal_and_own_EM_byte_SHA_freshly_verified=True,
        native_feature_manifests_and_indexes_SHA_freshly_verified=True,
        precommitted_native_trajectory_IDs_and_qids_fully_covered=True,
        native_feature_tensor_payloads_rehashed=False,
        final_EM_values_or_class_histogram_or_probe_scores_read=False,
        candidate_confirmation_scoring_performed=False, actor_optimizer_updates=0,
        source=proof(__file__), completed_epoch=time.time())
    output = BASE/'production-complete-blind.json'
    if output.exists():
        previous = read(output)
        assert previous['status'] == result['status'] and previous['events'] == result['events']
        print(json.dumps(dict(status='EXISTING_BLIND_PRODUCTION_SEAL_FRESHLY_REVERIFIED', complete=proof(output))))
        return
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, sort_keys=True, separators=(',', ':'))+'\n')
    temporary.replace(output)
    print(json.dumps(dict(status=result['status'], complete=proof(output), completed_shards=16,
        completed_batches=64, full_trajectories=40960,
        final_EM_values_or_probe_scores_read=False)))


if __name__ == '__main__':
    main()
