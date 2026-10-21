"""CPU file/proc fixtures for the future read-only generation observer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import guarded_generation_read as read
from . import follow_guarded_generation as follow


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value));return path


class FollowTests(unittest.TestCase):
    def test_registry_is_the_exact_manifest_declared_seventeen_units(self):
        rows = follow.registry()
        self.assertEqual(len(rows), 17)
        self.assertEqual(sum(x['role'] == 'model' and x['node'] == 'h100-5' for x in rows), 8)
        self.assertEqual(sum(x['role'] == 'model' and x['node'] == 'h100-10' for x in rows), 6)
        self.assertIn('sst-q7b-g5-guard-rollover-owner-v2-' + follow.control.RUN_ID[:12] + '.service', {x['unit'] for x in rows})
        self.assertTrue(all('-r1-' in x['unit'] for x in rows if x['role'] == 'model'))
        self.assertEqual(len(follow.KNOWN_HISTORY), 3)

    def test_exact_historical_proof_hashes_keep_twelve_failures_but_unknown_still_alerts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);reader = read.Reader(root / 'absent-alias', root, [root]);run = follow.control.RUN_ID
            failures, pins = [], []
            for i in range(13):
                job = root / 'pool/jobs' / ('job-old-%d' % i) / 'attempt-original'
                write(job / 'allocation.json', dict(native_unit='old-%d.service' % i))
                path = write(job / 'failure.json', dict(status='GENUINE_OLD_FAILURE', error_type='SystemExit'))
                failures.append(reader.metadata(path))
            for number, group in enumerate((failures[:4], failures[4:9], failures[9:12])):
                proof = write(root / 'preparation' / ('terminal-%d' % number) / 'complete.json', dict(run_id=run,
                    status='ACTUAL_TERMINAL_STATUS', original_files=[{k:row[k] for k in ('path', 'sha256')} for row in group],
                    release_observations=[dict(epoch=t, original_births_gone=True, own_cgroups_empty=True,
                        remaining_known_pids=[], own_gpu_contexts=[]) for t in (1, 4)]))
                pins.append(dict(path=str(proof), sha256=hashlib.sha256(proof.read_bytes()).hexdigest()))
            census = read.failure_census(reader, [], pins, run)
            self.assertEqual(census['total_original_failures'], 13)
            self.assertEqual(census['historical_attention'], 12)
            self.assertEqual(sum(x['classification'] == 'UNKNOWN_FAILURE_REQUIRES_ROOT' for x in census['failures']), 1)
            summary = follow.summarize({'h100-10': dict(historical=census)}, [])
            self.assertEqual([x['kind'] for x in summary['attention']], ['ROOT_ATTENTION_UNKNOWN_FAILURE'])
            retired_job = root / 'pool/jobs/job-retired-source/attempt-original'
            write(retired_job / 'allocation.json', dict(native_unit='retired-source.service'))
            retired = reader.metadata(write(retired_job / 'failure.json', dict(error_type='SystemExit', status='ACTUAL_143')))
            source_plan = write(root / 'preparation/rollover5/plan.json', dict(
                prior_failed_proof_path=pins[0]['path'], prior_failed_proof_sha256=pins[0]['sha256']))
            source_proof = write(root / 'preparation/rollover5/complete.json', dict(run_id=run,
                status='OWN_BOUND_COHORT_TERMINAL_RELEASED_V1', controlled_stop=False,
                original_files=[{k:retired[k] for k in ('path', 'sha256')}],
                source_plan=str(source_plan), source_plan_sha256=hashlib.sha256(source_plan.read_bytes()).hexdigest(),
                remote_node10_proof=pins[1]['path'], remote_node10_proof_sha256=pins[1]['sha256'],
                release_observations=[dict(epoch=t, original_births_gone=True, own_cgroups_empty=True, own_gpu_contexts=[]) for t in (7, 10)]))
            source_pin = dict(path=str(source_proof), sha256=hashlib.sha256(source_proof.read_bytes()).hexdigest())
            write(root / 'preparation/future-source-owner/submit-context.json', dict(predecessor_proof=source_pin['path'], predecessor_sha256=source_pin['sha256']))
            context = read.context_proofs(reader, [dict(role='owner', directory='future-source-owner', node='h100-5')])
            self.assertEqual(context, [source_pin])
            chained = read.failure_census(reader, [], [pins[2], *context], run)
            self.assertEqual(chained['historical_attention'], 13)
            self.assertEqual(sum(x['classification'] == 'UNKNOWN_FAILURE_REQUIRES_ROOT' for x in chained['failures']), 1)
            # A live-generation unit cannot borrow a historical exemption,
            # even if a caller supplies a matching old proof pin.
            current = [dict(unit='old-0.service')]
            self.assertEqual(read.failure_census(reader, current, pins, run)['historical_attention'], 11)
            write(Path(failures[0]['path']), dict(status='CHANGED_FAILURE'))
            self.assertEqual(read.failure_census(reader, [], pins, run)['historical_attention'], 11)
            self.assertTrue(all(Path(row['path']).exists() for row in failures))

    def test_actual_whole_and_next_requires_owned_posix_lock_and_current_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);reader = read.Reader(root, root, [root]);run = follow.control.RUN_ID
            unit, group, pid = 'future-model.service', '/system.slice/future-model.service', 1234
            job = root / 'pool/jobs/job-fixture/attempt-1'
            allocation = dict(native_unit=unit, native_cgroup=group, controller_pid=100, job_id='future-job', model_indices=[2], model_cuda_uuids=['GPU-fixture-2'])
            write(job / 'allocation.json', allocation);write(job / 'children.json', dict(own_process_groups=[pid]))
            inp = root / 'inputs/shard-0/batch-000000.json'
            write(job / 'progress.json', dict(completed_own_batches=1, latest_input=str(inp), questions=128, trajectories=640))
            write(root / 'sampling/shard-0/batches/batch-000000/complete.json', dict(run_id=run, batch='batch-000000', questions=128, trajectories=640))
            progress = root / 'sampling/shard-0/batches/batch-000001/outers/outer-000001/prefix/rank-0-progress.json'
            # Match retained multi_sampler's real cached device schema: it has
            # native_unit/GPU mapping, but no job_id or controller_pid.
            binding = dict(native_unit=unit, physical_gpu_indices=[2], gpu_uuids=['GPU-fixture-2'])
            write(progress, dict(completed_trajectories=50, actual_sampled_tokens=1000, device_binding=binding))
            write(job / 'sampler-launches/launch-00/failure.json', dict(error_type='OutOfMemoryError'))
            write(job / 'sampler-launches/launch-00/typed-model-initialization-oom.json', dict(actual_typed_cuda_oom=True))
            write(job / 'sampler-launches/launch-01/model-initialized.json', dict(actual_model_initialized=True, max_model_len=4096))
            proc = root / 'proc';p = proc / str(pid);(p / 'fd').mkdir(parents=True);(p / 'fdinfo').mkdir()
            (p / 'stat').write_text(str(pid) + ' (sampler) ' + ' '.join(['S'] + ['0'] * 18 + ['987']))
            (p / 'cgroup').write_text('0::' + group + '\n')
            lock = root / 'pool/batch-locks/shard-0/batch-000001.lock';lock.parent.mkdir(parents=True);lock.touch()
            (p / 'fd/7').symlink_to(lock);(p / 'fdinfo/7').write_text('lock: 1: POSIX ADVISORY WRITE 1234 01:01:10 0 EOF\n')
            expected = dict(key='source:fixture', unit=unit, role='model', node='h100-5')
            def summary():
                with patch.object(read, 'PROC', proc):item = read.attempt(reader, job, reader.metadata(job / 'allocation.json'))
                return follow.summarize({'h100-5': dict(units={'source:fixture': dict(unit=unit, native=dict(Id=unit, MainPID='100', ActiveState='active'), attempts=[item])}),
                    'h100-10': dict(global_sealed=dict(questions=128, trajectories=640, sealed_batches=1))}, [expected])
            current = summary()
            self.assertTrue(current['units']['source:fixture']['whole_and_next_observed'])
            self.assertFalse(current['attention'], 'Handled constructor OOM is telemetry, not a parent failure')
            # Actual input partition numbers restart per shard; the next own
            # input may have the same basename in a different shard.
            repeat = root / 'sampling/shard-1/batches/batch-000000/outers/outer-000001/prefix/rank-0-progress.json'
            write(repeat, dict(completed_trajectories=30, device_binding=binding))
            next_lock = root / 'pool/batch-locks/shard-1/batch-000000.lock';next_lock.parent.mkdir(parents=True);next_lock.touch()
            (p / 'fd/7').unlink();(p / 'fd/7').symlink_to(next_lock)
            self.assertTrue(summary()['units']['source:fixture']['whole_and_next_observed'])
            (p / 'fdinfo/7').write_text('lock: 1: POSIX ADVISORY WRITE 9999 01:01:10 0 EOF\n')
            self.assertFalse(summary()['units']['source:fixture']['whole_and_next_observed'])
            write(job / 'complete.json', dict(completed_own_batches=1))
            self.assertEqual(summary()['attention'][0]['kind'], 'ROOT_ATTENTION_RENEW_INSTANCE')
            channel = follow.summarize({'h100-5': dict(observation_error='SSH timeout')}, [expected])
            self.assertFalse(channel['attention']);self.assertIn('h100-5', channel['observation_channel_errors'])

    def test_not_loaded_submit_intent_is_not_a_native_exit_but_failed_is_attention(self):
        expected = dict(key='source:fixture', unit='future-model.service', role='model', node='h100-5')
        item = dict(unit=expected['unit'], submitted=True, staged_manifest_matches=True, attempts=[],
            native=dict(Id=expected['unit'], LoadState='not-found', MainPID='0', ActiveState='inactive'))
        def summary():return follow.summarize({'h100-5': dict(units={expected['key']: item})}, [expected])
        missing = summary()
        self.assertEqual(missing['units'][expected['key']]['state'], 'NOT_STARTED')
        self.assertFalse(missing['attention'])
        item['native'].update(LoadState='loaded', ActiveState='failed')
        self.assertEqual(summary()['attention'][0]['kind'], 'ROOT_ATTENTION_NEW_NATIVE_FAILURE')


if __name__ == '__main__':unittest.main()
