"""Local CPU tests only; every service operation and GPU query is mocked."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from reproduction.operations.qwen7b_resample.operational_reads import node10_primary_v2_readiness_repair as repair


def rows():
    return [dict(slot=slot, pid=pid, birth=birth,
        unit='sst-q7b-g5-primary10-' + slot + '-v2-' + repair.RUN_ID[:12] + '.service') for slot, pid, birth in repair.COHORT]


class ReadinessRepairTest(unittest.TestCase):
    def test_all_three_are_checked_before_exact_stop_and_intent_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            intent, receipt = Path(directory) / 'intent.json', Path(directory) / 'receipt.json'
            events = []
            def check(row):events.append(('check', row['pid']));return dict(pid=row['pid'], zero=True)
            def run(command, **kwargs):
                events.append(('stop', command));self.assertTrue(intent.exists())
                self.assertEqual(len(events), 7)
                return subprocess.CompletedProcess(command, 0, '', '')
            with mock.patch.object(repair, 'preflight', check), mock.patch.object(repair.subprocess, 'run', run):
                a, b = repair.stop_once(rows(), intent, receipt)
            self.assertEqual(events[-1][1], ['systemctl', 'stop', *[row['unit'] for row in rows()]])
            self.assertFalse(a['authorization_source']['human_direct_three_unit_stop_request'])
            self.assertEqual(a['authorization_source']['active_goal_created_epoch'], 1792511267)
            self.assertEqual(b['returncode'], 0)
            with mock.patch.object(repair.subprocess, 'run') as stop:
                with self.assertRaisesRegex(AssertionError, 'never replay'):repair.stop_once(rows(), intent, receipt)
                stop.assert_not_called()

    def test_any_new_work_in_any_slot_refuses_whole_stop(self):
        for fail_on in (3, 6):
            with self.subTest(fail_on=fail_on), tempfile.TemporaryDirectory() as directory:
                count = [0]
                def check(row):
                    count[0] += 1
                    assert count[0] != fail_on, 'Sampler started; root must decide'
                    return dict(pid=row['pid'])
                with mock.patch.object(repair, 'preflight', check), mock.patch.object(repair.subprocess, 'run') as stop:
                    with self.assertRaises(AssertionError):repair.stop_once(rows(), Path(directory) / 'intent.json', Path(directory) / 'receipt.json')
                    stop.assert_not_called()

    def test_foreign_unit_or_reordered_cohort_is_never_stopped(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(repair.subprocess, 'run') as stop:
            wrong = rows();wrong[0]['unit'] = 'healthy-cpu-relay.service'
            for value in (wrong, list(reversed(rows()))):
                with self.assertRaises(AssertionError):repair.stop_once(value, Path(directory) / 'i.json', Path(directory) / 'r.json')
            stop.assert_not_called()

    def test_zero_work_rejects_even_early_sampler_launch_and_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory);self.assertEqual(repair.zero_work(job)['sampler_children'], 0)
            for name in ('sampler-launches', 'sampler.log', 'progress.json', 'retriever-ready.json'):
                path = job / name
                path.mkdir() if name == 'sampler-launches' else path.write_text('{}')
                with self.assertRaises(AssertionError):repair.zero_work(job)
                path.rmdir() if path.is_dir() else path.unlink()
            (job / 'children.json').write_text(json.dumps(dict(own_process_groups=[123])))
            with self.assertRaises(AssertionError):repair.zero_work(job)
            (job / 'children.json').unlink();(job / 'failure.json').write_text('{}')
            with self.assertRaises(AssertionError):repair.zero_work(job)
            self.assertEqual(repair.zero_work(job, allow_terminal=True)['new_trajectories'], 0)

    def test_live_preflight_rejects_changed_birth_or_batch_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict(rows()[0], prehashed_originals=[], job_directory=directory)
            props = dict(Id=row['unit'], MainPID=str(row['pid']), ActiveState='active', ControlGroup='/system.slice/' + row['unit'])
            actual = dict(pid=row['pid'], birth=row['birth'], cgroup='0::/system.slice/' + row['unit'])
            with mock.patch.object(repair, 'query', return_value=props), mock.patch.object(repair, 'process', return_value=actual), \
                 mock.patch.object(repair, 'cgroup_pids', return_value=[row['pid']]), mock.patch.object(repair, 'batch_fds', return_value=[]):
                self.assertEqual(repair.preflight(row)['sampler_children'], 0)
                with mock.patch.object(repair, 'process', return_value=dict(actual, birth='different')):
                    with self.assertRaises(AssertionError):repair.preflight(row)
                with mock.patch.object(repair, 'batch_fds', return_value=[dict(fd=3, target='own batch lock')]):
                    with self.assertRaisesRegex(AssertionError, 'batch lock'):repair.preflight(row)

    def test_stable_inode_hash_and_two_release_scope_exclude_healthy_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bootstrap.log';path.write_bytes(b'waiting health readiness\n')
            record = repair.original(path);repair.stable([record]);self.assertEqual(record['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b'changed\n')
            with self.assertRaises(AssertionError):repair.stable([record])
        protected = [dict(pid=pid, birth='real', cgroup='protected') for pid in sorted(repair.PROTECTED_LOCAL)]
        targets = [dict(row, job_directory='/own/job') for row in rows()]
        def query(unit):return dict(Id=unit, MainPID='0', ActiveState='inactive', ControlGroup='')
        with mock.patch.object(repair, 'query', side_effect=query), mock.patch.object(repair, 'cgroup_pids', return_value=[]), \
             mock.patch.object(Path, 'exists', return_value=False), mock.patch.object(repair, 'zero_work', return_value={}), \
             mock.patch.object(repair, 'process', side_effect=lambda pid:next(row for row in protected if row['pid'] == pid)), \
             mock.patch.object(repair.subprocess, 'check_output', return_value='2882823\n2904782\n99999\n'):
            fact = repair.release(targets, protected)
        self.assertEqual(fact['own_gpu_contexts'], [])
        self.assertTrue(fact['port_28831_free_not_required'])
        self.assertEqual({row['unit'] for row in fact['units']}, {row['unit'] for row in rows()})


if __name__ == '__main__':unittest.main()
