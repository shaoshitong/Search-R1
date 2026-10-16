"""Local tiny fixtures only; no SSH, actual reserve, or large files."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retire_once', HERE / 'retire_once.py')
retire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retire)


class ExactProductionPlan(unittest.TestCase):
    def test_pinned_plan_is_45_old_state_files_and_preserves_topd(self):
        plan = retire.load_plan(HERE)
        self.assertEqual(len(plan['candidates']), 45)
        self.assertEqual(len(plan['protected_files']), 73)
        self.assertEqual(sum(x['size'] for x in plan['candidates']), 115049176725)
        protected_assets = [x for x in plan['protected_files']
                            if '/topd-20261008/' in x['source_path']
                            and x['path'].endswith('.safetensors')]
        self.assertEqual(len(protected_assets), 2)
        self.assertEqual(sum(x['size'] for x in protected_assets), 6171926992)
        self.assertTrue(all('/topd-' not in x['path'] for x in plan['candidates']))

    def test_cli_has_no_arbitrary_root_or_resume_switch(self):
        source = (HERE / 'retire_once.py').read_text()
        self.assertNotIn("add_argument('--root'", source)
        self.assertNotIn("add_argument('--resume'", source)
        self.assertNotIn('shutil.rmtree', source)
        self.assertNotIn('subprocess', source)


class TinyFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hk3-retire-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.root = self.home / 'old-reserve'
        self.payload = self.root / 'payload'
        self.control = self.root / 'control' / 'checkpoint-retire45-abb9f5806865'
        self.proc = self.home / 'proc'
        self.proc.mkdir()
        (self.proc / 'locks').write_text('')
        (self.root / 'control').mkdir(parents=True)
        self.birth_record = self.root / 'control' / 'process.json'
        self.birth_record.write_text(json.dumps([
            dict(pid=pid, start_ticks=ticks) for pid, ticks in retire.RECORDED_BIRTHS]))
        self.payload.mkdir()
        self.bundle = self.home / 'bundle'
        self.bundle.mkdir()
        sources = ['/data/user/jxu957/experiments/qprobe-teacher-warmup200-fixture/checkpoints/'
                   'search-g1-common8/actor/global_step_5/part-%s.pt' % i for i in range(3)]
        sources += ['/data/user/jxu957/experiments/topd-20261008/runtime/assets/model.safetensors',
                    '/data/user/jxu957/experiments/qprobe-teacher-warmup200-fixture/logs/eval.json']
        entries = []
        for index, source in enumerate(sources):
            path = self.payload / source.lstrip('/')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('fixture-%s' % index).encode())
            os.chmod(str(path), 0o600)
            entry = retire.observed_stat(path)
            entry.update(path=str(path), source_path=source)
            entries.append(entry)
        self.plan = dict(root=str(self.root), candidates=entries[:3], protected_files=entries[3:])
        self.plan_path = self.bundle / 'retirement-plan.json'
        self.plan_path.write_text(json.dumps(self.plan))
        evidence = self.bundle / 'hpc3-nonreference-evidence.json'
        evidence.write_text(json.dumps(dict(returncode=0, stdout=json.dumps(
            dict(old_hk3_reserve_references=[])))))
        changes = dict(ROOT=self.root, PAYLOAD=self.payload, CONTROL=self.control,
                       EXPECTED_UID=os.getuid(), EXPECTED_HOST=socket.gethostname().split('.')[0],
                       EXPECTED_CANDIDATES=3, EXPECTED_PROTECTED=2,
                       EXPECTED_BYTES=sum(x['size'] for x in entries[:3]),
                       EXPECTED_ALLOCATED=sum(x['allocated_bytes'] for x in entries[:3]),
                       PLAN_SHA=hashlib.sha256(self.plan_path.read_bytes()).hexdigest(),
                       NONREFERENCE_SHA=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                       PROCS=self.proc, BIRTH_RECORDS=(self.birth_record,))
        self.patches = mock.patch.multiple(retire, **changes)
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def loaded(self):
        return retire.load_plan(self.bundle)

    def test_success_deletes_only_explicit_members_and_refuses_second_attempt(self):
        plan = self.loaded()
        protected = {x['path']: Path(x['path']).read_bytes() for x in plan['protected_files']}
        proof = retire.preflight(plan)
        result = retire.execute(plan, proof)
        self.assertEqual(result['candidate_files'], 3)
        self.assertTrue(result['first_absent_pass'] and result['second_absent_pass'])
        self.assertTrue(all(not Path(x['path']).exists() for x in plan['candidates']))
        self.assertTrue(all(Path(path).read_bytes() == data for path, data in protected.items()))
        self.assertTrue(all(Path(x['path']).parent.is_dir() for x in plan['candidates']))
        persisted = json.loads((self.control / 'complete.json').read_text())
        self.assertEqual(persisted, result)
        with self.assertRaisesRegex(retire.Refuse, 'ONE_SHOT_CONTROL_ALREADY_EXISTS'):
            retire.preflight(plan)

    def test_same_size_and_mtime_inode_replacement_is_refused_before_claim(self):
        plan = self.loaded()
        entry = plan['candidates'][0]
        path = Path(entry['path'])
        data = path.read_bytes()
        path.rename(path.with_name('retained-original.pt'))
        path.write_bytes(data)
        os.chmod(str(path), 0o600)
        os.utime(str(path), ns=(entry['mtime_ns'], entry['mtime_ns']))
        with self.assertRaisesRegex(retire.Refuse, 'FRESH_STAT_DIFFERS'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())
        self.assertEqual(path.read_bytes(), data)

    def test_new_hardlink_is_refused_before_claim(self):
        plan = self.loaded()
        os.link(plan['candidates'][0]['path'], str(self.home / 'unrelated-hardlink'))
        with self.assertRaisesRegex(retire.Refuse, 'FRESH_STAT_DIFFERS'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())

    def test_symlink_replacement_is_refused_before_claim(self):
        plan = self.loaded()
        path = Path(plan['candidates'][0]['path'])
        path.unlink()
        path.symlink_to(plan['protected_files'][0]['path'])
        with self.assertRaisesRegex(retire.Refuse, 'SYMLINK_PATH'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())
        self.assertTrue(Path(plan['protected_files'][0]['path']).exists())

    def test_plan_tamper_cannot_expand_delete_scope(self):
        self.plan_path.write_bytes(self.plan_path.read_bytes() + b' ')
        with self.assertRaisesRegex(retire.Refuse, 'EXACT_PLAN_SHA_REQUIRED'):
            self.loaded()
        self.assertFalse(self.control.exists())

    def test_recorded_process_same_birth_is_refused(self):
        plan = self.loaded()
        pid, ticks = retire.RECORDED_BIRTHS[0]
        proc = self.proc / str(pid)
        proc.mkdir()
        fields = ['S'] + ['0'] * 18 + [str(ticks)]
        (proc / 'stat').write_text('%s (python3) %s' % (pid, ' '.join(fields)))
        with self.assertRaisesRegex(retire.Refuse, 'RECORDED_RESERVE_PROCESS_STILL_ACTIVE'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())

    def test_changed_process_record_is_refused_before_any_deletion(self):
        plan = self.loaded()
        self.birth_record.write_text(json.dumps([dict(pid=999888, start_ticks=123)]))
        with self.assertRaisesRegex(retire.Refuse, 'RECORDED_PROCESS_BINDING_CHANGED'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())
        self.assertTrue(all(Path(x['path']).exists() for x in plan['candidates']))

    def test_kernel_lock_on_candidate_inode_is_refused(self):
        plan = self.loaded()
        x = plan['candidates'][1]
        key = '%x:%x:%s' % (os.major(x['device']), os.minor(x['device']), x['inode'])
        (self.proc / 'locks').write_text('1: POSIX ADVISORY WRITE 9876 %s 0 EOF\n' % key)
        with self.assertRaisesRegex(retire.Refuse, 'CANDIDATE_HAS_PROC_LOCK'):
            retire.preflight(plan)

    def test_live_open_file_descriptor_is_refused(self):
        plan = self.loaded()
        proc = self.proc / '987654321'
        (proc / 'fd').mkdir(parents=True)
        (proc / 'comm').write_text('python3\n')
        (proc / 'maps').write_text('')
        (proc / 'cwd').symlink_to(self.home)
        (proc / 'fd' / '9').symlink_to(plan['candidates'][0]['path'])
        with self.assertRaisesRegex(retire.Refuse, 'LIVE_CANDIDATE_FILE_DESCRIPTOR'):
            retire.preflight(plan)
        self.assertFalse(self.control.exists())

    def test_partial_failure_is_durable_and_never_replayed(self):
        plan = self.loaded()
        proof = retire.preflight(plan)
        original_unlink = os.unlink
        counter = [0]

        def fail_second(*args, **kwargs):
            counter[0] += 1
            if counter[0] == 2:
                raise OSError('fixture disk error')
            return original_unlink(*args, **kwargs)

        with mock.patch.object(retire.os, 'unlink', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'fixture disk error'):
                retire.execute(plan, proof)
        self.assertFalse(Path(plan['candidates'][0]['path']).exists())
        self.assertTrue(Path(plan['candidates'][1]['path']).exists())
        failure = json.loads((self.control / 'failure.json').read_text())
        self.assertEqual(failure['removed_count_observed'], 1)
        self.assertFalse((self.control / 'complete.json').exists())
        with self.assertRaisesRegex(retire.Refuse, 'ONE_SHOT_CONTROL_ALREADY_EXISTS'):
            retire.preflight(plan)
        self.assertTrue(Path(plan['candidates'][1]['path']).exists())
        events = [json.loads(line)['event'] for line in (self.control / 'journal.jsonl').read_text().splitlines()]
        self.assertEqual(events, ['UNLINK_INTENT', 'UNLINK_FSYNC_DONE', 'UNLINK_INTENT'])

    def test_preexisting_unknown_claim_is_not_overwritten(self):
        plan = self.loaded()
        self.control.mkdir()
        claim = self.control / 'claim.json'
        claim.write_bytes(b'{partial unknown outcome')
        with self.assertRaisesRegex(retire.Refuse, 'ONE_SHOT_CONTROL_ALREADY_EXISTS'):
            retire.preflight(plan)
        self.assertEqual(claim.read_bytes(), b'{partial unknown outcome')
        self.assertTrue(all(Path(x['path']).exists() for x in plan['candidates']))


if __name__ == '__main__':
    unittest.main()
