import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .checkpoint20_pin import assets, pin


class Checkpoint20PinTests(unittest.TestCase):
    def fixture(self, base):
        def file(name, content=b'tensor fixture'):
            path = base / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
            return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        state = base / 'actor/outers/outer-0020-attempt-fixture'
        shard = file(str(state.relative_to(base))+'/huggingface/model.safetensors')
        rank_states = [file(str(state.relative_to(base))+'/rank-%d.pt'%r) for r in range(8)]
        driver = file(str(state.relative_to(base))+'/driver.pt')
        head = file('outers/outer-000020/probe.pt')
        pool = file('outers/outer-000020/pool.json', json.dumps(dict(next_outer=21, consumed_candidates=102400)).encode())
        for name in ('question-pool/binding.json', 'preparation/source-files.json'):
            file(name, b'{}')
        file('preparation/launch-config.json', json.dumps(dict(run_id='fixture',probe_learning_rate=1e-5,source_commit='commit')).encode())
        joint = dict(run_id='fixture', outer=20, source_commit='commit',
                     actor=dict(completed_outer=20, actual_adam_steps_per_rank=[400]*8,
                        scheduler_completed_outer=20, hf_files={'model.safetensors':shard},
                        hf_model_dir=str(state/'huggingface'), model_dir=str(state/'huggingface'),
                        actor_model_dir=str(state/'huggingface'), state_path=str(state), state_dir=str(state),
                        rank_states=rank_states,driver_state=driver),
                     probe=dict(head, completed_outer=20, actual_adam_steps_per_rank=[100]*8,learning_rate=1e-5),
                     probe_snapshot=head['path'], pool_snapshot=pool['path'])
        file('outers/outer-000020/complete.json', json.dumps(joint).encode())
        file('latest-joint.json', json.dumps(joint).encode())
        return joint

    def test_retained_model_and_full_clock_survive_normal_source_unlink(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); joint = self.fixture(base)
            root=base/'preparation/retained-joint20-once-v1';root.mkdir()
            result=pin(base,root)
            self.assertEqual(result['status'],'COMPLETE20_SAME_FILESYSTEM_PINNED')
            self.assertFalse(result['independent_backup'])
            for row in result['files']:
                self.assertEqual(Path(row['source']).stat().st_ino,Path(row['target']).stat().st_ino)
            for row in result['files']:
                Path(row['source']).unlink()
                self.assertTrue(Path(row['target']).is_file())
                self.assertEqual(hashlib.sha256(Path(row['target']).read_bytes()).hexdigest(),row['producer_sha256'])
            pinned=json.loads((root/'checkpoint/JOINT.json').read_text())
            self.assertTrue(Path(pinned['actor']['rank_states'][0]['path']).exists())
            self.assertTrue(Path(pinned['probe_snapshot']).exists())
            self.assertTrue(Path(pinned['pool_snapshot']).exists())

    def test_partial_clock_or_missing_optimizer_is_not_a_complete_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();joint=self.fixture(base)
            changed=copy.deepcopy(joint);changed['actor']['actual_adam_steps_per_rank'][7]=399
            with self.assertRaises(ValueError):assets(base,changed)
            Path(joint['actor']['rank_states'][3]['path']).unlink()
            with self.assertRaises(FileNotFoundError):assets(base,joint)

    def test_missing_joint_is_pending_without_a_pin_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();root=base/'preparation/pin';root.mkdir(parents=True)
            self.assertEqual(pin(base,root)['status'],'PENDING_REAL_JOINT20_COMMIT')
            self.assertFalse((root/'PIN-INTENT.json').exists())

    def test_file_creation_before_atomic_joint_publication_remains_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();self.fixture(base)
            (base/'latest-joint.json').write_text(json.dumps(dict(outer=19)))
            root=base/'preparation/pin';root.mkdir()
            self.assertEqual(pin(base,root)['status'],'PENDING_REAL_JOINT20_COMMIT')
            self.assertFalse((root/'PIN-INTENT.json').exists())


if __name__ == '__main__':
    unittest.main()
