"""CPU functional proof that a drain waits for the current whole G5 batch."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import draining_pool_worker as drain


class DrainTests(unittest.TestCase):
    def run_loop(self, invalid=False):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder);job = base / 'job';job.mkdir()
            publish = lambda p, v: (Path(p).parent.mkdir(parents=True, exist_ok=True), Path(p).write_text(json.dumps(v)))
            binding = dict(job_id='cpu-whole-batch', source_owner=dict(pid=900), source_owner_allocation_sha256='a' * 64)
            publish(job / 'allocation.json', binding)
            for index in range(1322):
                publish(base / 'inputs/shard-00' / ('batch-%06d.json' % index),
                    dict(split='train', model_revision=drain.REVISION, rows=[dict(split='train')]))
            args = argparse.Namespace(base=str(base), job_dir=str(job), handoff_marker='fixture',
                batch_reserve_seconds=300, deadline=time.time() + 3600)
            phases = []
            marker = dict(status='DRAIN_AT_WHOLE_BATCH_BOUNDARY', run_id='wrong' if invalid else drain.RUN_ID,
                job_id=binding['job_id'], allocation_sha256=drain.sha(job / 'allocation.json'))
            if invalid:publish(job / 'drain.json', marker)

            def phase(path, mode):
                phases.append(mode)
                if mode == 'prefix':publish(job / 'drain.json', marker)
                return dict(questions=1, trajectories=5)

            worker = SimpleNamespace(phase=phase, atomic=publish)
            def batch_config(_, base, inp, batch, config):
                dest = base / 'sampling' / inp.parent.name / 'batches' / inp.stem
                dest.mkdir(parents=True, exist_ok=True)
                return dest, dest / 'config.json'
            with patch.dict('os.environ', Q7B_POOL_INTERNAL=str(job)), \
                 patch.object(drain, 'single_gpu_sampler', return_value=worker), \
                 patch.object(drain._core, 'configuration', return_value={}), \
                 patch.object(drain._core, 'handoff', return_value={}), \
                 patch.object(drain._core, 'batch_config', side_effect=batch_config), \
                 patch.object(drain._core, 'completed', side_effect=lambda receipt, *a: {} if receipt.exists() else None), \
                 contextlib.redirect_stdout(io.StringIO()):
                if invalid:
                    with self.assertRaisesRegex(ValueError, 'Drain run identity'):
                        drain.sample_loop(args)
                    self.assertEqual(phases, [])
                    self.assertFalse((job / 'sampler-exit.json').exists())
                else:
                    drain.sample_loop(args)
                    result = json.loads((job / 'sampler-exit.json').read_text())
                    self.assertEqual(phases, ['prefix', 'continue'])
                    self.assertTrue((base / 'sampling/shard-00/batches/batch-000000/complete.json').exists())
                    self.assertFalse((base / 'sampling/shard-00/batches/batch-000001').exists())
                    self.assertEqual((result['completed_own_batches'], result['questions'], result['trajectories']), (1, 1, 5))
                    self.assertEqual(result['status'], 'POOL_WHOLE_BATCH_DRAINED')
                    self.assertFalse(result['global_goal_complete'])

    def test_drain_during_prefix_finishes_full_and_seals_before_return(self):
        self.run_loop()

    def test_wrong_run_drain_fails_without_a_completion_claim(self):
        self.run_loop(invalid=True)


if __name__ == '__main__':
    unittest.main()
