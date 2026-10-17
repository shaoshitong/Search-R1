import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import preserve_failed_730226 as op


class PreserveFailedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.binding = patch.object(op, 'BASE', self.base)
        self.binding.start()
        self.addCleanup(self.binding.stop)
        self.addCleanup(self.temp.cleanup)
        self.write('job.id', op.JOB)
        self.write('logs/run-manifest.json', dict(run_id=op.RUN, source_commit=op.COMMIT, training_world_size=4))
        self.write('logs/cleanup-status.json', dict(run_id=op.RUN, cleaned=True))
        required = ['training_state/rank-%d.pt' % r for r in range(4)] + ['training_state/driver.pt']
        for rel in required:
            self.write(str(op.CP / rel), 'tensor stays here')
        self.write(str(op.CP / 'complete.json'), dict(run_id=op.RUN, step=5,
            finite_all_saved_tensors=True, actual_optimizer_writes_per_rank=100,
            files_sha256={name: 'a' * 64 for name in required}))
        self.write(str(op.CP / 'training_state/manifest.json'), dict(completed_outer=5, question_probe=None))
        self.write('logs/actual-updates.jsonl', ''.join(json.dumps(dict(step=i, run_id=op.RUN,
            actual_optimizer_writes=20)) + '\n' for i in range(1, 6)))
        self.write('logs/search-g1-common8-training.log', 'traceback\n' + op.ERROR + '\n')
        for kind in ('optimizer', 'scheduler'):
            for rank in range(4):
                self.write('logs/%s-rank-%d.jsonl' % (kind, rank), '{"actual":true}\n')

    def write(self, rel, value):
        p = self.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value) if isinstance(value, dict) else value)

    def execute(self, argv):
        if argv[0] == 'sacct':
            return dict(returncode=0, stderr='', stdout=''.join('%s|%s|%s|%s|2026-10-18T03:29:16\n' %
                (job, state, code, op.NODE) for job, state, code in (
                (op.JOB, 'FAILED', '1:0'), (op.JOB + '.batch', 'FAILED', '1:0'),
                (op.JOB + '.extern', 'COMPLETED', '0:0'))))
        return dict(returncode=0, stdout='', stderr='')

    def test_exact_small_copies_sha_and_idempotent_readback(self):
        value = op.preserve(self.base, self.execute)
        self.assertEqual(value['completed_outer'], 5)
        self.assertTrue(value['checkpoint_available'])
        self.assertFalse(value['tensors_copied'])
        self.assertEqual(len([r for r in value['files'] if '-rank-' in r['relative_path']]), 8)
        self.assertFalse(list((self.base / 'failed-job-730226').rglob('*.pt')))
        for row in value['files']:
            self.assertEqual(op.sha(row['source']), op.sha(row['copy']))
        self.assertEqual(op.preserve(self.base, self.execute), value)

    def test_live_job_rejected_before_claim(self):
        def live(argv):
            v = self.execute(argv)
            if argv[0] == 'squeue':
                v['stdout'] = '730226|RUNNING|ACD1-61\n'
            return v
        with self.assertRaisesRegex(ValueError, 'terminal failed'):
            op.preserve(self.base, live)
        self.assertFalse((self.base / 'failed-job-730226').exists())

    def test_bad_checkpoint_rejected_before_claim(self):
        self.write(str(op.CP / 'training_state/manifest.json'), dict(completed_outer=4))
        with self.assertRaisesRegex(ValueError, 'checkpoint5'):
            op.preserve(self.base, self.execute)
        self.assertFalse((self.base / 'failed-job-730226').exists())

    def test_purged_queue_requires_fresh_exact_terminal_accounting(self):
        def purged(argv):
            v = self.execute(argv)
            if argv[0] == 'squeue':
                v.update(returncode=1, stderr='slurm_load_jobs error: Invalid job id specified\n')
            return v
        self.assertEqual(op.preserve(self.base, purged)['completed_outer'], 5)

    def test_unknown_queue_error_is_not_absence(self):
        def unavailable(argv):
            v = self.execute(argv)
            if argv[0] == 'squeue':
                v.update(returncode=1, stderr='slurm controller connection failed\n')
            return v
        with self.assertRaisesRegex(ValueError, 'terminal failed'):
            op.preserve(self.base, unavailable)
        self.assertFalse((self.base / 'failed-job-730226').exists())

    def test_partial_or_corrupt_copy_is_never_overwritten(self):
        target = self.base / 'failed-job-730226'
        target.mkdir()
        with self.assertRaisesRegex(ValueError, 'partial preservation'):
            op.preserve(self.base, self.execute)
        target.rmdir()
        value = op.preserve(self.base, self.execute)
        Path(value['files'][0]['copy']).write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            op.preserve(self.base, self.execute)


if __name__ == '__main__':
    unittest.main()
