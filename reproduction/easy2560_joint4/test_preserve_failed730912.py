"""Only this terminal bootstrap failure and its five small source files."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import preserve_failed730912 as module


class PreserveFailed730912Tests(unittest.TestCase):
    def fixture(self, root):
        (root / 'logs').mkdir(); (root / 'preparation').mkdir()
        (root / 'job.id').write_text(module.JOB + '\n')
        binding = dict(source_commit=module.COMMIT, actor_micro_batch_per_rank=16,
                       restore_receipt='/old/source-receipt.json', restore_receipt_sha256='a' * 64)
        (root / module.FILES[1]).write_text(json.dumps(binding))
        (root / module.FILES[2]).write_text(json.dumps(dict(binding, job=module.JOB, returncode=0)))
        (root / module.FILES[3]).write_text(json.dumps(dict(source_commit=module.COMMIT)))
        (root / module.FILES[4]).write_text('Traceback\nValueError: ' + module.ERROR + '\n')
        (root / 'repo').mkdir(); (root / 'repo/original.py').write_text('unchanged source')
        (root / 'unrelated.pt').write_bytes(b'not selected for preservation')
        rows = [(module.JOB, 'FAILED', '1:0'), (module.JOB + '.batch', 'FAILED', '1:0'),
                (module.JOB + '.extern', 'COMPLETED', '0:0')]
        result = dict(returncode=0, stdout=''.join('|'.join((*row, module.NODE,
            '2026-10-18T04:00:59', '2026-10-18T04:01:25')) + '\n' for row in rows), stderr='')
        return result

    def test_exact_five_files_once_SHA_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); result = self.fixture(root)
            originals = {name: module.sha(root / name) for name in module.FILES}
            with patch.object(module, 'BASE', root):
                receipt = module.preserve(root, lambda _: result)
                self.assertEqual(receipt, module.preserve(root, lambda _: self.fail('no repeat operation')))
            target = root / 'failed-job-730912'
            self.assertEqual({row['relative_path'] for row in receipt['files']}, set(module.FILES))
            self.assertEqual({name: module.sha(root / name) for name in module.FILES}, originals)
            self.assertEqual((root / 'repo/original.py').read_text(), 'unchanged source')
            self.assertFalse(list(target.rglob('*.pt')))
            self.assertFalse(receipt['actual_cuda_restore_accepted'])

    def test_nonterminal_or_open_extern_rejected_before_claim(self):
        for old, new in (('FAILED|1:0', 'RUNNING|0:0'), ('COMPLETED|0:0', 'RUNNING|0:0')):
            with self.subTest(old=old), tempfile.TemporaryDirectory() as td:
                root = Path(td); result = self.fixture(root); result['stdout'] = result['stdout'].replace(old, new)
                with patch.object(module, 'BASE', root), self.assertRaisesRegex(ValueError, 'extern closed'):
                    module.preserve(root, lambda _: result)
                self.assertFalse((root / 'failed-job-730912').exists())

    def test_run_manifest_or_cuda_training_evidence_is_not_this_failure(self):
        for name in ('run-manifest.json', 'native-allocation-admission.json', 'full-state-restored-workers.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td); result = self.fixture(root); (root / 'logs' / name).write_text('{}')
                with patch.object(module, 'BASE', root), self.assertRaisesRegex(ValueError, 'pre-manifest/no-CUDA'):
                    module.preserve(root, lambda _: result)
                self.assertFalse((root / 'failed-job-730912').exists())


if __name__ == '__main__': unittest.main()
