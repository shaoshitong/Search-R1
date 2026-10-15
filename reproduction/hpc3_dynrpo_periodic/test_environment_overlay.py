"""Pure CPU safety fixtures; actual package import/CV proof is remote --check."""
import tempfile
from pathlib import Path
import unittest

from reproduction.hpc3_dynrpo_periodic import environment_overlay as overlay


class OverlayTests(unittest.TestCase):
    def test_only_four_pinned_internal_wheels(self):
        command = overlay.pip_command('/remote/own/env/overlay')
        self.assertEqual(command[0], overlay.CONTROL)
        self.assertIn('--no-deps', command)
        self.assertIn('--only-binary=:all:', command)
        self.assertIn('--no-cache-dir', command)
        self.assertIn('--no-compile', command)
        self.assertEqual(command[command.index('--index-url')+1], overlay.INDEX)
        self.assertEqual(set(command[-4:]), {k+'=='+v for k, v in overlay.PINS.items()})
        self.assertFalse(any(x.startswith(('numpy==', 'torch==')) for x in command))

    def test_refuses_active_own_queue_and_existing_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            overlay.assert_inactive(base, '123|RUNNING|/foreign\n')
            with self.assertRaisesRegex(ValueError, 'active Slurm'):
                overlay.assert_inactive(base, '123|PENDING|'+str(base)+'\n')
            with self.assertRaisesRegex(ValueError, 'authenticate'):
                overlay.assert_inactive(base, 'ambiguous')
            (base/'logs').mkdir(); (base/'logs/run-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Existing run'):
                overlay.assert_inactive(base, '')

    def test_inventory_rejects_numpy_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'threadpoolctl.py').write_text('# test\n')
            self.assertEqual(set(overlay.inventory(root)), {'threadpoolctl.py'})
            (root/'numpy').mkdir()
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                overlay.inventory(root)
            (root/'numpy').rmdir()
            (root/'joblib').symlink_to(root/'threadpoolctl.py')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                overlay.inventory(root)

    def test_actual_calibration_import_coverage(self):
        repo = Path(__file__).resolve().parents[2]
        requests = overlay.import_requests(repo)
        for request in [('joblib', None), ('sklearn.neural_network', 'MLPClassifier'),
                        ('sklearn.model_selection', 'GroupShuffleSplit'), ('vllm', 'LLM'),
                        ('transformers', 'AutoModelForCausalLM'), ('fastapi', 'FastAPI')]:
            self.assertIn(request, requests)
        self.assertNotIn(('__future__', 'annotations'), requests)

    def test_check_requires_accepted_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'No accepted'):
                overlay.check(Path(temporary))


if __name__ == '__main__':
    unittest.main()
