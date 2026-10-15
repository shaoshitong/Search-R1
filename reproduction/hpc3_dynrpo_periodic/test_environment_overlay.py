"""Pure CPU safety fixtures; actual package import/CV proof is remote --check."""
import tempfile
import json
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

    def test_standard_runtime_pyc_does_not_change_installed_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'threadpoolctl.py').write_text('# accepted source\n')
            (root/'joblib').mkdir(); (root/'joblib/__init__.py').write_text('# accepted package\n')
            before = overlay.inventory(root)
            for source in (root/'threadpoolctl.py', root/'joblib/__init__.py'):
                cache = source.parent/'__pycache__'; cache.mkdir()
                (cache/(source.stem+'.cpython-310.pyc')).write_bytes(b'normal runtime cache')
                (cache/(source.stem+'.cpython-310.opt-1.pyc')).write_bytes(b'optimized runtime cache')
            self.assertEqual(overlay.inventory(root), before)
            # No broad exclusion of extra cache contents or sourceless bytecode.
            (root/'__pycache__/foreign.cpython-310.pyc').write_bytes(b'unbound')
            self.assertNotEqual(overlay.inventory(root), before)
            (root/'__pycache__/foreign.cpython-310.pyc').unlink()
            (root/'__pycache__/extra.py').write_text('# must stay hashed\n')
            self.assertNotEqual(overlay.inventory(root), before)

    def test_changed_source_still_rejects_the_accepted_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); root = base/'env/overlay'; root.mkdir(parents=True)
            source = root/'threadpoolctl.py'; source.write_text('# accepted\n')
            saved = dict(base=str(base), pins=overlay.PINS, files_sha256=overlay.inventory(root))
            (base/'env/overlay-installed.json').write_text(json.dumps(saved))
            cache = root/'__pycache__'; cache.mkdir()
            (cache/'threadpoolctl.cpython-310.pyc').write_bytes(b'normal cache')
            source.write_text('# changed implementation\n')
            with self.assertRaisesRegex(ValueError, 'content changed'):
                overlay.check(base)

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
