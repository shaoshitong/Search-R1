"""Regressions for long-running metadata observation; no SSH or GPUs."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .control import BASE
from .observe import QUERY, query_for


class ObserveTests(unittest.TestCase):
    def test_recovery_target_is_explicit_and_cannot_escape_this_experiment(self):
        target = str(BASE) + '-r2'
        self.assertIn("b=pathlib.Path('" + target + "')", query_for(target))
        for value in ('/tmp/another-task', str(BASE) + '/../other', str(BASE) + '-another'):
            with self.assertRaises(ValueError):
                query_for(value)

    def fixture(self, root):
        values = {
            'preparation/slurm-submission.json': {'job': '729343'},
            'preparation/source-publication.json': {'commit': 'c' * 40},
            'logs/pipeline-status.json': {'outer': 9, 'stage': 'actor'},
            'actor/latest.json': {'completed_outer': 8},
            'probe/latest.json': {'completed_outer': 0},
            'latest-joint.json': {'outer': 8},
            'logs/slurm-allocation.json': {'total_physical_gpus': 4},
        }
        for name, value in values.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value))

    def query(self, root):
        output = io.StringIO()
        queue = SimpleNamespace(returncode=0, stdout='729343|RUNNING', stderr='')
        with patch('subprocess.run', return_value=queue), patch(
                'subprocess.check_output', return_value='729343|RUNNING|0:0'), \
                contextlib.redirect_stdout(output):
            exec(QUERY.replace(str(BASE), str(root)), {})
        return output.getvalue()

    def test_cumulative_ledger_larger_than_scalar_cap_keeps_failure_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for exit_code in (None, 9):
                with self.subTest(exit_code=exit_code):
                    ledger = dict(jobs={'phase': dict(state='RUNNING' if exit_code is None else 'EXITED',
                                                     exit_code=exit_code)},
                                  padding='x' * (1024 ** 2 + 4096))
                    (root / 'logs/process-ledger.json').write_text(json.dumps(ledger))
                    output = self.query(root)
                    result = json.loads(output)
                    self.assertEqual(result['complete_outer'], 8)
                    self.assertEqual(len(result['failed_jobs']), int(exit_code is not None))
                    if exit_code is not None:
                        self.assertEqual(result['failed_jobs'][0]['exit_code'], 9)
                    self.assertNotIn('padding', output)
                    self.assertLess(len(output), 8192)

    def test_other_receipts_still_reject_oversized_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            p = root / 'probe/latest.json'
            p.write_text(json.dumps(dict(padding='x' * (1024 ** 2 + 1))))
            with self.assertRaisesRegex(ValueError, 'probe/latest.json'):
                self.query(root)

    def test_large_actor_and_joint_lineage_stays_remote_and_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            lineage = dict(source_actor_receipt=dict(actor_version='actor100', padding='x' * (2 * 1024**2)))
            (root / 'actor/latest.json').write_text(json.dumps(dict(completed_outer=101, grounding_state=lineage)))
            (root / 'latest-joint.json').write_text(json.dumps(dict(outer=101, grounding=lineage)))
            (root / 'logs/process-ledger.json').write_text(json.dumps(dict(jobs={'bad':dict(exit_code=9)})))
            output = self.query(root)
            result = json.loads(output)
            self.assertEqual(result['complete_outer'], 101)
            self.assertEqual(result['actor']['completed_outer'], 101)
            self.assertEqual(result['failed_jobs'][0]['exit_code'], 9)
            self.assertLess(len(output), 8192)
            self.assertNotIn('padding', output)

    def test_control_projection_retains_current_state_and_binds_omitted_lineage(self):
        definition = next(node for node in ast.parse(QUERY).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'project_metadata')
        namespace = {'json': json}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<projection>', 'exec'), namespace)
        source = dict(actor_version='actor100', completed_outer=100, padding='x' * (2 * 1024**2))
        original = dict(actor=dict(actual_adam_steps_per_rank=[2007]*4),
                        grounding=dict(source_actor_receipt=source),
                        ranks=[dict(grounding_state=dict(source_actor_receipt=source))])
        projected = namespace['project_metadata'](original)
        self.assertEqual(projected['actor'], original['actor'])
        self.assertEqual(projected['grounding']['source_actor_receipt']['actor_version'], 'actor100')
        digest = projected['grounding']['source_actor_receipt']['original_canonical_json_sha256']
        source['padding'] += 'changed'
        self.assertNotEqual(digest, namespace['project_metadata'](original)['grounding']['source_actor_receipt']['original_canonical_json_sha256'])
        self.assertLess(len(json.dumps(projected)), 8192)
        self.assertIn('padding', source)

    def test_reader_enforces_actual_bytes_and_missing_receipt(self):
        definition = next(node for node in ast.parse(QUERY).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'read')
        namespace = {'json': json}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<read>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'receipt.json'
            self.assertIsNone(namespace['read'](p))
            p.write_text(json.dumps(dict(value='x' * 40)))
            with self.assertRaisesRegex(ValueError, 'receipt.json'):
                namespace['read'](p, limit=32)
            self.assertEqual(namespace['read'](p, limit=128)['value'], 'x' * 40)


if __name__ == '__main__':
    unittest.main()
