"""Necessary scope checks; no remote state, model data or GPU."""
import unittest
from unittest.mock import patch

from . import deploy_joint169028_bank as d


class FullBankDeploymentTests(unittest.TestCase):
    def test_roles_exact_and_existing_training_separate(self):
        self.assertEqual(sum(v['questions'] for v in d.TEMPLATE['banks']), 169028)
        self.assertEqual(list(d.PATHS), list(d.COUNTS))
        self.assertTrue(all(d.BASE not in v for v in d.PATHS.values()))
        self.assertEqual(d.TEMPLATE['target'], d.BASE)
        self.assertEqual(d.TEMPLATE['full_raw_G10_acceptance']['sha256'],
            '3dabf439073b1c49b4b0be7e9885100a214f101443e3b7a983ddca1fbbf40b1b')

    def test_prepare_start_status_scripts_compile_and_no_GPU_submit(self):
        scripts = []
        def capture(script, *args, **kwargs):
            compile(script, '<actual-control>', 'exec')
            scripts.append(script)
            return {'test_only': True}
        with patch.object(d, 'remote', capture):
            for action in ('prepare', 'start', 'status'):
                d.action(action, 'a' * 40)
        self.assertEqual(len(scripts), 3)
        self.assertIn('--gres=gpu:0', scripts[1])
        self.assertIn('JobState=RUNNING', scripts[1])
        self.assertIn("job='745501'", scripts[1])
        self.assertNotIn('sbatch', scripts[1].split('args=')[1])


if __name__ == '__main__':
    unittest.main()
