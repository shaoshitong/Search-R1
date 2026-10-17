import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .terminal import writers_present


class TerminalTests(unittest.TestCase):
    def check(self, *, queue='', accounting=None, terminal_job='729343'):
        config = dict(execution_backend='slurm_hpc3', slurm_job='729343', slurm_node='ACD1-24')
        terminal = dict(execution_backend='slurm_hpc3', slurm_job=terminal_job, uid=os.getuid())
        if accounting is None:
            accounting = ('729343|%d|FAILED|1:0|ACD1-24\n' % os.getuid()
                          + '729343.batch||FAILED|1:0|ACD1-24\n'
                          + '729343.extern||COMPLETED|0:0|ACD1-24\n')
        with patch('subprocess.run', return_value=SimpleNamespace(
                returncode=0, stdout=queue, stderr='')), patch(
                'subprocess.check_output', return_value=accounting):
            return writers_present(terminal, config)

    def test_terminal_allocation_and_extern_prove_managed_writers_gone(self):
        self.assertEqual(self.check(), [])

    def test_queued_or_running_source_is_never_admitted(self):
        with self.assertRaises(RuntimeError):
            self.check(queue='729343|RUNNING')

    def test_unfinished_extern_is_never_treated_as_absence(self):
        rows = ('729343|%d|FAILED|1:0|ACD1-24\n' % os.getuid()
                + '729343.batch||FAILED|1:0|ACD1-24\n'
                + '729343.extern||RUNNING|0:0|ACD1-24\n')
        with self.assertRaises(RuntimeError):
            self.check(accounting=rows)

    def test_missing_accounting_or_wrong_identity_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.check(accounting='')
        with self.assertRaises(ValueError):
            self.check(terminal_job='unrelated')


if __name__ == '__main__':
    unittest.main()
