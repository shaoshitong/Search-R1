"""Scoped regression for transient ledger visibility, never identity admission."""
import json
import unittest
from unittest.mock import Mock, patch

from reproduction.common8.model_smoke import _read_process_ledger


class ProcessLedgerReadTests(unittest.TestCase):
    def test_transient_absence_retries_but_permanent_absence_and_corruption_fail(self):
        with patch('reproduction.common8.model_smoke.time.sleep') as sleep:
            path = Mock()
            path.read_text.side_effect = [FileNotFoundError(), '{"run_id":"actual-run"}']
            self.assertEqual(_read_process_ledger(path), {'run_id': 'actual-run'})
            self.assertEqual(path.read_text.call_count, 2)
            sleep.assert_called_once_with(0.1)

            sleep.reset_mock()
            path = Mock()
            path.read_text.side_effect = FileNotFoundError()
            with self.assertRaises(FileNotFoundError):
                _read_process_ledger(path, attempts=3)
            self.assertEqual(path.read_text.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

            sleep.reset_mock()
            path = Mock()
            path.read_text.return_value = '{broken'
            with self.assertRaises(json.JSONDecodeError):
                _read_process_ledger(path)
            self.assertEqual(path.read_text.call_count, 1)
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
