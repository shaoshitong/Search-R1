"""A failed transport must never make an old on-disk result look fresh."""
import json
import unittest
from .status_observation import accept_snapshot


class ObservationTests(unittest.TestCase):
    def test_transport_failure_and_empty_success_are_rejected(self):
        for code, text in ((255, ''), (0, 'connection closed\n'), (0, '')):
            with self.subTest(code=code, text=text), self.assertRaises(RuntimeError):
                accept_snapshot(code, text, 100.)

    def test_old_timestamp_is_rejected_even_with_successful_transport(self):
        row = dict(time=100., job_id='742046', queue='RUNNING', accounting='RUNNING')
        with self.assertRaises(RuntimeError):
            accept_snapshot(0, json.dumps(row), 100.)

    def test_fresh_pending_or_running_does_not_require_update_growth(self):
        for state in ('PENDING', 'RUNNING'):
            row = dict(time=101., job_id='742046', queue=state, accounting=state)
            self.assertEqual(accept_snapshot(0, '\n' + json.dumps(row), 100.), row)

    def test_partial_result_is_not_a_scheduler_snapshot(self):
        with self.assertRaises(RuntimeError):
            accept_snapshot(0, json.dumps(dict(time=101., job_id='742046')), 100.)


if __name__ == '__main__':
    unittest.main()
