import unittest
from reproduction.h1008_qprobe.watch_predecessor import terminal_record


class TerminalEvidenceTests(unittest.TestCase):
    def test_running_zero_is_not_terminal_success(self):
        self.assertIsNone(terminal_record('JobId=3395 JobState=RUNNING ExitCode=0:0', 3395))

    def test_actual_terminal_and_error_are_preserved(self):
        for state, code in [('COMPLETED', '0:0'), ('FAILED', '1:0'), ('CANCELLED', '0:9')]:
            raw = f'JobId=3395 JobState={state} ExitCode={code}'
            record = terminal_record(raw, 3395)
            self.assertEqual((record['state'], record['exit_code'], record['raw_scontrol']), (state, code, raw))

    def test_wrong_job_is_rejected(self):
        with self.assertRaises(ValueError):
            terminal_record('JobId=9999 JobState=COMPLETED ExitCode=0:0', 3395)


if __name__ == '__main__':
    unittest.main()
