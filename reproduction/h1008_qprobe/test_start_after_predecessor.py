import json
from pathlib import Path
import tempfile
import unittest
from reproduction.h1008_qprobe.start_after_predecessor import readiness


class ReadinessTests(unittest.TestCase):
    def test_never_launches_before_terminal_result_and_cleanup_are_all_present(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'preparation').mkdir()
            old = base / 'old'
            (old / 'journal').mkdir(parents=True)
            (old / 'logs').mkdir()
            (base / 'preparation/h1008-launch-config.json').write_text(json.dumps({'predecessor_evaluation_base': str(old)}))
            self.assertFalse(readiness(base))
            terminal = base / 'preparation/predecessor-exit0.json'
            terminal.write_text(json.dumps(dict(job_id=3395, state='COMPLETED', exit_code='0:0')))
            self.assertFalse(readiness(base))
            (old / 'journal/final-result.json').write_text('{}')
            self.assertFalse(readiness(base))
            (old / 'logs/cleanup-status.json').write_text('{}')
            self.assertTrue(readiness(base))
            terminal.write_text(json.dumps(dict(job_id=3395, state='FAILED', exit_code='1:0')))
            with self.assertRaises(RuntimeError):
                readiness(base)

    def test_unstarted_redeployment_uses_original_native_terminal_observer(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            (base / 'preparation').mkdir()
            observer = base / 'original-observer'
            (observer / 'preparation').mkdir(parents=True)
            old = base / 'old'
            (old / 'journal').mkdir(parents=True)
            (old / 'logs').mkdir()
            config = dict(predecessor_evaluation_base=str(old), predecessor_terminal_evidence_base=str(observer))
            (base / 'preparation/h1008-launch-config.json').write_text(json.dumps(config))
            (observer / 'preparation/predecessor-exit0.json').write_text(json.dumps(dict(job_id=3395, state='COMPLETED', exit_code='0:0')))
            (old / 'journal/final-result.json').write_text('{}')
            (old / 'logs/cleanup-status.json').write_text('{}')
            self.assertTrue(readiness(base))


if __name__ == '__main__':
    unittest.main()
