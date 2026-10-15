"""The user's independent review precedes any new job submission."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reproduction.h1008_qprobe import rerun_binary10 as control

class BinaryRerunTests(unittest.TestCase):
    def test_exact_scope_and_no_goal_pause_or_deletion(self):
        self.assertEqual(control.JOB,3401)
        self.assertEqual(control.RUN,'3a7a78c8ba9346379d72153a8abcd324')
        self.assertEqual(control.SOURCE,'cda611600853bd18a6c259598aa1626da2ae1d80')
        self.assertEqual(control.authority()['original_message_time'],'unknown')
        self.assertIn('goal remains active',control.authority()['scope'])
        self.assertFalse(control.authority()['deletion_executed'])

    def test_review_missing_or_blocked_never_submits(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'preparation').mkdir()
            control.once(base/'preparation/h1008-launch-config.json',{'source_commit':'new-source'})
            terminal=dict(job_id=3401,run_id=control.RUN,cleanup_true=True,cgroup_released=True)
            with patch.object(control,'validate'),patch.object(control,'capture_terminal',return_value=terminal),patch.object(control.subprocess,'run') as launch:
                with self.assertRaises(FileNotFoundError):control.submit_fresh(base)
                launch.assert_not_called()
                control.once(base/'preparation/binary-math-code-review.json',dict(reviewed_source_commit='new-source',math_approved=True,final_code_approved=False,unresolved_blocking_findings=['unresolved'],independent_reviewer='baseline512_audit'))
                with self.assertRaises(AssertionError):control.submit_fresh(base)
                launch.assert_not_called()

    def test_direct_submit_entry_cannot_bypass_binary_review(self):
        from reproduction.h1008_qprobe.submit import validate_binary_review
        import json
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'preparation').mkdir()
            config=dict(probe_architecture='qwen2_decoder3_binary_mix_v1',source_commit='new')
            with self.assertRaises(FileNotFoundError):validate_binary_review(base,config)
            path=base/'preparation/binary-math-code-review.json'
            valid=dict(reviewed_source_commit='new',math_approved=True,final_code_approved=True,
                       unresolved_blocking_findings=[],independent_reviewer='baseline512_audit')
            for bad in [dict(valid,final_code_approved=False),dict(valid,reviewed_source_commit='old'),dict(valid,unresolved_blocking_findings=['open'])]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):validate_binary_review(base,config)
            path.write_text(json.dumps(valid));validate_binary_review(base,config)

    def test_existing_stop_intent_does_not_replay_scancel(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d);(old/'preparation').mkdir()
            control.once(old/'preparation/binary-replacement-stop-intent.json',{'job_id':3401})
            with patch.object(control,'OLD',old),patch.object(control,'old_identity'),patch.object(control.subprocess,'run') as signal,patch.object(control.subprocess,'check_output') as native:
                self.assertTrue(control.request_stop(old)['existing_intent'])
                signal.assert_not_called();native.assert_not_called()

if __name__=='__main__':unittest.main()
