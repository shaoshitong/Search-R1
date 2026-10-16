"""Focused CPU checks for original terminal evidence and final metric accounting."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from .final_audit import expected_steps, native_state, stream_sha, validate_mixed


class FinalAuditTest(unittest.TestCase):
    def fixture(self,root):
        prep=root/'preparation';prep.mkdir()
        (prep/'submit-receipt.json').write_text(json.dumps(dict(job_id='3411')))
        (prep/'launch-config.json').write_text(json.dumps(dict(run_id='same-run')))
        proof=root/'original-terminal.json'
        raw='JobId=3411 JobState=COMPLETED ExitCode=0:0 Command='+str(prep/'native-launch.sh')
        proof.write_text(json.dumps(dict(base=str(root),job_id=3411,run_id='same-run',native_raw=raw,
            owned_births_present=0,native_cgroup_exists=False,owned_gpu_pids=[],
            latest=dict(replay_iteration=70,actual_adam_steps_per_rank=[14100]*8))))
        return proof,stream_sha(proof)['sha256']

    def test_purged_record_uses_exact_pinned_original_terminal_evidence(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);proof,digest=self.fixture(root)
            error=subprocess.CalledProcessError(1,['scontrol'],stderr=b'Invalid job id specified')
            with patch('subprocess.check_output',side_effect=error):
                result=native_state(root,proof,digest)
            self.assertEqual((result['state'],result['exit_code']),('COMPLETED','0:0'))
            self.assertEqual(result['evidence_source'],'pinned_original_native_record_after_scontrol_purge')

    def test_other_scontrol_errors_do_not_turn_into_claimed_completion(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);proof,digest=self.fixture(root)
            error=subprocess.CalledProcessError(1,['scontrol'],stderr=b'Permission denied')
            with patch('subprocess.check_output',side_effect=error):
                with self.assertRaises(subprocess.CalledProcessError):
                    native_state(root,proof,digest)

    def test_accuracy_comes_from_complete_decisions_and_final_adam_includes_origin(self):
        group=dict(count=512,threshold=.5,tp=123,fp=140,fn=60,tn=189,
                   mixed_count=183,nonmixed_count=329,accuracy=312/512)
        self.assertEqual(validate_mixed(group,512),.609375)
        self.assertEqual(expected_steps(70),[14100]*8)
        with self.assertRaises(ValueError):
            validate_mixed(dict(group,accuracy=.99),512)


if __name__=='__main__':
    unittest.main()
