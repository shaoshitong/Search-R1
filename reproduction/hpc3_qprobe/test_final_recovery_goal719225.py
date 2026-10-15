"""Explicit recovery/normal final evidence boundaries, no native execution."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import verify_goal as goal
from reproduction.hpc3_qprobe import final_control as control


class RecoveryGoalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.base=self.root/'producer';(self.base/'logs').mkdir(parents=True)
        self.run=dict(run_id='a'*32)
        self.full=dict(structural_accepted=True,actual_outer_updates=200,run_id=self.run['run_id'])
        self.write(self.base/'logs/pipeline-status.json',dict(run_id=self.run['run_id'],stage='FAILED'))
        self.write(self.base/'logs/cleanup-status.json',dict(run_id=self.run['run_id'],cleaned=True))
        self.path=self.root/'separate/structural-acceptance.json';self.write(self.path,self.full)
        self.commit='b'*40
        self.ref=dict(commit=self.commit,structural_acceptance=dict(path=str(self.path),sha256=control.digest(self.path)))

    def tearDown(self):self.temp.cleanup()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_normal_verifier_requires_original_receipt_and_finished_pipeline(self):
        with patch.object(control,'final_producer') as typed:
            with self.assertRaises(FileNotFoundError):goal.structural_acceptance(self.base,self.run)
            self.write(self.base/'logs/full-run-acceptance.json',self.full)
            full,ref,accounting=goal.structural_acceptance(self.base,self.run)
            self.assertEqual(full,self.full);self.assertIsNone(ref);self.assertIsNone(accounting)
            with self.assertRaisesRegex(ValueError,'not finished'):goal.producer_finished(self.base,self.run)
            typed.assert_not_called()

    def test_only_explicit_typed_recovery_reads_separate_structural_bytes_and_failed_pipeline(self):
        original=self.base/'logs/full-run-acceptance.json';original.write_text('preserved original failure')
        with patch.object(control,'final_producer',return_value=(self.run,{},'actual FAILED1:0',self.ref)) as typed:
            full,ref,accounting=goal.structural_acceptance(self.base,self.run,recovery_commit=self.commit)
            self.assertEqual(full,self.full);self.assertEqual(accounting,'actual FAILED1:0')
            goal.producer_finished(self.base,self.run,recovery_commit=self.commit,expected_recovery=ref)
            self.assertEqual(typed.call_count,2);self.assertEqual(original.read_text(),'preserved original failure')
            self.path.write_text(json.dumps(dict(self.full,actual_outer_updates=201)))
            with self.assertRaisesRegex(ValueError,'bytes/path differ'):
                goal.structural_acceptance(self.base,self.run,recovery_commit=self.commit)

    def test_changed_recovery_or_wrong_run_or_original_path_is_rejected(self):
        with patch.object(control,'final_producer',return_value=(self.run,{},'FAILED',dict(self.ref,commit='c'*40))):
            with self.assertRaisesRegex(ValueError,'changed during'):
                goal.producer_finished(self.base,self.run,recovery_commit=self.commit,expected_recovery=self.ref)
        with patch.object(control,'final_producer',return_value=({'run_id':'wrong'}, {}, 'FAILED', self.ref)):
            with self.assertRaisesRegex(ValueError,'another actual run'):
                goal.structural_acceptance(self.base,self.run,recovery_commit=self.commit)
        bad=copy.deepcopy(self.ref);bad['structural_acceptance']['path']=str(self.base/'logs/full-run-acceptance.json')
        with patch.object(control,'final_producer',return_value=(self.run,{},'FAILED',bad)):
            with self.assertRaisesRegex(ValueError,'bytes/path differ'):
                goal.structural_acceptance(self.base,self.run,recovery_commit=self.commit)

    def test_missing200_rejects_before_loading_typed_recovery_or_creating_output(self):
        self.write(self.base/'logs/run-manifest.json',dict(self.run,training_world_size=4))
        (self.base/'logs/actual-updates.jsonl').write_text('')
        with patch.object(goal.os,'getuid',return_value=14090),patch.object(control,'final_producer') as typed:
            with self.assertRaisesRegex(ValueError,'completed200 suffix missing'):
                goal.verify(self.base,self.root/'absent-native-audit',recovery_commit=self.commit)
            typed.assert_not_called()
            self.assertFalse((self.base/'logs/qprobe-final-goal-acceptance.json').exists())

    def test_recovered_goal_writes_only_the_existing_independent_once_scope(self):
        from reproduction.hpc3_qprobe import control as publisher
        original=self.base/'logs/qprobe-final-goal-acceptance.json';original.write_text('original receipt remains')
        value=dict(producer_final_cpu_recovery=self.ref,original_producer_accounting_remains_failed=True)
        path=self.root/'followthrough/recovery719225-finalcpu-20261015/goal-acceptance.json'
        with patch.object(control,'ROOT',self.root),patch.object(control,'OWNER',os.getuid()), \
                patch.object(publisher,'publish') as publish:
            with self.assertRaisesRegex(ValueError,'already own its once scope'):
                goal.publish_goal(self.base,value,recovery_commit=self.commit)
            publish.assert_not_called();path.parent.mkdir(parents=True)
            goal.publish_goal(self.base,value,recovery_commit=self.commit)
            publish.assert_called_once_with(path,value)
            self.assertEqual(original.read_text(),'original receipt remains')
            with self.assertRaisesRegex(ValueError,'normal goal cannot'):
                goal.publish_goal(self.base,value)

    def test_existing_separate_goal_receipt_is_never_overwritten(self):
        value=dict(producer_final_cpu_recovery=self.ref,original_producer_accounting_remains_failed=True)
        path=self.root/'followthrough/recovery719225-finalcpu-20261015/goal-acceptance.json'
        path.parent.mkdir(parents=True)
        with patch.object(control,'ROOT',self.root),patch.object(control,'OWNER',os.getuid()):
            goal.publish_goal(self.base,value,recovery_commit=self.commit)
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                goal.publish_goal(self.base,dict(value,changed=True),recovery_commit=self.commit)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse((self.base/'logs/qprobe-final-goal-acceptance.json').exists())


if __name__=='__main__':unittest.main()
