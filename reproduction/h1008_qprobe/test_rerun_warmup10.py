"""CPU controls tests; never contact management or touch training directories."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reproduction.h1008_qprobe import rerun_warmup10 as control


class Warmup10ControlTests(unittest.TestCase):
    def test_exact_stop_scope_and_original_time_not_invented(self):
        self.assertEqual(control.JOB,3400)
        self.assertEqual(control.RUN,'c025948ebc0947f69d86af84871a9237')
        self.assertEqual(control.SOURCE,'fb68e88b076917b2684f0fbf72d79a953c527f32')
        authority=control.authority()
        self.assertEqual(authority['original_message_time'],'unknown')
        self.assertFalse(authority['deletion_executed'])
        self.assertIn('goal remains active',authority['scope'])

    def test_once_never_overwrites_existing_intent(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'intent.json';control.once(path,{'attempt':1})
            with self.assertRaises(FileExistsError):control.once(path,{'attempt':2})
            self.assertEqual(control.read(path),{'attempt':1})

    def test_existing_stop_intent_never_replays_remote_signal(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'preparation').mkdir()
            control.once(base/'preparation/warmup10-stop-intent.json',{'job_id':3400})
            with patch.object(control,'validate'),patch.object(control.subprocess,'run') as run,patch.object(control.subprocess,'check_output') as read_native:
                result=control.request_stop(base)
            self.assertTrue(result['existing_intent']);run.assert_not_called();read_native.assert_not_called()

    def test_existing_submit_intent_never_replays_sbatch(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'preparation').mkdir()
            control.once(base/'preparation/submit-intent.json',{'attempt':1})
            terminal=dict(job_id=3400,run_id=control.RUN,cleanup_true=True,cgroup_released=True)
            with patch.object(control,'validate'),patch.object(control,'capture_terminal',return_value=terminal),patch.object(control.subprocess,'run') as run:
                result=control.submit_fresh(base)
            self.assertTrue(result['existing_submission']);run.assert_not_called()

    def test_only_code_deployment_never_deletes_actor_paths(self):
        tree=ast.parse(Path(control.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='rmtree' for n in ast.walk(tree)))
        source=Path(control.__file__).read_text()
        self.assertIn("teacher=read(cache/'identity.json')",source)
        self.assertIn("warmup_epochs=10",source)
        self.assertIn("restore_receipt=None",source)

if __name__=='__main__':unittest.main()
