import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.hpc3_qprobe import sigma3_experiment as e

class StageTests(unittest.TestCase):
    def test_state_changes_do_not_change_process_birth(self):
        a=dict(pid=2,start_ticks=4,sid=1,uid=14090,state='R')
        self.assertTrue(e.same_birth(a,dict(a,state='S')))
        self.assertFalse(e.same_birth(a,dict(a,start_ticks=9)))

    def test_complete_marker_does_not_replace_actual_wait_and_identity(self):
        parent=dict(pid=2,start_ticks=4,sid=2,uid=14090)
        child=dict(pid=3,start_ticks=5,sid=2,uid=14090)
        x={'sigma3-stage-controller.json':dict(commit='a',birth=parent),
           'sigma3-stage-child.json':dict(commit='a',birth=child,supervisor_birth=parent),
           'sigma3-stage-wait.json':dict(commit='a',actual_wait_exit_code=0,complete_receipt_exists=True,child_birth=child,supervisor_birth=parent),
           'sigma3-stage-complete.json':dict(result=dict(commit='a'))}
        self.assertTrue(e.accepted_stage(x))
        for name,changes in [('sigma3-stage-wait.json',dict(actual_wait_exit_code=1)),
                              ('sigma3-stage-child.json',dict(birth=dict(child,start_ticks=99))),
                              ('sigma3-stage-controller.json',dict(commit='b'))]:
            y=copy.deepcopy(x);y[name].update(changes);self.assertFalse(e.accepted_stage(y))
        del x['sigma3-stage-wait.json'];self.assertFalse(e.accepted_stage(x))

    def test_root_alias_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'original').mkdir();(root/'alias').symlink_to(root/'original')
            with patch.object(e,'BASE',root/'alias'):
                with self.assertRaisesRegex(ValueError,'ALIAS'):e.isolated_base()

if __name__=='__main__':unittest.main()
