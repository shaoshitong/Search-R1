import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from reproduction.hpc3_qprobe import raw_audit_followthrough_sigma3 as audit
from reproduction.hpc3_qprobe import raw_actor_readonly_sigma3 as reader

class Sigma3RawPlan(unittest.TestCase):
    def test_only_actual_fresh_prefix_is_verified_but_future200_is_declared(self):
        run={'run_id':'a'*32}
        history=Mock(return_value=[dict(base=audit.BASE,run=run,first=1,last=1)])
        binding=dict(producer_job='721485',producer_commit='aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12')
        with patch.dict(sys.modules,{'reproduction.hpc3_qprobe.verify_goal':SimpleNamespace(history_parts=history)}),patch.object(audit.ctl,'read',return_value=run),patch.object(audit,'rows',return_value=[dict(step=1)]),patch.object(audit.ctl,'producer',return_value=(run,binding,'RUNNING')):
            self.assertEqual(audit.plan(),[dict(first=1,last=200,producer=binding)])
        history.assert_called_once_with(audit.BASE,run,last=1)

    def test_raw_entry_has_single_fresh_producer_and_rejects201_before_read(self):
        self.assertEqual(reader.HISTORY,((1,200,audit.BASE.name,'aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12','721485'),))
        with self.assertRaises(ValueError):reader.main(201,'a'*32)

if __name__=='__main__':unittest.main()
