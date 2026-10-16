"""Regression for the actual200 long-JXU/early-MOU freshness failure."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import owner_partition_storage_v2 as storage


class SerialAdmission(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.path=self.root/'request.json'
        self.uid=os.getuid();self.parent=dict(pid=100,start_ticks=20,sid=100,uid=self.uid)
        self.request=dict(archive_child=self.parent,excluded_paths=[],created_unix=0)
        self.du=dict(actual_parent=self.parent,actual_du_wait_exit_code=0,stderr='',
                     stdout='4096\t'+str(self.root)+'\n',started_unix=1,completed_unix=700)
        self.proof={'sha256':'a'*64}
        self.patches=[patch.object(storage,'JXU',self.uid),patch.object(storage,'MEASURE',self.root),
                      patch.object(storage.time,'time',return_value=710)]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def ready(self,value=None):
        (self.root/'jxu-du-result.json').write_text('{}')
        with patch.object(storage,'consume_du',return_value=(value or self.du,self.proof)):
            return storage.complement_ready_for_owner(self.path,self.request)

    def test_missing_result_is_wait_without_reading_partial_streams(self):
        (self.root/'jxu-du-stdout').write_text('1234\tpartial')
        with patch.object(storage,'consume_du') as consume:
            self.assertIsNone(storage.complement_ready_for_owner(self.path,self.request))
            consume.assert_not_called()

    def test_actual_wait0_complete_coverage_is_required_before_grant(self):
        self.assertEqual(self.ready(),self.proof)
        self.assertIsNone(self.ready(dict(self.du,actual_du_wait_exit_code=1)))
        self.assertIsNone(self.ready(dict(self.du,stderr='No such file or directory')))
        for stdout in ('4096\t'+str(self.root), '4096\t/another/root\n',
                       self.du['stdout']+self.du['stdout']):
            with self.subTest(stdout=stdout),self.assertRaises(ValueError):
                self.ready(dict(self.du,stdout=stdout))

    def test_wrong_parent_request_time_stale_or_live_du_refuses(self):
        for value in (dict(self.du,actual_parent=dict(self.parent,pid=101)),
                      dict(self.du,started_unix=-1)):
            with self.assertRaisesRegex(ValueError,'another request/child'):self.ready(value)
        with patch.object(storage.time,'time',return_value=1301):
            with self.assertRaisesRegex(ValueError,'stale'):self.ready()
        (self.root/'jxu-du-result.json').write_text('{}')
        with patch.object(storage,'consume_du',side_effect=ValueError('original owner DU is still live')):
            with self.assertRaisesRegex(ValueError,'still live'):
                storage.complement_ready_for_owner(self.path,self.request)

    def test_original600_and900_bounds_remain_and_serial_order_avoids_race(self):
        self.assertEqual(storage.MAX_RESULT_AGE_SECONDS,600)
        self.assertEqual(storage.MAX_TRANSACTION_SECONDS,900)
        # Actual failure shape: the early MOU result is589s old initially;
        # a full graph read pushes it beyond600s. Serial MOU stays fresh.
        old_owner={'completed_unix':101};late_owner={'completed_unix':705}
        self.assertTrue(storage.fresh(self.request,self.du,old_owner,700))
        self.assertFalse(storage.fresh(self.request,self.du,old_owner,712))
        self.assertTrue(storage.fresh(self.request,self.du,late_owner,712))
        self.assertFalse(storage.fresh(self.request,self.du,late_owner,901))


@unittest.skipUnless(sys.platform=='linux','actual GNU DU runs on the published Linux runtime')
class ActualHeldComplement(unittest.TestCase):
    def test_own_real_subprocess_and_full_streams_precede_owner_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();scan=root/'scan';scan.mkdir();(scan/'payload').write_bytes(b'x'*8192)
            records=root/'records';records.mkdir();uid=os.getuid()
            with patch.object(storage,'JXU',uid),patch.object(storage,'MEASURE',scan):
                started=time.time();command=storage.du_command([scan],[])
                value=storage.run_du(records,'jxu-du',command,uid)
                self.assertEqual(value['actual_du_wait_exit_code'],0)
                self.assertEqual(value['stderr'],'')
                request=dict(archive_child=value['actual_parent'],excluded_paths=[],created_unix=started)
                proof=storage.complement_ready_for_owner(records/'request.json',request)
                self.assertEqual(proof['sha256'],storage.digest(records/'jxu-du-result.json'))
                self.assertFalse(storage.live(value['actual_du']))
                with (records/'jxu-du-stdout').open('ab') as f:f.write(b'bad')
                with self.assertRaisesRegex(ValueError,'complete DU bytes'):
                    storage.complement_ready_for_owner(records/'request.json',request)


if __name__=='__main__':unittest.main()
