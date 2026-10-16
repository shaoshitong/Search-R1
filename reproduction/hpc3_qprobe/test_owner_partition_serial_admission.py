"""Regression for the actual200 long-JXU/early-MOU freshness failure."""
from contextlib import ExitStack
import json
import os
import re
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
            self.assertIsNone(self.ready())
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


class StaleTransactionRetry(unittest.TestCase):
    def exercise(self,phase):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp).resolve();archive=root/'archive-200';archive.mkdir()
            private=root/'private';private.mkdir(mode=0o700)
            uid=os.getuid();source={'commit':'a'*40};clock=[0.]
            own=dict(pid=123,start_ticks=5,sid=123,uid=uid,state='S')
            claim=dict(source=source,step=200,token='token',storage_wait_protocol=storage.PROTOCOL)
            claim_path=archive/'submit-claim.json';claim_path.write_text('{}')
            for name,value in [('JXU',uid),('MOU',uid),('GID',os.getgid()),('MEASURE',root),
                               ('ALLOWED',re.compile(re.escape(str(private))))]:
                stack.enter_context(patch.object(storage,name,value))
            stack.enter_context(patch.object(storage,'code_guard',return_value=(root,source)))
            stack.enter_context(patch.object(storage,'process',return_value=own))
            stack.enter_context(patch.object(storage,'live',return_value=True))
            stack.enter_context(patch.object(storage,'initial_frontier',return_value=([private],{})))
            stack.enter_context(patch.object(storage,'document',side_effect=lambda path,*a,**k:
                (claim,{'sha256':'b'*64}) if Path(path)==claim_path else
                (dict(actual_process=storage.birth(own),claim_sha256='b'*64),{})))
            stack.enter_context(patch.object(storage,'digest',side_effect=lambda p:
                'b'*64 if Path(p)==claim_path else __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()))
            stack.enter_context(patch.object(storage.time,'time',side_effect=lambda:clock[0]))
            stack.enter_context(patch.object(storage.time,'sleep'))
            scans=[];owners=[]
            def scan(directory,*args):
                scans.append(directory)
                if len(scans)==2:raise RuntimeError('next full transaction reached')
                clock[0]=901 if phase=='long_du' else 100
                value=dict(actual_parent=storage.birth(own),actual_du_wait_exit_code=0,stderr='',
                    stdout='4096\t'+str(root)+'\n',started_unix=1,completed_unix=clock[0])
                storage.publish(directory/'jxu-du-result.json',value);return value
            def owner(path,request,proof,**kwargs):
                owners.append(kwargs)
                if phase=='owner_wait':clock[0]=701;return None
                if not kwargs:clock[0]=110
                (Path(path).parent/'mou-control').mkdir(exist_ok=True)
                return {'completed_unix':110,'allocated_bytes':4096}
            def crossed(*args,**kwargs):
                if phase=='bad_graph':raise ValueError('retained complete DU bytes differ')
                clock[0]=711
                raise ValueError('OWNER_PARTITION_STORAGE: complete measurement became stale before copy admission')
            stack.enter_context(patch.object(storage,'run_du',side_effect=scan))
            stack.enter_context(patch.object(storage,'consume_owner',side_effect=owner))
            stack.enter_context(patch.object(storage,'consume_partition',side_effect=crossed))
            expected=ValueError if phase=='bad_graph' else RuntimeError
            message='retained complete DU bytes differ' if phase=='bad_graph' else 'next full transaction reached'
            with self.assertRaisesRegex(expected,message):
                storage.measure(claim_path,'token',storage.RUN,200)
            first=archive/'owner-storage-attempt-1'
            if phase=='bad_graph':
                self.assertEqual(len(scans),1)
                self.assertFalse((first/'stale.json').exists())
                self.assertFalse((archive/'owner-storage-attempt-2').exists())
                self.assertFalse((archive/'storage-owner-partition-accepted.json').exists())
                return
            closed=json.loads((first/'request-closed.json').read_text())
            self.assertEqual(closed['status'],'ACTUAL_STALE_COMPLEMENT_REQUEST_CLOSED')
            self.assertFalse(json.loads((first/'stale.json').read_text())['usage_accepted'])
            self.assertTrue((archive/'owner-storage-attempt-2/request.json').exists())
            self.assertFalse((archive/'storage-owner-partition-accepted.json').exists())
            if phase=='long_du':self.assertEqual(owners,[])
            if phase=='owner_wait':self.assertEqual(len(owners),1)
            if phase=='candidate_read':
                self.assertTrue(closed['owner_claim_present'])
                self.assertEqual(owners[-1],dict(current=False,require_usage=False))
                self.assertTrue((first/'complete.json').exists())

    def test_901_second_du_closes_without_owner_and_remeasures(self):self.exercise('long_du')
    def test_owner_wait_crossing600_cannot_deadlock(self):self.exercise('owner_wait')
    def test_full_graph_read_crossing600_retains_candidate_without_acceptance(self):self.exercise('candidate_read')
    def test_nonstale_graph_corruption_is_fatal_not_hidden_by_remeasurement(self):self.exercise('bad_graph')

    def test_claimed_owner_without_real_terminal_prevents_next_transaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp).resolve();(folder/'mou-control').mkdir()
            (folder/'owner-admission.lock').touch();(folder/'owner-admission.lock').chmod(0o640)
            request=dict(source={'commit':'a'*40});proof={'sha256':'b'*64}
            calls=[]
            def consume(*args,**kwargs):
                calls.append(kwargs)
                if len(calls)==1:return None
                raise RuntimeError('original owner still has no terminal confirmation')
            with patch.object(storage,'JXU',os.getuid()),patch.object(storage,'GID',os.getgid()),\
                 patch.object(storage,'consume_owner',side_effect=consume),patch.object(storage.time,'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'no terminal confirmation'):
                    storage.close_stale_request(folder/'request.json',request,proof)
                sleep.assert_called_once_with(5)
            self.assertTrue(json.loads((folder/'request-closed.json').read_text())['owner_claim_present'])
            self.assertFalse((folder/'stale.json').exists())
            self.assertEqual(calls,[dict(current=False,require_usage=False)]*2)


if __name__=='__main__':unittest.main()
