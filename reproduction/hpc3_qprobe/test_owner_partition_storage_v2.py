"""Owner visibility, nonce/TTL, terminal-proof and failed-scan ordering attacks."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import owner_partition_storage_v2 as storage
from reproduction.hpc3_qprobe import test_owner_partition_storage as legacy


class StorageV2Mixin:
    def setUp(self):
        self.storage_patch=patch.object(legacy,'storage',storage);self.storage_patch.start()
        self.addCleanup(self.storage_patch.stop)
        super().setUp()


class Contracts(StorageV2Mixin,legacy.ContractTests):pass
class ActualGNU(StorageV2Mixin,legacy.ActualGNUPartitions):pass
class ActualChildren(StorageV2Mixin,legacy.ActualHeldChildren):
    def test_binding_publication_failure_still_reaps_the_real_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();children=[];original=storage.subprocess.Popen;publish=storage.publish
            def popen(*a,**kw):
                child=original(*a,**kw);children.append(child);return child
            def fail_binding(path,value):
                if Path(path).name.endswith('-binding.json'):raise OSError('actual binding publication failure')
                return publish(path,value)
            with patch.object(storage.subprocess,'Popen',side_effect=popen),patch.object(storage,'publish',side_effect=fail_binding):
                with self.assertRaisesRegex(OSError,'publication failure'):
                    storage.run_du(root,'actual',[sys.executable,'-c','import time;time.sleep(.2)'],os.getuid())
            self.assertEqual(len(children),1);self.assertEqual(children[0].returncode,0)
            self.assertFalse(storage.live(dict(pid=children[0].pid,start_ticks=1,sid=1,uid=os.getuid())))
            self.assertTrue((root/'actual-launch-intent.json').exists())


class OwnerProof(unittest.TestCase):
    write=legacy.HeldOwnerProof.write
    alter=legacy.HeldOwnerProof.alter

    def setUp(self):
        with patch.object(legacy,'storage',storage):legacy.HeldOwnerProof.setUp(self)
        self.stack.enter_context(patch.object(storage,'JXU',os.getuid()))
        self.host=dict(host='fixture',boot='fixture',namespace='fixture')
        self.stack.enter_context(patch.object(storage,'host_binding',return_value=self.host))
        self.stack.enter_context(patch.object(storage,'code_guard',return_value=(self.root,self.source)))
        self.request.update(archive_child=dict(pid=2147481000,start_ticks=1,sid=2147481000,uid=os.getuid()),
            archive_claim_sha256='e'*64,archive_token='f'*32,created_unix=0)
        self.request_path.write_text(json.dumps(self.request));self.request_path.chmod(0o640)
        self.proof=dict(path=str(self.request_path),sha256=storage.digest(self.request_path),bytes=self.request_path.stat().st_size)
        self.grant_dir=self.root/'authorizations';self.grant_dir.mkdir();self.nonce='a'*32
        self.grant_path=self.grant_dir/(self.nonce+'.json')
        self.grant=dict(protocol=storage.PROTOCOL,source=self.source,request=self.proof,archive_child=self.request['archive_child'],
            archive_claim_sha256=self.request['archive_claim_sha256'],archive_token=self.request['archive_token'],
            host=self.host,nonce=self.nonce,authorized_by=self.request['archive_child'],created_unix=9,
            ttl_seconds=storage.AUTHORIZATION_TTL,training_actions=0,permission_changes=0)
        jxu=dict(actual_parent=self.request['archive_child'],actual_du=dict(pid=2147481001,start_ticks=2,
            sid=2147481000,uid=os.getuid()),command=storage.du_command([storage.MEASURE],self.request['excluded_paths']),
            actual_du_wait_exit_code=0,uid=os.getuid(),started_unix=1,completed_unix=8,training_actions=0,permission_changes=0)
        for stream,raw in [('stdout',('4096\t'+str(storage.MEASURE)+'\n').encode()),('stderr',b'')]:
            (self.root/('jxu-du-'+stream)).write_bytes(raw)
            jxu.update({stream:raw.decode(),stream+'_bytes':len(raw),stream+'_sha256':hashlib.sha256(raw).hexdigest()})
        storage.publish(self.root/'jxu-du-result.json',jxu)
        storage.publish(self.root/'jxu-du-binding.json',{k:jxu[k] for k in ('actual_parent','actual_du','command','started_unix')})
        storage.publish(self.root/'jxu-du-launch-intent.json',{k:jxu[k] for k in ('actual_parent','command','started_unix')})
        _,jp=storage.document(self.root/'jxu-du-result.json',(os.getuid(),),protected=True)
        _,jb=storage.document(self.root/'jxu-du-binding.json',(os.getuid(),),protected=True)
        _,jl=storage.document(self.root/'jxu-du-launch-intent.json',(os.getuid(),),protected=True)
        self.grant['completed_complement']=dict(jp,binding=jb,launch_intent=jl)
        storage.publish(self.grant_path,self.grant);_,grant_proof=storage.document(self.grant_path,(os.getuid(),),protected=True)
        storage.publish(self.root/'authorization-active.json',grant_proof)
        for name in ('claim.json','process.json','child.json','worker-result.json','complete.json'):
            path=self.folder/name;value=json.loads(path.read_bytes());value['request']=self.proof
            if name=='claim.json':value.update(authorization=grant_proof,authorization_nonce=self.nonce,created_unix=11)
            path.write_text(json.dumps(value))
        for path in self.folder.glob('*.json'):path.chmod(0o640)
        result=json.loads((self.folder/'mou-du-result.json').read_bytes())
        storage.publish(self.folder/'mou-du-launch-intent.json',
            {key:result[key] for key in ('actual_parent','command','started_unix')})
        self.stack.enter_context(patch.object(storage,'request_guard',return_value=(self.request,self.proof)))
        reporter=dict(pid=os.getpid(),sid=os.getpid(),start_ticks=123,uid=os.getuid(),state='S')
        self.stack.enter_context(patch.object(storage,'process',side_effect=lambda pid:reporter if pid==os.getpid() else None))

    def confirm(self):return storage.owner_confirm(self.request_path,'c'*40)

    def consume(self,**kwargs):return storage.consume_owner(self.request_path,self.request,self.proof,**kwargs)

    def test_foreign_proc_never_consulted_after_actual_owner_terminal_proof(self):
        self.confirm()
        with patch.object(storage.os,'getuid',return_value=os.getuid()+1),\
             patch.object(storage,'live',side_effect=AssertionError('foreign /proc was consulted')):
            result=self.consume()
        self.assertEqual(result['allocated_bytes'],36864)
        self.assertEqual(result['actual_terminal_births'],[self.supervisor,self.task,self.du])
        self.assertEqual(result['authorization_nonce'],self.nonce)
        self.assertTrue(result['terminal_confirmation']['sha256'])

    def test_missing_foreign_receipt_is_wait_not_false_death_or_acceptance(self):
        with patch.object(storage,'live',side_effect=AssertionError('foreign /proc was consulted')):
            self.assertIsNone(self.consume())
        (self.folder/'worker-result.json').unlink()
        self.assertIsNone(self.confirm().get('sha256'))

    def test_each_live_original_owner_birth_prevents_terminal_confirmation(self):
        for live in (self.supervisor,self.task,self.du):
            with self.subTest(live=live),patch.object(storage,'live',side_effect=lambda row:row==live):
                if live==self.du:
                    with self.assertRaisesRegex(ValueError,'DU is still live'):self.confirm()
                else:self.assertEqual(self.confirm()['status'],'ACTUAL_OWN_HELPERS_NOT_YET_TERMINAL')
                self.assertFalse((self.folder/'terminal-confirmation.json').exists())

    def test_expired_grant_cannot_admit_but_historical_valid_claim_can_be_read(self):
        with patch.object(storage.time,'time',return_value=1000):
            with self.assertRaisesRegex(ValueError,'expired at first claim'):
                storage.authorization_guard(self.request_path,self.request,self.proof)
            self.confirm();self.assertEqual(self.consume()['allocated_bytes'],36864)

    def test_nonce_host_source_claim_request_and_shared_write_attacks_refuse(self):
        original=self.grant_path.read_bytes()
        for key,bad in (('nonce','b'*32),('host',{}),('source',{}),('request',{}),('archive_token','wrong')):
            self.grant_path.write_text(json.dumps(dict(self.grant,**{key:bad})))
            with self.subTest(key=key),self.assertRaises(ValueError):self.confirm()
            self.grant_path.write_bytes(original)
        self.grant_path.chmod(0o660)
        with self.assertRaisesRegex(ValueError,'protected owner mode'):self.confirm()

    def test_new_owner_claim_cannot_use_grant_missing_completed_complement(self):
        value=dict(self.grant);value.pop('completed_complement')
        self.grant_path.write_text(json.dumps(value))
        _,proof=storage.document(self.grant_path,(os.getuid(),),protected=True)
        (self.root/'authorization-active.json').write_text(json.dumps(proof))
        with patch.object(storage.time,'time',return_value=12):
            with self.assertRaisesRegex(ValueError,'completed complement proof differs'):
                storage.authorization_guard(self.request_path,self.request,self.proof)

    def test_owner_grant_rejects_modified_complement_sha_nonzero_wait_and_streams(self):
        self.confirm()
        path=self.root/'jxu-du-result.json';original=path.read_bytes()
        value=json.loads(original);value['actual_du_wait_exit_code']=1;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'complement proof differs'):self.consume()
        path.write_bytes(original)
        (self.root/'jxu-du-stdout').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'complement streams differ'):self.consume()

    def test_confirmed_report_cannot_replace_modified_original_held_bytes(self):
        self.confirm();self.alter('worker-result.json','actual_task_wait_exit_code',1)
        with self.assertRaisesRegex(ValueError,'owner helper failed'):self.consume()

    def test_confirmed_original_claim_process_child_full_bytes_cannot_change(self):
        self.confirm()
        for name in ('claim.json','process.json','child.json'):
            path=self.folder/name;raw=path.read_bytes();value=json.loads(raw)
            value['extra_same_length_byte']='a'
            path.write_text(json.dumps(value))
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'original held bytes'):self.consume()
            path.write_bytes(raw)
        raw=(self.folder/'claim.json').read_bytes();value=json.loads(raw);value['created_unix']=12
        (self.folder/'claim.json').write_text(json.dumps(value))
        self.assertEqual((self.folder/'claim.json').stat().st_size,len(raw))
        with self.assertRaisesRegex(ValueError,'original held bytes'):self.consume()
        (self.folder/'claim.json').write_bytes(raw)

    def test_same_length_synchronized_command_tamper_cannot_preserve_confirmation(self):
        self.confirm()
        for name in ('claim.json','child.json'):
            path=self.folder/name;raw=path.read_bytes();value=json.loads(raw)
            value['command'][0]='X'+value['command'][0][1:];path.write_text(json.dumps(value))
            self.assertEqual(path.stat().st_size,len(raw))
        with self.assertRaisesRegex(ValueError,'original held bytes'):self.consume()

    def test_original_du_binding_and_launch_intent_full_bytes_are_pinned(self):
        self.confirm()
        for name in ('mou-du-binding.json','mou-du-launch-intent.json'):
            path=self.folder/name;raw=path.read_bytes();value=json.loads(raw)
            path.write_text(json.dumps(value,indent=1))
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'original held bytes'):self.consume()
            path.write_bytes(raw)

    def test_failed_owner_without_du_result_but_launch_intent_or_streams_is_unknown(self):
        self.alter('worker-result.json','actual_task_wait_exit_code',1)
        for name in ('mou-du-result.json','mou-du-binding.json','complete.json'):(self.folder/name).unlink()
        with self.assertRaisesRegex(ValueError,'launch outcome unknown'):self.confirm()
        (self.folder/'mou-du-launch-intent.json').unlink()
        with self.assertRaisesRegex(ValueError,'launch outcome unknown'):self.confirm()
        (self.folder/'mou-du-stdout').unlink();(self.folder/'mou-du-stderr').unlink()
        self.confirm();self.assertEqual(self.consume(require_usage=False)['actual_terminal_births'],[self.supervisor,self.task])

    def test_failed_owner_and_du_are_terminal_evidence_but_never_usage(self):
        self.alter('worker-result.json','actual_task_wait_exit_code',1)
        self.alter('mou-du-result.json','actual_du_wait_exit_code',1)
        (self.folder/'complete.json').unlink()
        self.confirm()
        actual=self.consume(require_usage=False,current=False)
        self.assertEqual(actual['actual_task_wait_exit_code'],1);self.assertEqual(actual['actual_du_wait_exit_code'],1)
        self.assertIsNone(actual['allocated_bytes'])
        with self.assertRaisesRegex(ValueError,'no usage'):self.consume()

    def test_terminal_confirmation_mode_host_and_exact_birth_set_attacks_refuse(self):
        self.confirm();path=self.folder/'terminal-confirmation.json';original=path.read_bytes()
        value=json.loads(original)
        for key,bad in (('host',{}),('source',{}),('request',{})):
            path.write_text(json.dumps(dict(value,**{key:bad})))
            with self.subTest(key=key),self.assertRaises(ValueError):self.consume()
            path.write_bytes(original)
        changed=copy.deepcopy(value);changed['owner_evidence']['actual_terminal_births'].append(self.task)
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'exact terminal birth set'):self.consume()
        path.write_bytes(original);path.chmod(0o660)
        with self.assertRaisesRegex(ValueError,'protected owner mode'):self.consume()

    def test_closed_request_cannot_admit_owner_or_reuse_nonce(self):
        storage.publish(self.root/'request-closed.json',dict(closed=True))
        with patch.object(storage.time,'time',return_value=12),self.assertRaisesRegex(ValueError,'already closed'):
            storage.authorization_guard(self.request_path,self.request,self.proof)


class FailedComplementOrdering(unittest.TestCase):
    def test_v2_cannot_be_combined_with_legacy_storage_or_used_for_other_steps(self):
        from reproduction.hpc3_qprobe import final_control as control
        with patch.object(control,'source') as source:
            for args in (dict(step=100,owner_partition_v2_protocol=True),
                         dict(step=150,owner_partition_v2_protocol=True,owner_partition_protocol=True),
                         dict(step=200,owner_partition_v2_protocol=True,storage_wait_protocol=True)):
                with self.subTest(args=args),self.assertRaises(ValueError):control.launch_archive('/fixture','a'*40,**args)
            source.assert_not_called()
    def test_new_denial_is_closed_and_full_attempt_advanced_without_owner_claim_or_wait(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp).resolve();archive=root/'archive-150';archive.mkdir()
            paths=[root/'private1',root/'private2']
            for path in paths:path.mkdir(mode=0o700)
            uid=os.getuid();source=dict(commit='a'*40)
            own=dict(pid=123,start_ticks=5,sid=123,uid=uid,state='S')
            claim=dict(source=source,step=150,token='token',storage_wait_protocol=storage.PROTOCOL)
            claim_path=archive/'submit-claim.json';claim_path.write_text('{}')
            for name,value in (('JXU',uid),('MOU',uid),('GID',os.getgid()),('MEASURE',root),
                               ('ALLOWED',re.compile(re.escape(str(root))+r'/private[12]'))):
                stack.enter_context(patch.object(storage,name,value))
            stack.enter_context(patch.object(storage,'code_guard',return_value=(root,source)))
            stack.enter_context(patch.object(storage,'process',return_value=own))
            stack.enter_context(patch.object(storage,'initial_frontier',return_value=([paths[0]],{})))
            stack.enter_context(patch.object(storage,'document',side_effect=lambda path,*a,**k:
                (claim,dict(sha256='b'*64)) if Path(path)==claim_path else
                (dict(actual_process=storage.birth(own),claim_sha256='b'*64),{})))
            consume=stack.enter_context(patch.object(storage,'consume_owner',side_effect=AssertionError('useless owner wait')))
            calls=[]
            def scan(directory,*args):
                calls.append(directory)
                if len(calls)==2:raise RuntimeError('new complete attempt reached')
                value=dict(actual_du_wait_exit_code=1,stderr="du: cannot read directory '"+str(paths[1])+"': Permission denied\n")
                storage.publish(directory/'jxu-du-result.json',value);return value
            stack.enter_context(patch.object(storage,'run_du',side_effect=scan))
            with self.assertRaisesRegex(RuntimeError,'new complete attempt reached'):
                storage.measure(claim_path,'token',storage.RUN,150)
            consume.assert_not_called()
            first=archive/'owner-storage-attempt-1';second=archive/'owner-storage-attempt-2'
            closed=json.loads((first/'request-closed.json').read_bytes())
            self.assertFalse(closed['owner_claim_present'])
            self.assertFalse(json.loads((first/'failed.json').read_bytes())['usage_accepted'])
            request=json.loads((second/'request.json').read_bytes())
            self.assertEqual(set(request['excluded_paths']),set(map(str,paths)))
            self.assertFalse((archive/'storage-owner-partition-accepted.json').exists())

    def test_foreign_live_api_fails_closed_and_ray_smoke_scope_is_exact(self):
        with self.assertRaisesRegex(ValueError,'foreign /proc'):
            storage.live(dict(pid=1,uid=os.getuid()+1))
        root=str(storage.MEASURE)+'/qprobe-teacher-warmup200-periodic25-thread5-20261016-r16/tmp/'
        self.assertTrue(storage.ALLOWED.fullmatch(root+'ray-thread-smoke-72d37ba2'))
        self.assertFalse(storage.ALLOWED.fullmatch(root+'ray-thread-smoke-not-observed'))


@unittest.skipUnless(sys.platform=='linux' and Path('/proc/self/stat').exists(),'real Linux launch/lock lifecycle required')
class ActualLaunchAndLock(unittest.TestCase):
    def test_real_worker_loss_after_popen_before_binding_is_unknown_and_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();marker=root/'actual-child.json'
            script=r'''import json,os,sys,time
from pathlib import Path
from reproduction.hpc3_qprobe import owner_partition_storage_v2 as s
root=Path(sys.argv[1]);original=s.subprocess.Popen
def window(*args,**kwargs):
 child=original(*args,**kwargs)
 (root/'actual-child.json').write_text(json.dumps(s.birth(s.process(child.pid))))
 time.sleep(120)
 return child
s.subprocess.Popen=window
s.run_du(root,'window',[sys.executable,'-c','import time;time.sleep(3)'],os.getuid())
'''
            worker=subprocess.Popen([sys.executable,'-c',script,str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                deadline=time.monotonic()+10
                while not marker.exists():
                    self.assertIsNone(worker.poll());self.assertLess(time.monotonic(),deadline);time.sleep(.01)
                actual=json.loads(marker.read_bytes());self.assertTrue(storage.live(actual))
                worker.terminate();self.assertEqual(worker.wait(timeout=5),-15)
                self.assertTrue(storage.live(actual))
                self.assertTrue((root/'window-launch-intent.json').exists())
                self.assertFalse((root/'window-binding.json').exists())
                with self.assertRaisesRegex(ValueError,'launch outcome unknown'):storage.no_du_launched(root,'window')
                # The deliberately orphaned tiny test child exits naturally;
                # no PID or process-group signal is sent to it.
                deadline=time.monotonic()+6
                while storage.live(actual):
                    self.assertLess(time.monotonic(),deadline);time.sleep(.02)
                with self.assertRaisesRegex(ValueError,'launch outcome unknown'):storage.no_du_launched(root,'window')
            finally:
                if worker.poll() is None:worker.terminate();worker.wait(timeout=5)
                worker.stdout.close();worker.stderr.close()

    def test_real_0640_readonly_fd_exclusive_flock_blocks_then_releases_on_actual_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(storage,'JXU',os.getuid()),patch.object(storage,'GID',os.getgid()):
            root=Path(tmp).resolve();request=root/'request.json'
            lock=root/'owner-admission.lock';lock.touch();lock.chmod(0o640)
            script=r'''import json,os,sys
from pathlib import Path
from reproduction.hpc3_qprobe import owner_partition_storage_v2 as s
root=Path(sys.argv[1]);s.JXU=os.getuid();s.GID=os.getgid();original=s.os.open;flags=[]
def opened(path,mode,*args):
 flags.append(mode);return original(path,mode,*args)
s.os.open=opened
(root/'ready').write_text('ready')
with s.admission_lock(root/'request.json'):
 (root/'acquired').write_text(json.dumps(dict(readonly=(flags[-1]&os.O_ACCMODE)==os.O_RDONLY,filesystem_device=root.stat().st_dev)))
'''
            with storage.admission_lock(request):
                child=subprocess.Popen([sys.executable,'-c',script,str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                deadline=time.monotonic()+10
                while not (root/'ready').exists():
                    self.assertIsNone(child.poll());self.assertLess(time.monotonic(),deadline);time.sleep(.01)
                time.sleep(.1);self.assertIsNone(child.poll());self.assertFalse((root/'acquired').exists())
            try:
                self.assertEqual(child.wait(timeout=10),0,child.stderr.read().decode())
                actual=json.loads((root/'acquired').read_bytes());self.assertTrue(actual['readonly'])
                self.assertEqual(actual['filesystem_device'],root.stat().st_dev)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
                child.stdout.close();child.stderr.close()


if __name__=='__main__':unittest.main()
