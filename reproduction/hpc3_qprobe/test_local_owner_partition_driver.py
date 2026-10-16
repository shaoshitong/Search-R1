"""Driver account/scope admission, unknown outcomes and read-only decisions."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import local_owner_partition_driver as local
from reproduction.hpc3_qprobe import owner_partition_storage_v2 as storage
from reproduction.hpc3_qprobe import final_control as ctl

COMMIT='c'*40
SOURCE={'commit':COMMIT}


class DriverOnce(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.driver=local.Driver(COMMIT,self.root,self.root)
        self.request={'path':str(local.ROOT/'final-control'/local.RUN/'archive-150/owner-storage-attempt-1/request.json'),
                      'sha256':'a'*64,'bytes':1}

    def state(self,action):
        return dict(source=SOURCE,protocol=local.PROTOCOL,step=150,training_actions=0,permission_changes=0,
                    goal_complete=False,action=action,request=self.request)

    def test_accounts_and_paths_and_quoted_command(self):
        for account,mode in [('jxu957','authorize-owner'),('mou176','owner-start'),('mou176','owner-confirm')]:
            args=shlex.split(local.command(account,mode,COMMIT,request=self.request['path']))
            self.assertIn(str(local.ROOT/'source'/COMMIT),[a.split('=',1)[1] for a in args if a.startswith('QPROBE_SOURCE=')])
            self.assertIn('/usr/bin/python3' if account=='mou176' else str(local.BASE/'env/bin/python'),args)
            self.assertEqual(args[-2:],['--request',self.request['path']])
        for account,mode in [('mou176','authorize-owner'),('jxu957','owner-start'),('jxu957','owner-confirm')]:
            with self.assertRaisesRegex(ValueError,'role differs'):local.command(account,mode,COMMIT,request=self.request['path'])
        for path in ['/tmp/request.json',self.request['path'].replace('archive-150','archive-100'),
                     self.request['path'].replace('attempt-1/','attempt-17/')]:
            with self.assertRaisesRegex(ValueError,'bounded scope'):local.command('mou176','owner-start',COMMIT,request=path)

    def test_unknown_start_is_durably_intended_and_never_replayed(self):
        result=subprocess.CompletedProcess([],124,'','observation timeout')
        with patch.object(local.bridge,'run',return_value=result) as run:
            with self.assertRaisesRegex(ValueError,'outcome unknown'):self.driver.act(150,self.state('START_OWNER'))
            self.assertTrue((self.root/('owner-start-'+'a'*64+'.json')).is_file())
            with self.assertRaisesRegex(ValueError,'already attempted'):self.driver.act(150,self.state('START_OWNER'))
            self.assertEqual(run.call_count,1)

    def test_confirm_and_authorize_roles_and_completed_is_not_goal(self):
        with patch.object(self.driver,'call',return_value={}) as call:
            self.assertFalse(self.driver.act(150,self.state('AUTHORIZE_OWNER')))
            self.assertFalse(self.driver.act(150,self.state('CONFIRM_OWNER')))
            self.assertEqual([c.args[:2] for c in call.call_args_list],[('jxu957','authorize-owner'),('mou176','owner-confirm')])
            self.assertTrue(self.driver.act(150,self.state('ARCHIVE_COMPLETED')))

    def test_proven_closed_request_refusal_follows_new_request_without_replay(self):
        refusal=dict(status='ACTUAL_OWNER_REQUEST_NOT_ADMITTED',protocol=local.PROTOCOL,source=SOURCE,
            request=self.request,owner_claim_created=False,reason='closed_or_superseded_request_with_no_owner_claim',
            training_actions=0,permission_changes=0)
        with patch.object(self.driver,'call',return_value=refusal) as call:
            self.assertFalse(self.driver.act(150,self.state('AUTHORIZE_OWNER')))
            self.assertFalse(self.driver.act(150,self.state('START_OWNER')))
            with self.assertRaisesRegex(ValueError,'already attempted'):self.driver.act(150,self.state('START_OWNER'))
            self.assertEqual(call.call_count,2)
            self.request=dict(self.request,path=self.request['path'].replace('attempt-1','attempt-2'),sha256='b'*64)
            call.return_value=dict(refusal,request=self.request)
            self.assertFalse(self.driver.act(150,self.state('START_OWNER')))
            self.assertEqual(call.call_count,3)

    def test_unknown_or_failed_state_and_source_mismatch_never_launch(self):
        with patch.object(self.driver,'call') as call:
            for action in ['ARCHIVE_FAILED','UNKNOWN_OWNER_ADMISSION','UNKNOWN_ARCHIVE_ADMISSION','not-known']:
                with self.subTest(action=action),self.assertRaises(ValueError):self.driver.act(150,self.state(action))
            state=self.state('START_OWNER');state['source']={'commit':'d'*40}
            with self.assertRaisesRegex(ValueError,'source/scope'):self.driver.act(150,state)
            call.assert_not_called()

    def test_long_complement_wait_cannot_authorize_or_start_owner(self):
        with patch.object(self.driver,'call') as call:
            self.assertFalse(self.driver.act(150,self.state('WAIT_FOR_JXU_PARTITION')))
            call.assert_not_called()
        self.assertFalse(list(self.root.glob('owner-start-*.json')))

    def test_explicit200_scope_does_not_inspect_or_replay_old150(self):
        folder=self.root/'new200-driver';seen=[]
        driver=local.Driver(COMMIT,self.root,folder)
        def inspect(account,mode,**kwargs):
            seen.append((account,mode,kwargs))
            return dict(self.state('ARCHIVE_COMPLETED'),step=200)
        with patch.object(local,'source_guard'),patch.object(driver,'call',side_effect=inspect):
            self.assertEqual(driver.run(steps=(200,)),0)
        self.assertEqual(seen,[('jxu957','inspect-storage',{'step':200})])
        self.assertEqual(json.loads((folder/'claim.json').read_text())['steps'],[200])
        self.assertEqual(json.loads((folder/'result.json').read_text())['steps'],[200])
        with patch.object(local,'source_guard'):
            for steps in ((150,),(200,150),(100,200),()):
                with self.subTest(steps=steps),self.assertRaisesRegex(ValueError,'explicit new200'):
                    driver.run(steps=steps)


class InspectionDecisions(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.archive=self.root/'final-control'/storage.RUN/'archive-150'
        self.archive.mkdir(parents=True)
        self.claim=dict(source=SOURCE,step=150,storage_wait_protocol=storage.PROTOCOL,role='detached_cpu_pair_archive',
                        producer=dict(run_id=storage.RUN,producer_job='721485'),training_permitted=False)
        self.request_path=self.archive/'owner-storage-attempt-1/request.json';self.request_path.parent.mkdir()
        self.proof=dict(path=str(self.request_path),sha256='a'*64,bytes=1)
        self.child=dict(pid=111,start_ticks=11,sid=110,uid=storage.JXU)
        self.request=dict(attempt=1,step=150,source=SOURCE,archive_child=self.child,roots=['fixture'])
        self.state=dict(actual_task_exit_code=None,task_alive=True,actual_task=self.child)
        self.files={str(self.archive/'submit-claim.json'):self.claim,
                    str(self.archive/'owner-storage-active.json'):dict(request_path=str(self.request_path),
                        request_sha256='a'*64,attempt=1,archive_child=self.child)}
        for name in self.files:Path(name).write_text('{}')
        self.patches=[patch.object(storage,'ROOT',self.root),patch.object(storage,'code_guard',return_value=(self.root,SOURCE)),
                      patch.object(ctl,'controller_state',return_value=self.state),
                      patch.object(storage,'document',side_effect=lambda path,*a,**k:(self.files[str(path)],{})),
                      patch.object(storage,'request_guard',return_value=(self.request,self.proof)),
                      patch.object(storage,'live',side_effect=AssertionError('inspection must never use foreign /proc')),
                      patch.object(storage,'complement_ready_for_owner',return_value={'sha256':'c'*64})]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_unclaimed_request_needs_explicit_jxu_grant(self):
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'AUTHORIZE_OWNER')
        (self.request_path.parent/'authorization-active.json').write_text('{}')
        with patch.object(storage,'authorization_guard',return_value=({},{})):
            self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'START_OWNER')
        with patch.object(storage,'authorization_guard',side_effect=ValueError('OWNER_PARTITION_STORAGE: owner authorization was expired at first claim')):
            self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'AUTHORIZE_OWNER')
        with patch.object(storage,'authorization_guard',side_effect=ValueError('tampered grant')):
            with self.assertRaisesRegex(ValueError,'tampered'):storage.inspect_storage(150,COMMIT)

    def test_pending_or_failed_complement_never_grants_owner(self):
        with patch.object(storage,'complement_ready_for_owner',return_value=None),\
             patch.object(storage,'authorization_guard') as grant:
            self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'WAIT_FOR_JXU_PARTITION')
            (self.request_path.parent/'authorization-active.json').write_text('{}')
            self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'WAIT_FOR_JXU_PARTITION')
            grant.assert_not_called()

    def test_owner_confirmation_uses_original_protected_graph(self):
        directory=self.request_path.parent/'mou-control';directory.mkdir();path=directory/'claim.json';path.write_text('{}')
        self.files[str(path)]=dict(source=SOURCE,request=self.proof,role='read_only_MOU_partition_du',training_actions=0,permission_changes=0)
        with patch.object(storage,'authorization_guard',return_value=({},{})):
            self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'CONFIRM_OWNER')
            (directory/'terminal-confirmation.json').write_text('{}')
            with patch.object(storage,'consume_owner',return_value={'status':'ACTUAL_OWNER_FULL_PARTITION_HELD0'}) as consume:
                self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'OWNER_CONFIRMED')
                consume.assert_called_once_with(self.request_path,self.request,self.proof,current=False,require_usage=False)

    def test_closed_or_empty_frontier_never_owner_starts(self):
        (self.request_path.parent/'request-closed.json').write_text('{}')
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'WAIT_FOR_NEXT_REQUEST')
        (self.request_path.parent/'request-closed.json').unlink();self.request['roots']=[]
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'WAIT_FOR_JXU_ONLY_SCAN')

    def test_actual_held_failure_or_success_and_bad_source(self):
        self.state.update(actual_task_exit_code=1,status='CPU_CHILD_FAILED_OR_MISSING_RECEIPT')
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'ARCHIVE_FAILED')
        self.state.update(actual_task_exit_code=0,status='ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0')
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'ARCHIVE_COMPLETED')
        self.claim['source']={'commit':'d'*40}
        with self.assertRaisesRegex(ValueError,'claim source/scope'):storage.inspect_storage(150,COMMIT)

    def test_scope_and_missing_owner_claim_are_unknown(self):
        with self.assertRaisesRegex(ValueError,'scoped to150/200'):storage.inspect_storage(100,COMMIT)
        (self.request_path.parent/'mou-control').mkdir()
        self.assertEqual(storage.inspect_storage(150,COMMIT)['action'],'UNKNOWN_OWNER_ADMISSION')

    def test_zero_claim_closed_and_superseded_admissions_are_typed_and_no_popen(self):
        closed=self.request_path.parent/'request-closed.json';closed.write_text('{}')
        self.files[str(closed)]=dict(status='ACTUAL_FAILED_COMPLEMENT_REQUEST_CLOSED',source=SOURCE,
                                   request=self.proof,owner_claim_present=False)
        with patch.object(storage,'admission_lock'),patch.object(storage.subprocess,'Popen') as popen:
            for action in (storage.authorize_owner,storage.owner_start):
                value=action(self.request_path,COMMIT)
                self.assertEqual(value['status'],'ACTUAL_OWNER_REQUEST_NOT_ADMITTED')
                self.assertFalse(value['owner_claim_created'])
            popen.assert_not_called()
        closed.unlink();new=self.archive/'owner-storage-attempt-2/request.json'
        new_proof=dict(self.proof,path=str(new),sha256='b'*64);new_request=dict(self.request,attempt=2,step=150)
        self.request['step']=150
        self.files[str(self.archive/'owner-storage-active.json')].update(request_path=str(new),request_sha256='b'*64,attempt=2)
        with patch.object(storage,'request_guard',side_effect=lambda path,*a,**k:(new_request,new_proof) if Path(path)==new else (self.request,self.proof)),\
             patch.object(storage,'admission_lock'),patch.object(storage.subprocess,'Popen') as popen:
            for action in (storage.authorize_owner,storage.owner_start):
                value=action(self.request_path,COMMIT)
                self.assertEqual(value['active_request'],new_proof)
            popen.assert_not_called()

    def test_existing_claim_or_unproven_closed_metadata_is_not_benign(self):
        (self.request_path.parent/'mou-control').mkdir()
        with self.assertRaisesRegex(ValueError,'existing owner claim'):storage.unclaimed_admission_refusal(self.request_path,COMMIT)


if __name__=='__main__':unittest.main()
