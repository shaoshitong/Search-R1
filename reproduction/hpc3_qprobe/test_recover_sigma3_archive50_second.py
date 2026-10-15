"""Real-byte fixtures of the observed pre-copy failure; no remote work."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import recover_sigma3_archive50_second as recovery
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder
from reproduction.hpc3_qprobe.test_independent_goal_control_sigma3 import output_fixture


class ActualFailureBytes(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.stack=ExitStack();self.addCleanup(self.stack.close)
        files=json.loads(Path(__file__).with_name('archive50_second_failure_fixture.json').read_text())
        self.blobs={}
        for folder,items in files.items():
            for name,row in items.items():
                if name.endswith('.json'):
                    candidates=[json.dumps(row['value'],indent=2).encode(),json.dumps(row['value']).encode()]
                    raw=next((b for b in candidates if hashlib.sha256(b).hexdigest()==row['sha256']),b'')
                else:raw=row['value'].encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
                original=Path(folder)/name;target=self.map(original);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw);self.blobs[original]=target
        self.claim=self.read(recovery.ARCHIVE/'submit-claim.json')
        self.fclaim=self.read(recovery.OLD_FOLLOW/'claim.json')
        self.observed=dict(actual_live_birth=False,process=self.read(recovery.OLD_FOLLOW/'process.json'),
            result=self.read(recovery.OLD_FOLLOW/'result.json'),state=self.read(recovery.OLD_FOLLOW/'state.json'))
        original_read=reader.read_file;original_snapshot=recovery.snapshot
        def read_file(path,**kwargs):return original_read(self.map(Path(path)),**kwargs)
        def digest(path):
            if 'raw-control' in str(path):return 'a'*64
            return hashlib.sha256(self.map(Path(path)).read_bytes()).hexdigest()
        self.stack.enter_context(patch.object(recovery.control,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(reader.shared,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(recovery.control,'read',side_effect=self.read))
        self.stack.enter_context(patch.object(recovery.control,'digest',side_effect=digest))
        self.stack.enter_context(patch.object(reader,'read_file',side_effect=read_file))
        self.stack.enter_context(patch.object(recovery,'snapshot',side_effect=lambda p:original_snapshot(self.map(Path(p)))))
        self.stack.enter_context(patch.object(recovery.control,'live',return_value=False))
        self.stack.enter_context(patch.object(reader,'publication',return_value=(Path('/fixture'),self.claim['source'])))
        self.stack.enter_context(patch.object(recovery.control,'producer',return_value=(dict(run_id=recovery.RUN,
            source_commit=recovery.PRODUCER),self.claim['producer'],'fixture')))
        self.stack.enter_context(patch.object(recovery.control,'pair',return_value=(Path('/fixture'),self.claim['pair'])))
        self.stack.enter_context(patch.object(recovery,'fixed_binding',return_value=self.fclaim['binding']))
        self.stack.enter_context(patch.object(recovery,'service',return_value=SimpleNamespace(host_guard=lambda:None)))
        self.old=SimpleNamespace(inspect=lambda:self.observed,consume=lambda commit:self.fclaim['recovery'])
        self.stack.enter_context(patch.object(reader,'load_fixed',return_value=(self.old,{})))

    def map(self,path):
        if str(path).startswith(str(recovery.ROOT)):
            return self.root/path.relative_to(recovery.ROOT)
        return path

    def read(self,path):return json.loads(self.map(Path(path)).read_bytes())

    def test_exact_actual_held1_failure_and_original_bytes_accept(self):
        result=recovery.failure_evidence()
        self.assertEqual(result['original_held_exit_code'],1)
        self.assertEqual(result['observer_recovery'],self.fclaim['recovery']['observer_recovery'])
        self.assertEqual(result['prior_archive50_recovery'],self.fclaim['recovery'])
        self.assertEqual(result['archive_files']['task.log']['sha256'],
                         '81276ebf2457200eb29048293af60ef2a10a79ae53c8bb61c9c66a58ec1ef12e')
        self.assertEqual(set(result['raw_immutable_anchors']),{'claim.json','process.json'})

    def test_live_archive_or_follower_cannot_be_recovered_from_independent_du_failure(self):
        with patch.object(recovery.control,'live',side_effect=lambda row:row==recovery.OLD_RUNNER):
            with self.assertRaisesRegex(ValueError,'live or unknown'):recovery.failure_evidence()
        self.observed['actual_live_birth']=True
        with self.assertRaisesRegex(ValueError,'follower terminal'):recovery.failure_evidence()

    def test_nonmatching_wait_or_same_size_log_tamper_refuses(self):
        path=self.map(recovery.ARCHIVE/'worker-result.json');original=path.read_bytes()
        value=json.loads(original);value['actual_task_exit_code']=0;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'held1'):recovery.failure_evidence()
        path.write_bytes(original)
        log=self.map(recovery.ARCHIVE/'task.log');raw=log.read_bytes();log.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError,'failure log'):recovery.failure_evidence()

    def test_any_existing_payload_refuses_even_before_first_copied_file(self):
        with tempfile.TemporaryDirectory() as temp:
            payload=Path(temp)/'payload';payload.mkdir()
            with patch.object(recovery,'PAYLOAD',payload):
                with self.assertRaisesRegex(ValueError,'partial archive'):recovery.no_payload()

    def test_existing_unknown_intent_refuses_before_source_or_popen(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(recovery,'INTENT',Path(temp)),\
             patch.object(recovery.control,'owner'),patch.object(recovery.control,'caps'),\
             patch.object(recovery.control,'source') as source,patch.object(recovery.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'existing recovery'):recovery.start('c'*40,'/fixture')
        source.assert_not_called();popen.assert_not_called()

    def test_real_atomic_preservation_consume_and_second_start_no_replay(self):
        failure=recovery.failure_evidence();old_archive=self.map(recovery.ARCHIVE)
        preserved=self.root/'preserved';intent=self.root/'intent';directory=self.root/'new-follower'
        du=self.root/'fresh-du.json';du.write_text(json.dumps(dict(completed_unix=time.time()-1)))
        record=reader.document(du)[1];commit='c'*40
        publication=self.root/'new-publication.json';publication.write_text(json.dumps(dict(files={'search_g1/fixture.py':'b'*64})))
        source=dict(commit=commit,publication_path=str(publication));repo=Path(recovery.__file__).parents[2]
        raw=SimpleNamespace(STATUS='ALL200',inspect=lambda commit:dict(status='ACTUAL_READONLY_AUDIT_LIVE'))
        follow_before=recovery.snapshot(recovery.OLD_FOLLOW)
        with patch.multiple(recovery,ARCHIVE=old_archive,PRESERVED=preserved,INTENT=intent,DIRECTORY=directory,
                            PAYLOAD=self.root/'no-payload'),\
             patch.object(recovery,'failure_evidence',side_effect=lambda archive=None:dict(failure,
                 archive_files=recovery.snapshot(archive or old_archive))),\
             patch.object(recovery,'successful_du',return_value=record),\
             patch.object(recovery.control,'owner'),patch.object(recovery.control,'caps'),\
             patch.object(recovery.control,'source',return_value=(repo,source)),\
             patch.object(recovery.control,'producer',return_value=(dict(source_sha256={'search_g1/fixture.py':'b'*64}),
                 self.claim['producer'],'721485|RUNNING|0:0|')),\
             patch.object(reader,'load_fixed',return_value=(raw,{})),\
             patch.object(recovery.subprocess,'Popen',return_value=Mock(pid=900001)) as popen,\
             patch.object(recovery.control,'process',return_value=dict(pid=900001,start_ticks=123,sid=900001,uid=os.getuid())):
            recovery.start(commit,du)
            self.assertFalse(old_archive.exists())
            self.assertEqual(recovery.snapshot(preserved),failure['archive_files'])
            accepted=recovery.consume(commit)
            self.assertEqual(accepted['observer_recovery'],self.fclaim['recovery']['observer_recovery'])
            self.assertEqual(accepted['prior_archive50_recovery'],self.fclaim['recovery'])
            (self.root/'no-payload').mkdir();old_archive.mkdir();(old_archive.parent/'archive-100').mkdir()
            self.assertEqual(recovery.consume(commit),accepted)
            retained_log=preserved/'task.log';old_log=retained_log.read_bytes();retained_log.write_bytes(b'X'+old_log[1:])
            with self.assertRaisesRegex(ValueError,'failure bytes changed'):recovery.consume(commit)
            retained_log.write_bytes(old_log)
            old_du=du.read_bytes();du.write_text(json.dumps(dict(completed_unix=time.time())))
            with self.assertRaisesRegex(ValueError,'du receipt'):recovery.consume(commit)
            du.write_bytes(old_du)
            with self.assertRaisesRegex(ValueError,'existing recovery'):recovery.start(commit,du)
            self.assertEqual(popen.call_count,1)
            migration=intent/'migration.json';v=json.loads(migration.read_text());v['after_files']={};migration.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'intent/migration'):recovery.consume(commit)
        self.assertEqual(recovery.snapshot(recovery.OLD_FOLLOW),follow_before)

    def test_du_must_be_real_held0_empty_stderr_dead_and_fresh_at_admission(self):
        now=time.time();value=dict(source=dict(path='/fixture/storage_observer_sigma3_outer50_second_permission_repaired.py',
            commit='d'*40),actual_du_exit_code=0,stderr='',completed_unix=now-1,allocated_bytes=1405047443456)
        proof={'path':'/fixture/result.json','sha256':'a'*64,'bytes':512}
        actual=dict(result=value,status='ACTUAL_DU_COMPLETED0_USAGE_MEASURED',actual_live_birth=False,actual_du_live_birth=False)
        observer=SimpleNamespace(inspect=lambda:actual)
        with patch.object(reader,'document',return_value=(value,proof)),patch.object(reader,'load_fixed',return_value=(observer,{})):
            self.assertEqual(recovery.successful_du('/fixture/result.json',now=now),proof)
            for key,bad in (('stderr','Permission denied'),('actual_du_exit_code',1),('completed_unix',now-601)):
                before=value[key];value[key]=bad
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'held0'):recovery.successful_du('/fixture/result.json',now=now)
                value[key]=before
            actual['actual_du_live_birth']=True
            with self.assertRaisesRegex(ValueError,'held0'):recovery.successful_du('/fixture/result.json',now=now)



if __name__=='__main__':unittest.main()
