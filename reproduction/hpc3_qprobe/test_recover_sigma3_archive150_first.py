"""Actual24 failure bytes and original5f100 held receipts; no remote actions."""
from contextlib import ExitStack
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import recover_sigma3_archive150_first as recovery
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader


class Actual150Failure(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.files=json.loads(recovery.FIXTURE.read_bytes())
        for folder,files in self.files.items():
            for name,row in files.items():
                raw=base64.b64decode(row['raw_base64'],validate=True)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
                target=self.map(Path(folder)/name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                row['inode']=target.stat().st_ino
        fixture=self.root/'fixture.json';fixture.write_text(json.dumps(self.files))
        self.stack.enter_context(patch.object(recovery,'FIXTURE',fixture))
        inheritance=json.loads(Path(__file__).with_name('archive100_original_inheritance_fixture.json').read_bytes())
        for name,row in inheritance['files'].items():
            target=self.map(Path(inheritance['folder'])/name);target.parent.mkdir(parents=True,exist_ok=True)
            raw=base64.b64decode(row['raw_base64'],validate=True)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256']);target.write_bytes(raw)
        self.claim=self.read(recovery.ARCHIVE/'submit-claim.json');self.fclaim=self.read(recovery.OLD_FOLLOW/'claim.json')
        self.hundred=self.read(recovery.ARCHIVE.parent/'archive-100/submit-claim.json')
        original_read=reader.read_file;original_snapshot=recovery.snapshot;original_document=recovery.storage.document
        self.stack.enter_context(patch.object(recovery.control,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(reader.shared,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(recovery.control,'read',side_effect=self.read))
        self.stack.enter_context(patch.object(recovery.control,'digest',side_effect=self.digest))
        self.stack.enter_context(patch.object(reader,'read_file',side_effect=lambda p,**kw:original_read(self.map(Path(p)),**kw)))
        self.stack.enter_context(patch.object(recovery,'snapshot',side_effect=lambda p:original_snapshot(self.map(Path(p)))))
        self.stack.enter_context(patch.object(recovery.storage,'document',side_effect=lambda p,*_:original_document(self.map(Path(p)),(os.getuid(),))))
        self.stack.enter_context(patch.object(recovery.storage,'host_guard'))
        self.stack.enter_context(patch.object(recovery.storage,'live',return_value=False))
        self.stack.enter_context(patch.object(recovery.control,'live',return_value=False))
        self.stack.enter_context(patch.object(reader,'publication',return_value=(Path('/fixture'),self.claim['source'])))
        self.stack.enter_context(patch.object(recovery.control,'producer',return_value=(dict(run_id=recovery.RUN,
            source_commit=recovery.PRODUCER),self.claim['producer'],'fixture')))
        self.stack.enter_context(patch.object(recovery.control,'pair',return_value=(Path('/fixture'),self.claim['pair'])))
        self.stack.enter_context(patch.object(recovery,'fixed_binding',return_value=self.fclaim['binding']))
        self.stage=Mock(return_value=self.fclaim['recovery'])
        self.stack.enter_context(patch.object(recovery.lineage,'consume',self.stage))
        original_exists=Path.exists;original_link=Path.is_symlink
        self.stack.enter_context(patch.object(Path,'exists',lambda p:original_exists(self.map(p))))
        self.stack.enter_context(patch.object(Path,'is_symlink',lambda p:original_link(self.map(p))))

    def map(self,path):
        path=Path(path)
        return self.root/path.relative_to(recovery.ROOT) if str(path).startswith(str(recovery.ROOT)+'/') else path

    def read(self,path):return json.loads(self.map(path).read_bytes())

    def digest(self,path):
        if str(path).endswith('/global_step_100/complete.json'):return recovery.ARCHIVE100_TRANSPORT_SHA
        if 'raw-control' in str(path):return 'a'*64
        return hashlib.sha256(self.map(path).read_bytes()).hexdigest()

    def test_all_actual_failure_bytes_and_original100_binding_accept(self):
        value=recovery.failure_evidence()
        self.assertEqual(sum(len(value[key]) for key in ('archive_files','follower_files','du_files')),24)
        self.assertEqual(value['original_held_exit_code'],1)
        self.assertEqual(value['inherited_archive100']['source']['commit'],recovery.OLD)
        self.assertEqual(value['inherited_archive100']['controller'],recovery.OLD_CONTROLLER)
        self.assertEqual(value['prior_archive100_fourth_recovery'],self.fclaim['recovery'])
        self.stage.assert_called_once_with(recovery.OLD,'recover_sigma3_archive100_fourth')

    def test_same_size_original_failure_tamper_and_extra_file_refuse(self):
        path=self.map(recovery.ARCHIVE/'task.log');raw=path.read_bytes();path.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError,'bytes/inode'):recovery.failure_evidence()
        path.write_bytes(raw)
        extra=self.map(recovery.ARCHIVE/'extra.json');extra.write_text('{}')
        with self.assertRaisesRegex(ValueError,'fileset'):recovery.failure_evidence()

    def test_inode_replacement_with_identical_bytes_refuses(self):
        path=self.map(recovery.ARCHIVE/'task.log');raw=path.read_bytes()
        replacement=path.with_suffix('.replacement');replacement.write_bytes(raw);replacement.replace(path)
        with self.assertRaisesRegex(ValueError,'bytes/inode'):recovery.failure_evidence()

    def test_live_old_birth_and_live_observer_refuse(self):
        with patch.object(recovery.control,'live',side_effect=lambda p:p==recovery.OLD_RUNNER):
            with self.assertRaisesRegex(ValueError,'still live'):recovery.failure_evidence()
        with patch.object(recovery.storage,'live',return_value=True):
            with self.assertRaisesRegex(ValueError,'terminal denied-frontier'):recovery.failure_evidence()

    def test_original100_may_not_be_retagged_to_new_controller_or_source(self):
        path=self.map(recovery.ARCHIVE.parent/'archive-100/submit-claim.json');raw=path.read_bytes()
        for key,bad in (('source',{'commit':'a'*40}),('controller',dict(recovery.OLD_CONTROLLER,pid=111))):
            value=json.loads(raw);value[key]=bad;path.write_text(json.dumps(value))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'successful100 claim'):
                recovery.inherited_archive100(self.claim['producer'])
        path.write_bytes(raw)

    def test_only_new150_and200_can_use_owner_partition_protocol(self):
        with patch.object(recovery.control,'launch_archive') as launch:
            private=recovery.core();private.control.launch_archive('/fixture','a'*40,150)
            launch.assert_called_once_with('/fixture','a'*40,150,owner_partition_protocol=True)
            with self.assertRaisesRegex(ValueError,'only claim150/200'):
                private.control.launch_archive('/fixture','a'*40,100)

    def test_unknown_existing_intent_refuses_before_source_or_popen(self):
        with patch.object(recovery,'INTENT',self.root),patch.object(recovery.control,'owner'),\
             patch.object(recovery.control,'caps'),patch.object(recovery.control,'source') as source,\
             patch.object(recovery.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'existing150'):recovery.start('a'*40)
            source.assert_not_called();popen.assert_not_called()

    def test_any_existing_or_dangling_symlink_payload_refuses(self):
        payload=self.root/'payload';payload.symlink_to(self.root/'absent')
        with patch.object(recovery,'PAYLOAD',payload),self.assertRaisesRegex(ValueError,'partial150'):
            recovery.no_payload()

    def test_real_atomic_preservation_then_consumption_and_second_start_no_replay(self):
        failure=recovery.failure_evidence();original_archive=self.map(recovery.ARCHIVE)
        preserved=self.root/'preserved';intent=self.root/'intent';folder=self.root/'new-follower';commit='c'*40
        publication=self.root/'publication.json';publication.write_text(json.dumps({'files':{'search_g1/test.py':'b'*64}}))
        source={'commit':commit,'publication_path':str(publication)};repo=Path(recovery.__file__).parents[2]
        raw=SimpleNamespace(STATUS='ALL200',inspect=lambda _:dict(status='ACTUAL_READONLY_AUDIT_LIVE'))
        before=recovery.snapshot(recovery.OLD_FOLLOW)
        with patch.multiple(recovery,ARCHIVE=original_archive,PRESERVED=preserved,INTENT=intent,DIRECTORY=folder,
                            PAYLOAD=self.root/'no-payload'),\
             patch.object(recovery,'failure_evidence',side_effect=lambda archive=None:dict(failure,
                 archive_files=recovery.snapshot(archive or original_archive))),\
             patch.object(recovery.control,'owner'),patch.object(recovery.control,'caps'),\
             patch.object(recovery.control,'source',return_value=(repo,source)),\
             patch.object(recovery.control,'producer',return_value=(dict(source_sha256={'search_g1/test.py':'b'*64}),
                 self.claim['producer'],'721485|RUNNING|0:0|')),\
             patch.object(reader,'load_fixed',return_value=(raw,{})),\
             patch.object(recovery.subprocess,'Popen',return_value=Mock(pid=900001)) as popen,\
             patch.object(recovery.control,'process',return_value=dict(pid=900001,start_ticks=123,sid=900001,uid=os.getuid())):
            recovery.start(commit)
            self.assertFalse(original_archive.exists());self.assertEqual(recovery.snapshot(preserved),failure['archive_files'])
            value=recovery.consume(commit)
            self.assertEqual(value['prior_archive100_fourth_recovery'],self.fclaim['recovery'])
            self.assertEqual(value['inherited_archive100']['source']['commit'],recovery.OLD)
            with self.assertRaisesRegex(ValueError,'existing150'):recovery.start(commit)
            self.assertEqual(popen.call_count,1)
            migration=intent/'migration.json';changed=json.loads(migration.read_text());changed['after_files']={}
            migration.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'intent/migration'):recovery.consume(commit)
        self.assertEqual(recovery.snapshot(recovery.OLD_FOLLOW),before)


if __name__=='__main__':unittest.main()
