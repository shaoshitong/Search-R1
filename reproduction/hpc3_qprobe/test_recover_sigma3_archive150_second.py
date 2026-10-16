"""Original S failure bytes, preserved-path consumption and once-only admission."""
from contextlib import ExitStack
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from reproduction.hpc3_qprobe import recover_sigma3_archive150_second as recovery
from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader


class ActualSecond150Failure(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.files=json.loads(recovery.FIXTURE.read_bytes())
        for folder,files in self.files.items():
            for name,row in files.items():
                raw=base64.b64decode(row['raw_base64'],validate=True)
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
                target=self.map(Path(folder)/name);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw);target.chmod(row['mode']);s=target.stat()
                row.update(inode=s.st_ino,device=s.st_dev,uid=s.st_uid,gid=s.st_gid)
        fixture=self.root/'fixture.json';fixture.write_text(json.dumps(self.files))
        self.stack.enter_context(patch.object(recovery,'FIXTURE',fixture))
        inheritance=json.loads(Path(__file__).with_name('archive100_original_inheritance_fixture.json').read_bytes())
        for name,row in inheritance['files'].items():
            target=self.map(Path(inheritance['folder'])/name);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(base64.b64decode(row['raw_base64'],validate=True))
        self.claim=self.read(recovery.ARCHIVE/'submit-claim.json');self.fclaim=self.read(recovery.OLD_FOLLOW/'claim.json')
        self.hundred=self.read(recovery.ARCHIVE.parent/'archive-100/submit-claim.json')
        self.w1=self.read(recovery.RETIREMENT/'complete.json')
        original_read=reader.read_file;original_snapshot=recovery.snapshot
        self.stack.enter_context(patch.object(recovery.control,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(reader.shared,'OWNER',os.getuid()))
        self.stack.enter_context(patch.object(recovery.control,'read',side_effect=self.read))
        self.stack.enter_context(patch.object(recovery.control,'digest',side_effect=self.digest))
        self.stack.enter_context(patch.object(reader,'read_file',side_effect=lambda p,**kw:original_read(self.map(p),**kw)))
        self.stack.enter_context(patch.object(recovery,'snapshot',side_effect=lambda p:original_snapshot(self.map(p))))
        self.stack.enter_context(patch.object(recovery,'retirement_snapshot',return_value={
            name:{k:v for k,v in row.items() if k!='raw_base64'} for name,row in self.files[str(recovery.RETIREMENT)].items()}))
        self.stack.enter_context(patch.object(recovery.retirement,'protected_document',side_effect=self.protected))
        self.stack.enter_context(patch.object(recovery.storage,'host_guard'))
        self.stack.enter_context(patch.object(recovery.storage,'host_binding',return_value=self.read(recovery.RETIREMENT/'mou-report.json')['host']))
        self.stack.enter_context(patch.object(recovery.storage,'live',return_value=False))
        self.stack.enter_context(patch.object(recovery.control,'live',return_value=False))
        sources={recovery.OLD:self.claim['source'],recovery.ARCHIVE100_SOURCE:self.hundred['source'],
                 recovery.RETIREMENT_SOURCE:self.w1['source']}
        self.stack.enter_context(patch.object(reader,'publication',side_effect=lambda commit:(Path('/fixture'),sources[commit])))
        self.stack.enter_context(patch.object(recovery.control,'producer',return_value=(dict(run_id=recovery.RUN,
            source_commit=recovery.PRODUCER),self.claim['producer'],'fixture')))
        self.stack.enter_context(patch.object(recovery.control,'pair',return_value=(Path('/fixture'),self.claim['pair'])))
        self.stack.enter_context(patch.object(recovery,'fixed_binding',return_value=self.fclaim['binding']))
        self.stage=Mock(return_value=self.fclaim['recovery']);self.stack.enter_context(patch.object(recovery.lineage,'consume',self.stage))
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

    def protected(self,path,owner):
        expected=recovery.storage.MOU if Path(path).name=='mou-report.json' else recovery.storage.JXU
        self.assertEqual(owner,expected)
        raw=self.map(path).read_bytes()
        return json.loads(raw),dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

    def test_full_failure_and_original100_and_active_retirement_accept(self):
        value=recovery.failure_evidence()
        self.assertEqual(sum(len(value[k]) for k in ('archive_files','follower_files','retirement_files')),28)
        self.assertEqual(value['original_held_exit_code'],-15)
        self.assertTrue(value['active_CPU_retirement']['active_retirement'])
        self.assertFalse(value['active_CPU_retirement']['spontaneous_failure'])
        self.assertEqual(value['inherited_archive100']['source']['commit'],recovery.ARCHIVE100_SOURCE)
        self.assertEqual(value['inherited_archive100']['controller'],recovery.ARCHIVE100_CONTROLLER)
        self.assertEqual(value['prior_archive150_first_recovery'],self.fclaim['recovery'])
        self.stage.assert_called_once_with(recovery.OLD,'recover_sigma3_archive150_first')

    def test_preserved_failure_consumption_ignores_new_canonical150(self):
        before=recovery.failure_evidence();old=self.map(recovery.ARCHIVE);new=self.map(recovery.PRESERVED)
        old.rename(new);old.mkdir();(old/'worker-result.json').write_text('{"not":"original"}')
        self.assertEqual(recovery.failure_evidence(recovery.PRESERVED),before)
        self.assertEqual((old/'worker-result.json').read_text(),'{"not":"original"}')

    def test_original_byte_tamper_inode_replacement_and_extra_member_refuse(self):
        path=self.map(recovery.ARCHIVE/'task.log');raw=path.read_bytes();path.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError,'bytes/inode'):recovery.failure_evidence()
        path.write_bytes(raw);other=path.with_suffix('.replacement');other.write_bytes(raw);other.chmod(path.stat().st_mode)
        other.replace(path)
        with self.assertRaisesRegex(ValueError,'bytes/inode'):recovery.failure_evidence()

    def test_extra_failure_member_refuses(self):
        (self.map(recovery.ARCHIVE)/'unexpected').write_text('x')
        with self.assertRaisesRegex(ValueError,'fileset'):recovery.failure_evidence()

    def test_live_old_birth_or_du_refuses(self):
        with patch.object(recovery.control,'live',side_effect=lambda p:p==recovery.OLD_CHILD):
            with self.assertRaisesRegex(ValueError,'still live'):recovery.failure_evidence()
        with patch.object(recovery.storage,'live',return_value=True):
            with self.assertRaisesRegex(ValueError,'failed complete complement'):recovery.failure_evidence()

    def test_old100_source_and_controller_may_not_be_retagged(self):
        path=self.map(recovery.ARCHIVE.parent/'archive-100/submit-claim.json');raw=path.read_bytes()
        for key,value in [('source',self.claim['source']),('controller',recovery.OLD_CONTROLLER)]:
            changed=json.loads(raw);changed[key]=value;path.write_text(json.dumps(changed))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'successful100 claim'):
                recovery.inherited_archive100(self.claim['producer'])
        path.write_bytes(raw)

    def test_original_w1_mode_and_report_bytes_must_be_pinned_by_graph(self):
        path=self.map(recovery.RETIREMENT/'mou-report.json');value=json.loads(path.read_bytes())
        value['foreign_process_declared_dead']=True;path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'protected graph'):recovery.failure_evidence()

    def test_only150_and200_use_v2(self):
        with patch.object(recovery.control,'launch_archive') as launch:
            private=recovery.core()
            for step in (150,200):private.control.launch_archive('/fixture','a'*40,step)
            self.assertEqual([call.kwargs for call in launch.call_args_list],[{'owner_partition_v2_protocol':True}]*2)
            for step in (5,50,100):
                with self.subTest(step=step),self.assertRaisesRegex(ValueError,'only claim150/200'):
                    private.control.launch_archive('/fixture','a'*40,step)

    def test_existing_intent_forbids_source_lookup_and_launch(self):
        with patch.object(recovery,'INTENT',self.root),patch.object(recovery.control,'owner'),\
             patch.object(recovery.control,'caps'),patch.object(recovery.control,'source') as source,\
             patch.object(recovery.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'existing150'):recovery.start('a'*40)
            source.assert_not_called();popen.assert_not_called()

    def test_existing_or_dangling_payload_refuses(self):
        payload=self.root/'payload';payload.symlink_to(self.root/'absent')
        with patch.object(recovery,'PAYLOAD',payload),self.assertRaisesRegex(ValueError,'partial150'):recovery.no_payload()

    def test_real_atomic_preservation_then_once_consumption(self):
        failure=recovery.failure_evidence();original=self.map(recovery.ARCHIVE);preserved=self.root/'preserved'
        intent=self.root/'intent';folder=self.root/'new-follower';commit='c'*40
        publication=self.root/'publication.json';publication.write_text(json.dumps({'files':{'search_g1/science.py':'b'*64}}))
        source={'commit':commit,'publication_path':str(publication)};repo=Path(recovery.__file__).parents[2]
        raw=SimpleNamespace(STATUS='ALL200',inspect=lambda _:dict(status='ACTUAL_READONLY_AUDIT_LIVE'))
        before=recovery.snapshot(recovery.OLD_FOLLOW)
        with patch.multiple(recovery,ARCHIVE=original,PRESERVED=preserved,INTENT=intent,DIRECTORY=folder,PAYLOAD=self.root/'absent'),\
             patch.object(recovery,'failure_evidence',side_effect=lambda archive=None:dict(failure,archive_files=recovery.snapshot(archive or original))),\
             patch.object(recovery.control,'owner'),patch.object(recovery.control,'caps'),\
             patch.object(recovery.control,'source',return_value=(repo,source)),\
             patch.object(recovery.control,'producer',return_value=(dict(source_sha256={'search_g1/science.py':'b'*64}),self.claim['producer'],'721485|RUNNING|0:0|')),\
             patch.object(reader,'load_fixed',return_value=(raw,{})),\
             patch.object(recovery.subprocess,'Popen',return_value=Mock(pid=900001)) as popen,\
             patch.object(recovery.control,'process',return_value=dict(pid=900001,start_ticks=123,sid=900001,uid=os.getuid())):
            recovery.start(commit);self.assertFalse(original.exists())
            self.assertEqual(recovery.snapshot(preserved),failure['archive_files'])
            original.mkdir();(original/'new-claim').write_text('new canonical150')
            value=recovery.consume(commit)
            self.assertEqual(value['original_archive_child_exit'],-15)
            self.assertEqual(value['prior_archive150_first_recovery'],self.fclaim['recovery'])
            self.assertEqual(value['inherited_archive100']['source']['commit'],recovery.ARCHIVE100_SOURCE)
            with self.assertRaisesRegex(ValueError,'existing150'):recovery.start(commit)
            self.assertEqual(popen.call_count,1)
        self.assertEqual(recovery.snapshot(recovery.OLD_FOLLOW),before)


if __name__=='__main__':unittest.main()
