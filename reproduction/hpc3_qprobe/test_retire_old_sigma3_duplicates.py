"""Temporary byte trees and real local held waits; no remote cleanup acceptance."""
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
from reproduction.hpc3_qprobe import retire_old_sigma3_duplicates as retire


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=value if isinstance(value,bytes) else json.dumps(value).encode();path.write_bytes(data)
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


class ByteTrees(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.top=Path(self.temp.name).resolve();self.archives=self.top/'archives'/retire.OLD_RUN
        self.stack=ExitStack()
        for target,key,value in ((retire,'ARCHIVES',self.archives),(retire.ctl,'OWNER',os.getuid()),
                (retire.reader.shared,'OWNER',os.getuid()),(retire,'stopped',Mock(return_value={'actual':'human cancelled boundary'}))):
            self.stack.enter_context(patch.object(target,key,value))
        self.documents={};self.rows=[]
        for step in (5,50,100):
            folder=self.archives/('global_step_'+str(step));source=self.top/'original'/str(step)
            original=write(source/'a.bin',('authentic-'+str(step)).encode())
            original_receipt=write(source/'source-receipt.json',dict(run_id=retire.OLD_RUN,step=step))
            files=[];receipts={};sets={}
            for role in ('archive','target'):
                directory=folder/role
                write(directory/'a.bin',Path(original['path']).read_bytes())
                receipt=write(directory/'relocated-source-receipt.json',dict(original=str(source),role=role))
                receipts[role+'_receipt']={k:receipt[k] for k in ('path','sha256')}
                sets[role+'_file_set']=[str(directory/'a.bin'),receipt['path']]
            f=dict(source_path=original['path'],archive_path=str(folder/'archive/a.bin'),
                target_path=str(folder/'target/a.bin'),bytes=original['bytes'],sha256=original['sha256']);files=[f]
            transport=dict(status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',run_id=retire.OLD_RUN,
                step=step,world_size=4,files=files,source_receipt={k:original_receipt[k] for k in ('path','sha256')},**receipts,**sets)
            write(folder/'complete.json',transport);self.documents[step]=transport
        for step,role in retire.PAIRS:
            directory=self.archives/('global_step_'+str(step))/role;observed=retire.tree(directory)
            snapshots=[{k:record[k] for k in ('path','bytes','inode','mtime_ns')} for record in observed['files'].values()]
            self.rows.append(dict(step=step,role=role,path=str(directory),file_snapshot=snapshots,
                files=len(snapshots),bytes=sum(x['bytes'] for x in snapshots),expected_content_hashes=self.documents[step]['files'],
                remaining_original_receipt=self.documents[step]['source_receipt']))
        self.total=sum(r['bytes'] for r in self.rows);self.stack.enter_context(patch.object(retire,'TOTAL_BYTES',self.total))
        self.plan=dict(directories=self.rows,total_bytes=self.total)
        self.plan_proof=write(self.top/'plan.json',self.plan)
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def guards(self):
        with redirect_stdout(io.StringIO()):return retire.preflight(self.plan,self.plan_proof)
    def test_full_preflight_then_only_exact_five_trees_removed_sources_archive100_and_transport_proofs_preserved(self):
        guards=self.guards();self.assertTrue(guards['all_guards_before_first_unlink'])
        unrelated=self.archives/'unrelated';write(unrelated/'must-stay.txt',b'keep')
        retained={r['path']:Path(r['path']).read_bytes() for r in guards['retained_files']}
        with redirect_stdout(io.StringIO()):
            for target in guards['targets']:retire.exact_delete(target)
        self.assertTrue(all(not Path(r['path']).exists() for r in self.rows))
        for path,data in retained.items():self.assertEqual(Path(path).read_bytes(),data)
        self.assertEqual((unrelated/'must-stay.txt').read_bytes(),b'keep')
        self.assertTrue((self.archives/'global_step_100/archive/a.bin').exists())
    def test_same_size_source_or_retained_archive100_corruption_refuses_before_any_unlink(self):
        for path in (self.documents[5]['files'][0]['source_path'],self.documents[100]['files'][0]['archive_path']):
            original=Path(path).read_bytes();Path(path).write_bytes(b'X'*len(original))
            with self.assertRaisesRegex(ValueError,'fresh original full SHA differs'):self.guards()
            self.assertTrue(all(Path(r['path']).exists() for r in self.rows))
            Path(path).write_bytes(original)
    def test_added_member_wrong_original_snapshot_and_symlink_refuse_before_any_unlink(self):
        directory=Path(self.rows[0]['path']);extra=directory/'extra';extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError,'full fileset differs'):self.guards()
        extra.unlink()
        changed=copy.deepcopy(self.plan);changed['directories'][0]['file_snapshot'][0]['inode']+=1
        with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'original snapshot identity changed'):
            retire.preflight(changed,self.plan_proof)
        link=directory/'link';link.symlink_to(directory/'a.bin')
        with self.assertRaisesRegex(ValueError,'unsafe tree member'):self.guards()
        self.assertTrue((directory/'a.bin').exists())
    def test_after_preflight_equal_length_content_change_fails_before_unlink_of_that_file(self):
        guards=self.guards();target=guards['targets'][0];path=Path(target['path'])/'a.bin'
        original=path.read_bytes();path.write_bytes(b'X'*len(original))
        # Emulate equal metadata (known on remote storage) with the prior mtime.
        st=path.stat();expected=target['files'][str(path)];os.utime(path,ns=(st.st_atime_ns,expected['mtime_ns']))
        with self.assertRaisesRegex(ValueError,'duplicate content changed before unlink'):retire.exact_delete(target)
        self.assertEqual(path.read_bytes(),b'X'*len(original))
        self.assertTrue((Path(target['path'])/'relocated-source-receipt.json').exists())
    def test_unexpected_target_tree_member_after_preflight_never_expands_delete_scope(self):
        target=self.guards()['targets'][0];extra=Path(target['path'])/'not-approved';extra.write_bytes(b'keep')
        with self.assertRaisesRegex(ValueError,'tree changed after all guards'):retire.exact_delete(target)
        self.assertEqual(extra.read_bytes(),b'keep')


class AdmissionAndHolder(unittest.TestCase):
    def test_existing_unknown_claim_never_replays_or_reverifies_big_weights(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d).resolve();write(directory/'claim.json',dict(source=dict(commit='a'*40)))
            with patch.object(retire,'DIRECTORY',directory),patch.object(retire,'caps'),patch.object(retire.ctl,'owner'),\
                 patch.object(retire.ctl,'OWNER',os.getuid()),patch.object(retire,'baseline') as baseline,\
                 patch.object(retire.subprocess,'Popen') as launch:
                self.assertEqual(retire.start('a'*40,authorize_exact_five=True)['status'],'UNKNOWN_OR_FAILED_NO_REPLAY')
                baseline.assert_not_called();launch.assert_not_called()
    def test_missing_authorization_refuses_before_source_claim_or_spawn(self):
        with patch.object(retire,'DIRECTORY',Path('/nonexistent-five-retirement-test')),patch.object(retire,'caps'),\
             patch.object(retire.ctl,'source') as source,patch.object(retire.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(ValueError,'authorization required'):retire.start('a'*40)
            source.assert_not_called();launch.assert_not_called()
    def test_actual_human_cancelled_binding_refuses_running_failed_or_wrong_owner(self):
        with patch.object(retire.reader.shared,'OWNER',os.getuid()):
            for state in ('RUNNING','FAILED','COMPLETED'):
                raw='719225|'+state+'|0:0|jxu957|14090|jxu957|formal-user|emergency_acd|ACD1-56\n'
                with patch.object(retire.subprocess,'check_output',return_value=raw),self.assertRaisesRegex(ValueError,'must actually remain human CANCELLED'):
                    retire.stopped()
            raw='719225|CANCELLED by 14090|0:15|jxu957|14090|jxu957|formal-user|emergency_acd|ACD1-56\n'
            with patch.object(retire.subprocess,'check_output',return_value=raw):
                self.assertEqual(retire.stopped()['actual_cancelled_accounting'],raw)
    def test_fixed_plan_matches_exact_user_authorized_paths_and_bytes(self):
        with patch.object(retire.reader.shared,'OWNER',os.getuid()):
            value,proof=retire.baseline()
            self.assertEqual(value['total_bytes'],204223139168)
            self.assertEqual([(r['step'],r['role']) for r in value['directories']],list(retire.PAIRS))
            self.assertEqual(proof['sha256'],retire.PLAN_SHA)
    def test_supervisor_uses_real_child_wait_and_preserves_nonzero_actual_exit(self):
        with tempfile.TemporaryDirectory() as d:
            directory=Path(d).resolve();source=dict(commit='a'*40);own=dict(pid=os.getpid(),start_ticks=100,sid=os.getpid(),uid=os.getuid())
            write(directory/'claim.json',dict(source=source,cwd=str(directory),exact_five_destructive_scope_authorized=True,created_unix=0))
            write(directory/'process.json',own)
            children=[];real_popen=subprocess.Popen
            def launch(command,**kwargs):
                self.assertEqual(command,[sys.executable,'-m',retire.MODULE,'worker','--commit','a'*40])
                child=real_popen([sys.executable,'-c','import sys,time;time.sleep(.02);sys.exit(7)'],**kwargs)
                child.wait=Mock(wraps=child.wait);children.append(child);return child
            def birth(pid):return own if pid==own['pid'] else dict(pid=pid,start_ticks=101,sid=own['sid'],uid=own['uid'])
            with patch.object(retire,'DIRECTORY',directory),patch.object(retire,'caps'),\
                 patch.object(retire.ctl,'source',return_value=(directory,source)),patch.object(retire.ctl,'process',side_effect=birth),\
                 patch.object(retire.reader,'OWNER',os.getuid()),patch.object(retire.ctl,'OWNER',os.getuid()),patch.object(retire.subprocess,'Popen',side_effect=launch),\
                 patch.object(retire.ctl,'live',return_value=False),redirect_stderr(io.StringIO()):
                self.assertEqual(retire.supervise('a'*40),1)
            self.assertTrue(children,json.loads((directory/'result.json').read_text()))
            children[0].wait.assert_called_once_with()
            result=json.loads((directory/'result.json').read_text())
            self.assertEqual(result['actual_child_exit_code'],7);self.assertEqual(result['supervisor_exit_code'],1)
            self.assertIn('did not actually return clean0',result['error'])


    def test_real_held_wait0_consume_requires_dead_both_births_and_unchanged_retired_receipt(self):
        # Protocol-only synthetic retirement receipt; temporary byte-tree tests
        # above independently exercise every actual delete/content guard.
        with tempfile.TemporaryDirectory() as d:
            top=Path(d).resolve();directory=top/'holder';directory.mkdir();archives=top/'archives'
            source=dict(commit='a'*40);own=dict(pid=os.getpid(),start_ticks=100,sid=os.getpid(),uid=os.getuid())
            plan=dict(path=str(top/'published-plan.json'),sha256=retire.PLAN_SHA,bytes=1)
            guards=dict(plan=plan,all_guards_before_first_unlink=True,total_apparent_bytes=retire.TOTAL_BYTES,
                targets=[dict(path=str(archives/('global_step_'+str(step))/role),files={}) for step,role in retire.PAIRS],retained_files=[])
            preflight=write(directory/'preflight.json',guards)
            claim=dict(source=source,plan=plan,cwd=str(directory),exact_five_destructive_scope_authorized=True,
                training_control_permitted=False,created_unix=0)
            claim_record=write(directory/'claim.json',claim);write(directory/'process.json',own)
            receipt=dict(status=retire.STATUS,source=source,claim_sha256=claim_record['sha256'],
                preflight_sha256=preflight['sha256'],retired_directories=[t['path'] for t in guards['targets']],
                retired_files=0,retired_apparent_bytes=retire.TOTAL_BYTES,
                retained_original_sources_and_archive100_full_sha_pass=True,original_transport_proofs_not_overwritten=True,
                training_control_actions=0,model_or_gpu_accesses=0)
            program='import json,os,time;from pathlib import Path;time.sleep(.02);r=json.loads('+repr(json.dumps(receipt))+');'
            program+='r.update(actual_process=dict(pid=os.getpid(),start_ticks=101,sid='+str(own['sid'])+',uid=os.getuid()),completed_unix=time.time());'
            program+='Path('+repr(str(directory/'retired-receipt.json'))+').write_text(json.dumps(r));print(json.dumps(r))'
            children=[];real_popen=subprocess.Popen
            def launch(command,**kwargs):
                self.assertEqual(command,[sys.executable,'-m',retire.MODULE,'worker','--commit','a'*40])
                child=real_popen([sys.executable,'-c',program],**kwargs);child.wait=Mock(wraps=child.wait);children.append(child);return child
            def birth(pid):return own if pid==own['pid'] else dict(pid=pid,start_ticks=101,sid=own['sid'],uid=own['uid'])
            with ExitStack() as stack:
                for target,key,value in ((retire,'DIRECTORY',directory),(retire,'ARCHIVES',archives),(retire,'caps',Mock()),
                    (retire.ctl,'source',Mock(return_value=(directory,source))),(retire.ctl,'process',Mock(side_effect=birth)),
                    (retire.ctl,'OWNER',os.getuid()),(retire.reader,'OWNER',os.getuid()),
                    (retire.ctl,'live',Mock(return_value=False)),(retire.subprocess,'Popen',Mock(side_effect=launch))):
                    stack.enter_context(patch.object(target,key,value))
                with redirect_stderr(io.StringIO()):self.assertEqual(retire.supervise('a'*40),0)
                children[0].wait.assert_called_once_with()
                self.assertEqual(retire.consume('a'*40)['status'],retire.HELD_STATUS)
                result=json.loads((directory/'result.json').read_text())
                for active in (result['process'],result['child']):
                    with patch.object(retire.ctl,'live',side_effect=lambda row:row==active),self.assertRaisesRegex(ValueError,'actual terminal held wait'):
                        retire.consume('a'*40)
                actual=json.loads((directory/'retired-receipt.json').read_text());actual['retired_files']=1
                write(directory/'retired-receipt.json',actual)
                with self.assertRaisesRegex(ValueError,'retired receipt/preflight/authorized scope bytes differ'):retire.consume('a'*40)


if __name__=='__main__':unittest.main()
