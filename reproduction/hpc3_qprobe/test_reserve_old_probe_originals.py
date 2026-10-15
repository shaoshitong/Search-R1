"""Non-network checks for the fixed backup scope and actual-held acceptance."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.hpc3_qprobe import reserve_old_probe_originals as r


class ReserveTests(unittest.TestCase):
    def test_committed_plan_is_exact_and_non_destructive(self):
        p=r.plan()
        self.assertEqual((len(p['files']),p['total_bytes']),(118,121296682925))
        self.assertFalse(p['deletion_authorized'])
        self.assertFalse(p['source_cleanup_performed'])
        self.assertFalse(any(r.below(Path(x['path']),r.BASE) for x in p['files']))

    def test_scope_changes_refused(self):
        original=r.plan()
        cases=[]
        p=copy.deepcopy(original);p['deletion_authorized']=True;cases.append(p)
        p=copy.deepcopy(original);p['files'][0]['path']=str(r.BASE/'weights.pt');cases.append(p)
        p=copy.deepcopy(original);p['files'][0]['path']=str(r.ROOT.parent/'..'/'escape');cases.append(p)
        p=copy.deepcopy(original);p['files'][0]['bytes']+=1;cases.append(p)
        p=copy.deepcopy(original);p['files'][1]['path']=p['files'][0]['path'];cases.append(p)
        p=copy.deepcopy(original);p['files'][0]['uid']=0;cases.append(p)
        p=copy.deepcopy(original);p['groups'][0]['role']='target';cases.append(p)
        for p in cases:
            with self.subTest(p=p.get('deletion_authorized')):
                with self.assertRaises(ValueError):r.validate_plan(p)

    def test_sender_copy_has_no_source_mutation_and_dry_run_is_same_protocol(self):
        plan=r.plan()
        with patch.object(r,'digest',return_value=plan['known_hosts_sha256']):
            args=(plan,'a'*40,'b'*32,'c'*64,Path('/control/files'),Path('/target/payload'))
            actual=r.copy_command(*args);dry=r.copy_command(*args,dry_run=True)
        self.assertEqual([x for x in dry if x!='--dry-run'],actual)
        self.assertIn('--no-links',actual)
        self.assertTrue(any('rsync-sender' in x for x in actual))
        self.assertFalse(any(x.startswith(('--delete','--remove-source')) for x in actual))
        self.assertFalse(any('SSHPASS=' in x for x in actual))

    def test_existing_main_claim_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)/'control';control.mkdir()
            with patch.object(r,'CONTROL',control),patch.object(r,'PAYLOAD',Path(tmp)/'payload'),\
                 patch.object(r,'code_guard',return_value=(r.plan(),Path(tmp))),\
                 patch.object(r.subprocess,'Popen') as spawn:
                with self.assertRaises(ValueError):r.start('a'*40,'ram-only-test')
                spawn.assert_not_called()

    def supervise_case(self,actual_exit,invalid_result=False,binding_failure=False):
        supervisor=dict(pid=101,start_ticks=1001,sid=101,uid=1002)
        worker=dict(pid=102,start_ticks=1002,sid=101,uid=1002)
        token='b'*32;commit='a'*40
        child=Mock(pid=102,stdin=io.BytesIO());child.wait.return_value=actual_exit
        metadata={'claim.json':dict(token=token,source_commit=commit),
                  'process.json':dict(token=token,source_commit=commit,actual_process=supervisor),
                  'worker-result.json':dict(actual_process=worker,exit_code=0,status=r.STATUS,
                                            source_cleanup_performed=False,complete_sha256='c'*64)}
        if invalid_result:metadata['worker-result.json']['actual_process']=dict(worker,pid=999)
        original_publish=r.publish
        def publish(path,value,replace=False):
            if binding_failure and Path(path).name=='worker-process.json':raise OSError('injected worker binding failure')
            return original_publish(path,value,replace)
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)
            with patch.object(r,'CONTROL',control),patch.object(r,'code_guard',return_value=(r.plan(),control)),\
                 patch.object(r,'read_owned',side_effect=lambda path,uid:metadata[Path(path).name]),\
                 patch.object(r,'birth',side_effect=lambda pid:worker if pid==102 else supervisor),\
                 patch.object(r.os,'getsid',return_value=101),patch.object(r.subprocess,'Popen',return_value=child),\
                 patch.object(r,'digest',return_value='c'*64),patch.object(r,'publish',side_effect=publish):
                code=r.supervise(commit,token,'ram-only-test')
                result=json.loads((control/'result.json').read_text())
                self.assertNotIn('ram-only-test',(control/'result.json').read_text())
                child.wait.assert_called_once_with()
                return code,result

    def test_worker_nonzero_cannot_be_accepted(self):
        code,result=self.supervise_case(1)
        self.assertEqual((code,result['actual_child_exit_code']),(1,1))
        self.assertNotIn('complete_sha256',result)

    def test_worker_zero_needs_matching_actual_process_and_receipt(self):
        code,result=self.supervise_case(0,True)
        self.assertEqual(code,1)
        self.assertEqual(result['actual_child_exit_code'],0)
        self.assertIn('validation_error',result)

    def test_actual_wait_zero_and_verified_receipt_accept(self):
        code,result=self.supervise_case(0)
        self.assertEqual(code,0)
        self.assertEqual(result['status'],'ACTUAL_HELD_REMOTE_RESERVE_WORKER_COMPLETED0')
        self.assertEqual(result['complete_sha256'],'c'*64)

    def test_supervisor_binding_failure_closes_stdin_and_still_waits(self):
        code,result=self.supervise_case(0,binding_failure=True)
        self.assertEqual(code,1)
        self.assertEqual(result['actual_child_exit_code'],0)
        self.assertIn('injected worker binding failure',result['binding_error'])

    def test_phase_binding_publish_failure_still_waits_for_direct_child(self):
        row=dict(pid=102,start_ticks=1002,sid=101,uid=1002)
        child=Mock(pid=102);child.wait.return_value=0
        original_publish=r.publish
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)
            def failing_publish(path,value,replace=False):
                if Path(path).name=='state.json':raise OSError('injected publication failure')
                return original_publish(path,value,replace)
            with patch.object(r,'birth',return_value=row),patch.object(r.subprocess,'Popen',return_value=child),\
                 patch.object(r,'publish',side_effect=failing_publish):
                with self.assertRaises(OSError):r.child_wait('copy',['safe-command'],'ram-only-test',control)
            child.wait.assert_called_once_with()
            receipt=json.loads((control/'copy.wait.json').read_text())
            self.assertEqual(receipt['actual_exit_code'],0)
            self.assertIn('injected publication failure',receipt['binding_error'])
            self.assertNotIn('ram-only-test',(control/'copy.wait.json').read_text())

    def terminal_case(self,mutation):
        commit='a'*40;token='b'*32;supervisor=dict(pid=101,start_ticks=1001,sid=101,uid=1002)
        worker=dict(pid=102,start_ticks=1002,sid=101,uid=1002)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();control=base/'control';control.mkdir();payload=base/'payload';payload.mkdir()
            target=payload/'data/source';target.parent.mkdir();target.write_bytes(b'abc')
            s=target.stat();file=dict(path='/data/source',bytes=3,sha256=r.digest(target))
            p=dict(files=[file],groups=[dict(step=5)],total_bytes=3)
            manifest=dict(files=p['files'],groups=p['groups'],source_commit=commit,plan_sha256=r.PLAN_SHA)
            manifest_sha=r.hashlib.sha256(r.canonical(manifest)).hexdigest()
            complete=dict(status=r.STATUS,verified_bytes=3,verified_files=1,source_commit=commit,
                plan_sha256=r.PLAN_SHA,actual_process=worker,source_cleanup_performed=False,deletion_authorized=False,
                manifest_sha256=manifest_sha,files=[dict(source_path=file['path'],target_path=str(target),bytes=3,
                    sha256=file['sha256'],inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns)])
            for phase in ('before','after'):
                complete['source_'+phase]=dict(status='ORIGINAL_SOURCE_FULL_SHA_PASS',files=[file],
                    source_commit=commit,token=token,phase=phase,manifest_sha256=manifest_sha,source_cleanup_performed=False)
            metadata={'complete.json':complete,'source-manifest.json':manifest}
            for index,name in enumerate(('source-before','rsync-copy','source-after')):
                row=dict(pid=201+index,start_ticks=2001+index,sid=101,uid=1002)
                metadata[name+'.child.json']=dict(actual_process=row)
                metadata[name+'.wait.json']=dict(actual_process=row,actual_held_pid=row['pid'],actual_exit_code=0,binding_error=None)
            result=dict(status='ACTUAL_HELD_REMOTE_RESERVE_WORKER_COMPLETED0',actual_child_exit_code=0,
                binding_error=None,actual_supervisor=supervisor,actual_worker=worker,complete_sha256='c'*64)
            if mutation=='missing':target.unlink()
            elif mutation=='inode':complete['files'][0]['inode']+=1
            elif mutation=='wait':metadata['rsync-copy.wait.json']['actual_exit_code']=1
            elif mutation=='members':complete['files']=[]
            elif mutation=='phase':complete['source_after']['token']='d'*32
            original_lstat=Path.lstat
            def lstat(path):
                v=original_lstat(path)
                if path==target:return SimpleNamespace(st_mode=v.st_mode,st_uid=1002,st_ino=v.st_ino,
                    st_dev=v.st_dev,st_size=v.st_size,st_mtime_ns=v.st_mtime_ns)
                return v
            with patch.object(r,'CONTROL',control),patch.object(r,'PAYLOAD',payload),\
                 patch.object(r,'read_owned',side_effect=lambda path,uid:metadata[Path(path).name]),\
                 patch.object(r,'digest',return_value='c'*64),patch.object(r,'live',return_value=False),\
                 patch.object(Path,'lstat',lstat):
                return r.verify_terminal_metadata(p,commit,dict(token=token),dict(actual_process=supervisor),result,worker)

    def test_terminal_refuses_missing_changed_or_incomplete_target(self):
        for mutation in ('missing','inode','wait','members','phase'):
            with self.subTest(mutation=mutation):
                with self.assertRaises((ValueError,FileNotFoundError)):self.terminal_case(mutation)

    def test_terminal_accepts_full_current_metadata_and_all_held_phase_waits(self):
        self.assertEqual(self.terminal_case(None)['verified_files'],1)

    def test_ssh_timeout_saves_unknown_audit_without_replay(self):
        from reproduction.hpc3_qprobe import local_old_original_reserve as local
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            args=['reserve','inspect','--commit','a'*40,'--audit-dir',tmp]
            with patch('sys.argv',args),patch.object(local,'ssh',side_effect=subprocess.TimeoutExpired(
                    ['ssh'],55,output=b'partial observation',stderr=b'timeout evidence')) as remote:
                with self.assertRaises(ValueError):local.main()
            remote.assert_called_once()
            files=list(Path(tmp).glob('old-original-reserve-inspect-*.json'));self.assertEqual(len(files),1)
            result=json.loads(files[0].read_text())
            self.assertEqual(result['status'],'UNKNOWN_NO_REPLAY')
            self.assertIsNone(result['returncode'])
            self.assertEqual(result['stdout'],'partial observation')


if __name__=='__main__':unittest.main()
