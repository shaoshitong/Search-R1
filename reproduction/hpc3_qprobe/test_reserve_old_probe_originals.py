"""Non-network checks for the fixed backup scope and actual-held acceptance."""
import copy
import io
import json
from pathlib import Path
import tempfile
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

    def supervise_case(self,actual_exit,invalid_result=False):
        supervisor=dict(pid=101,start_ticks=1001,sid=101,uid=1002)
        worker=dict(pid=102,start_ticks=1002,sid=101,uid=1002)
        token='b'*32;commit='a'*40
        child=Mock(pid=102,stdin=io.BytesIO());child.wait.return_value=actual_exit
        metadata={'claim.json':dict(token=token,source_commit=commit),
                  'process.json':dict(token=token,source_commit=commit,actual_process=supervisor),
                  'worker-result.json':dict(actual_process=worker,exit_code=0,status=r.STATUS,
                                            source_cleanup_performed=False,complete_sha256='c'*64)}
        if invalid_result:metadata['worker-result.json']['actual_process']=dict(worker,pid=999)
        with tempfile.TemporaryDirectory() as tmp:
            control=Path(tmp)
            with patch.object(r,'CONTROL',control),patch.object(r,'code_guard',return_value=(r.plan(),control)),\
                 patch.object(r,'read_owned',side_effect=lambda path,uid:metadata[Path(path).name]),\
                 patch.object(r,'birth',side_effect=lambda pid:worker if pid==102 else supervisor),\
                 patch.object(r.os,'getsid',return_value=101),patch.object(r.subprocess,'Popen',return_value=child),\
                 patch.object(r,'digest',return_value='c'*64):
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


if __name__=='__main__':unittest.main()
