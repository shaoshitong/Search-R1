"""Byte-copy/relocation CPU tests; fixtures are text, no actual model states."""
import copy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe.archive_completed_pair import copy_checked, destination, relocate, file_records
from reproduction.hpc3_qprobe import archive_completed_pair as transport
from search_g1.qprobe_features import sha_file


class ArchiveTests(unittest.TestCase):
    def measurement(self, results):
        output=io.StringIO()
        with patch.object(transport.subprocess,'run',side_effect=results) as run,contextlib.redirect_stdout(output):
            value=transport.measured_owner_storage('fixture',5)
        return value,run,[json.loads(line) for line in output.getvalue().splitlines()]

    def test_complete_zero_error_measurement_admits_exact_bytes(self):
        value,run,logs=self.measurement([subprocess.CompletedProcess([],0,'1234\t'+str(transport.ROOT.parent)+'\n','')])
        self.assertEqual(value,1234);self.assertEqual(run.call_count,1)
        self.assertTrue(logs[0]['accepted']);self.assertEqual(logs[0]['actual_returncode'],0)

    def test_live_enoent_requires_new_complete_zero_error_walk(self):
        error='du: cannot access '+str(transport.ROOT.parent)+"/cache/old': No such file or directory\n"
        good='4321\t'+str(transport.ROOT.parent)+'\n'
        value,run,logs=self.measurement([subprocess.CompletedProcess([],1,'999\tbad\n',error),
                                       subprocess.CompletedProcess([],0,good,'')])
        self.assertEqual(value,4321);self.assertEqual(run.call_count,2)
        self.assertEqual(logs[0]['stderr'],error);self.assertFalse(logs[0]['accepted'])
        self.assertTrue(logs[1]['accepted'])

    def test_permission_malformed_and_mixed_errors_fail_without_retry(self):
        root=str(transport.ROOT.parent)
        for rc,stdout,stderr in [(1,'123\t'+root+'\n','du: cannot read '+root+'/private: Permission denied\n'),
                                 (0,'123\t/other\n',''),(0,'123\t'+root+'\n','\n'),
                                 (1,'','du: '+root+'/a: No such file or directory\ndu: Permission denied\n')]:
            output=io.StringIO()
            with self.subTest(stderr=stderr),patch.object(transport.subprocess,'run',return_value=
                    subprocess.CompletedProcess([],rc,stdout,stderr)) as run,contextlib.redirect_stdout(output):
                with self.assertRaises(ValueError):transport.measured_owner_storage('fixture',5)
                self.assertEqual(run.call_count,1)
                logged=json.loads(output.getvalue());self.assertEqual(logged['stderr'],stderr)
                self.assertFalse(logged['accepted'])

    def test_enoent_exhaustion_and_timeout_are_preserved_not_accepted(self):
        error='du: '+str(transport.ROOT.parent)+'/gone: No such file or directory\n'
        output=io.StringIO()
        with patch.object(transport.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'',error)) as run,contextlib.redirect_stdout(output):
            with self.assertRaises(ValueError):transport.measured_owner_storage('fixture',5)
            self.assertEqual(run.call_count,3)
        self.assertEqual(len(output.getvalue().splitlines()),3)
        output=io.StringIO()
        with patch.object(transport.subprocess,'run',side_effect=subprocess.TimeoutExpired(['du'],1800,output=b'partial',stderr=b'error')),contextlib.redirect_stdout(output):
            with self.assertRaises(subprocess.TimeoutExpired):transport.measured_owner_storage('fixture',5)
        record=json.loads(output.getvalue());self.assertFalse(record['accepted']);self.assertIn('partial',record['stdout'])

    def test_two_independent_copies_and_actual_receipt_relocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            path=source/'cp/model.txt';path.parent.mkdir();path.write_bytes(b'synthetic-file-content')
            st=path.stat();row=dict(path=str(path),relative_path='model.txt',inode=st.st_ino,
                bytes=st.st_size,mtime_ns=st.st_mtime_ns,full_read_sha256=sha_file(path))
            receipt=dict(source_base=str(source),checkpoint=str(path.parent),question_probe_state_path=str(path),
                head_path=str(path),files=[row],reward_files=[copy.deepcopy(row)],actual_cuda_restore_accepted=False)
            for label in ('archive','target'):
                target=root/label;target.mkdir();dest=destination(path,source,target)
                copy_checked(path,dest,row['full_read_sha256'],row['bytes'])
                rebound=relocate(receipt,target)
                self.assertEqual(rebound['files'][0]['path'],str(dest))
                self.assertNotEqual(rebound['files'][0]['inode'],row['inode'])
                self.assertEqual(rebound['files'][0]['full_read_sha256'],row['full_read_sha256'])
                self.assertEqual(dest.read_bytes(),path.read_bytes())
                self.assertEqual(len(list(file_records(rebound))),2)
                self.assertFalse(rebound['actual_cuda_restore_accepted'])
            self.assertEqual(receipt['files'][0],row)

    def test_source_tamper_existing_copy_and_target_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'source.txt';p.write_bytes(b'original')
            sha=sha_file(p);size=p.stat().st_size
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):copy_checked(p,root/'bad.txt',sha,size)
            self.assertFalse((root/'bad.txt').exists())
            p.write_bytes(b'original');copy_checked(p,root/'copy.txt',sha,size)
            with self.assertRaises(ValueError):copy_checked(p,root/'copy.txt',sha,size)

    def test_external_encoder_shards_keep_one_parent_and_safe_names(self):
        source=Path('/data/own/source');target=Path('/data/own/archive')
        a=destination('/data/shared/model/index.json',source,target)
        b=destination('/data/shared/model/shard.safetensors',source,target)
        self.assertEqual(a.parent,b.parent);self.assertTrue(a.is_relative_to(target))
        self.assertNotEqual(a.parent,destination('/data/other/model/index.json',source,target).parent)

    def test_retained_source_checkpoint_keeps_hf_and_training_state_together(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source=root/'new-run';source.mkdir()
            cp=root/'old-run/checkpoints/actor/global_step_5';(cp/'training_state').mkdir(parents=True)
            paths=[cp/'model.safetensors',cp/'training_state/rank-0.pt']
            records=[];target=root/'archive';target.mkdir()
            for path in paths:
                path.write_bytes(b'fixture'+path.name.encode());s=path.stat()
                row=dict(path=str(path),inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,
                    full_read_sha256=sha_file(path));records.append(row)
                copy_checked(path,destination(path,source,target,cp),row['full_read_sha256'],row['bytes'])
            receipt=dict(source_base=str(source),checkpoint=str(cp),question_probe_state_path=str(paths[1]),
                head_path=str(paths[1]),files=records,actual_cuda_restore_accepted=False)
            moved=relocate(receipt,target)
            copied_cp=Path(moved['checkpoint'])
            self.assertEqual(copied_cp,target/'restored-checkpoint')
            self.assertEqual(Path(moved['files'][1]['path']),copied_cp/'training_state/rank-0.pt')
            self.assertEqual({str(p.relative_to(copied_cp)) for p in copied_cp.rglob('*') if p.is_file()},
                             {'model.safetensors','training_state/rank-0.pt'})


if __name__=='__main__':unittest.main()
