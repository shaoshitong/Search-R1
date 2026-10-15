import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import retire_exact78_old_originals as r


class ExactRetirement(unittest.TestCase):
    def test_frozen_scope_authorization_and_keep_exclusion(self):
        with patch.object(r.reader,'OWNER',os.getuid()):plan,proof,auth=r.baseline()
        self.assertEqual(proof['sha256'],r.PLAN_SHA)
        self.assertEqual(auth['sha256'],r.HUMAN_SHA)
        self.assertEqual(len(plan['candidate_files']),78)
        self.assertEqual(sum(row['bytes'] for row in plan['candidate_files']),115100222440)
        self.assertFalse({x['path'] for x in plan['candidate_files']} &
                         {x['path'] for x in plan['retained_source100_dependency_metadata']})

    def fixture(self,directory):
        root=Path(directory).resolve();p=root/'payload';p.write_bytes(b'protected exact bytes')
        s=p.stat();row=dict(path=str(p),bytes=s.st_size,inode=s.st_ino,device=s.st_dev,
            uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        return root,p,row,r.old.tree(root)

    def test_full_hash_held_fd_exact_unlink_and_keep(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);keep=root/'complete.json';keep.write_text('retain')
            item=r.open_hash(row,root,snapshot)
            try:r.unlink_held(item)
            finally:os.close(item[2]);os.close(item[1])
            self.assertFalse(p.exists());self.assertEqual(keep.read_text(),'retain')

    def test_changed_content_refuses_before_unlink(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);row['sha256']='0'*64
            with self.assertRaises(ValueError):r.open_hash(row,root,snapshot)
            self.assertTrue(p.exists())

    def test_replaced_name_after_hash_refuses_unlink(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);item=r.open_hash(row,root,snapshot)
            p.rename(root/'old');p.write_text('new')
            try:
                with self.assertRaises(ValueError):r.unlink_held(item)
                self.assertEqual(p.read_text(),'new')
            finally:os.close(item[2]);os.close(item[1])

    def test_mutation_after_hash_refuses_unlink(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);item=r.open_hash(row,root,snapshot);p.write_text('changed')
            try:
                with self.assertRaises(ValueError):r.unlink_held(item)
                self.assertTrue(p.exists())
            finally:os.close(item[2]);os.close(item[1])

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);p.rename(root/'real');p.symlink_to(root/'real')
            with self.assertRaises(OSError):r.open_hash(row,root,snapshot)

    def test_same_reported_signature_changed_bytes_still_refuses(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d)
            with patch.object(r.reader,'signature',return_value=(1,2,3,4,5)):
                item=r.open_hash(row,root,snapshot);p.write_bytes(b'x'*row['bytes'])
                try:
                    with self.assertRaisesRegex(ValueError,'content changed'):r.unlink_held(item)
                    self.assertTrue(p.exists())
                finally:os.close(item[2]);os.close(item[1])

    def test_later_candidate_hash_failure_prevents_first_unlink(self):
        with tempfile.TemporaryDirectory() as d,patch.object(r,'OWNER',os.getuid()),patch.object(r.old.ctl,'OWNER',os.getuid()):
            root,p,row,snapshot=self.fixture(d);item=r.open_hash(row,root,snapshot)
            plan=dict(candidate_files=[row,dict(row,path=str(root/'second'))],prospective_trees=[str(root)])
            binding=dict(actual_process={'pid':123});claim=dict(source={},plan={},authorization={},admission_sha256='a')
            with patch.object(r,'DIRECTORY',root),patch.object(r,'code_guard',return_value={}),\
                 patch.object(r.reserve,'read_owned',side_effect=[claim,binding]),\
                 patch.object(r.reserve,'birth',return_value=binding['actual_process']),\
                 patch.object(r,'baseline',return_value=(plan,{},{})),patch.object(r,'admission'),\
                 patch.object(r,'consumers'),patch.object(r,'snapshots',return_value={str(root):snapshot}),\
                 patch.object(r,'retained_full',return_value=[]),patch.object(r,'progress'),\
                 patch.object(r,'open_hash',side_effect=[item,ValueError('later candidate SHA failed')]),\
                 patch.object(r,'unlink_held') as unlink:
                with self.assertRaisesRegex(ValueError,'later candidate'):r.worker('a'*40)
                unlink.assert_not_called()
            self.assertTrue(p.exists())
            with self.assertRaises(OSError):os.fstat(item[2])

    def test_supervisor_binding_publish_failure_still_waits_actual_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();birth=dict(pid=77,sid=77,uid=14090,start_ticks=1)
            child=type('Child',(),{})();child.pid=78;child.stdin=io.BytesIO();waits=[]
            child.wait=lambda:waits.append(child.pid) or 0
            published=[]
            def publish(path,value):
                if path.name=='worker-process.json':raise OSError('binding write failed')
                published.append(value)
            with patch.object(r,'DIRECTORY',root),patch.object(r,'code_guard'),\
                 patch.object(r.reserve,'read_owned',return_value=dict(actual_process=birth)),\
                 patch.object(r.reserve,'birth',side_effect=[birth,dict(birth,pid=78)]),\
                 patch.object(r.subprocess,'Popen',return_value=child),patch.object(r.reserve,'publish',side_effect=publish):
                self.assertEqual(r.supervise('a'*40),1)
            self.assertEqual(waits,[78]);self.assertEqual(published[0]['actual_child_exit_code'],0)
            self.assertIn('binding write failed',published[0]['error'])


if __name__=='__main__':unittest.main()
