import hashlib
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


if __name__=='__main__':unittest.main()
