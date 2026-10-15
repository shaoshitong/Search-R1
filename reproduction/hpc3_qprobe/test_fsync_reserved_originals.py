import os
from pathlib import Path
import tempfile
import unittest

from reproduction.hpc3_qprobe import fsync_reserved_originals as f


class DurabilityTests(unittest.TestCase):
    def test_fsync_keeps_exact_owned_bytes_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'example';p.write_bytes(b'bounded test data');s=p.stat()
            row=dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=f.digest(p))
            value=f.flush_file(p,row,os.getuid())
            self.assertEqual(p.read_bytes(),b'bounded test data')
            self.assertEqual(f.digest(p),value['sha256_from_completed_full_read'])

    def test_wrong_inode_or_symlink_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'example';p.write_bytes(b'data');s=p.stat()
            row=dict(device=s.st_dev,inode=s.st_ino+1,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=f.digest(p))
            with self.assertRaises(ValueError):f.flush_file(p,row,os.getuid())
            link=p.with_name('alias');link.symlink_to(p);row['inode']=s.st_ino
            with self.assertRaises(ValueError):f.flush_file(link,row,os.getuid())


if __name__=='__main__':unittest.main()
