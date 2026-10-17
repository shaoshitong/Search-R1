"""Byte stability tests for timestamp settlement and genuine in-read mutation."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import final_cpu_acceptance as cpu
from reproduction.easy2560_joint4 import final_restore_acceptance as restore


def metadata(stat, **changes):
    fields = ('st_dev', 'st_ino', 'st_size', 'st_uid', 'st_mtime_ns', 'st_ctime_ns')
    return SimpleNamespace(**dict({key:getattr(stat,key) for key in fields}, **changes))


class FinalReaderTests(unittest.TestCase):
    DATA = b'{"step": 1}\n{"step": 2}\n'

    def invoke(self, kind, path):
        if kind == 'lines': return list(cpu.lines(path))
        if kind == 'small': return cpu.small(path)
        return restore._read(path)

    def raw(self, kind):
        return self.DATA if kind == 'lines' else b'{"step": 1}\n'

    def test_same_fd_mtime_ctime_settlement_passes_only_after_full_byte_comparison(self):
        for kind in ('small','lines','restore'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';raw=self.raw(kind);path.write_bytes(raw)
                real_fstat=os.fstat;calls=[]
                def settled(fd):
                    value=real_fstat(fd);calls.append(fd)
                    if len(calls)==1:
                        # Actual DPC observation: cached pre-settlement times
                        # differed156931ns; dev/inode/length/uid did not change.
                        return metadata(value,st_mtime_ns=value.st_mtime_ns-156931,
                                        st_ctime_ns=value.st_ctime_ns-156931)
                    return value
                with patch.object(os,'fstat',side_effect=settled): result=self.invoke(kind,path)
                self.assertGreaterEqual(len(calls),3)
                self.assertEqual(len(set(calls)),1)
                if kind=='lines':self.assertEqual(result,[dict(step=1),dict(step=2)])
                else:self.assertEqual(result[1]['sha256'],hashlib.sha256(raw).hexdigest())

    def test_path_only_timestamp_settlement_takes_two_path_snapshots_and_same_fd_reread(self):
        for kind in ('small','lines'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';path.write_bytes(self.raw(kind))
                real_stat=Path.stat;real_fstat=os.fstat;path_calls=[];fd_calls=[]
                def settle_path(current,*args,**kwargs):
                    value=real_stat(current,*args,**kwargs)
                    if current==path:
                        path_calls.append(current)
                        return metadata(value,st_mtime_ns=value.st_mtime_ns+156931,
                                        st_ctime_ns=value.st_ctime_ns+156931)
                    return value
                def fstat(fd):fd_calls.append(fd);return real_fstat(fd)
                with patch.object(Path,'stat',new=settle_path),patch.object(os,'fstat',side_effect=fstat):
                    self.invoke(kind,path)
                self.assertEqual(len(path_calls),2)
                self.assertGreaterEqual(len(fd_calls),3)

    def test_path_replacement_during_timestamp_fallback_reread_is_rejected(self):
        for kind in ('small','lines','restore'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';original=self.raw(kind);path.write_bytes(original)
                real_fstat=os.fstat;calls=[]
                def settle_then_replace(fd):
                    value=real_fstat(fd);calls.append(fd)
                    if len(calls)==1:
                        return metadata(value,st_mtime_ns=value.st_mtime_ns-156931,
                                        st_ctime_ns=value.st_ctime_ns-156931)
                    if len(calls)==3:
                        # Same-fd reread still sees the original bytes. The
                        # final path snapshot must detect its replacement.
                        path.rename(path.with_suffix('.old'));path.write_bytes(original)
                    return value
                with patch.object(os,'fstat',side_effect=settle_then_replace),self.assertRaisesRegex(ValueError,'CHANGED_DURING_READ'):
                    self.invoke(kind,path)
                self.assertEqual(len(calls),3)

    def test_same_size_byte_rewrite_during_read_is_rejected_with_original_prefix(self):
        for kind in ('small','lines','restore'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';original=self.raw(kind);path.write_bytes(original)
                real_fstat=os.fstat;calls=[]
                def change(fd):
                    calls.append(fd)
                    if len(calls)==2:
                        previous=path.stat();path.write_bytes(original.replace(b'1',b'9'))
                        os.utime(path,ns=(previous.st_atime_ns,previous.st_mtime_ns+1000000))
                    return real_fstat(fd)
                prefix='EASY2560_FINAL_RESTORE_SMALL_PROOF_CHANGED' if kind=='restore' else 'EASY2560_FINAL_CPU_.*CHANGED'
                with patch.object(os,'fstat',side_effect=change),self.assertRaisesRegex(ValueError,prefix):
                    self.invoke(kind,path)

    def test_append_and_path_replacement_during_read_are_not_timestamp_only(self):
        for kind in ('small','lines','restore'):
            for action in ('append','replace'):
                with self.subTest(kind=kind,action=action),tempfile.TemporaryDirectory() as directory:
                    path=Path(directory).resolve()/'evidence.json';original=self.raw(kind);path.write_bytes(original)
                    real_fstat=os.fstat;calls=[]
                    def change(fd):
                        calls.append(fd)
                        if len(calls)==2:
                            if action=='append':
                                with path.open('ab') as out:out.write(b'\n')
                            else:path.rename(path.with_suffix('.old'));path.write_bytes(original)
                        return real_fstat(fd)
                    with patch.object(os,'fstat',side_effect=change),self.assertRaisesRegex(ValueError,'CHANGED_DURING_READ'):
                        self.invoke(kind,path)

    def test_uid_change_is_never_accepted_even_if_bytes_and_timestamps_are_equal(self):
        for kind in ('small','lines','restore'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';path.write_bytes(self.raw(kind))
                real_fstat=os.fstat;calls=[]
                def changed_owner(fd):
                    value=real_fstat(fd);calls.append(fd)
                    return metadata(value,st_uid=value.st_uid+1) if len(calls)==2 else value
                with patch.object(os,'fstat',side_effect=changed_owner),self.assertRaisesRegex(ValueError,'CHANGED_DURING_READ'):
                    self.invoke(kind,path)

    def test_generator_rewrite_after_first_yield_still_fails_at_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'ledger.jsonl';path.write_bytes(self.DATA)
            iterator=cpu.lines(path);self.assertEqual(next(iterator),dict(step=1))
            previous=path.stat();path.write_bytes(self.DATA.replace(b'1',b'9'))
            os.utime(path,ns=(previous.st_atime_ns,previous.st_mtime_ns+1000000))
            with self.assertRaisesRegex(ValueError,'FINAL_LEDGER_CHANGED_DURING_READ'):list(iterator)

    def test_unchanged_metadata_never_triggers_an_additional_read_or_path_stat(self):
        for kind in ('small','lines'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                path=Path(directory).resolve()/'evidence.json';path.write_bytes(self.raw(kind))
                # Fix the metadata input to this branch. A real DPC settlement
                # is deliberately covered above and correctly needs a reread.
                stable=path.stat();calls=[];real_stat=Path.stat;path_calls=[]
                def fstat(fd):calls.append(fd);return stable
                def stat(current,*args,**kwargs):
                    if current==path:path_calls.append(current);return stable
                    return real_stat(current,*args,**kwargs)
                with patch.object(os,'fstat',side_effect=fstat),patch.object(Path,'stat',new=stat):self.invoke(kind,path)
                self.assertEqual(len(calls),2)
                self.assertEqual(len(path_calls),1)


if __name__=='__main__':unittest.main()
