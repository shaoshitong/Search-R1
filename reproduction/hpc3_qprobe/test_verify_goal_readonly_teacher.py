"""A final CPU reader cannot recreate missing cache evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe.verify_goal import read_complete_teacher_cache
from search_g1.qprobe_warmup import TeacherCache, moments


class ReadonlyTeacherTests(unittest.TestCase):
    def fixture(self, root):
        root=root.resolve()
        questions=[dict(qid=str(i),question_hash=hashlib.sha256(str(i).encode()).hexdigest())
                   for i in range(5120)]
        identity=dict(questions=questions,training_world_size=4)
        (root/'identity.json').write_text(json.dumps(identity))
        cache=object.__new__(TeacherCache);cache.root=root;cache.identity=identity
        for index in range(10):
            raw=root/('raw-%02d.pt'%index);raw.write_bytes(('fixture raw '+str(index)).encode())
            groups=[]
            for i in range(index*512,(index+1)*512):
                vector=[1]*(i%6)+[0]*(5-i%6);count,mean,std=moments(vector)
                groups.append(dict(questions[i],strict_em=vector,correct_count=count,mean=mean,std=std,
                    trajectory_ids=[str(i)+'-'+str(j) for j in range(5)]))
            chunk=dict(index=index,groups=groups,raw_rollout=dict(relative_path=raw.name,
                bytes=raw.stat().st_size,sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
                contracts=dict(sampling_contract_sha='sampling',strict_scoring_contract_sha='strict',retriever_version='retriever'))
            (root/('chunk-%02d.json'%index)).write_text(json.dumps(chunk))
        expected=cache.verify_complete()
        return cache,expected

    def test_existing_science_receipt_is_consumed_without_any_writer_or_file_change(self):
        with tempfile.TemporaryDirectory() as folder:
            cache,expected=self.fixture(Path(folder))
            before={p.name:p.read_bytes() for p in cache.root.iterdir()}
            with patch('search_g1.qprobe_warmup.write_once',side_effect=AssertionError('writer reached')) as writer:
                self.assertEqual(read_complete_teacher_cache(cache),expected)
                writer.assert_not_called()
            self.assertEqual({p.name:p.read_bytes() for p in cache.root.iterdir()},before)

    def test_missing_complete_and_midread_disappearance_are_never_recreated(self):
        with tempfile.TemporaryDirectory() as folder:
            cache,_=self.fixture(Path(folder));complete=cache.root/'complete.json'
            saved=complete.read_bytes();complete.unlink()
            with patch('search_g1.qprobe_warmup.write_once',side_effect=AssertionError('writer reached')) as writer:
                with self.assertRaises(FileNotFoundError):read_complete_teacher_cache(cache)
                self.assertFalse(complete.exists());writer.assert_not_called()
                complete.write_bytes(saved)
                original_read=cache.read_chunk
                def disappearing(index):
                    chunk=original_read(index)
                    if index==9:complete.unlink()
                    return chunk
                with patch.object(cache,'read_chunk',side_effect=disappearing):
                    with self.assertRaises(FileNotFoundError):read_complete_teacher_cache(cache)
                self.assertFalse(complete.exists());writer.assert_not_called()

    def test_changed_original_raw_is_rejected_even_with_an_unchanged_complete_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            cache,_=self.fixture(Path(folder));raw=cache.root/'raw-07.pt'
            content=raw.read_bytes();raw.write_bytes(b'x'*len(content))
            with self.assertRaisesRegex(ValueError,'FULL_SHA_DIFFERS'):read_complete_teacher_cache(cache)

    def test_cross_chunk_trajectory_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache,_=self.fixture(Path(folder));p=cache.root/'chunk-01.json'
            chunk=json.loads(p.read_text());chunk['groups'][0]['trajectory_ids'][0]='0-0'
            p.write_text(json.dumps(chunk))
            with self.assertRaisesRegex(ValueError,'full pool trajectory IDs'):read_complete_teacher_cache(cache)


if __name__=='__main__':unittest.main()
