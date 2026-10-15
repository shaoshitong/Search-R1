"""Nested original consumers plus byte/source/live boundary attacks; no remote I/O."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import recovery_lineage_stage as lineage

PINS=list(lineage.RECOVERIES)[1:]


class ValidationStage(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.calls=Counter();self.loads=Counter()
        self.host=True;self.dead=True;self.new_dependency=False;self.modules=[]
        self.evidence=self.root/'evidence.json';self.evidence.write_bytes(b'original bytes')
        self.code=self.root/'consumer.py';self.code.write_bytes(b'original code!')
        self.code_sha=self.digest(self.code)
        self.headers={}
        for pin in PINS:
            p=self.root/(pin+'.json');p.write_text(pin)
            self.headers[pin]=dict(commit=pin,publication_path=str(p),publication_sha256=self.digest(p))
        self.owner=patch.object(lineage.reader,'OWNER',os.getuid());self.owner.start();self.addCleanup(self.owner.stop)
        self.shared_owner=patch.object(lineage.reader.shared,'OWNER',os.getuid());self.shared_owner.start();self.addCleanup(self.shared_owner.stop)
        self.load_patch=patch.object(lineage.reader,'load_fixed',side_effect=self.load);self.load_patch.start();self.addCleanup(self.load_patch.stop)
        self.pub_patch=patch.object(lineage.reader,'publication',side_effect=self.publication);self.pub_patch.start();self.addCleanup(self.pub_patch.stop)

    @staticmethod
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def publication(self,pin):
        if self.digest(self.code)!=self.code_sha:raise ValueError('full source SHA differs')
        return self.root,self.headers[pin]

    def load(self,pin,name,attribute):
        self.publication(pin);self.loads[pin]+=1
        control=SimpleNamespace(digest=self.digest,live=lambda _:not self.dead,
                                read=lambda path:json.loads(Path(path).read_bytes()))
        module=SimpleNamespace(reader=lineage.reader,**{attribute:control})
        index=PINS.index(pin)
        def consume(commit):
            self.calls[pin]+=1
            if not self.host:raise ValueError('wrong actual host/kernel')
            if control.live(None):raise ValueError('original birth is live')
            sha=control.digest(self.evidence)
            dependencies=[]
            if index:
                for _ in range(4):
                    old,_=module.reader.load_fixed(PINS[index-1],lineage.RECOVERIES[PINS[index-1]],'control')
                    dependencies.append(old.consume(PINS[index-1]))
            if self.new_dependency and index==len(PINS)-1:
                module.reader.load_fixed('a'*40,'unseen','control')
            return dict(source={'commit':commit},evidence_sha256=sha,dependencies=dependencies)
        module.consume=consume;self.modules.append(module)
        return module,self.headers[pin]

    def stage(self):
        stage=lineage.Stage();module,_=stage.load_fixed(PINS[-1],lineage.RECOVERIES[PINS[-1]],'control')
        return stage,module,module.consume(PINS[-1])

    def test_original_nested_proof_equal_with_each_original_consumer_twice_and_private_modules(self):
        original,_=self.load(PINS[-1],lineage.RECOVERIES[PINS[-1]],'control')
        expected=original.consume(PINS[-1]);baseline=self.calls[PINS[0]]
        self.assertEqual(baseline,4**(len(PINS)-1));self.calls.clear()
        stage,module,actual=self.stage();self.assertEqual(actual,expected)
        # A caller cannot mutate the private memoized proof.
        actual['source']['commit']='tampered return'
        self.assertEqual(module.consume(PINS[-1]),expected)
        stage.seal();self.assertEqual(dict(self.calls),{pin:2 for pin in PINS})
        self.assertIs(original.reader,lineage.reader)
        with self.assertRaisesRegex(ValueError,'closed stage'):module.consume(PINS[-1])
        with self.assertRaises(ValueError):stage.seal()

    def test_same_length_metadata_tamper_with_restored_mtime_refuses_at_seal(self):
        stage,_,_=self.stage();s=self.evidence.stat()
        self.evidence.write_bytes(b'changed! bytes');os.utime(self.evidence,ns=(s.st_atime_ns,s.st_mtime_ns))
        self.assertEqual(self.evidence.stat().st_size,s.st_size)
        with self.assertRaises(ValueError):stage.seal()

    def test_final_rehash_catches_change_after_all_original_consumers_reran(self):
        stage,_,_=self.stage();original=self.publication
        def mutate(pin):
            result=original(pin);self.evidence.write_bytes(b'changed! bytes');return result
        with patch.object(lineage.reader,'publication',side_effect=mutate),self.assertRaises(ValueError):stage.seal()

    def test_full_source_change_and_publication_identity_change_refuse(self):
        stage,_,_=self.stage();s=self.code.stat();self.code.write_bytes(b'changed! code!')
        os.utime(self.code,ns=(s.st_atime_ns,s.st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'full source'):stage.seal()
        self.code.write_bytes(b'original code!');stage,_,_=self.stage()
        self.headers[PINS[0]]=dict(self.headers[PINS[0]],publication_sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'publication changed'):stage.seal()

    def test_host_and_actual_live_birth_are_rechecked_at_the_boundary(self):
        stage,_,_=self.stage();self.host=False
        with self.assertRaisesRegex(ValueError,'host/kernel'):stage.seal()
        self.host=True;stage,_,_=self.stage();self.dead=False
        with self.assertRaisesRegex(ValueError,'birth is live'):stage.seal()

    def test_new_dependency_during_seal_and_unknown_top_level_refuse(self):
        stage,_,_=self.stage();self.new_dependency=True
        with self.assertRaisesRegex(ValueError,'new source dependency'):stage.seal()
        with self.assertRaisesRegex(ValueError,'unregistered'):lineage.consume('a'*40,'unknown')

    def test_no_cross_boundary_memoization(self):
        stage,_,_=self.stage();stage.seal();self.calls.clear()
        other,module,_=self.stage();other.seal()
        self.assertEqual(dict(self.calls),{pin:2 for pin in PINS})
        self.assertIsNot(stage,other)

    def test_plain_owned_json_is_rehashed_after_original_consumers_reran(self):
        stage,_,_=self.stage();module,_=stage.load_fixed(PINS[-1],lineage.RECOVERIES[PINS[-1]],'control')
        path=self.root/'worker-result.json';path.write_text('{"exit":0}')
        self.assertEqual(module.control.read(path),{'exit':0})
        original=self.publication
        def mutate(pin):
            result=original(pin);s=path.stat();path.write_text('{"exit":1}')
            os.utime(path,ns=(s.st_atime_ns,s.st_mtime_ns));return result
        with patch.object(lineage.reader,'publication',side_effect=mutate),self.assertRaises(ValueError):stage.seal()

    def test_observer_plain_json_and_with_sha_reads_are_both_retained(self):
        path=self.root/'observer.json';path.write_text('{"value":0}')
        original=self.load
        def load(pin,name,attribute):
            module,proof=original(pin,name,attribute)
            module.read_json=lambda p,with_sha=False:(json.loads(Path(p).read_bytes()),self.digest(p)) if with_sha else json.loads(Path(p).read_bytes())
            return module,proof
        with patch.object(lineage.reader,'load_fixed',side_effect=load):
            stage,module,_=self.stage()
        self.assertEqual(module.read_json(path),{'value':0})
        self.assertEqual(module.read_json(path,with_sha=True),({'value':0},self.digest(path)))
        original=self.publication
        def mutate(pin):
            result=original(pin);path.write_text('{"value":1}');return result
        with patch.object(lineage.reader,'publication',side_effect=mutate),self.assertRaises(ValueError):stage.seal()

    def test_original50_five_held_json_files_rehash_even_when_shared_validator_hashes_them(self):
        fixture=json.loads(Path(__file__).with_name('archive100_third_failure_fixture.json').read_text())
        entries=next(value for name,value in fixture.items() if name.endswith('/archive-50'))
        folder=self.root/'final-control'/lineage.reader.RUN/'archive-50';folder.mkdir(parents=True)
        names=('submit-claim.json','runner-binding.json','task-binding.json','worker-result.json','task-result.json')
        for name in names:
            raw=entries[name]['value'].encode();self.assertEqual(self.digest_bytes(raw),entries[name]['sha256'])
            (folder/name).write_bytes(raw)
        claim=json.loads((folder/names[0]).read_bytes());original_load=self.load
        def load(pin,name,attribute):
            module,proof=original_load(pin,name,attribute);original_consume=module.consume
            def consume(commit):
                value=original_consume(commit)
                if commit==PINS[-1]:
                    docs=[module.control.read(folder/n) for n in names]
                    held=lineage.reader.validate_archive_documents(*docs,step=50,binding=claim['producer'],
                        publication=claim['source'],allowed_controllers=[claim['controller']],alive=lambda _:False)
                    self.assertTrue(held['ready']);value['held50']=held
                return value
            module.consume=consume;return module,proof
        with patch.object(lineage.reader,'ROOT',self.root),\
             patch.object(lineage.reader.ctl,'command',return_value=claim['command']),\
             patch.object(lineage.reader,'birth_ok',side_effect=lambda b:isinstance(b,dict) and b.get('uid')==14090),\
             patch.object(lineage.reader,'load_fixed',side_effect=load):
            for name in names:
                with self.subTest(file=name):
                    stage,_,_=self.stage()
                    self.assertTrue(all(str(folder/n) in stage.retained for n in names))
                    path=folder/name;raw=path.read_bytes();original=self.publication
                    def mutate(pin):
                        result=original(pin);s=path.stat();path.write_bytes(b'X'+raw[1:])
                        os.utime(path,ns=(s.st_atime_ns,s.st_mtime_ns));return result
                    with patch.object(lineage.reader,'publication',side_effect=mutate),self.assertRaises(ValueError):stage.seal()
                    path.write_bytes(raw)

    @staticmethod
    def digest_bytes(value):return hashlib.sha256(value).hexdigest()


if __name__=='__main__':unittest.main()
