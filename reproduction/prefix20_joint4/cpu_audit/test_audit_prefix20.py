"""Small adversarial fixtures. No Torch, model files, GPU or remote access."""
import copy
import importlib.abc
import io
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

if __package__:
    from . import audit_prefix20 as consumer
    from .safe_pt import Archive, UnsafeArchive, Inert
else:
    import audit_prefix20 as consumer
    from safe_pt import Archive, UnsafeArchive, Inert

Reads,Rejected,acquisition,early_history,main = (
    consumer.Reads,consumer.Rejected,consumer.acquisition,consumer.early_history,consumer.main)
evaluate_identity,final_answer,question_id,sha_value,strict,verify_selection = (
    consumer.evaluate_identity,consumer.final_answer,consumer.question_id,
    consumer.sha_value,consumer.strict,consumer.verify_selection)


class ImportedTorchMustFail(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('CPU consumer attempted Torch import')


_guard=ImportedTorchMustFail()

def setUpModule():sys.meta_path.insert(0,_guard)

def tearDownModule():sys.meta_path.remove(_guard)


class EarlyGateTests(unittest.TestCase):
    def test_missing200_cli_writes_immutable_rejected_receipt(self):
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();output=root/'audit.json'
            argv=['audit_prefix20.py','--base',str(root),'--commit','b'*40,'--run-id','a'*32,'--output',str(output)]
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):self.assertEqual(main(),1)
            receipt=json.loads(output.read_text())
            self.assertFalse(receipt['accepted']);self.assertFalse(receipt['torch_imported'])
            self.assertEqual(receipt['gpu_allocations'],0)
            self.assertIn('MISSING_COMPLETE200',receipt['reason'])
            prior=output.read_bytes()
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):main()
            self.assertEqual(output.read_bytes(),prior)

    def test_missing200_fails_before_optional_dependencies_or_pt_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'logs').mkdir()
            (root/'logs/actual-updates.jsonl').write_text(json.dumps(dict(step=1,run_id='a'*32))+'\n')
            with patch.object(consumer,'Archive', side_effect=AssertionError('PT touched too early')):
                with self.assertRaisesRegex(Rejected,'MISSING_COMPLETE200'):
                    early_history(Reads(root),'b'*40,'a'*32)

    def test_update201_rejected_even_when_first200_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'logs').mkdir()
            (root/'logs/actual-updates.jsonl').write_text(''.join(json.dumps(dict(step=i,run_id='a'*32))+'\n' for i in range(1,202)))
            with self.assertRaisesRegex(Rejected,'UPDATE201'):
                early_history(Reads(root),'b'*40,'a'*32)

    def test_truncated_jsonl_and_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(Rejected,'DUPLICATE'):
            strict(b'{"accepted":false,"accepted":true}')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'x').write_text('{}')
            with self.assertRaisesRegex(Rejected,'TRUNCATED'):
                Reads(root).rows('x')


class SelectionTests(unittest.TestCase):
    def fixture(self):
        eligible=[dict(qid='q%d'%i,source_index=i,normalized_question_hash='h%d'%i) for i in range(5120)]
        questions=[dict(ordinal=i,qid='q%d'%i,content_sha256='h%d'%i,pool_index=i,
            canonical_source_index=i,pool_cycle=0,trajectory_ids=['q%d-t%d'%(i,j) for j in range(5)],
            em_probabilities=[.5]*5,selected=i<512) for i in range(5120)]
        selected=[dict(qid=q['qid'],trajectory_ids=q['trajectory_ids'],em_probabilities=q['em_probabilities'],
            mixed_probability=acquisition(q['em_probabilities']),candidate_ordinal=q['ordinal']) for q in questions[:512]]
        return dict(stage='prefix20_selection_frozen_before_continuation',candidate_questions=5120,
            selected_questions=512,exploration_questions=0,questions=questions,selected=selected,
            prefix_freeze=dict(actual_optimizer_and_scheduler_unchanged=True,actor_rng_unchanged=False),
            prediction_freeze=dict(actual_optimizer_and_scheduler_unchanged=True,actor_rng_unchanged=True)),eligible

    def test_all_tied_probabilities_preserve_candidate_order(self):
        selection,eligible=self.fixture()
        self.assertEqual(verify_selection(selection,eligible),list(range(512)))

    def test_top448_plus64_exploration_cannot_pass_top512(self):
        selection,eligible=self.fixture();selection['exploration_questions']=64
        with self.assertRaisesRegex(Rejected,'TOP512_ALL'):
            verify_selection(selection,eligible)

    def test_low_probability_group_cannot_replace_top_group(self):
        selection,eligible=self.fixture();q=selection['questions'][0];q['em_probabilities']=[1.]*5
        with self.assertRaisesRegex(Rejected,'EXACT_TOP512'):
            verify_selection(selection,eligible)

    def test_gold_or_boolean_probabilities_do_not_silently_become_valid(self):
        with self.assertRaisesRegex(Rejected,'INVALID_G5'):
            acquisition([True,.5,.5,.5,.5])
        selection,eligible=self.fixture();selection['questions'][0]['canonical_source_index']=7
        with self.assertRaisesRegex(Rejected,'MEMBERSHIP'):
            verify_selection(selection,eligible)


class IdentityAndStrictEMTests(unittest.TestCase):
    def test_duplicate_final_questions_are_retained_and_order_matters(self):
        expected=[dict(prompt=[dict(content='Question: A?')],data_source='nq'),
                  dict(prompt=[dict(content='Question: A?')],data_source='nq'),
                  dict(prompt=[dict(content='Question: B?')],data_source='hotpotqa')]
        ids=[question_id(r['prompt']) for r in expected]
        proof=dict(run_id='a'*32,split='final',step=200,final=True,question_ids=ids,
            data_sources=['nq','nq','hotpotqa'],validation_sha256='d'*64,
            counts=dict(nq=2,hotpotqa=1),em=[1.,0.,0.],f1=[1.,.5,0.],correct=1,
            weighted_em=1/3,weighted_f1=.5)
        manifest=dict(splits=dict(final=dict(sha256='d'*64)))
        result=evaluate_identity(proof,expected,'final',200,'a'*32,manifest)
        self.assertEqual(result['rows'],3)
        self.assertAlmostEqual(result['macro_em'],.25)
        changed=copy.deepcopy(proof);changed['question_ids']=[ids[2],ids[1],ids[0]]
        with self.assertRaisesRegex(Rejected,'ORDERED'):
            evaluate_identity(changed,expected,'final',200,'a'*32,manifest)

    def test_seven_padded_rows_cannot_be_native_final(self):
        expected=[dict(prompt=[dict(content='Question: A?')],data_source='nq')]
        proof=dict(run_id='a'*32,split='final',step=200,final=True,question_ids=['padding']*8,
            data_sources=['nq']*8,validation_sha256='d'*64,counts=dict(nq=8),em=[0.]*8,f1=[0.]*8,
            correct=0,weighted_em=0.,weighted_f1=0.)
        with self.assertRaisesRegex(Rejected,'ORDERED'):
            evaluate_identity(proof,expected,'final',200,'a'*32,dict(splits=dict(final=dict(sha256='d'*64))))

    def test_nested_nonterminal_and_multiple_answers_do_not_count(self):
        self.assertEqual(final_answer('<think>reason</think><answer>yes</answer>'),'yes')
        for text in ('<information><answer>yes</answer></information>',
                     '<answer>yes</answer> trailing', '<answer>yes</answer><answer>yes</answer>',
                     '<answer><search>yes</search></answer>'):
            self.assertIsNone(final_answer(text))


class SafePTTests(unittest.TestCase):
    def zip_bytes(self,payload,storage=None):
        target=io.BytesIO()
        with zipfile.ZipFile(target,'w') as archive:
            archive.writestr('archive/data.pkl',payload)
            archive.writestr('archive/byteorder','little')
            if storage is not None:archive.writestr('archive/data/0',storage)
        return target.getvalue()

    def test_malicious_reduce_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel=Path(directory)/'must-not-exist'
            class Exploit:
                def __reduce__(self):return os.system,('touch '+str(sentinel),)
            with self.assertRaisesRegex(UnsafeArchive,'Disallowed pickle global'):
                Archive(self.zip_bytes(pickle.dumps(Exploit(),protocol=2)))
            self.assertFalse(sentinel.exists())

    def test_actual_scalar_adam_counter_read_without_torch(self):
        # Produce a modern Torch-style ZIP using protocol2 and inert fake
        # serialization names; no Torch package is loaded or executed.
        fake=types.ModuleType('torch');utils=types.ModuleType('torch._utils')
        exec('class FloatStorage: pass',fake.__dict__)
        exec('def _rebuild_tensor_v2(*args): raise AssertionError("must never execute")',utils.__dict__)
        fake._utils=utils
        marker=object()
        class Value:
            def __reduce__(self):return utils._rebuild_tensor_v2,(marker,0,(),(),False,{})
        class Writer(pickle.Pickler):
            def persistent_id(self,value):
                if value is marker:return ('storage',fake.FloatStorage,'0','cpu',1)
        output=io.BytesIO()
        with patch.dict(sys.modules,{'torch':fake,'torch._utils':utils}):
            Writer(output,protocol=2).dump(dict(step=Value()))
        with Archive(self.zip_bytes(output.getvalue(),struct.pack('<f',2000.))) as archive:
            self.assertEqual(archive.scalar(archive.value['step']),2000.)
            self.assertEqual(archive.value['step'].shape,())
        self.assertNotIn('torch',sys.modules)

    def test_numpy_rng_inert_global_marker_is_hashable_without_numpy_import(self):
        with Archive(self.zip_bytes(pickle.dumps({},protocol=2))) as archive:
            state=Inert(Inert,(0,),b'b');state.__setstate__((1,(4,),Inert('u4'),False,b'\x00'*16))
            self.assertEqual(archive.signature(state),archive.signature(state))
            self.assertIn('inert_global_type',repr(archive.signature(state)))

    def test_protocol2_private_rng_bytes_reconstruct_without_codec_execution(self):
        payload=pickle.dumps(dict(raw=b'\x00\xffrng'),protocol=2)
        with Archive(self.zip_bytes(payload)) as archive:
            self.assertEqual(archive.value['raw'],b'\x00\xffrng')

    def test_pickle_nonexperiment_globals_fail_closed(self):
        with self.assertRaisesRegex(UnsafeArchive,'Disallowed pickle global'):
            Archive(self.zip_bytes(b'\x80\x02cbuiltins\neval\n.'))


if __name__=='__main__':
    guard=ImportedTorchMustFail();sys.meta_path.insert(0,guard)
    try:unittest.main()
    finally:sys.meta_path.remove(guard)
