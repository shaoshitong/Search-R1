"""Scalar fixture tests only; no GPU, model, actual data, or quality claims."""
import copy
import tempfile
import unittest
from pathlib import Path
from reproduction.compatible_async_evaluation.contract import (
    protocol, PROTOCOL_PATH, digest, verify_loaded_workers, verify_source_tree)
from reproduction.compatible_async_evaluation.journal import (
    make_binding,append_batch,load_prefix,aggregate_final)


class CompatibleJournalTests(unittest.TestCase):
    def setUp(self):
        self.ids=['fixture'+str(i//2) for i in range(512)]
        self.sources=['nq']*256+['hotpotqa']*256
        self.binding=make_binding('fixture','a'*64,protocol()['datasets']['validation']['sha256'],self.ids,self.sources,5,'validation')
        self.rows=[dict(ordinal=i,question_id=q,data_source=s,em=0,f1=0.,actual_calls=0,
            valid_action=1,turns=1,answer_present=False,answer_sha256=None,response_token_count=1,
            model_token_count=1) for i,(q,s) in enumerate(zip(self.ids,self.sources))]

    def test_fixed_occurrences_and_separate_recipe(self):
        with tempfile.TemporaryDirectory() as path:
            append_batch(path,self.binding,self.ids,self.sources,self.rows[:64])
            append_batch(path,self.binding,self.ids,self.sources,self.rows[64:])
            result=aggregate_final(self.binding,self.ids,self.sources,load_prefix(path,self.binding,self.ids,self.sources))
            self.assertEqual(result['qa_count'],512)
            self.assertEqual(result['recipe_id'],'old-context-fp32-async')
            self.assertTrue(result['old_content_fp32_async'])
            self.assertNotIn('canonical_legacy_evaluation',result)
            changed=dict(self.binding,recipe_id='strict-old-synchronous')
            with self.assertRaises(ValueError):load_prefix(path,changed,self.ids,self.sources)

    def test_actual_async_and_fp32_worker_proof_required(self):
        rows=[dict(rank=i,world_size=8,cuda_uuid=str(i),hf_parameter_dtypes=['torch.float32'],
            vllm_parameter_dtypes=['torch.float32'],hf_parameter_numel=1,
            optimizer_constructed=False,optimizer_steps=0,async_phase_enabled=True,
            recipe_sha256=digest(PROTOCOL_PATH)) for i in range(8)]
        self.assertTrue(verify_loaded_workers(rows))
        for field,value in [('async_phase_enabled',False),('hf_parameter_dtypes',['torch.bfloat16']),('optimizer_constructed',True)]:
            bad=copy.deepcopy(rows);bad[0][field]=value
            with self.assertRaises(ValueError):verify_loaded_workers(bad)

    def test_final_count_step_and_seven_sources(self):
        ids=[str(i) for i in range(51713)];sources=['fixture'+str(i%7) for i in range(51713)]
        result=make_binding('fixture','a'*64,protocol()['datasets']['final']['sha256'],ids,sources,200,'final')
        self.assertEqual(result['question_count'],51713)
        with self.assertRaises(ValueError):make_binding('fixture','a'*64,protocol()['datasets']['final']['sha256'],ids,sources,150,'final')

    def test_source_tree_must_be_current_published_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'source.py';path.write_text('# fixture\n')
            mapping={'source.py':digest(path)}
            self.assertEqual(verify_source_tree(root,mapping)['source_files'],1)
            path.write_text('# changed\n')
            with self.assertRaises(ValueError):verify_source_tree(root,mapping)


if __name__=='__main__':unittest.main()
