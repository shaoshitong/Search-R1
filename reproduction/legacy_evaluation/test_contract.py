"""CPU fixture checks; never assert real evaluator/precision acceptance."""
import copy
import tempfile
from pathlib import Path
import unittest
from reproduction.legacy_evaluation.contract import (PROTOCOL_PATH,digest,protocol,fp32_loader_options,
    verify_loaded_workers,verify_result,verify_source_tree,occurrence_sha256)

class LegacyEvaluationTests(unittest.TestCase):
    def test_actual_dtype_not_just_declared_config_is_required(self):
        rows=[dict(rank=i,world_size=8,cuda_uuid='fixture'+str(i),hf_parameter_dtypes=['torch.float32'],
            vllm_parameter_dtypes=['torch.float32'],hf_parameter_numel=1,optimizer_constructed=False,
            optimizer_steps=0,recipe_sha256=digest(PROTOCOL_PATH)) for i in range(8)]
        self.assertTrue(verify_loaded_workers(rows))
        for key,value in [('hf_parameter_dtypes',['torch.bfloat16']),('vllm_parameter_dtypes',['torch.bfloat16']),('optimizer_constructed',True)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):verify_loaded_workers(bad)
    def test_new_inline_metrics_cannot_be_relabelled_legacy(self):
        ids=[str(i) for i in range(512)];sources=['nq']*256+['hotpotqa']*256
        spec=protocol();result=dict(recipe_id=spec['recipe_id'],recipe_sha256=digest(PROTOCOL_PATH),step=5,
            split='validation',qa_count=512,question_count=512,validation_sha256=spec['datasets']['validation']['sha256'],
            canonical_occurrence_order_verified=True,occurrence_sha256=occurrence_sha256(ids,sources),source_counts={'nq':256,'hotpotqa':256})
        self.assertTrue(verify_result(result,'validation',5,ids,sources))
        for key,value in [('recipe_sha256',None),('qa_count',511),('canonical_occurrence_order_verified',False),('validation_sha256','b'*64),('occurrence_sha256','0'*64),('source_counts',{'nq':255,'hotpotqa':257})]:
            with self.assertRaises(ValueError):verify_result(dict(result,**{key:value}),'validation',5,ids,sources)
    def test_new_source_tree_is_rejected_without_any_gpu_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'LEGACY_SOURCE_MISMATCH'):verify_source_tree(Path(directory))

if __name__=='__main__':unittest.main()
