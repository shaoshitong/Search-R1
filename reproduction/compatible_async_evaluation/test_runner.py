"""CPU-only evaluator controls: no old-source activation or CUDA execution."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.compatible_async_evaluation import runner


class RunnerTests(unittest.TestCase):
    def test_wrapper_config_is_historical_run_and_hash_bound(self):
        config={'max_turns':4}
        proof=dict(run_id='historical',resolved_config=config,
            resolved_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        with patch.object(runner,'protocol',return_value={'protocol_authoritative_source':{'historical_run':'historical'}}):
            self.assertEqual(runner.resolved_document(proof,'current'),config)
            with self.assertRaises(ValueError):runner.resolved_document(dict(proof,run_id='other'),'current')
            proof['resolved_config']['max_turns']=2
            with self.assertRaises(ValueError):runner.resolved_document(proof,'current')

    def test_result_reentry_accepts_identical_actual_result_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'compatible-result.json'
            result={'qa_count':51713,'actual_gpu_execution_verified':True,'old_content_fp32_async':True}
            runner.finalize_result(path,result)
            runner.finalize_result(path,result)
            with self.assertRaises(ValueError):runner.finalize_result(path,dict(result,qa_count=512))
            self.assertEqual(json.loads(path.read_text()),result)

    def test_result_symlink_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'target.json';target.write_text('{}')
            path=Path(directory)/'compatible-result.json';path.symlink_to(target)
            with self.assertRaises(ValueError):runner.finalize_result(path,{})

    def test_new_controller_and_worker_only_no_old_tree_activation_or_fit(self):
        source=Path(runner.__file__).read_text();tree=ast.parse(source)
        for forbidden in ('legacy_eval_control','activate(', 'setup_worker','legacy_source_root','canonical_legacy_evaluation'):
            self.assertNotIn(forbidden,source)
        self.assertIn('from reproduction.compatible_async_evaluation.async_controller import run as original_loop',source)
        self.assertIn('from reproduction.compatible_async_evaluation.worker_runtime import make_worker_class',source)
        self.assertIn("os.environ['SEARCH_G1_PROFILE'] = 'compatible_async_evaluation'",source)
        self.assertIn("os.environ['SEARCH_G1_COMPAT_EVAL_BINDING']",source)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','update_actor','update_policy') for n in ast.walk(tree)))
        self.assertIn("binding['source_sha256']",source)
        self.assertIn('old_content_fp32_async=True',source)

if __name__=='__main__':unittest.main()
