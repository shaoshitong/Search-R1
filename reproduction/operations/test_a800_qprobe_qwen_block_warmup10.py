"""Tiny local checks of the operations contract; no model, SSH or GPU access."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import re
import unittest

PATH=Path(__file__).with_name('a800_qprobe_qwen_block_warmup10.py')
spec=importlib.util.spec_from_file_location('qwen_warmup10_ops_tested',PATH)
ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)


class QwenWarmupOperationsTest(unittest.TestCase):
    def test_all_remote_programs_parse_and_preserve_actual_parent(self):
        scaffold,base=ops.load_support()
        for action in ('stage','prepare','run-once','observe'):
            with self.subTest(action=action):
                code=ops.program(scaffold,base,action);ast.parse(code)
                self.assertIn(ops.PREFIX,code)
                self.assertNotIn('pidfd_send_signal',code)
        stage=ops.program(scaffold,base,'stage')
        self.assertIn("prior=Path('/data/shaoshitong/Search-G1-qprobe-token-binary-warmup10-a799f7aff3e94045b64791655b878c31')",stage)
        self.assertIn("item=archive.next()",stage)
        self.assertNotIn("for item in archive:",stage)

    def test_latest_native_human_contract_rejects_manual_or_old_budget(self):
        human=json.loads(ops.HUMAN.read_bytes());ops.validate_human(human)
        for key,value in (('native_block_required',False),('manual_transformer_reimplementation_forbidden',False),('warmup_epochs',20),('expected_actual_head_adam_updates',400),('public_source_publication_explicitly_authorized',False)):
            changed=copy.deepcopy(human);changed[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):ops.validate_human(changed)

    def test_cpu_source_proof_distinguished_from_actual_copy(self):
        node=next(node for node in ast.parse(ops.SUPPORT).body if isinstance(node,ast.FunctionDef) and node.name=='native_proof')
        environment={'re':re};exec(compile(ast.Module(body=[node],type_ignores=[]),'<native-proof>','exec'),environment)
        proof=dict(native_class=ops.NATIVE_CLASS,transformers_version='4.46.3',block_index=35,
            official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',source_tensors_verified=True,
            official_block_copy_verified=False,head_initialized=False,fresh_linear=False,fresh_adam=False)
        environment['native_proof'](proof,copied=False)
        with self.assertRaises(AssertionError):environment['native_proof'](proof,copied=True)
        copied=dict(proof,official_block_copy_verified=True,head_initialized=True,fresh_linear=True,fresh_adam=True)
        environment['native_proof'](copied,copied=True)
        copied['native_class']='manual.DecoderLayer'
        with self.assertRaises(AssertionError):environment['native_proof'](copied,copied=True)


if __name__=='__main__':unittest.main()
