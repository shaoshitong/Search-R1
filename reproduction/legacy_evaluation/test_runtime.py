"""CPU fixtures only: no GPU acceptance, remote action, or real question data."""
import copy
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from reproduction.legacy_evaluation.contract import PROTOCOL_PATH,digest,protocol
from reproduction.legacy_evaluation.journal import make_binding,append_batch,load_prefix,aggregate_final
from reproduction.legacy_evaluation.runner import reuse_complete_result,finalize_result


class LegacyJournalTests(unittest.TestCase):
    def setUp(self):
        self.ids=['fixture-question-'+str(i) for i in range(512)]
        self.sources=['nq']*256+['hotpotqa']*256
        self.binding=make_binding('fixture-run','1'*64,protocol()['datasets']['validation']['sha256'],self.ids,self.sources,5,'validation')
        self.rows=[dict(question_id=q,data_source=s,em=0,f1=0.,actual_calls=0,valid_action=1,turns=1,
            answer_present=False,answer_sha256=None,response_token_count=2,model_token_count=2) for q,s in zip(self.ids,self.sources)]

    def test_checkpoint_and_recipe_change_cannot_reuse_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            append_batch(d,self.binding,self.ids,self.sources,self.rows[:64])
            other=make_binding('fixture-run','2'*64,self.binding['validation_sha256'],self.ids,self.sources,50,'validation')
            with self.assertRaises(ValueError):load_prefix(d,other,self.ids,self.sources)
            bad=dict(self.binding,recipe_sha256='0'*64)
            with self.assertRaises(ValueError):load_prefix(d,bad,self.ids,self.sources)
            with self.assertRaises(ValueError):append_batch(d,self.binding,self.ids,self.sources,self.rows[:64])

    def test_complete_reentry_reuses_original_proof_without_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            for start in range(0,512,64):append_batch(d,self.binding,self.ids,self.sources,self.rows[start:start+64])
            rows=load_prefix(d,self.binding,self.ids,self.sources)
            result=aggregate_final(self.binding,self.ids,self.sources,rows)
            cp=dict(checkpoint='/remote/fixture/checkpoint5',checkpoint_manifest_sha256='1'*64,generation_config_sha256='2'*64)
            workers=[dict(rank=i,world_size=8,cuda_uuid='fixture'+str(i),hf_parameter_dtypes=['torch.float32'],
                vllm_parameter_dtypes=['torch.float32'],hf_parameter_numel=1,optimizer_constructed=False,
                optimizer_steps=0,recipe_sha256=digest(PROTOCOL_PATH),actual_vllm_version='0.6.3',actual_vllm_config_dtype='torch.float32',actual_engine_max_model_len=4596,**cp) for i in range(8)]
            proof=Path(d)/'loaded-workers-fixture.json';proof.write_text(json.dumps(workers))
            result.update(actual_gpu_execution_verified=True,canonical_legacy_evaluation=True,loaded_workers_sha256=digest(proof),control_source_commit='fixture')
            target=Path(d)/'legacy-result.json';finalize_result(target,result);before=target.read_bytes()
            binding=dict(split='validation',checkpoint_step=5,**cp)
            same=reuse_complete_result(d,self.binding,self.ids,self.sources,rows,binding)
            finalize_result(target,same);self.assertEqual(before,target.read_bytes())
            workers[0]['hf_parameter_dtypes']=['torch.bfloat16'];proof.write_text(json.dumps(workers))
            with self.assertRaises(ValueError):reuse_complete_result(d,self.binding,self.ids,self.sources,rows,binding)

    def test_final_requires200_and_exact_51713(self):
        ids=['f'+str(i) for i in range(51713)];sources=['fixture-source'+str(i%7) for i in range(51713)]
        self.assertEqual(make_binding('f','1'*64,protocol()['datasets']['final']['sha256'],ids,sources,200,'final')['question_count'],51713)
        with self.assertRaises(ValueError):make_binding('f','1'*64,protocol()['datasets']['final']['sha256'],ids,sources,150,'final')
        with self.assertRaises(ValueError):make_binding('f','1'*64,protocol()['datasets']['final']['sha256'],ids[:-1],sources[:-1],200,'final')


class LegacyPackageTests(unittest.TestCase):
    def test_regular_old_reproduction_package_is_explicitly_extended(self):
        control=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)
            for name in protocol()['code_sha256']:
                path=old/name;path.parent.mkdir(parents=True,exist_ok=True)
                source=os.environ.get('SEARCH_G1_LEGACY_SOURCE_ROOT')
                raw=(Path(source)/name).read_bytes() if source else subprocess.check_output(['git','show',protocol()['protocol_authoritative_source']['source_commit']+':'+name],cwd=control)
                path.write_bytes(raw)
            (old/'reproduction/__init__.py').write_text('old_regular_package=True\n')
            script="""import sys
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import reproduction
assert reproduction.old_regular_package
from legacy_eval_control import activate
activate(sys.argv[1])
import reproduction.legacy_evaluation.contract as c
from pathlib import Path
assert Path(c.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
assert reproduction.__path__[0]==str(Path(sys.argv[1]).resolve()/'reproduction')
"""
            subprocess.run([sys.executable,'-c',script,str(old),str(control)],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
