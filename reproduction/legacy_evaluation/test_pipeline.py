"""CPU fixtures for sharing/asset identity. No native job or GPU is started."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from reproduction.legacy_evaluation import pipeline
from reproduction.legacy_evaluation.contract import protocol


class AssetIdentityTests(unittest.TestCase):
    def test_changed_index_or_encoder_is_rejected_before_gpu_load(self):
        original=protocol();root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);legacy=base/'legacy';source=legacy/'search_r1/search/retrieval_server.py'
            source.parent.mkdir(parents=True)
            original_root=os.environ.get('SEARCH_G1_LEGACY_SOURCE_ROOT')
            raw=(Path(original_root)/'search_r1/search/retrieval_server.py').read_bytes() if original_root else subprocess.check_output(['git','show','2de313e:search_r1/search/retrieval_server.py'],cwd=root)
            source.write_bytes(raw)
            index=base/'index';corpus=base/'corpus';encoder=base/'encoder';encoder.mkdir()
            index.write_bytes(b'fixture-index');corpus.write_bytes(b'fixture-corpus');(encoder/'model').write_bytes(b'fixture-encoder')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            spec=copy.deepcopy(original);spec['protocol_authoritative_source']['base']=str(base)
            spec['retrieval']['index_sha256']=sha(index);spec['retrieval']['corpus_sha256']=sha(corpus)
            receipt=dict(accepted=True,run_id=original['protocol_authoritative_source']['historical_run'],index_sha256=sha(index),corpus_sha256=sha(corpus),encoder_assets=dict(model_id=spec['retrieval']['encoder'],revision=spec['retrieval']['revision'],files={'model':dict(sha256=sha(encoder/'model'),size=15)}))
            reference=base/'logs/retrieval-admission.json';reference.parent.mkdir();reference.write_text(json.dumps(receipt))
            manifest=dict(legacy_source_root=str(legacy),retrieval_server_source=str(source),retrieval_reference_admission=str(reference),retrieval_reference_admission_sha256=sha(reference),retrieval_index_path=str(index),retrieval_corpus_path=str(corpus),retrieval_encoder=str(encoder),run_id='fixture',execution_id='fixture')
            with patch.object(pipeline,'protocol',return_value=spec):
                proof=pipeline.verify_retrieval_assets(manifest);self.assertEqual(len(proof['files']),3)
                index.write_bytes(b'changed-index')
                with self.assertRaises(AssertionError):pipeline.verify_retrieval_assets(manifest)
                index.write_bytes(b'fixture-index');(encoder/'model').write_bytes(b'changed-encoder')
                with self.assertRaises(AssertionError):pipeline.verify_retrieval_assets(manifest)


class SharedAdmissionTests(unittest.TestCase):
    def test_shared_budget_no_second_gres_and_forbidden_training_refusal(self):
        uuids=['fixture-gpu-'+str(i) for i in range(8)]
        own=dict(NodeList='h100-8',JobId='fixture-job',JobState='RUNNING',NumNodes='1',NumCPUs='96',TRES='fixture',OverSubscribe='OK')
        foreign=dict(NodeList='h100-8',JobId='video',JobState='RUNNING',UserId='fixture-owner')
        def output(args,**kwargs):
            if args[0]=='squeue':return 'fixture-job\nvideo\n'
            return 'foreign' if args[-1]=='video' else 'own'
        with tempfile.TemporaryDirectory() as d:
            manifest=dict(allocation_mode='shared',preserved_foreign_allocations=[dict(job_id='video',user_id='fixture-owner')],forbidden_training_job_ids=['3400'],physical_gpu_uuids=uuids,preserved_foreign_gpu_processes=[],role='fixture',run_id='fixture',execution_id='fixture',checkpoint_step=5,split='validation',recipe_sha256='fixture')
            with patch.dict(os.environ,SLURM_JOB_ID='fixture-job',CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(8))),patch.object(pipeline.socket,'gethostname',return_value='h100-8'),patch.object(pipeline.subprocess,'check_output',side_effect=output),patch.object(pipeline,'fields',side_effect=lambda text:foreign if text=='foreign' else own),patch.object(pipeline,'memory_bytes',return_value=512*2**30),patch.object(pipeline,'cuda_uuids',return_value=uuids),patch.object(pipeline,'rows',side_effect=lambda query,kind='gpu':[[str(i),u] for i,u in enumerate(uuids)] if query=='index,uuid' else []),patch.object(pipeline,'publish'):
                result=pipeline.native_admission(Path(d),manifest)
                self.assertEqual(result['allocation_mode'],'shared');self.assertFalse(result['gres_secondary_reservation_required'])
                bad=dict(manifest,forbidden_training_job_ids=['video'])
                with self.assertRaises(AssertionError):pipeline.native_admission(Path(d),bad)
                bad=dict(manifest,preserved_foreign_allocations=[])
                with self.assertRaises(AssertionError):pipeline.native_admission(Path(d),bad)

class OwnedExitTests(unittest.TestCase):
    def test_exit0_with_known_children_is_cleanup_pending_not_failure(self):
        class Owned:
            jobs={'evaluator':{'state':'RUNNING'}}
            def observe(self,name):return [dict(pid=123,start_ticks=456)]
            def save(self):pass
        class Process:
            def poll(self):return 0
        instance=object.__new__(pipeline.FinalPipeline);instance.owned=Owned()
        instance.wait('evaluator',Process())
        self.assertEqual(instance.owned.jobs['evaluator']['state'],'EXIT0_CLEANUP_PENDING')
        self.assertEqual(instance.owned.jobs['evaluator']['exit_code'],0)

if __name__=='__main__':unittest.main()
