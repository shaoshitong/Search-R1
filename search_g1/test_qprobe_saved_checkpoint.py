"""CPU provenance/real head-stream readbacks; no native model/GPU claim.

The admission fixture has tiny synthetic actor files and explicit synthetic
failure metadata. Runtime tests genuinely fit/serialize a small CPU head.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from search_g1 import qprobe_saved_checkpoint as saved
from search_g1.qprobe_checkpoint import _file_record,digest
from search_g1.test_qprobe_checkpoint import PairFixture
from search_g1.test_qprobe_runtime import runtime,stage_pending
from search_g1.training_state import driver_rng
from reproduction.hpc3_qprobe.formal_audit_failure718650 import TERMINAL


def write(path,value):Path(path).write_text(json.dumps(value))


class SavedFixture:
    def __init__(self,directory):
        self.fixture=PairFixture(Path(directory).resolve(),outer=5);f=self.fixture;self.base=f.base;self.cp=f.cp
        self.directory=f.base/'failed-job-718650';self.directory.mkdir()
        for i in range(9):(f.cp/('synthetic-extra-%d'%i)).write_bytes(b'tiny fixture; no native model claim')
        complete=json.loads((f.cp/'complete.json').read_text())
        complete.update(validated_tensor_count=435,actual_optimizer_writes_per_rank=100,
                        question_probe=dict(feature_spec_sha='f'*64,head_version=f.trainer.qprobe_runtime.version))
        complete['files_sha256']={str(p.relative_to(f.cp)):digest(p) for p in f.cp.rglob('*') if p.is_file() and p.name!='complete.json'}
        write(f.cp/'complete.json',complete)
        live=json.loads((f.base/'artifacts/live/manifest.json').read_text())
        f.run['initial_model_sha256']=live['encoder_sha256'];write(f.base/'logs/run-manifest.json',f.run)
        ranks=[dict(rank=i,actual_optimizer_writes=100,scheduler_last_epoch=5,scheduler_step_count=6,
            completed_outer=5,last_attempt_index=5,used_attempt_ids=['fixture-%d'%j for j in range(5)],
            policy_rejections=0,optimizer_tensors_finite=True) for i in range(4)]
        self.cpu=dict(status='ACTUAL_SAVED_NATIVE4_CHECKPOINT5_AND_CORRECTED_STRUCTURAL_PREFIX_CPU_PASS',
            run_id=f.run_id,source_commit=f.run['source_commit'],ranks=ranks,head_version=f.trainer.qprobe_runtime.version,
            head_replay_groups=7680,candidate_source_cursor=25600,
            actual_checkpoint_rng_boundary='pre-dev512-after5',head_private_rng_cpu_readback_equal=True,
            head_adam_model_replay_cpu_readback_equal=True,selector_rng_cpu_readback_equal=True,
            driver_candidate_saved_state_equal=True,corrected_first_five_structural_reader_accepted=True,
            actual_constructed_stream_restored=False,actual_cuda_restore_accepted=False,
            completed_pair_available=False,post_evaluation_rng_boundary_available=False)
        files=[dict(_file_record(p),relative_path=str(p.relative_to(f.base)))
               for p in sorted(f.cp.rglob('*')) if p.is_file()]
        self.closed=dict(status='ACTUAL_FAILED_NATIVE4_CHECKPOINT5_AND_RAW_EVIDENCE_FULL_SHA_PRESERVED',
            cpu_state_validation_pending=False,job='718650',run_id=f.run_id,source_commit=f.run['source_commit'],
            actual_accounting='|'.join(TERMINAL)+'\n',base=str(f.base),checkpoint=str(f.cp),
            actor_head_completed_outers=5,checkpoint_file_count=21,owned_components_stopped=True,
            private_tmpfs_cleaned=True,first_five_acceptance_complete=False,completed_pair_available=False,
            actual_cuda_restore_accepted=False,post_evaluation_rng_boundary_available=False,
            checkpoint_rng_boundary='pre-dev512-after5',files=files,saved_prefix_cpu_validation=self.cpu)
        self.corrected=dict(original_source_commit=f.run['source_commit'],original_raw_evidence_modified=False,
            actual_cuda_restore_accepted=False,safety=dict(structural_accepted=True,actual_complete_outers=5,
                actual_optimizer_writes_per_rank=100,unique_global_monitor_count=200))
        write(self.directory/'terminal-and-files.json',self.closed)
        write(self.directory/'saved-prefix-cpu-validation.json',self.cpu)
        write(self.directory/'corrected-reader-prefix.json',self.corrected)
        for name in ('formal-inline-development-before.json','formal-inline-development-after.json'):
            write(f.base/'logs'/name,dict(fixture_only=True))
        self.path=self.directory/'saved-checkpoint-source.json'
        with patch('reproduction.hpc3_qprobe.formal_audit_failure718650.BASE',f.base),\
             patch('reproduction.hpc3_qprobe.formal_audit_failure718650.CHECKPOINT',f.cp),\
             patch('reproduction.hpc3_qprobe.formal_audit_failure718650.RUN',f.run_id),\
             patch('reproduction.hpc3_qprobe.formal_audit_failure718650.SOURCE',f.run['source_commit']),\
             patch('reproduction.hpc3_qprobe.formal_audit_failure718650.metadata',return_value={}):
            saved.build_source(self.path)
        self.receipt=json.loads(self.path.read_text())

    def update(self,key,path,value):
        write(path,value);self.receipt[key]=_file_record(path);write(self.path,self.receipt)


class SavedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_distinct_source_type_dispatch_and_explicit_lost_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            f=SavedFixture(folder)
            from search_g1.full_state_restore import source
            _,r,cp,base,_=source(f.path)
            self.assertEqual(r['kind'],saved.SAVED_KIND);self.assertEqual(cp,f.cp);self.assertEqual(base,f.base)
            self.assertEqual(len(r['files']),21)
            self.assertFalse(r['completed_pair_available']);self.assertFalse(r['actual_cuda_restore_accepted'])
            self.assertFalse(r['bitwise_continuation_from_failed_post_evaluation'])

    def test_failure_cpu_and_structural_proofs_cannot_be_rebound_by_rehashing(self):
        for key,name,field,value in (
            ('failure_evidence','terminal-and-files.json','run_id','another-run'),
            ('saved_state_cpu_validation','saved-prefix-cpu-validation.json','head_replay_groups',7679),
            ('corrected_structural_prefix','corrected-reader-prefix.json','original_raw_evidence_modified',True)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                f=SavedFixture(folder);path=f.directory/name;body=json.loads(path.read_text());body[field]=value
                f.update(key,path,body)
                with self.assertRaisesRegex(ValueError,'PROOF_DIFFERS'):saved.read_source(f.path)

    def test_checkpoint_complete_and_receipt_cannot_disagree(self):
        with tempfile.TemporaryDirectory() as folder:
            f=SavedFixture(folder);f.receipt['head_path']=str(f.cp/'config.json');write(f.path,f.receipt)
            with self.assertRaisesRegex(ValueError,'SAVED_STATE_BINDING_DIFFERS'):saved.read_source(f.path)

    def test_lost_postdev_rng_cannot_be_marked_bitwise(self):
        with tempfile.TemporaryDirectory() as folder:
            f=SavedFixture(folder);f.receipt['bitwise_continuation_from_failed_post_evaluation']=True;write(f.path,f.receipt)
            with self.assertRaisesRegex(ValueError,'SAVED_STATE_BINDING_DIFFERS'):saved.read_source(f.path)

    def test_once_controller_rejects_another_same_schema_saved_source(self):
        from reproduction.hpc3_qprobe import recover_saved718650 as controller
        with tempfile.TemporaryDirectory() as folder:
            f=SavedFixture(folder)
            with patch.object(controller,'SOURCE_RECEIPT',f.path):
                with self.assertRaisesRegex(ValueError,'SOURCE_COUNTERS_OR_TYPE_DIFFER'):controller.existing_source()

    def actual_runtime(self,folder):
        value=runtime(folder)
        for step in range(1,6):
            attempt='saved-test-'+str(step);stage_pending(value,outer=step,attempt_id=attempt)
            value.complete_outer(step,'actual-small-cpu-head:'+str(step),dict(attempt_id=attempt))
        stream=dict(candidate_order=list(range(8)),candidate_cursor=2,epoch=1,
                    consumed_source_rows=25600,pending_source=['one','two'],pending_offset=1)
        state_path=Path(folder)/'state.pt';value.save(state_path,'actual-small-cpu-head:5',stream,'reward')
        driver_path=Path(folder)/'driver.pt';rng=driver_rng();torch.save(dict(rng=rng),driver_path)
        receipt=dict(kind=saved.SAVED_KIND,step=5,run_id='runtime-fixture',question_probe_state_path=str(state_path),
            head_path=str(state_path)+'.head.pt',head_version=value.version,driver_file={'path':str(driver_path)})
        trainer=SimpleNamespace(qprobe_runtime=value,mixed_stream=SimpleNamespace(state=lambda:copy.deepcopy(stream),consumed=25600))
        (Path(folder)/'logs').mkdir(exist_ok=True)
        return trainer,receipt,stream,rng

    def test_actual_head_adam_private_rng_selector_stream_and_driver_equal_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer,r,_,_=self.actual_runtime(folder)
            with patch.dict(os.environ,SEARCH_G1_BASE=folder),\
                 patch('search_g1.full_state_restore.source',return_value=(None,r,None,None,None)),\
                 patch('search_g1.full_state_restore.digest',return_value='fixture-sha'):
                proof=saved.record_runtime_restore(trainer,'fixture')
            self.assertEqual(proof['readback_passes'],2);self.assertTrue(proof['actual_postconstruction_driver_rng_equal'])

    def test_mutated_actual_stream_or_rng_fails_before_runtime_receipt(self):
        for mutation in ('stream','driver','selector','head'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                trainer,r,stream,rng=self.actual_runtime(folder)
                try:
                    if mutation=='stream':stream['pending_offset']=0
                    if mutation=='driver':torch.rand(1)
                    if mutation=='selector':trainer.qprobe_runtime.selection_rng.random()
                    if mutation=='head':trainer.qprobe_runtime.head.completed_outer=4
                    with patch.dict(os.environ,SEARCH_G1_BASE=folder),\
                         patch('search_g1.full_state_restore.source',return_value=(None,r,None,None,None)):
                        with self.assertRaises((AssertionError,ValueError)):
                            saved.record_runtime_restore(trainer,'fixture')
                    self.assertFalse((Path(folder)/'logs/full-state-restored-question-probe.json').exists())
                finally:
                    import random
                    torch.set_rng_state(rng['torch_cpu']);np.random.set_state(rng['numpy']);random.setstate(rng['python'])


if __name__=='__main__':unittest.main()
