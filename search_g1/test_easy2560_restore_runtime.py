import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import easy2560_restore_runtime as runtime


class RestoreReadbackTests(unittest.TestCase):
    def fixture(self, base):
        (base/'logs').mkdir()
        source = base/'source-receipt.json'; source.write_text('{}')
        cp = base/'original-cp5'
        progress = dict(completed_outer=5, cumulative_writes=100, attempt_index=5,
                        aborted_attempts=0, full_budget_outer=5)
        receipt = dict(run_id='r', source_job='730226', step=5, attempt_progress=progress,
                       hf_saved_tensor_count=435, driver_state=dict(source_consumed=12800),
                       ranks=[dict(used_attempt_ids=['saved-1','saved-2','saved-3','saved-4','saved-5'])])
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        run = dict(run_id='r', full_state_restore=dict(kind='easy2560_native4',
                   source_receipt=str(source), source_receipt_sha256=sha))
        rows = [dict(rank=r, actual_cuda_uuid='gpu'+str(r), process_identity=dict(pid=10+r,start_ticks=123),
                     slurm_job_id='new', source_run_id='r', source_checkpoint=str(cp), source_step=5,
                     actual_adam_scheduler_readback_equal=True, actual_rng_readback_equal=True,
                     actual_optimizer_writes=100,scheduler_last_epoch=5,last_attempt_index=5,
                     model_hf_exact_tensors_checked_on_rank0=435 if r==0 else 0,policy_rejections=0)
                for r in range(4)]
        driver = dict(actual_driver_rng_readback_equal=True,actual_candidate_stream_readback_equal=True,
                      completed_outer=5,attempt_progress=progress,actual_candidate_source_cursor=12800)
        allocation=dict(slurm_job_id='new',actual_cuda_runtime_uuids=['gpu'+str(r) for r in range(4)])
        def write():
            for name,value in (('full-state-restored-workers.json',dict(ranks=rows)),
                               ('full-state-restored-driver.json',driver),
                               ('native-allocation-admission.json',allocation)):
                (base/'logs'/name).write_text(json.dumps(value))
        write()
        return run,rows,driver,write,(source,receipt,cp,base/'old',dict(source_commit='oldsource'))

    def test_four_actual_readbacks_authenticate_saved_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); run,rows,driver,write,source=self.fixture(base)
            with patch('search_g1.easy2560_restore.source',return_value=source):
                value=runtime.verified_runtime_prefix(base,run)
            self.assertEqual((value['completed_outer'],value['writes'],value['attempt_index']),(5,100,5))
            self.assertTrue(value['actual_model_optimizer_scheduler_rng_data_equal'])
            self.assertEqual(value['current_world_size'],4)

    def test_live_process_or_one_equal_field_does_not_substitute_for_restore(self):
        for mutation in ('missing-rank','adam','rng','data','same-process','wrong-cursor'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as td:
                base=Path(td); run,rows,driver,write,source=self.fixture(base)
                if mutation=='missing-rank':rows.pop()
                if mutation=='adam':rows[2]['actual_adam_scheduler_readback_equal']=False
                if mutation=='rng':rows[3]['actual_rng_readback_equal']=False
                if mutation=='data':driver['actual_candidate_stream_readback_equal']=False
                if mutation=='same-process':rows[1]['process_identity']=rows[0]['process_identity']
                if mutation=='wrong-cursor':driver['actual_candidate_source_cursor']=10240
                write()
                with patch('search_g1.easy2560_restore.source',return_value=source),self.assertRaises(AssertionError):
                    runtime.verified_runtime_prefix(base,run)

    def test_old_qa5_stays_historical_and_does_not_invent_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); run,rows,driver,write,source=self.fixture(base)
            path,receipt,cp,old,parent=source
            after=base/'old-qa5.json';after.write_text(json.dumps(dict(run_id='r',step=5,correct=142)))
            receipt['historical_artifacts']=dict(development=dict(after5=dict(path=str(after),
                full_read_sha256=hashlib.sha256(after.read_bytes()).hexdigest())))
            with patch('search_g1.easy2560_restore.source',return_value=source):
                result=runtime.record_development_history(base,run)
            self.assertFalse(result['before0_available']);self.assertFalse(result['current_evaluation_performed'])
            self.assertFalse(result['post_evaluation_rng_available'])
            self.assertEqual(result['original_after5']['correct'],142)
            self.assertFalse((base/'logs/formal-inline-development-before.json').exists())


class ResumedRawLedgerTests(unittest.TestCase):
    def test_suffix_numbers_keep_original_outers_and_authenticate_receipt(self):
        from reproduction.easy2560_joint4.retire_candidate_raw import completed_updates
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();(base/'logs').mkdir()
            receipt=base/'receipt.json';receipt.write_text('{}')
            run=dict(full_state_restore=dict(kind='easy2560_native4',completed_outer=5,
                source_receipt=str(receipt),source_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()))
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            ledger=base/'logs/actual-updates.jsonl'
            ledger.write_text(''.join(json.dumps(dict(step=n,attempt_id=str(n)))+'\n' for n in (6,7,8)))
            self.assertEqual([r['step'] for r in completed_updates(base)[0]],[6,7,8])
            receipt.write_text('changed')
            with self.assertRaisesRegex(ValueError,'RESTORED_PREFIX_RECEIPT_CHANGED'):completed_updates(base)


if __name__=='__main__':unittest.main()
