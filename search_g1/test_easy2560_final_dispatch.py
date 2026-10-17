"""CPU wiring tests; these do not assert that any CUDA recovery has occurred."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import full_state_restore as sdk


class FinalDispatchTests(unittest.TestCase):
    def test_peer_state_read_failure_is_propagated_before_model_collectives(self):
        def gather(rows,local):rows[:]=[local,'OSError: peer rank state unreadable',None,None]
        dist=SimpleNamespace(get_world_size=lambda:4,all_gather_object=gather)
        torch=SimpleNamespace(distributed=dist)
        with patch.dict('sys.modules',{'torch':torch,'torch.distributed':dist}):
            with self.assertRaisesRegex(AssertionError,'peer rank state unreadable'):
                sdk.final_collective_inputs(lambda:('proof','saved'))

    def test_final_receipt_uses_dedicated_source_before_legacy_eight_rank_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'receipt.json';p.write_text(json.dumps({'status':sdk.FINAL_EASY_STATUS}))
            expected=(p,{'step':200},Path(directory),None,None)
            stub=SimpleNamespace(source=lambda path: expected)
            with patch.dict('sys.modules',{'search_g1.easy2560_final_checkpoint':stub}):
                self.assertEqual(sdk.source(p),expected)

    def test_final_receipt_cannot_be_bound_as_a_new_training_continuation(self):
        proof=(Path('/fixture'),{'status':sdk.FINAL_EASY_STATUS},None,None,None)
        with patch.object(sdk,'source',return_value=proof):
            with self.assertRaisesRegex(ValueError,'NOT_A_TRAINING_BIND'):sdk.bind({},'/fixture')
            cfg=SimpleNamespace(trainer={'full_state_restore_receipt':'/fixture'})
            with self.assertRaisesRegex(ValueError,'NOT_A_TRAINING_BIND'):sdk.from_config(cfg)

    def test_final200_uses_empty_optimizer_constructor_without_changing_saved_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'logs').mkdir();cp=root/'cp200';cp.mkdir()
            p=root/'receipt.json';p.write_text('{}')
            binding={'kind':sdk.FINAL_EASY_KIND,'source_receipt':str(p),
                     'source_receipt_sha256':sdk.digest(p),'completed_outer':200}
            (root/'logs/run-manifest.json').write_text(json.dumps({'full_state_restore':binding}))
            row={'status':sdk.FINAL_EASY_STATUS,'step':200,'restore_only':True}
            config={'completed_updates':200,'lr':1e-6,'lr_warmup_steps':57}
            with patch.dict(os.environ,SEARCH_G1_BASE=str(root)),patch.object(sdk,'source',return_value=(p,row,cp,None,None)):
                actual=sdk.optimizer_construction_config(config,cp)
                self.assertEqual(actual['completed_updates'],0)
                self.assertEqual(config['completed_updates'],200)
                row['restore_only']=False
                with self.assertRaises(AssertionError):sdk.optimizer_construction_config(config,cp)

    def test_driver_dispatch_uses_actual_saved_file_and_does_not_claim_cuda_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir();(root/'artifacts/live').mkdir(parents=True)
            p=root/'receipt.json';p.write_text('{}');saved=root/'driver.pt';saved.write_text('{"fixture": true}')
            record={'path':str(saved),'full_read_sha256':sdk.digest(saved)}
            receipt={'status':sdk.FINAL_EASY_STATUS,'step':200,'restore_only':True,'run_id':'r',
                     'driver_file':record,'source_job':'j','source_commit':'a'*40,
                     'sampling_rng_stage':'saved_checkpoint_before_final51713'}
            p.write_text(json.dumps(receipt))
            run={'run_id':'r','full_state_restore':{'kind':sdk.FINAL_EASY_KIND,'source_receipt_sha256':sdk.digest(p)}}
            (root/'logs/run-manifest.json').write_text(json.dumps(run))
            (root/'artifacts/live/manifest.json').write_text('{"round": 150}')
            calls=[]
            def restore(trainer,state,target,**kwargs):
                calls.append((state,target,kwargs))
                return {'completed_outer':200,'reward_refresh_pending':False,
                        'actual_driver_rng_readback_equal':True,'actual_candidate_stream_readback_equal':True}
            modules={'torch':SimpleNamespace(load=lambda path,**kw:json.loads(Path(path).read_text())),
                     'search_g1.training_state':SimpleNamespace(restore_easy2560_driver=restore)}
            with patch.dict(os.environ,SEARCH_G1_BASE=str(root)),patch.dict('sys.modules',modules), \
                 patch.object(sdk,'source',return_value=(p,receipt,root/'cp200',root,run)):
                result=sdk.restore_driver(SimpleNamespace(),p)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0][0],{'fixture':True})
            self.assertTrue(result['actual_driver_dataequal']);self.assertFalse(result['actual_cuda_restore_accepted'])
            self.assertFalse(result['post_callback_rng_captured'])
            self.assertEqual(result['source_step'],200)


if __name__=='__main__':unittest.main()
