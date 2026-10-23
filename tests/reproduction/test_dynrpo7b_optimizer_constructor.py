import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from search_g1.full_state_restore import optimizer_construction_config


class DynRPOOptimizerConstructor(unittest.TestCase):
    def test_native_scheduler_continues_real_saved_adam_and_attempt_history(self):
        import torch
        from reproduction.common8.scheduler import OuterSchedulerEvidence, fixed_outer_warmup_scheduler, expected_lr
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'logs').mkdir()
            run=dict(profile='common8',training_world_size=4,budget={'world_size':4},
                     full_state_restore=dict(kind='dynrpo7b_native4',completed_outer=3,
                                             actual_optimizer_writes_per_rank=60))
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=1e-6)
            scheduler=fixed_outer_warmup_scheduler(opt,warmup_steps=57)
            for outer in range(3):
                for _ in range(20):p.grad=torch.ones_like(p);opt.step()
                scheduler.step()
            prefix=dict(completed_outer=3,writes=60,attempt_index=3,used_attempt_ids=['a'*32])
            identity=dict(policy_safety_mode='inline_first_five',slurm_job_id='123')
            with patch('reproduction.common8.scheduler._identity',return_value=identity), \
                 patch('reproduction.common8.scheduler._start_ticks',return_value=123), \
                 patch('search_g1.full_state_restore.verified_runtime_prefix',return_value=prefix):
                evidence=OuterSchedulerEvidence(base,opt,scheduler,rank=0,world_size=4,job_id='123')
                try:
                    self.assertEqual(evidence.used_attempt_ids,{'a'*32})
                    evidence.begin_outer(4,optimizer_writes=60,batch_gate_sha256='b'*64,
                                         attempt_id='c'*32,attempt_index=4)
                    for _ in range(20):p.grad=torch.ones_like(p);opt.step()
                    evidence.complete_outer(optimizer_writes=80,opportunities_processed=20)
                    self.assertEqual(int(opt.state[p]['step']),80)
                    self.assertEqual(scheduler.last_epoch,4)
                    self.assertEqual(opt.param_groups[0]['lr'],expected_lr(4))
                finally:evidence.close()

    def test_actual_constructor_accepts_native_outer_and_preserves_resolved_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'logs').mkdir();receipt=base/'boundary.json';receipt.write_text('{}')
            cp=(base/'actor').resolve()
            for outer in (3,10,50,190,200):
                with self.subTest(outer=outer):
                    binding=dict(kind='dynrpo7b_native4',source_receipt=str(receipt),source_receipt_sha256='a'*64,completed_outer=outer)
                    (base/'logs/run-manifest.json').write_text(json.dumps(dict(full_state_restore=binding)))
                    config=dict(completed_updates=outer,lr=1e-6,lr_warmup_steps=57)
                    with patch.dict(os.environ,SEARCH_G1_BASE=str(base)),patch('search_g1.full_state_restore.source',return_value=(receipt,dict(step=outer),cp,None,None)),patch('search_g1.full_state_restore.digest',return_value='a'*64):
                        construction=optimizer_construction_config(config,str(cp))
                        self.assertEqual(config['completed_updates'],outer)
                        self.assertEqual(construction,dict(config,completed_updates=0))
                        with self.assertRaises(AssertionError):
                            optimizer_construction_config(dict(config,completed_updates=outer-1),str(cp))
                        with self.assertRaises(AssertionError):
                            optimizer_construction_config(config,str(base/'another-actor'))


if __name__=='__main__':unittest.main()
