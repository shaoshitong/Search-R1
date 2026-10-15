"""Focused CPU fixtures for completed-head recovery and exact pool boundaries."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from .pool import next_selection
from .recovery import stage, sha, binding_sha, validate_pool_boundary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def pool_fixture(completed=7):
    pool = dict(records=[dict(qid='q%d'%i, question_hash='h%d'%i) for i in range(12)],
                canonical_train_sha256='train',fixed_test_qids_sha256='testids',
                fixed_test_normalized_text_sha256='testtext')
    state = dict(schema_version=1, seed=1729, cycle=0, order=list(range(12)),cursor=5,
                 next_epoch=2, consumed_questions=5,epoch_questions=5)
    for epoch in range(2,completed+1):
        _,state=next_selection(pool,state,epoch)
    selection, after = next_selection(pool,state,completed+1)
    selection['pool_sha256']='poolsha'
    return pool,state,selection,after


class PoolBoundaryTests(unittest.TestCase):
    def test_live_pending_and_selection_before_state_are_both_exact(self):
        pool,head,selected,after=pool_fixture()
        self.assertTrue(validate_pool_boundary(pool,head,after,selected,7,'poolsha')['pending_selection_reused'])
        self.assertEqual(validate_pool_boundary(pool,head,head,selected,7,'poolsha')['pool_phase'],
                         'reserved_selection_published_before_pool_state')
        self.assertFalse(validate_pool_boundary(pool,head,head,None,7,'poolsha')['pending_selection_reused'])
        self.assertEqual(len({row['qid'] for row in selected['rows']}),5)

    def test_same_epoch_with_wrong_permutation_or_question_order_is_rejected(self):
        pool,head,selected,after=pool_fixture()
        bad=copy.deepcopy(after);bad['order'][0],bad['order'][1]=bad['order'][1],bad['order'][0]
        with self.assertRaises(ValueError):validate_pool_boundary(pool,head,bad,selected,7,'poolsha')
        bad=copy.deepcopy(selected);bad['rows'].reverse()
        with self.assertRaises(ValueError):validate_pool_boundary(pool,head,after,bad,7,'poolsha')


class LatestRecoveryTests(unittest.TestCase):
    def fixture(self,root,completed=7):
        source,base=root/'source',root/'target'
        for path in (source/'preparation',source/'logs',source/'train/epochs'/('epoch-%04d'%completed),
                     source/'question-pool',source/'epochs'/('epoch-%04d'%completed),base/'preparation'):
            path.mkdir(parents=True,exist_ok=True)
        binding=dict(learning_rate=.0003, global_batch=256, world_size=8, seed=1729,
                     loss='single_mean_BCEWithLogits',epochs=1000,save_every=10)
        checkpoint=source/'train/latest.pt';checkpoint.write_bytes(b'complete-head-adam-eight-RNG')
        receipt=dict(epoch=completed,sha256=sha(checkpoint),binding_sha256=binding_sha(binding),
                     actual_adam_steps_per_rank=[completed*100]*8)
        write(source/'train/latest.json',receipt)
        write(source/'train/run-claim.json',dict(binding=binding,binding_sha256=binding_sha(binding)))
        write(source/'train/epochs'/('epoch-%04d'%completed)/'complete.json',receipt)
        # Uncommitted temporary files and partial next-epoch writes stay original.
        (source/'train/latest.pt.partial').write_bytes(b'uncommitted-head')
        (source/'train/latest.json.partial').write_bytes(b'uncommitted-receipt')
        for rank in range(8):
            path=source/'train/epochs'/('epoch-%04d'%(completed+1))/('rank-%d-updates.jsonl'%rank)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('\n'.join(json.dumps(dict(epoch=completed+1,actual_adam_steps=completed*100+i+1)) for i in range(rank%3)))
        pool,head,selected,after=pool_fixture(completed)
        write(source/'question-pool/pool.json',pool);poolsha=sha(source/'question-pool/pool.json')
        selected['pool_sha256']=poolsha
        write(source/'question-pool/state.json',after)
        write(source/'epochs'/('epoch-%04d'%completed)/'pool-state-at-head-completion.json',head)
        write(source/'epochs'/('epoch-%04d'%(completed+1))/'selection.json',selected)
        write(source/'epochs'/('epoch-%04d'%(completed+1))/'questions.json',dict(epoch=completed+1,questions=['gold-and-question']))
        write(source/'logs/cleanup-status.json',dict(cleaned=True,owned_births_all_gone=False))
        write(source/'logs/process-ledger.json',dict(jobs={}))
        write(source/'preparation/submit-receipt.json',dict(job_id='3408'))
        config=dict(base=str(base),run_id='a'*32,source_commit='2'*40,model_revision='r',model_files_sha256={'config':'hash'},
                    canonical_train_sha256='train',cached_features_manifest_sha256='testcache',seed=1729,global_batch=256,
                    learning_rate=.00003,epochs=1000,save_every=10)
        write(source/'preparation/launch-config.json',dict(config,base=str(source),source_commit='1'*40,learning_rate=.0003))
        proof=source/'preparation/cpu-proof.json'
        write(proof,dict(full_sha256=receipt['sha256'],epoch=completed,actual_adam_steps_per_rank=[completed*100]*8,
                         rank_rng=list(range(8)),cpu_load_performed=True))
        config['restore']=dict(source_base=str(source),source_job_id='3408',completed_epoch=completed,
            checkpoint_sha256=receipt['sha256'],checkpoint_cpu_proof_path=str(proof),checkpoint_cpu_proof_sha256=sha(proof),
            allow_learning_rate_change=True,source_learning_rate=.0003,target_learning_rate=.00003,
            discard_uncommitted_head_updates=True)
        return source,base,config,receipt

    def test_arbitrary_latest_full_state_lr_only_and_partial_discard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,base,config,receipt=self.fixture(Path(tmp),completed=7)
            result=stage(config,terminal=dict(JobId='3408',JobState='CANCELLED',ExitCode='0:9'))
            self.assertEqual(config['start_epoch'],8)
            self.assertEqual(result['actual_adam_steps_per_rank'],[700]*8)
            self.assertEqual(result['discarded_uncommitted_head']['observed_updates_per_rank'],[0,1,2,0,1,2,0,1])
            self.assertFalse((base/'train/latest.pt.partial').exists())
            self.assertFalse((base/'train/latest.json.partial').exists())
            self.assertTrue((source/'train/latest.pt.partial').exists())
            transition=json.loads((base/'train/learning-rate-transition.json').read_text())
            self.assertEqual(transition['source_learning_rate'],3e-4)
            self.assertEqual(transition['target_learning_rate'],3e-5)
            self.assertEqual(transition['source_complete_epoch'],7)
            claim=json.loads((base/'train/run-claim.json').read_text())
            self.assertEqual(claim['binding_sha256'],transition['target_binding_sha256'])
            self.assertEqual(claim['binding']['learning_rate'],3e-5)
            self.assertEqual(json.loads((source/'train/run-claim.json').read_text())['binding']['learning_rate'],3e-4)
            self.assertEqual(sha(base/'train/latest.pt'),receipt['sha256'])
            self.assertEqual((base/'question-pool/state.json').read_bytes(),(source/'question-pool/state.json').read_bytes())
            self.assertEqual((base/'epochs/epoch-0008/selection.json').read_bytes(),(source/'epochs/epoch-0008/selection.json').read_bytes())
            target=base/'train/new-latest.pt';target.write_bytes(b'new head after full next epoch');os.replace(target,base/'train/latest.pt')
            self.assertEqual(sha(source/'train/latest.pt'),receipt['sha256'])

    def test_latest_epoch_is_not_hardcoded_to_early_checkpoint(self):
        for completed in (23,137):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as tmp:
                _,_,config,_=self.fixture(Path(tmp),completed=completed)
                result=stage(config,terminal=dict(JobId='3408',JobState='CANCELLED'))
                self.assertEqual(result['start_epoch'],completed+1)
                self.assertEqual(result['actual_adam_steps_per_rank'],[completed*100]*8)

    def test_unauthorized_lr_or_discard_is_rejected_before_restore_intent(self):
        for flag in ('allow_learning_rate_change','discard_uncommitted_head_updates'):
            with tempfile.TemporaryDirectory() as tmp:
                _,base,config,_=self.fixture(Path(tmp),completed=23)
                config['restore'][flag]=False
                with self.assertRaises(ValueError):stage(config,terminal=dict(JobId='3408',JobState='CANCELLED'))
                self.assertFalse((base/'preparation/restore-intent.json').exists())

    def test_source_still_running_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,base,config,_=self.fixture(Path(tmp),completed=31)
            with self.assertRaises(ValueError):stage(config,terminal=dict(JobId='3408',JobState='RUNNING'))
            self.assertFalse((base/'preparation/restore-intent.json').exists())


if __name__=='__main__':unittest.main()
