"""CPU mechanics only: paired SHA/count/provenance and next-attempt restoration.

Synthetic HF byte files carry a checkpoint-complete fixture. This suite does not
exercise CUDA, vLLM, FSDP loading, or validate real model weights.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from search_g1.qprobe_checkpoint import (PAIR_STATUS,bind_restore,digest,install_restored_reward_probe,
    publish_completed_pair,read_source,restore_driver,validate_rank_state,boundary_worker_rng)
from search_g1.qprobe_model import OnlineQProbe
from search_g1.training_state import atomic_save,driver_rng


class PairFixture:
    def __init__(self,base,outer=50,size=4):
        self.base=Path(base);self.outer=outer;self.size=size;self.run_id='a'*32
        (self.base/'logs').mkdir(parents=True);(self.base/'artifacts/live').mkdir(parents=True)
        self.cp=self.base/'checkpoints/actor'/f'global_step_{outer}';self.state=self.cp/'training_state'
        self.state.mkdir(parents=True)
        self.run=dict(run_id=self.run_id,source_commit='b'*40,profile='common8',training_world_size=size,
            actor_micro_batch_per_rank=4 if size==4 else 2,completed_updates=0,
            budget={'world_size':size,'learning_rate':1e-6},input_sha256={'manifest.json':'c'*64},
            initial_model_sha256={'weights':'d'*64},question_probe=dict(candidate_questions=5120,
                selected_questions=512,exploration_questions=64,responses_per_question=5),
            retriever_version='immutable-index-version',retriever_manifest_sha256='9'*64,retriever_topk=3,
            initial_probe_version=f'{self.run_id}:0:old')
        (self.base/'logs/run-manifest.json').write_text(json.dumps(self.run))
        self.stream=dict(candidate_order=list(range(8)),candidate_cursor=4,epoch=2,
            consumed_source_rows=outer*5120,pending_source=['p0','p1'],pending_offset=1)
        self.attempt_index=outer+2;self.writes=outer*18
        self.ids=[f'{i:032x}' for i in range(1,self.attempt_index+1)]
        parameter=torch.nn.Parameter(torch.tensor([1.]));optimizer=torch.optim.AdamW([parameter],lr=1e-6)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda count:min(count/57,1))
        for _ in range(outer):
            for _ in range(18):parameter.grad=torch.ones(1);optimizer.step()
            scheduler.step()
        proofs=[]
        self.rank_states=[]
        for rank in range(size):
            state=dict(run_id=self.run_id,source_commit='b'*40,rank=rank,world_size=size,
                actual_optimizer_writes=self.writes,completed_outer=outer,last_attempt_index=self.attempt_index,
                used_attempt_ids=self.ids,policy_rejections=7,optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),rng=dict(driver_rng(),cuda_placeholder=torch.tensor([1])))
            self.rank_states.append(state)
            proof=atomic_save(self.state/f'rank-{rank}.pt',state);proof.update(rank=rank)
            proofs.append(proof)
        head=OnlineQProbe(4,dict(head_epochs_per_outer=1),feature_spec_sha='f'*64)
        labels=[dict(qid='q1',question_hash='e'*64,complete=True,selection_source='explore',strict_em=[1,1,0,0,0])]
        head.update_completed_outer(outer,labels,torch.tensor([[1.,2.,3.,4.]]),policy_snapshot_id='sample-before-50',
            sampling_contract_sha='s',strict_scoring_contract_sha='c',retriever_version='r',run_id=self.run_id,attempt_id=self.ids[-1])
        actor_id=self.run_id+':'+str(outer)
        head_receipt=head.save(self.state/'question-probe-state.pt.head.pt',actor_snapshot_id=actor_id)
        qstate=dict(schema_version=1,actor_snapshot_id=actor_id,completed_outer=outer,feature_spec_sha='f'*64,
            head_version=head.version,head_file='question-probe-state.pt.head.pt',head_sha256=head_receipt['checkpoint_sha'],
            config=self.run['question_probe'],candidate_state=self.stream,completed_attempts=[self.ids[-1]],
            reward_probe_version=f'{self.run_id}:0:old',selection_rng={})
        qproof=atomic_save(self.state/'question-probe-state.pt',qstate)
        head_proof=dict(path=str(self.state/'question-probe-state.pt.head.pt'),bytes=(self.state/'question-probe-state.pt.head.pt').stat().st_size,
            sha256=head_receipt['checkpoint_sha'])
        qproof.update(completed_outer=outer,actor_snapshot_id=actor_id,feature_spec_sha='f'*64,head_version=head.version,head=head_proof)
        self.progress=dict(completed_outer=outer,cumulative_writes=self.writes,attempt_index=self.attempt_index,aborted_attempts=2)
        driver=atomic_save(self.state/'driver.pt',dict(completed_outer=outer,candidate_stream=self.stream,rng=driver_rng(),
            probe_version=f'{self.run_id}:0:old',question_probe=qproof,attempt_progress=self.progress))
        manifest=dict(completed_outer=outer,worker_files=proofs,driver_file=driver,question_probe=qproof)
        (self.state/'manifest.json').write_text(json.dumps(manifest))
        (self.cp/'config.json').write_text('{}')
        (self.cp/'weights.safetensors').write_bytes(b'synthetic weights; no CUDA or model tensor claim')
        (self.cp/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'weights.safetensors'}}))
        complete=dict(step=outer,run_id=self.run_id,finite_all_saved_tensors=True,validated_tensor_count=1,
            actual_optimizer_writes_per_rank=self.writes,
            files_sha256={str(p.relative_to(self.cp)):digest(p) for p in self.cp.rglob('*') if p.is_file()})
        (self.cp/'complete.json').write_text(json.dumps(complete))
        heads=self.base/'artifacts/old-reward-heads';heads.mkdir()
        for name in ('commit_probe.pkl','boundary_probe.pkl'):(heads/name).write_bytes(b'fixed old reward head bytes '+name.encode())
        self.reward_version=f'{self.run_id}:{min(outer//50,3)*50}:adopted'
        live=dict(run_id=self.run_id,round=min(outer//50,3)*50,version=self.reward_version,encoder=str(self.cp),
            commit=str(heads/'commit_probe.pkl'),boundary=str(heads/'boundary_probe.pkl'),
            head_sha256={name:digest(heads/name) for name in ('commit_probe.pkl','boundary_probe.pkl')},
            encoder_sha256={name:digest(self.cp/name) for name in ('config.json','weights.safetensors','model.safetensors.index.json')})
        live.update(calibration_report={'fixture_only':True},label_proof={'fixture_only':True})
        (self.base/'artifacts/live/manifest.json').write_text(json.dumps(live))
        self.trainer=SimpleNamespace(global_steps=outer,mixed_stream=SimpleNamespace(state=lambda:self.stream),
            qprobe_runtime=SimpleNamespace(completed_outer=outer,version=head.version),
            inline_attempt_progress=SimpleNamespace(**self.progress,pending=None),
            reward_fn=SimpleNamespace(expected_probe_version=self.reward_version,no_self=True),
            actor_rollout_wg=SimpleNamespace(capture_mixed_state=self.capture_boundary))
        # Explicit synthetic raw prefix for the new provenance snapshot. These
        # tiny fixture rows are never a formal monitor/native execution claim.
        updates=[dict(run_id=self.run_id,step=i,attempt_id=self.ids[i+1]) for i in range(1,outer+1)]
        attempts=[dict(run_id=self.run_id,logical_update=1,attempt_id=tag,complete_outer=False)
                  for tag in self.ids[:2]]
        attempts += [dict(run_id=self.run_id,logical_update=i,attempt_id=self.ids[i+1],complete_outer=True)
                     for i in range(1,outer+1)]
        for name,rows in (('actual-updates.jsonl',updates),('actual-attempts.jsonl',attempts)):
            (self.base/'logs'/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def capture_boundary(self,directory,checkpoint=False):
        assert checkpoint is False
        result=[]
        for rank,original in enumerate(self.rank_states):
            saved=copy.deepcopy(original);saved.pop('optimizer');saved.pop('scheduler')
            saved['rng']['cuda_placeholder']=torch.tensor([2])
            proof=atomic_save(Path(directory)/f'rank-{rank}.pt',saved);proof.update(rank=rank)
            result.append(proof)
        return result

    def publish(self):
        with patch.dict(os.environ,SEARCH_G1_BASE=str(self.base),SEARCH_G1_TRAINING_WORLD_SIZE=str(self.size)):
            return publish_completed_pair(self.trainer,self.cp)


class PairCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_generic50_records_actual_adam900_attempt52_and_postrefit_reward50(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name);old_driver=digest(fixture.state/'driver.pt');old_complete=digest(fixture.cp/'complete.json')
            proof=fixture.publish();self.assertEqual(proof['status'],PAIR_STATUS)
            self.assertFalse(Path(proof['path']).is_relative_to(fixture.cp))
            p,receipt,cp,base,run=read_source(proof['path'])
            self.assertEqual(receipt['attempt_progress'],fixture.progress)
            self.assertEqual(receipt['ranks'][0]['actual_optimizer_writes'],900)
            self.assertEqual(receipt['ranks'][0]['last_attempt_index'],52)
            self.assertEqual(receipt['reward_probe_round'],50)
            self.assertEqual(digest(fixture.state/'driver.pt'),old_driver)
            self.assertEqual(digest(fixture.cp/'complete.json'),old_complete)
            self.assertFalse(receipt['actual_cuda_restore_accepted'])
            self.assertEqual(boundary_worker_rng(receipt,0,fixture.rank_states[0])['cuda_placeholder'].item(),2)
            self.assertEqual(fixture.rank_states[0]['rng']['cuda_placeholder'].item(),1)
            from search_g1.full_state_restore import source
            self.assertEqual(source(p)[1]['step'],50)

    def test_source_hash_metadata_rejects_changed_head(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name,outer=7);proof=fixture.publish()
            head=fixture.state/'question-probe-state.pt.head.pt';head.write_bytes(head.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'SOURCE_FILE_CHANGED'):read_source(proof['path'])

    def test_worker_count_and_head_actor_mismatch_fail_before_pair_publication(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name,outer=7)
            malformed=copy.deepcopy(fixture.rank_states[0]);malformed['actual_optimizer_writes']+=1
            with self.assertRaisesRegex(ValueError,'ADAM_STEP_READBACK'):
                validate_rank_state(malformed,0,4,7)
            malformed=copy.deepcopy(fixture.rank_states[0]);malformed['last_attempt_index']=10
            with self.assertRaisesRegex(ValueError,'ATTEMPT_IDS_INCOMPLETE'):
                validate_rank_state(malformed,0,4,7)
            malformed=copy.deepcopy(fixture.rank_states[0]);malformed['completed_outer']=8
            with self.assertRaisesRegex(ValueError,'OUTER_SCHEDULER'):
                validate_rank_state(malformed,0,4,7)
            fixture.trainer.qprobe_runtime.version='different-head'
            with self.assertRaisesRegex(ValueError,'PUBLISHED_HEAD_CHANGED'):fixture.publish()

    def test_refit_boundary_cannot_be_published_before_new_reward_adoption(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name)
            fixture.trainer.reward_fn.expected_probe_version=f'{fixture.run_id}:0:old'
            with self.assertRaisesRegex(ValueError,'FINAL_REWARD_PROBE_NOT_ADOPTED'):fixture.publish()

    def test_bind_pristine_matching_method_and_restore_exact_driver_prefix(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(Path(name)/'source',outer=7);proof=fixture.publish()
            child=copy.deepcopy(fixture.run);child['run_id']='new-pristine'
            bind_restore(child,proof['path'])
            self.assertEqual(child['full_state_restore']['completed_outer'],7)
            self.assertEqual(child['full_state_restore']['attempt_index'],9)
            self.assertEqual(child['full_state_restore']['actual_optimizer_writes_per_rank'],126)
            target=Path(name)/'target';(target/'logs').mkdir(parents=True)
            (target/'logs/run-manifest.json').write_text(json.dumps(child))
            install_restored_reward_probe(target,child)
            trainer=SimpleNamespace(reward_fn=SimpleNamespace(no_self=True),val_reward_fn=SimpleNamespace())
            before=driver_rng()
            try:
                with patch.dict(os.environ,SEARCH_G1_BASE=str(target)):
                    row=restore_driver(trainer,proof['path'])
                self.assertEqual(row['completed_outer'],7)
                self.assertEqual(trainer._restored_attempt_progress,fixture.progress)
                self.assertEqual(trainer._restored_qprobe_state['actor_snapshot_id'],fixture.run_id+':7')
                self.assertEqual(trainer._restored_candidate_state['pending_source'],['p0','p1'])
                self.assertEqual(trainer._restored_candidate_state['pending_offset'],1)
                self.assertEqual(trainer.reward_fn.expected_probe_version,fixture.reward_version)
            finally:
                import random,numpy as np
                torch.set_rng_state(before['torch_cpu']);np.random.set_state(before['numpy']);random.setstate(before['python'])
            wrong=copy.deepcopy(fixture.run);wrong['question_probe']['exploration_questions']=128
            with self.assertRaisesRegex(ValueError,'HEAD_CONFIGURATION'):bind_restore(wrong,proof['path'])
            wrong=copy.deepcopy(fixture.run);wrong['training_world_size']=8
            with self.assertRaisesRegex(ValueError,'BUDGET_MISMATCH'):bind_restore(wrong,proof['path'])

    def test_bind_inherits_immutable_retriever_semantics_and_rejects_topk_change(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name,outer=7);proof=fixture.publish()
            child=copy.deepcopy(fixture.run)
            for key in ('retriever_version','retriever_manifest_sha256','retriever_topk'):child.pop(key)
            bind_restore(child,proof['path'])
            self.assertEqual(child['retriever_version'],'immutable-index-version')
            self.assertEqual(child['retriever_manifest_sha256'],'9'*64)
            self.assertEqual(child['retriever_topk'],3)
            changed=copy.deepcopy(fixture.run);changed['retriever_topk']=4
            with self.assertRaisesRegex(ValueError,'TARGET_RETRIEVER_TOPK_CHANGED'):bind_restore(changed,proof['path'])

    def test_history_pins_available_artifact_without_fabricating_missing_prefix(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name,outer=7)
            original=fixture.base/'logs/formal-inline-development-before.json'
            original.write_text(json.dumps({'step':0,'fixture_only':True,'correct':17}))
            proof=fixture.publish();receipt=read_source(proof['path'])[1]
            history=receipt['historical_artifacts']
            self.assertFalse(history['raw_training_prefix_verified'])
            self.assertEqual(set(history['development']),{'before'})
            self.assertEqual(history['development']['before']['full_read_sha256'],digest(original))
            self.assertNotIn(50,{item['step'] for item in history['checkpoints']})
            original.write_text(json.dumps({'step':0,'fixture_only':True,'correct':99}))
            with self.assertRaisesRegex(ValueError,'SOURCE_FILE_CHANGED'):read_source(proof['path'])

    def test_four_and_eight_rank_generic_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            for size in (4,8):
                fixture=PairFixture(Path(name)/str(size),outer=3,size=size)
                proof=fixture.publish();receipt=read_source(proof['path'])[1]
                self.assertEqual(receipt['world_size'],size)
                self.assertEqual(len(receipt['ranks']),size)

    def test_new_attempt_ledger_starts_after_real_attempt_not_outer(self):
        from reproduction.common8.recoverable_attempt import record_attempt
        with tempfile.TemporaryDirectory() as name:
            base=Path(name);(base/'logs').mkdir()
            run=dict(run_id='a'*32,profile='common8',training_world_size=4,
                policy_safety_mode='inline_first_five',execution_backend='ssh_local',execution_id='f'*32,
                slurm_job_id=None,
                full_state_restore=dict(kind='question_probe_pair',completed_outer=50,attempt_index=52))
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            result=dict(all_ranks_agreed=True,attempt_index=53,attempt_id='b'*32,real_rank_count=4)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(base),SEARCH_G1_TRAINING_WORLD_SIZE='4'):
                row=record_attempt(result)
            self.assertEqual(row['attempt_index'],53)

    def test_generic_probe_admission_and_archive_use_pinned_round50(self):
        from reproduction.a800.probe_admission import initial_evaluator
        from reproduction.common8.formal_inline_safety import archive_initial_probe
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(Path(name)/'source');proof=fixture.publish()
            child=copy.deepcopy(fixture.run);child.update(source_sha256={'fixture.py':'1'*64},
                model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1')
            bind_restore(child,proof['path'])
            round_id,encoder,hashes=initial_evaluator(child)
            self.assertEqual(round_id,50);self.assertEqual(encoder,str(fixture.cp))
            target=Path(name)/'target';(target/'logs').mkdir(parents=True)
            install_restored_reward_probe(target,child)
            receipt,files=archive_initial_probe(target,child)
            self.assertEqual(receipt['restored_probe_round'],50)
            self.assertEqual(receipt['version'],fixture.reward_version)
            self.assertEqual(len(files),4)
            self.assertTrue((target/receipt['archive']).exists())

    def test_generic_scheduler_resumes_real900_writes_and_attempt53(self):
        from reproduction.common8.test_scheduler import ReceiptTests
        from reproduction.common8.scheduler import OuterSchedulerEvidence
        test=ReceiptTests(methodName='test_prepare_is_durable_and_does_not_advance');test.setUp()
        try:
            for _ in range(50):test.writes(18);test.scheduler.step()
            test.run.update(policy_safety_mode='inline_first_five',
                full_state_restore=dict(kind='question_probe_pair',completed_outer=50,
                                        actual_optimizer_writes_per_rank=900,attempt_index=52))
            test._save_run()
            prefix=dict(completed_outer=50,writes=900,attempt_index=52,
                        used_attempt_ids=[f'{i:032x}' for i in range(1,53)])
            with patch('search_g1.full_state_restore.verified_runtime_prefix',return_value=prefix):
                evidence=OuterSchedulerEvidence(test.base,test.optimizer,test.scheduler,rank=0,world_size=8,job_id='12345')
            try:
                self.assertEqual((evidence.completed_outer,evidence.actual_writes,evidence.last_attempt_index),(50,900,52))
                evidence.begin_outer(51,optimizer_writes=900,batch_gate_sha256='d'*64,attempt_id='e'*32,attempt_index=53)
                test.writes(17);evidence.complete_outer(optimizer_writes=917,opportunities_processed=20)
                self.assertEqual(test.scheduler.last_epoch,51)
                self.assertEqual(int(test.optimizer.state[test.parameter]['step']),917)
            finally:evidence.close()
        finally:test.tearDown()

    def test_late199_source_binds_one_remaining_outer_and_200_is_readable_only(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(Path(name)/'source199',outer=199)
            proof=fixture.publish();child=copy.deepcopy(fixture.run)
            bind_restore(child,proof['path'])
            self.assertEqual(child['full_state_restore']['followup_acceptance_outer'],200)
            self.assertEqual(child['full_state_restore']['completed_outer'],199)
            terminal=PairFixture(Path(name)/'source200',outer=200)
            terminal_proof=terminal.publish()
            self.assertEqual(read_source(terminal_proof['path'])[1]['step'],200)
            with self.assertRaisesRegex(ValueError,'TARGET_ALREADY_COMPLETE'):
                bind_restore(copy.deepcopy(terminal.run),terminal_proof['path'])

    def test_post_evaluation_worker_rng_hash_is_immutable(self):
        with tempfile.TemporaryDirectory() as name:
            fixture=PairFixture(name,outer=7);proof=fixture.publish()
            receipt=read_source(proof['path'])[1]
            selected=receipt['boundary_worker_files'][0]
            path=Path(selected['path']);path.write_bytes(path.read_bytes()+b'corrupt')
            with self.assertRaisesRegex(ValueError,'SOURCE_FILE_CHANGED'):
                boundary_worker_rng(receipt,0,fixture.rank_states[0])


if __name__=='__main__':unittest.main()
