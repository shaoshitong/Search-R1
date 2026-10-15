"""Tiny CPU-only validation of fresh-actor and independent-release contracts."""
import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest

from reproduction.a800_common.qprobe_qwen_formal_ops import CPU_MODULES,CPU_PROGRAM,validate_cpu_checks,validate_fresh,validate_release,once


def fresh():
    return dict(execution_backend='ssh_local',slurm_job_id=None,execution_host=socket.gethostname(),profile='common8',completed_updates=0,target_updates=200,training_world_size=4,actor_micro_batch_per_rank=4,initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',initial_probe_version=None,auxiliary_physical_indices=[5,6],question_probe=None,budget=dict(world_size=4,per_rank_actor_micro_batch=4,ppo_micro_batch_size=16,ppo_mini_batch_size=128,lr_warmup_steps=57,per_rank_log_prob_micro_batch=8))


class NativeFormalOpsTest(unittest.TestCase):
    def test_pristine_actor_budget_and_native_logical_gpu_requirement(self):
        run=fresh();validate_fresh(run,bound=False)
        for name,value in (('completed_updates',5),('full_state_restore',dict(saved=True)),('actor_micro_batch_per_rank',2)):
            bad=copy.deepcopy(run);bad[name]=value
            with self.subTest(name=name),self.assertRaises(AssertionError):validate_fresh(bad,bound=False)
        q=dict(head_kind='qwen_native_lastblock_binary_v1',head_device='cuda:0',candidate_questions=5120,selected_questions=512,responses_per_question=5,exploration_questions=0,ranking='binary_mixed_probability',source_layer=35,head_epochs_per_outer=2,replay_window_outer=8,teacher_replay_excluded=True,std_loss_weight=.3,count_aux_loss_weight=0.,target_statistic='sample_std',initial_head_checkpoint='/remote/initial.pt',initial_head_ready='/remote/ready.json')
        run.update(question_probe=q,mixed_first_selection=False);validate_fresh(run,bound=True)
        for key,value in (('head_device','cpu'),('head_kind','token_binary_v1'),('replay_window_outer',16),('teacher_replay_excluded',False)):
            bad=copy.deepcopy(run);bad['question_probe'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_fresh(bad,bound=True)

    def test_release_binds_exact_new_source_head_bank_and_independent_actor(self):
        intent=dict(source_commit='a'*40,human_intent_sha256='h'*64)
        qualified=dict(source_map_sha256='s'*64,warmup_ready_sha256='r'*64,warmup_head_sha256='w'*64,bank_manifest_sha256='b'*64)
        review=dict(role='native_qwen_formal200_independent_release_review',approved=True,independent=True,context_free=True,reviewer='reviewer',implementation_actor='builder',source_commit=intent['source_commit'],human_intent_sha256=intent['human_intent_sha256'],authorized_source_publication_approved=True,actual_deployment_review_approved=True,**qualified)
        validate_release(review,intent,qualified)
        for key,value in (('warmup_head_sha256','wrong'),('bank_manifest_sha256','wrong'),('context_free',False),('reviewer','builder')):
            bad=dict(review);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_release(bad,intent,qualified)

    def test_existing_claim_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'claim.json';once(path,dict(attempt=1))
            with self.assertRaises(FileExistsError):once(path,dict(attempt=2))
            self.assertEqual(json.loads(path.read_bytes()),dict(attempt=1))

    def test_cpu_compatibility_requires_actual_non_skipped_pass_and_no_cuda(self):
        expected=dict(source_commit='a'*40,source_map_sha256='b'*64,modules=list(CPU_MODULES),test_source_sha256={name:'c'*64 for name in CPU_MODULES},runner_source_sha256='d'*64)
        proof=dict(expected,stage='NATIVE_FORMAL_CPU_COMPATIBILITY_CHECKS',tests_run=12,failures=0,errors=0,skips=0,passed=True,cuda_visible_devices='',cuda_visible_devices_after='',cuda_initialized_before=False,cuda_initialized_after=False)
        validate_cpu_checks(proof,expected)
        for key,value in (('tests_run',0),('skips',1),('failures',1),('errors',1),('cuda_visible_devices','0'),('cuda_visible_devices_after','0'),('cuda_initialized_before',True),('cuda_initialized_after',True),('source_map_sha256','wrong'),('check_failure_type','ImportError')):
            bad=dict(proof);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_cpu_checks(bad,expected)
        compile(CPU_PROGRAM,'native-formal-cpu-checks','exec')


if __name__=='__main__':unittest.main()
