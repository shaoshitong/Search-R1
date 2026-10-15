"""Tiny CPU-only validation of fresh-actor and independent-release contracts."""
import copy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.a800_common import qprobe_qwen_formal_ops as ops
from reproduction.a800_common.qprobe_qwen_formal_ops import CPU_MODULES,CPU_PROGRAM,validate_cpu_checks,validate_fresh,validate_release,once


def fresh():
    return dict(execution_backend='ssh_local',slurm_job_id=None,execution_host=socket.gethostname(),profile='common8',completed_updates=0,target_updates=200,training_world_size=4,actor_micro_batch_per_rank=4,initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',initial_probe_version=None,auxiliary_physical_indices=[5,6],question_probe=None,budget=dict(world_size=4,per_rank_actor_micro_batch=4,ppo_micro_batch_size=16,ppo_mini_batch_size=128,lr_warmup_steps=57,per_rank_log_prob_micro_batch=8))


class NativeFormalOpsTest(unittest.TestCase):
    def test_qualified_bank_origin_requires_terminal_proof_and_unchanged_file_seal(self):
        cache=ops.QUALIFIED_BANK_BASE+'/token-features'
        admitted=dict(stage='FULL_BANK_SHA_AND_ENCODER_RESOURCE_RELEASE_ADMITTED',complete_sha256=ops.QUALIFIED_BANK_COMPLETE,
            cache_dir=cache,feature_manifest_sha256=ops.QUALIFIED_BANK_MANIFEST,feature_spec_sha256=ops.SPEC_SHA,
            owned_sessions_remaining=0,owned_cuda_contexts_remaining=0,
            files=[dict(path=cache+'/'+name,device=7,inode=index+1,bytes=100+index,mtime_ns=1000+index) for index,name in enumerate(ops.BANK_FILES)])
        origin=dict(role='native_qwen_preupdate_qualified_bank_origin_v1',base=ops.QUALIFIED_BANK_ORIGIN,
            source_commit=ops.QUALIFIED_BANK_SOURCE,source_map_sha256=ops.QUALIFIED_BANK_MAP,run_id=ops.QUALIFIED_BANK_RUN,
            execution_id=ops.QUALIFIED_BANK_EXECUTION,bank_base=ops.QUALIFIED_BANK_BASE,complete_sha256=ops.QUALIFIED_BANK_COMPLETE,
            feature_manifest_sha256=ops.QUALIFIED_BANK_MANIFEST,admitted_sha256=ops.QUALIFIED_BANK_ADMIT,
            terminal_audit_sha256=ops.QUALIFIED_BANK_TERMINAL_AUDIT,producer_source_sha256=ops.QUALIFIED_BANK_PRODUCERS,
            formal_completed_outers=0,actor_or_head_updates_inherited=False,old_handle_sha256='a'*64,
            old_intent_canonical_sha256='b'*64,bank_admitted_observation_sha256='c'*64,terminal_audit_bytes=50000,admitted=admitted)
        audit=dict(role='native_qwen92bb_actual_terminal_failure_and_onlyowned_resource_audit',source_commit=origin['source_commit'],
            run_id=origin['run_id'],execution_id=origin['execution_id'],terminal_owned_cleanup_accepted=True,owned_birth_records=34,
            live_owned_session_processes=[],owned_cuda_contexts_remaining=[],all_project_trainers_live=[],
            one_nvml_read_only_query=True,no_tensor_or_gpu_test=True,no_stop_or_restart=True,metadata_files={
                'logs/cleanup-status.json':dict(value=dict(cleaned=True,execution_backend='ssh_local',slurm_job_id=None,run_id=origin['run_id'],execution_id=origin['execution_id'])),
                'logs/pipeline-status.json':dict(value=dict(stage='FAILED')),
                'logs/process-ledger.json':dict(value=dict(jobs=dict(trainer=dict(state='STOPPED'))))})
        ops.validate_qualified_bank_origin(origin,ops.QUALIFIED_BANK_PRODUCERS);ops.validate_terminal_bank_origin(origin,audit)
        for key,value in (('source_commit','9'*40),('feature_manifest_sha256','9'*64),('formal_completed_outers',1)):
            with self.subTest(key=key),self.assertRaises(AssertionError):ops.validate_qualified_bank_origin(dict(origin,**{key:value}),ops.QUALIFIED_BANK_PRODUCERS)
        with self.assertRaises(AssertionError):ops.validate_qualified_bank_origin(origin,dict(ops.QUALIFIED_BANK_PRODUCERS,**{'search_g1/qprobe_token_features.py':'9'*64}))
        with self.assertRaises(AssertionError):ops.validate_terminal_bank_origin(origin,dict(audit,live_owned_session_processes=[dict(pid=1)]))
        seals={row['path']:SimpleNamespace(st_dev=row['device'],st_ino=row['inode'],st_size=row['bytes'],st_mtime_ns=row['mtime_ns']) for row in admitted['files']}
        with patch.object(ops,'owned',side_effect=lambda path:SimpleNamespace(stat=lambda:seals[path])):
            ops.check_bank_file_seal(admitted)
            seals[cache+'/packed_features.npy'].st_mtime_ns+=1
            with self.assertRaises(AssertionError):ops.check_bank_file_seal(admitted)
        with self.assertRaises(AssertionError):ops.bank_command(Path('/unused'),Path('/unused'),dict(qualified_bank_origin=origin),prepare_only=True)

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
