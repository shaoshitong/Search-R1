"""CPU contracts for the independent terminal evaluation entry.

These fixtures verify refusal/identity/order/ownership behavior only. They do
not claim a real CP200 exists or that CUDA inference/retrieval has passed.
"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reproduction.hpc3_dynrpo_periodic import final_eval as f


def boundary():
    parent = dict(run_id='a'*32, training_world_size=4, execution_site='hpc3',
                  question_probe=dict(head_kind=f.HEAD_KIND))
    receipt = dict(status=f.PAIR_STATUS, step=200, world_size=4, run_id=parent['run_id'],
        execution_phase='FINAL_EVALUATION_PENDING', actor_snapshot_id=parent['run_id']+':200',
        attempt_progress=dict(completed_outer=200, cumulative_writes=4000),
        ranks=[dict(rank=rank, completed_outer=200, scheduler_last_epoch=200,
                    scheduler_step_count=201, actual_optimizer_writes=4000) for rank in range(4)])
    return receipt, parent


def row(qid, source='nq'):
    return dict(question_id=qid, data_source=source, em=1, f1=1., actual_calls=1,
        valid_action=2, turns=2, answer_sha256='b'*64, answer_present=True,
        response_token_count=20, model_token_count=10)


class FinalEvalTests(unittest.TestCase):
    def test_terminal_source_is_limited_to_this_independent_experiment(self):
        for value in [f.PREFIX,f.PREFIX+'-r3']:
            self.assertEqual(f.source_namespace(value),Path(value))
        for value in ['/data/user/jxu957/experiments/another-thread',f.PREFIX+'-r0',
                      f.PREFIX+'-r3/../another-thread',f.PREFIX+'-other']:
            with self.assertRaisesRegex(ValueError,'SOURCE_MUST_BELONG'):
                f.source_namespace(value)

    def test_missing_checkpoint_is_not_ready_before_import_or_remote_action(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                f.authenticate(Path(directory)/'missing-source-receipt.json')

    def test_tensor_auth_requires_actual_running_owned_allocated_node_and_64GiB(self):
        job = dict(JobId='123456', JobState='RUNNING', UserId='mou176(21930)',
                   NodeList='ACD1-[12-13]', AllocTRES='cpu=12,mem=128G,node=2')
        args = dict(job_id='123456', uid=21930, username='mou176', hostname='ACD1-12.domain',
                    cgroup_memory_limit_bytes=128*1024**3)
        proof = f.validate_state_auth_resources(job, ['ACD1-12', 'ACD1-13'], **args)
        self.assertEqual(proof['actual_allocated_memory_bytes'], 128*1024**3)
        self.assertFalse(proof['gpu_allocation_required'])
        self.assertFalse(proof['host_total_memory_used_as_evidence'])
        boundary_job = dict(job, AllocTRES='cpu=6,mem=65536M,node=1')
        f.validate_state_auth_resources(boundary_job, ['ACD1-12'], **dict(args, cgroup_memory_limit_bytes=64*1024**3))
        for change in (dict(JobState='PENDING'), dict(JobId='123457'), dict(UserId='jxu957(14090)'),
                       dict(AllocTRES='cpu=6,mem=8G,node=1'), dict(AllocTRES='cpu=6,node=1'),
                       dict(AllocTRES='', ReqTRES='cpu=12,mem=128G,node=1', TRES='cpu=12,mem=128G,node=1')):
            with self.subTest(change=change), self.assertRaises(ValueError):
                f.validate_state_auth_resources(dict(job, **change), ['ACD1-12'], **args)
        for change in (dict(hostname='login1'), dict(uid=14090), dict(username='jxu957'),
                       dict(cgroup_memory_limit_bytes=8*1024**3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                f.validate_state_auth_resources(job, ['ACD1-12'], **dict(args, **change))

    def test_login_tensor_auth_preflight_rejects_before_slurm_queries(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(f.subprocess, 'check_output') as query:
            with self.assertRaisesRegex(ValueError, 'NOT_LOGIN_HOST'):
                f.state_auth_resource_preflight()
            query.assert_not_called()

    def test_existing_receipt_still_runs_resource_gate_before_any_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); base = root/'task-r1'; base.mkdir()
            receipt = base/'source-receipt.json'; receipt.write_text('{}')
            with mock.patch.object(f, 'PREFIX', str(root/'task')), \
                 mock.patch.object(f, 'state_auth_resource_preflight', side_effect=RuntimeError('resource gate first')) as gate, \
                 mock.patch.dict('sys.modules', {'torch':None}):
                with self.assertRaisesRegex(RuntimeError, 'resource gate first'):
                    f.authenticate(receipt)
                gate.assert_called_once_with()

    def test_cgroup_v1_v2_read_actual_effective_ancestor_limit(self):
        for unified in (True, False):
            with self.subTest(unified=unified), tempfile.TemporaryDirectory() as directory:
                mount = Path(directory).resolve(); leaf = mount/'job123/step_batch'; leaf.mkdir(parents=True)
                filename = 'memory.max' if unified else 'memory.limit_in_bytes'
                (mount/filename).write_text(str(128*1024**3))
                (leaf.parent/filename).write_text(str(8*1024**3))
                (leaf/filename).write_text('max' if unified else str(128*1024**3))
                member = '0::/job123/step_batch\n' if unified else '5:memory:/job123/step_batch\n'
                kind = 'cgroup2' if unified else 'cgroup'
                mountinfo = f'29 23 0:26 / {mount} rw - {kind} cgroup rw,memory\n'
                original = Path.read_text
                def proc_or_fixture(path, *args, **kwargs):
                    if path == Path('/proc/self/cgroup'): return member
                    if path == Path('/proc/self/mountinfo'): return mountinfo
                    return original(path, *args, **kwargs)
                with mock.patch.object(Path, 'read_text', proc_or_fixture):
                    limit, evidence = f.cgroup_memory_limits()
                self.assertEqual(limit, 8*1024**3)
                self.assertEqual(len(evidence), 3)

    def test_terminal200_only_requires_complete_four_rank_native_pair(self):
        receipt, parent = boundary()
        value = f.qualify_boundary(receipt, parent, Path('/remote/global_step_200'))
        self.assertEqual(value['completed_student_outers'], 200)
        self.assertFalse(value['reconstruction_of_original_update_row200'])
        self.assertFalse(value['training_resume'])
        mutations = [dict(step=199), dict(step=201), dict(step=True), dict(world_size=8),
                     dict(status='MODEL_WEIGHTS_ONLY'), dict(execution_phase='PROBE_PENDING')]
        for mutation in mutations:
            bad = dict(receipt, **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                f.qualify_boundary(bad, parent, Path('/remote/global_step_200'))
        bad = copy.deepcopy(receipt); bad['ranks'][3]['actual_optimizer_writes'] = 3999
        with self.assertRaises(ValueError):
            f.qualify_boundary(bad, parent, Path('/remote/global_step_200'))
        bad_parent = copy.deepcopy(parent); bad_parent['question_probe']['head_kind'] = 'legacy'
        with self.assertRaises(ValueError):
            f.qualify_boundary(receipt, bad_parent, Path('/remote/global_step_200'))

    def test_optimizer_objects_and_all_update_attempts_are_forbidden(self):
        guard = f.NoUpdates()
        guard._is_actor, guard._is_ref, guard._is_rollout = False, False, True
        guard.actor_optimizer = guard.actor_lr_scheduler = None
        guard.assert_no_optimizer()
        for intent in ('actor.step', 'probe.step', 'fit', 'scheduler.step'):
            with self.subTest(intent=intent), self.assertRaises(RuntimeError):
                guard.reject_update(intent)
        for field, value in (('_is_actor', True), ('_is_ref', True), ('_is_rollout', False),
                             ('actor_optimizer', object()), ('actor_lr_scheduler', object())):
            old = getattr(guard, field); setattr(guard, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                guard.assert_no_optimizer()
            setattr(guard, field, old)

    def test_worker_requires_active_admitted_four_gpu_controller_birth(self):
        controller = dict(pid=123, start_ticks=987, sid=123, uid=os.getuid(), state='S')
        claim = dict(role=f.ROLE, job='123456', controller=controller)
        admission = dict(accepted=True, slurm_job_id='123456', training_world_size=4)
        f.validate_worker_execution(claim, admission, '123456', controller)
        for wrong in (dict(controller, start_ticks=988), dict(controller, state='Z'),
                      dict(controller, uid=os.getuid()+1)):
            with self.assertRaises(ValueError):
                f.validate_worker_execution(claim, admission, '123456', wrong)
        for wrong in (dict(admission, training_world_size=8), dict(admission, accepted=False)):
            with self.assertRaises(ValueError):
                f.validate_worker_execution(claim, wrong, '123456', controller)

    def test_HF_safetensors_tied_alias_is_verified_without_skipping_parameters(self):
        declared = {'model.embed_tokens.weight':'model-1.safetensors',
                    'model.layers.0.input_layernorm.weight':'model-1.safetensors'}
        actual = set(declared) | {'lm_head.weight'}
        mapping = f.hf_model_mapping(actual, declared, tie_word_embeddings=True)
        self.assertEqual(set(mapping), actual)
        self.assertEqual(mapping['lm_head.weight'], 'model.embed_tokens.weight')
        for names, tied in ((actual, False), (actual | {'unknown.weight'}, True),
                            (actual - {'model.embed_tokens.weight'}, True)):
            with self.subTest(names=names, tied=tied), self.assertRaises(ValueError):
                f.hf_model_mapping(names, declared, tie_word_embeddings=tied)
        self.assertEqual(f.hf_model_mapping(set(declared), declared, tie_word_embeddings=False),
                         {name:name for name in declared})

    def test_canonical_512_and_51713_exact_cardinality_and_source_counts(self):
        for split, count in f.COUNTS.items():
            ids = ['repeat-'+str(index % 999) for index in range(count)]
            sources = ['nq' if index % 2 else 'hotpotqa' for index in range(count)]
            record = dict(rows=count, source_counts=dict(f.Counter(sources)))
            f.validate_product_binding(record, ids, sources, split)
            for bad_ids in (ids[:-1], ids+['extra']):
                with self.subTest(split=split), self.assertRaises(ValueError):
                    f.validate_product_binding(record, bad_ids, sources, split)
            with self.assertRaises(ValueError):
                f.validate_product_binding(record, ids, ['nq']*count, split)
        with self.assertRaises(ValueError):
            f.validate_product_binding(dict(rows=64, source_counts={'nq':64}), ['id']*64, ['nq']*64, 'stage')

    def test_ordered_occurrences_retained_and_scoring_not_em_as_f1(self):
        ids, sources = ['same', 'other', 'same'], ['nq', 'hotpotqa', 'nq']
        rows = [row(qid, source) for qid, source in zip(ids, sources)]
        f.measured_rows(rows, ids, sources)
        wrong = copy.deepcopy(rows); wrong[1], wrong[2] = wrong[2], wrong[1]
        with self.assertRaises(ValueError):
            f.measured_rows(wrong, ids, sources)
        wrong = copy.deepcopy(rows); wrong[0]['f1'] = .9
        with self.assertRaises(ValueError):
            f.measured_rows(wrong, ids, sources)
        wrong = copy.deepcopy(rows); wrong[0]['actual_calls'] = 5
        with self.assertRaises(ValueError):
            f.measured_rows(wrong, ids, sources)
        wrong = copy.deepcopy(rows); wrong[0]['f1'] = float('nan')
        with self.assertRaises(ValueError):
            f.measured_rows(wrong, ids, sources)

    def test_owned_namespace_escape_symlink_lock_and_immutable_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); base = root/'task-r1'; base.mkdir(); (base/'preparation').mkdir()
            with mock.patch.object(f, 'PREFIX', str(root/'task')):
                self.assertEqual(f.owned(base), base)
                with self.assertRaises(ValueError):
                    f.owned(root)
                with self.assertRaises(ValueError):
                    f.output(base, root/'outside.json')
                outside = root/'outside'; outside.mkdir(); (base/'escape').symlink_to(outside)
                with self.assertRaises(ValueError):
                    f.output(base, base/'escape/wrong.json')
                target = base/'result.json'; f.write_once(base, target, dict(accepted=1))
                f.write_once(base, target, dict(accepted=1))
                with self.assertRaises(ValueError):
                    f.write_once(base, target, dict(accepted=2))
                self.assertEqual(f.read(target), dict(accepted=1))
                with f.execution_guard(base):
                    with self.assertRaises(ValueError):
                        with f.execution_guard(base):
                            self.fail('second writer admitted')

    def test_chunks_bind_context_exact_interval_rows_and_real_gpu_evidence(self):
        ids = [str(index) for index in range(512)]; sources = ['nq']*512
        rows = [row(qid) for qid in ids[:64]]
        binding = dict(source_context_sha256='c'*64)
        value = dict(role=f.ROLE, source_context_sha256='c'*64, split='validation', start=0, end=64,
            rows=rows, rows_sha256=hashlib.sha256(f.canonical(rows)).hexdigest(),
            actual_gpu_inference_verified=True,
            gpu_readback=[dict(rank=rank, actual_cuda_uuid='GPU-'+str(rank), no_optimizer_constructed=True,
                               vllm_readback=dict(accepted=True)) for rank in range(4)])
        f.validate_chunk(value, binding, 'validation', 0, ids, sources)
        for mutation in (dict(source_context_sha256='d'*64), dict(end=63),
                         dict(actual_gpu_inference_verified=False), dict(gpu_readback=[])):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                f.validate_chunk(dict(value, **mutation), binding, 'validation', 0, ids, sources)
        bad = copy.deepcopy(value); bad['rows'][0]['f1'] = .95; bad['rows'][0]['em'] = 0
        with self.assertRaises(ValueError):
            f.validate_chunk(bad, binding, 'validation', 0, ids, sources)
        with self.assertRaises(ValueError):
            f.validate_chunk(value, binding, 'validation', 64, ids, sources)

    def test_source_score_requires_CP200_canonical_product_no_stage_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); (base/'logs').mkdir()
            self.assertIsNone(f.accepted_source_score(base, dict(run_id='run'), 'validation'))
            ids = [str(i) for i in range(512)]; sources = ['nq']*512
            record = dict(sha256='a'*64, source_counts={'nq':512})
            score = dict(run_id='run', step=200, split='validation', final=False,
                validation_sha256='a'*64, question_ids=ids, data_sources=sources, em=[1.]*512, f1=[1.]*512)
            path = base/'logs/validation-step200.json'; path.write_bytes(f.canonical(score))
            with mock.patch.object(f, 'canonical_product', return_value=(record, None, ids, sources)):
                result = f.accepted_source_score(base, dict(run_id='run'), 'validation')
                self.assertEqual(result['questions'], 512)
                score.update(step=199); path.write_bytes(f.canonical(score))
                with self.assertRaises(ValueError):
                    f.accepted_source_score(base, dict(run_id='run'), 'validation')

    def test_partial_results_context_binds_scorer_generation_data_and_CP200(self):
        from reproduction.common8.source_files import runtime_files
        repo = Path(f.__file__).resolve().parents[2]
        run = dict(source_sha256={str(path.relative_to(repo)):'a'*64 for path in runtime_files(repo)},
            input_sha256={'validation.parquet':'v'*64, 'final.parquet':'f'*64},
            budget=dict(common_manifest_sha256='m'*64), infra_rollout=dict(mode='same'))
        proof = dict(source_receipt_sha256='p'*64, run_id='r'*32)
        value, original = f.context(run, proof)
        self.assertEqual(value['max_response_length'], 500)
        self.assertEqual(value['validation_sampling'], 'canonical_greedy_one_response')
        for name in ('reproduction/common8/calibration_evaluation.py',
                     'reproduction/common8/context_budget.py', 'search_g1/signals.py'):
            altered = copy.deepcopy(run); altered['source_sha256'][name] = 'b'*64
            self.assertNotEqual(f.context(altered, proof)[1], original)
        altered = copy.deepcopy(run); altered['input_sha256']['final.parquet'] = 'z'*64
        self.assertNotEqual(f.context(altered, proof)[1], original)
        self.assertNotEqual(f.context(run, dict(proof, source_receipt_sha256='q'*64))[1], original)
        incomplete = copy.deepcopy(run)
        incomplete['source_sha256'].pop('reproduction/common8/calibration_evaluation.py')
        with self.assertRaises(ValueError):
            f.context(incomplete, proof)


if __name__ == '__main__':
    unittest.main()
