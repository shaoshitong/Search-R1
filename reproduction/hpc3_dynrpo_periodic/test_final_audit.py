"""Pure scalar/file fixtures for the independent periodic final audit.

These are rejection/contract tests. They are not a GPU run, an actual checkpoint
reload, or a substitute for invoking final_audit on HPC3's original artifacts.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from reproduction.hpc3_dynrpo_periodic import final_audit as audit


def sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def phases():
    return [dict(outer=outer, fit_epochs=20, head_adam_steps_before=i * 400,
        head_adam_steps_after=(i + 1) * 400,
        **(dict(groups=5120, head_optimizer_updates=400, replay_groups=0) if i else {}))
        for i, outer in enumerate(audit.REFIT_OUTERS)]


def refit_fixture():
    outer = 25
    run = dict(run_id='fixture-run', source_sha256={k:sha(k) for k in audit.STRICT_SCORER_FILES},
        retriever_version=sha('index'), retriever_topk=3)
    labels = [dict(qid='q' + str(i), question_hash=sha(i), strict_em=[0, 1, 0, 1, 0], correct_count=2)
              for i in range(5120)]
    ranks = [dict(rank=r, actual_optimizer_writes=500) for r in range(4)]
    pre = dict(step=outer, execution_phase='PROBE_PENDING', checkpoint='/original/cp25', ranks=ranks)
    post = dict(step=outer, execution_phase='STUDENT_READY', checkpoint='/original/cp25')
    before = dict(adam_steps=400, fingerprint=sha('before'))
    after = dict(adam_steps=800, fingerprint=sha('after'))
    complete = dict(stage='PERIODIC_PROBE_REFIT20_ACCEPTED', outer=outer, fit_epochs=20,
        groups=5120, trajectories=25600, head_optimizer_updates=400, head_adam_steps_before=400,
        head_adam_steps_after=800, replay_groups=0, student_updated=False,
        student_parameters_copied=False, optimizer_reset=False, dynamic_feature_tensors_persisted=False)
    state = dict(outer=outer, head_adam_steps=800, head_fingerprint=after['fingerprint'],
        phase=dict(start_steps=400, epoch=20, batch=0, order=None,
            summaries=[dict(epoch=e, training_groups=5120, actual_head_adam_updates=400 + 20 * e,
                target_statistic='sample_std', auxiliary_weight=.3) for e in range(1, 21)],
            minibatches=[dict(epoch=e, batch=b, loss=.1, adam_steps=400 + (e - 1) * 20 + b + 1)
                         for e in range(1, 21) for b in range(20)]),
        labels=labels, indices=list(range(5120)), feature_hashes={r['qid']:sha(r['qid']) for r in labels},
        dynamic_tensor_fields_present=False,
        frozen_before=[dict(rank=r, completed_outer=outer, parameter_sha256=sha('parameters' + str(r)),
            scheduler_sha256=sha('scheduler'), actual_optimizer_writes=500, adam_steps=[500]) for r in range(4)])
    scoring = dict(source_sha256=run['source_sha256'], target='strict_terminal_em',
        responses_per_question=5, std_ddof=1, common_data_manifest_sha256=audit.DATA_SHA)
    sampling = dict(temperature=1., top_p=1., max_turns=4)
    retrieval = dict(topk=3, recorded_index_version=run['retriever_version'])
    contracts = dict(run_id=run['run_id'], strict_scoring_contract=scoring,
        strict_scoring_contract_sha=audit.document_sha(scoring), sampling_contract=sampling,
        sampling_contract_sha=audit.document_sha(sampling), retriever_contract=retrieval,
        retriever_version=audit.document_sha(retrieval))
    chunks = []
    for i in range(10):
        parents = [str(j) + '-' + sha('original-parent' + str(j)) for j in range(i * 512, (i + 1) * 512)]
        identity = dict(run_id=run['run_id'], logical_update=outer, validation=False,
            source_sha256=run['source_sha256'], parent_group_ids=parents,
            policy_snapshot_id=run['run_id'] + ':25:policy-' + 'a' * 32,
            trajectory_ids=[run['run_id'] + '-25-probe-refit-' + p + '-sample-' + str(sample)
                            for p in parents for sample in range(5)])
        chunks.append(dict(outer=outer, questions=512, trajectories=2560, policy_tokens=1000,
            identity=identity, contracts=contracts, labels=labels[i * 512:(i + 1) * 512]))
    state['canonical_parent_group_ids'] = [p for c in chunks for p in c['identity']['parent_group_ids']]
    return complete, state, chunks, pre, post, run, outer, before, after


class PeriodicDocumentTests(unittest.TestCase):
    def test_exact_eight_phases_and_final_no_refit(self):
        self.assertEqual(audit.exact_phases(phases(), 200), 3200)
        self.assertEqual(audit.exact_phases(phases()[:1], 25, pending=True), 400)
        self.assertNotIn(200, audit.REFIT_OUTERS)

    def test_missing25_phase_and_short_valid_prefix_rejected(self):
        for bad in ([p for p in phases() if p['outer'] != 25], phases()[:1]):
            with self.assertRaises(audit.Incomplete):audit.exact_phases(bad, 200)

    def test_wrong_adam_continuity_or_epoch_budget_rejected(self):
        for key, value in (('head_adam_steps_before', 0), ('head_adam_steps_after', 799), ('fit_epochs', 10)):
            broken = phases();broken[1][key] = value
            with self.subTest(key=key), self.assertRaises(audit.Incomplete):audit.exact_phases(broken, 200)

    def test_real_ten_chunk_twenty_epoch_contract_accepted(self):
        audit.check_refit_documents(*refit_fixture())

    def test_refit_missing_chunk_duplicate_question_or_wrong_adam_rejected(self):
        for mutation in ('missing_chunk', 'duplicate_question', 'wrong_adam', 'partial_epoch', 'replay'):
            args = list(refit_fixture())
            if mutation == 'missing_chunk':args[2].pop()
            if mutation == 'duplicate_question':args[1]['indices'][-1] = 0
            if mutation == 'wrong_adam':args[0]['head_adam_steps_before'] = 0
            if mutation == 'partial_epoch':args[1]['phase']['epoch'] = 19
            if mutation == 'replay':args[0]['replay_groups'] = 5120
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_refit_documents(*args)

    def test_refit_persisted_feature_tensors_or_nonfrozen_student_rejected(self):
        for mutation in ('features', 'student', 'reset', 'missing_pending'):
            args = list(refit_fixture())
            if mutation == 'features':args[1]['dynamic_tensor_fields_present'] = True
            if mutation == 'student':args[1]['frozen_before'][0]['completed_outer'] = 0
            if mutation == 'reset':args[0]['optimizer_reset'] = True
            if mutation == 'missing_pending':args[3]['execution_phase'] = 'STUDENT_READY'
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_refit_documents(*args)

    def test_refit_chunk_uses_actual_native_identity_schema(self):
        args = list(refit_fixture())
        # Native prepare_identity carries parent/trajectory IDs, not invented
        # repeats/probe_refit/question_ids fields. Reordered parents are invalid.
        self.assertNotIn('probe_refit', args[2][0]['identity'])
        args[2][0]['identity']['parent_group_ids'].reverse()
        with self.assertRaises(audit.Incomplete):audit.check_refit_documents(*args)

    def inherited_chunk_fixture(self):
        args = list(refit_fixture())
        parent = copy.deepcopy(args[5])
        parent['source_sha256']['reproduction/hpc3_dynrpo_periodic/final_audit.py'] = sha('previous audit')
        current = copy.deepcopy(parent)
        current['source_sha256']['reproduction/hpc3_dynrpo_periodic/final_audit.py'] = sha('published repair')
        args[5] = current
        origins = [parent] * 2 + [current] * 8
        for chunk, origin in zip(args[2], origins):
            chunk['identity']['source_sha256'] = copy.deepcopy(origin['source_sha256'])
        return args, origins

    def test_inherited_prefix_binds_original_run_not_repaired_full_source_map(self):
        args, origins = self.inherited_chunk_fixture()
        self.assertNotEqual(origins[0]['source_sha256'], args[5]['source_sha256'])
        audit.check_refit_documents(*args, chunk_runs=origins)
        # Prefix provenance cannot be silently rebound to the newer producer.
        with self.assertRaises(audit.Incomplete):audit.check_refit_documents(*args)
        wrong_origins = [args[5]] + origins[1:]
        with self.assertRaises(audit.Incomplete):audit.check_refit_documents(*args, chunk_runs=wrong_origins)

    def test_inherited_chunk_cannot_change_scientific_identity_or_generation_contract(self):
        for mutation in ('run', 'source', 'sampling', 'scorer', 'missing_origin'):
            args, origins = self.inherited_chunk_fixture()
            if mutation == 'run':
                origins[0] = dict(origins[0], retriever_version=sha('different index'))
            elif mutation == 'source':
                args[2][0]['identity']['source_sha256'] = copy.deepcopy(args[5]['source_sha256'])
            elif mutation == 'sampling':
                contract = args[2][0]['contracts'] = copy.deepcopy(args[2][0]['contracts'])
                contract['sampling_contract']['temperature'] = .5
                contract['sampling_contract_sha'] = audit.document_sha(contract['sampling_contract'])
            elif mutation == 'scorer':
                origin = origins[0] = copy.deepcopy(origins[0])
                origin['source_sha256'][audit.STRICT_SCORER_FILES[0]] = sha('different strict scorer')
                args[2][0]['identity']['source_sha256'] = copy.deepcopy(origin['source_sha256'])
                contract = args[2][0]['contracts'] = copy.deepcopy(args[2][0]['contracts'])
                contract['strict_scoring_contract']['source_sha256'][audit.STRICT_SCORER_FILES[0]] = \
                    origin['source_sha256'][audit.STRICT_SCORER_FILES[0]]
                contract['strict_scoring_contract_sha'] = audit.document_sha(contract['strict_scoring_contract'])
            else:origins.pop()
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_refit_documents(*args, chunk_runs=origins)

    def test_refit_shaped_reward_as_em_and_different_scorer_rejected(self):
        for mutation in ('soft_label', 'wrong_scorer', 'wrong_retriever'):
            args = list(copy.deepcopy(refit_fixture()))
            if mutation == 'soft_label':args[1]['labels'][0]['strict_em'][0] = .5
            if mutation == 'wrong_scorer':
                scoring = args[2][0]['contracts']['strict_scoring_contract']
                scoring['source_sha256'] = dict(scoring['source_sha256'])
                scoring['source_sha256'][audit.STRICT_SCORER_FILES[0]] = sha('foreign scorer')
                args[2][0]['contracts']['strict_scoring_contract_sha'] = audit.document_sha(scoring)
            if mutation == 'wrong_retriever':args[2][0]['contracts']['retriever_contract']['topk'] = 10
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_refit_documents(*args)

    def test_current_student_features_required_not_initial_bank(self):
        questions = [dict(qid='q' + str(i), question_hash=sha(i), content_sha256=sha('c' + str(i)),
            predicted_mixed_probability=.5, selected=i < 512) for i in range(5120)]
        selection = dict(stage='question_probe_selection_frozen', step=26, head_version='v25',
            attempted_candidates=5120, selected_groups=512, prediction_before_label=True,
            post_rollout_em_filter=False, questions=questions,
            online_features=dict(source='current_student_online', student_completed_outer=25,
                student_optimizer_writes=500, questions=5120, feature_layer=27,
                dynamic_tensors_persisted=False, student_gradient_enabled=False))
        audit.check_online_features(selection, 26, 500, 'v25')
        for key, value in (('source', 'initial_fixed_feature_bank'), ('student_completed_outer', 0),
                ('feature_layer', 35), ('student_gradient_enabled', True), ('dynamic_tensors_persisted', True)):
            broken = copy.deepcopy(selection);broken['online_features'][key] = value
            with self.subTest(key=key), self.assertRaises(audit.Incomplete):
                audit.check_online_features(broken, 26, 500, 'v25')

    def test_teacher_full_model_sha_required_in_addition_to_revision(self):
        files = {'file' + str(i):sha(i) for i in range(9)}
        run = dict(initial_model_sha256=files, input_sha256={'train.parquet':sha('train')})
        identity = dict(schema=2, model_revision=audit.REVISION, model_files_sha256=files,
            data_manifest_sha256=audit.DATA_SHA, train_parquet_sha256=sha('train'),
            teacher_frozen_evidence_required=True,
            questions=[dict(qid='q' + str(i), question_hash=sha('q' + str(i))) for i in range(5120)])
        audit.check_teacher_identity(identity, files, run)
        broken = copy.deepcopy(identity);broken['model_files_sha256']['file0'] = sha('teacher checkpoint different')
        with self.assertRaises(audit.Incomplete):audit.check_teacher_identity(broken, files, run)


class EvaluationTests(unittest.TestCase):
    def value(self, final=False):
        count = 51713 if final else 512
        canonical = dict(sha256=sha('final' if final else 'dev'), question_ids=['q' + str(i) for i in range(count)],
            data_sources=['nq'] * count, counts={'nq':count})
        value = dict(run_id='run', step=200 if final else 50, final=final,
            split='final' if final else 'validation', validation_sha256=canonical['sha256'],
            question_ids=list(canonical['question_ids']), data_sources=canonical['data_sources'], counts=canonical['counts'],
            em=[0] * count, f1=[.5] * count, correct=0, weighted_em=0., weighted_f1=.5)
        return value, canonical

    def test_exact512_and51713_ordered_em_f1(self):
        for final in (False, True):
            value, canonical = self.value(final)
            self.assertEqual(audit.check_evaluation(value, canonical, run_id='run', step=value['step'], final=final)['questions'],
                             51713 if final else 512)

    def test_reordered_questions_missing_scores_and_em_as_f1_rejected(self):
        for mutation in ('order', 'missing', 'invalid_f1', 'run'):
            value, canonical = self.value()
            if mutation == 'order':value['question_ids'].reverse()
            if mutation == 'missing':value['f1'].pop()
            if mutation == 'invalid_f1':value['em'][0] = 1;value['f1'][0] = 0
            if mutation == 'run':value['run_id'] = 'foreign'
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_evaluation(value, canonical, run_id='run', step=50)


class RetainedAttemptTests(unittest.TestCase):
    def fixture(self):
        counter, history, actual = 0, [], []
        for index, (outer, delta, complete) in enumerate(((1, 2, False), (1, 20, True),
                                                        (2, 3, False), (2, 20, True)), 1):
            attempt = 'attempt-' + str(index)
            begin = dict(kind='BEGIN', attempt_id=attempt, attempt_index=index, logical_update=outer,
                writes_before=counter, writes_after=counter, actual_optimizer_writes=0,
                opportunities_processed=0, scheduler_advances=0, completed_outer_after=outer - 1)
            end = dict(begin, kind='COMPLETE' if complete else 'ABORTED_ATTEMPT', writes_after=counter + delta,
                actual_optimizer_writes=delta, opportunities_processed=20 if complete else delta,
                scheduler_advances=int(complete), completed_outer_after=outer if complete else outer - 1)
            history.append(begin)
            if complete:
                history.append(dict(end, kind='COMMIT_PREPARED', scheduler_advances=0,
                                    completed_outer_after=outer - 1))
                actual.append(dict(step=outer, attempt_id=attempt, attempt_index=index,
                    optimizer_writes_before=counter, optimizer_writes_after=counter + delta))
            history.append(end);counter += delta
        return actual, dict(cumulative_writes=counter, counter_history=history)

    def test_retained_partial_adam_before_and_between_complete_outers_is_valid(self):
        rows, proof = self.fixture()
        self.assertEqual(rows[0]['optimizer_writes_before'], 2)
        self.assertGreater(rows[1]['optimizer_writes_before'], rows[0]['optimizer_writes_after'])
        self.assertEqual(audit.check_counter_history(rows, proof, 0), 45)

    def test_unexplained_gap_not_hidden_by_backend_cumulative_counter(self):
        rows, proof = self.fixture()
        proof['counter_history'] = [r for r in proof['counter_history'] if r['attempt_index'] != 3]
        with self.assertRaises(audit.Incomplete):audit.check_counter_history(rows, proof, 0)


class RNGCursorAndAllocationTests(unittest.TestCase):
    class UInt32Seeds(list):dtype = 'uint32'

    def rng(self, worker=False):
        result = dict(torch_cpu=b'cpu-state',
            numpy=('MT19937', self.UInt32Seeds(range(624)), 624, 0, 0.), python=random.Random(10).getstate())
        if worker:
            result.update(torch_cuda=b'cuda', actor_cuda_saved=b'actor', rollout_cuda_saved=b'rollout',
                request_generators={'request1':dict(device='cuda:0', state=b'generator')}, cuda_device=0,
                vllm_version='0.6.3', boundary='idle synchronous generation engine')
        return result

    @staticmethod
    def tensor_check(value, *, cpu_generator=False):return isinstance(value, bytes) and bool(value)

    def test_driver_and_four_rank_rng_schema_requires_actual_payloads(self):
        for worker in (False, True):
            audit.check_rng_schema(self.rng(worker), worker=worker, tensor_check=self.tensor_check)
        for mutation in ('cuda', 'cpu', 'numpy', 'python', 'request'):
            value = self.rng(True)
            if mutation == 'cuda':value['rollout_cuda_saved'] = None
            if mutation == 'cpu':value['torch_cpu'] = b''
            if mutation == 'numpy':value['numpy'] = ('MT19937', [], 0, 0, 0.)
            if mutation == 'python':value['python'] = ('missing-real-state',)
            if mutation == 'request':value['request_generators']['request1']['state'] = None
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_rng_schema(value, worker=True, tensor_check=self.tensor_check)

    def test_sampler_permutation_and_compact_pending_offset_are_checked(self):
        cursor = dict(candidate_order=list(range(169028)), candidate_cursor=25800, epoch=0,
            consumed_source_rows=25600, pending_length=512, pending_offset=312,
            pending_indices=list(range(25288, 25800)))
        audit.check_cursor_schema(cursor, source_consumed=25600, outer=5)
        for mutation in ('permutation', 'offset', 'pending_indices', 'consumed'):
            bad = copy.deepcopy(cursor)
            if mutation == 'permutation':bad['candidate_order'][0] = 1
            if mutation == 'offset':bad['pending_offset'] = 512
            if mutation == 'pending_indices':bad['pending_indices'].reverse()
            if mutation == 'consumed':bad['consumed_source_rows'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.check_cursor_schema(bad, source_consumed=25600, outer=5)

    def test_actual_cpu_slurm_allocation_rejects_login_limits_and_foreign_job(self):
        job = dict(JobId='123', JobState='RUNNING', UserId='mou176(' + str(os.getuid()) + ')',
                   AllocTRES='cpu=12,mem=128G,billing=12')
        kwargs = dict(job_id='123', uid=os.getuid(), hostname='ACD1-12', nodes=['ACD1-12'],
                      cgroup_memory=128 * 1024**3)
        self.assertTrue(audit.validate_audit_allocation(job, **kwargs)['accepted'])
        for mutation in ('small_cgroup', 'small_allocation', 'wrong_node', 'pending', 'foreign_uid'):
            value, arguments = copy.deepcopy(job), copy.deepcopy(kwargs)
            if mutation == 'small_cgroup':arguments['cgroup_memory'] = 8 * 1024**3
            if mutation == 'small_allocation':value['AllocTRES'] = 'cpu=12,mem=32G'
            if mutation == 'wrong_node':arguments['hostname'] = 'login'
            if mutation == 'pending':value['JobState'] = 'PENDING'
            if mutation == 'foreign_uid':value['UserId'] = 'foreign(99999999)'
            with self.subTest(mutation=mutation), self.assertRaises(audit.Incomplete):
                audit.validate_audit_allocation(value, **arguments)

    def test_wrong_restore_prefix_or_silent_counter_reset_rejected(self):
        rows, proof = RetainedAttemptTests().fixture()
        with self.assertRaises(audit.Incomplete):audit.check_counter_history(rows, proof, 100)
        proof['counter_history'][0]['writes_before'] = 5
        with self.assertRaises(audit.Incomplete):audit.check_counter_history(rows, proof, 0)


class FileAndParentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.prefix = Path(self.temp.name).resolve() / 'dynrpo'
        self.prefix.mkdir()
        self.reader = audit.Reader(self.prefix)

    def tearDown(self):self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def record(self, path):
        stat = path.stat()
        return dict(path=str(path), inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    full_read_sha256=self.reader.digest(path))

    def run_document(self, base, offset=0):
        value = dict(execution_site='hpc3', training_world_size=4, target_updates=200,
            model_revision=audit.REVISION, question_probe={'head_kind':audit.HEAD_KIND}, run_id='run',
            budget={}, input_sha256={}, initial_model_sha256={}, retriever_version='retriever', source_commit='a' * 40)
        return value

    def test_missing_file_modified_file_and_symlink_rejected(self):
        with self.assertRaises(audit.Incomplete):self.reader.file(self.prefix / 'missing')
        path = self.write(self.prefix / 'evidence.json', {'real':'old'})
        record = self.record(path)
        path.write_text(json.dumps({'real':'new'}))
        with self.assertRaises(audit.Incomplete):self.reader.record(record)
        (self.prefix / 'alias').symlink_to(path)
        with self.assertRaises(audit.Incomplete):self.reader.file(self.prefix / 'alias')

    def test_foreign_namespace_and_wrong_parent_rejected(self):
        foreign = Path(self.temp.name).resolve() / 'foreign';foreign.mkdir()
        with self.assertRaises(audit.Incomplete):self.reader.namespace(foreign)
        child = Path(str(self.prefix) + '-r1');child.mkdir()
        parent_run = self.run_document(self.prefix)
        parent_path = self.write(self.prefix / 'logs/run-manifest.json', parent_run)
        receipt = dict(status=audit.PAIR_STATUS, world_size=4, source_base=str(self.prefix), run_id='run', step=25,
            checkpoint=str(self.prefix / 'cp25'), source_run_manifest=self.record(parent_path))
        receipt_path = self.write(self.prefix / 'pair.json', receipt)
        value = self.run_document(child)
        value['full_state_restore'] = dict(kind='question_probe_pair', completed_outer=25,
            source_receipt=str(receipt_path), source_receipt_sha256=self.reader.digest(receipt_path),
            source_base=str(child), source_checkpoint=receipt['checkpoint'],
            source_run_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'])
        self.write(child / 'logs/run-manifest.json', value)
        class Backend:
            def run(self, base, run):pass
            def pair(self, path):return json.loads(Path(path).read_text())
        auditor = audit.Auditor(child, reader=self.reader, backend=Backend())
        with self.assertRaises(audit.Incomplete):auditor.ancestors()

    def test_inherited_receipt_cycle_rejected_before_recursive_native_loader(self):
        first, second = self.prefix / 'first.json', self.prefix / 'second.json'
        self.write(first, dict(source_base=str(self.prefix), inherited_checkpoint_receipt=str(second)))
        self.write(second, dict(source_base=str(self.prefix), inherited_checkpoint_receipt=str(first)))
        auditor = audit.Auditor(self.prefix, reader=self.reader, backend=object())
        with self.assertRaises(audit.Incomplete) as caught:auditor.read_pair(first)
        self.assertEqual(caught.exception.code, 'PARENT_RECEIPT_CYCLE')

    def test_shallow_legacy_green_does_not_satisfy_any_requirement(self):
        self.write(self.prefix / 'logs/full-run-acceptance.json', dict(structural_accepted=True, actual_outer_updates=200))
        auditor = audit.Auditor(self.prefix, reader=self.reader, backend=object())
        result = auditor.execute()
        self.assertFalse(result['accepted'])
        self.assertFalse(result['structural_accepted'])
        self.assertEqual(len(result['incomplete']), 8)

    def test_cp200_does_not_fabricate_missing_raw200_row(self):
        rows = [{'step':i} for i in range(1, 200)]
        self.write(self.prefix / 'logs/actual-updates.jsonl', {})
        (self.prefix / 'logs/actual-updates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.write(self.prefix / 'checkpoints/cp200/complete.json', dict(step=200, finite_all_saved_tensors=True))
        auditor = audit.Auditor(self.prefix, reader=self.reader, backend=object())
        auditor.chain = [dict(base=self.prefix, run={}, first=1, last=200)]
        with self.assertRaises(audit.Incomplete) as caught:auditor.train_segments()
        self.assertEqual(caught.exception.code, 'RAW_OUTER200_ROW_MISSING_CP200_IS_NOT_A_REPLACEMENT')
        self.assertEqual(len(self.reader.jsonl(self.prefix / 'logs/actual-updates.jsonl')), 199)

    def test_hash_byte_and_time_budget_fail_closed(self):
        path = self.prefix / 'large';path.write_bytes(b'abcd')
        limited = audit.Reader(self.prefix, max_hash_bytes=3)
        with self.assertRaises(audit.Incomplete):limited.digest(path)
        expired = audit.Reader(self.prefix, max_seconds=-1)
        with self.assertRaises(audit.Incomplete):expired.file(path)

    def test_actual_checkpoint_owner_layouts_not_parent_index(self):
        for suffix in ('checkpoints/actor/global_step_25', 'checkpoints/search-g1-common8/actor/global_step_200'):
            for base in (self.prefix, Path(str(self.prefix) + '-r3')):
                with self.subTest(base=base, suffix=suffix):
                    self.assertEqual(audit.checkpoint_namespace(base / suffix, self.prefix), base)
        for suffix in ('checkpoints/actor/global_step_201', 'foreign/actor/global_step_25',
                       'checkpoints/actor/global_step_0'):
            with self.subTest(suffix=suffix), self.assertRaises(audit.Incomplete):
                audit.checkpoint_namespace(self.prefix / suffix, self.prefix)

    def test_only_exact_bootstrap_shared_data_alias_allowed(self):
        canonical = Path(self.temp.name).resolve() / 'canonical/train.parquet'
        canonical.parent.mkdir();canonical.write_bytes(b'canonical shared input')
        alias = self.prefix / 'data/train.parquet';alias.parent.mkdir();alias.symlink_to(canonical)
        self.assertEqual(self.reader.shared_input(alias, canonical), canonical)
        self.assertEqual(self.reader.digest(self.reader.shared_input(alias, canonical)),
                         hashlib.sha256(canonical.read_bytes()).hexdigest())
        with self.assertRaises(audit.Incomplete):self.reader.file(alias)
        foreign = Path(self.temp.name).resolve() / 'foreign/train.parquet'
        foreign.parent.mkdir();foreign.write_bytes(canonical.read_bytes())
        alias.unlink();alias.symlink_to(foreign)
        with self.assertRaises(audit.Incomplete):self.reader.shared_input(alias, canonical)

    def test_authenticated_refit_parent_prefix_and_current_suffix(self):
        child = Path(str(self.prefix) + '-r1');child.mkdir()
        parent_root = self.prefix / 'artifacts/question-probe/periodic-refits/25'
        current_root = child / 'artifacts/question-probe/periodic-refits/25'
        labels = [dict(qid='q' + str(i)) for i in range(5120)]
        current = dict(outer=25, pre_receipt_sha256=sha('accepted-pre'), labels=labels,
            indices=list(range(5120)), frozen_before=[], feature_hashes={r['qid']:sha(r['qid']) for r in labels})
        parent = dict(current, labels=labels[:1024], feature_hashes=None)
        saved = self.write(parent_root / 'latest-state.pt', dict(fixture='scalar state placeholder'))
        self.write(parent_root / 'latest-state.json', dict(sha256=self.reader.digest(saved)))
        for number in range(10):
            root = parent_root if number < 2 else current_root
            self.write(root / ('chunk-' + str(number)) / 'real-rollout-audit.json', dict(chunk=number))
        class Backend:
            def phase_state(self, path):return copy.deepcopy(parent)
        chain = [dict(base=self.prefix, first=1, last=25), dict(base=child, first=26, last=200)]
        chunks, files = audit.resolve_refit_chunks(self.reader, Backend(), chain,
            current_root, 25, current, sha('accepted-pre'))
        self.assertEqual([v['chunk'] for v in chunks], list(range(10)))
        self.assertTrue(files[0]['path'].startswith(str(parent_root)))
        self.assertTrue(files[-1]['path'].startswith(str(current_root)))
        self.assertEqual(files[0]['source_base'], str(self.prefix))
        self.assertEqual(files[-1]['source_base'], str(child))
        parent['pre_receipt_sha256'] = sha('different-student-pre')
        with self.assertRaises(audit.Incomplete):
            audit.resolve_refit_chunks(self.reader, Backend(), chain, current_root, 25, current, sha('accepted-pre'))

    def test_unrelated_directory_is_not_a_refit_chunk_fallback(self):
        current_root = self.prefix / 'artifacts/question-probe/periodic-refits/25'
        foreign = Path(self.temp.name).resolve() / 'foreign/periodic-refits/25/chunk-0/real-rollout-audit.json'
        self.write(foreign, dict(chunk=0))
        with self.assertRaises(audit.Incomplete):
            audit.resolve_refit_chunks(self.reader, object(), [dict(base=self.prefix, first=1, last=200)],
                current_root, 25, {}, sha('accepted-pre'))


if __name__ == '__main__':unittest.main()
