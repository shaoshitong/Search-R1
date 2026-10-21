"""CPU scalar/receipt fixtures use published producer schemas, never model loading.

These opaque fixtures have no generated selected traces. Their ``verify``
helper explicitly isolates selection verification; separate integration and
selection-evidence tests exercise that added final-consumer requirement.
"""
import builtins
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import final_cpu_acceptance as core

PUBLISHED_MICRO16_COMMIT = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
OPAQUE_SELECTION_FIXTURE_SCOPE = dict(
    fixture_scope='selection verification mocked: scalar/archive fixtures contain no selected traces')


class FinalAcceptanceTests(unittest.TestCase):
    def put(self, relative, value):
        path = self.base/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else core.encoded(value))
        return path

    def ledger(self, relative, rows, suffix=b''):
        return self.put(relative, b''.join(json.dumps(r, allow_nan=False).encode()+b'\n' for r in rows)+suffix)

    def fixture(self, directory, *, training_commit=None, physical=32):
        self.base = Path(directory).resolve(); self.run_id = 'a'*32; self.job = '730080'
        self.commit = training_commit or core.REVIEWED_COMMIT
        self.run = dict(run_id=self.run_id, source_commit=self.commit,
            easy2560_selection=True, training_world_size=4, data_selection_mode='mixed_first',
            actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4,
            all_correct_virtual_zero=False, checkpoint_dir=str(self.base/'checkpoints/search-g1-common8'))
        if self.commit != core.REVIEWED_COMMIT:
            code = self.put('repo/reviewed.py', b'fixture published source')
            self.run['source_sha256'] = {'reviewed.py':core.digest(code.read_bytes())}
            published = dict(source_commit=self.commit, fresh_source_all_files_sha_verified=True,
                files={'reviewed.py':dict(bytes=code.stat().st_size, sha256=core.digest(code.read_bytes()))})
            self.put('source/'+self.commit+'/source-publication.json', published)
            self.put('preparation/run-source.json', dict(source_commit=self.commit,
                files=published['files'], runtime_repo=str(self.base/'repo')))
        self.run_path = self.put('logs/run-manifest.json', self.run)
        self.put('logs/native-allocation-admission.json', dict(run_id=self.run_id, accepted=True, slurm_job_id=self.job))
        self.identities = {}
        fixed_sources = ['nq']*256 + ['hotpotqa']*256
        # Legitimate duplicated normalized IDs are retained as occurrences.
        self.identities['validation'] = dict(rows=512, sha256='b'*64,
            question_ids=['same-fixed-question']*512, data_sources=fixed_sources,
            source_counts=dict(nq=256, hotpotqa=256))
        final_counts = dict(nq=3610, triviaqa=10000, popqa=10000, hotpotqa=10000,
                            musique=6000, bamboogle=100, two_wiki=12003)
        final_sources = [source for source, count in final_counts.items() for _ in range(count)]
        self.assertEqual(len(final_sources), 51713)
        self.identities['final'] = dict(rows=51713, sha256='c'*64,
            question_ids=['same-final-question']*51713, data_sources=final_sources, source_counts=final_counts)
        self.validations = []
        for step in core.STEPS:
            self.validations.append(self.measure('validation', step))
        self.final = self.measure('final', 200)
        self.validations.append(self.final)
        self.ledger('logs/validation-identities.jsonl', self.validations)
        self.put('logs/final-identity.json', self.final)
        fixed5 = self.validations[0]
        self.after = dict(role='formal_inline_development_evaluation', step=5, run_id=self.run_id,
            validation_sha256=self.identities['validation']['sha256'], question_ids=fixed5['question_ids'],
            data_sources=fixed5['data_sources'], weighted_em=fixed5['weighted_em'], weighted_f1=fixed5['weighted_f1'],
            rows=[dict(question_id=qid, data_source=source, em=em, f1=f1)
                  for qid, source, em, f1 in zip(fixed5['question_ids'], fixed5['data_sources'], fixed5['em'], fixed5['f1'])])
        self.put('logs/formal-inline-development-after.json', self.after)
        self.updates = []; self.attempts = []; self.schedulers = [[] for _ in range(4)]; self.workers = [[] for _ in range(4)]
        for step in range(1, 201):
            aid = f'{step:032x}'
            attempt = dict(run_id=self.run_id, slurm_job_id=self.job, attempt_id=aid, attempt_index=step,
                logical_update=step, completed_outer=step, complete_outer=True, scheduler_advances=1,
                all_ranks_agreed=True, real_rank_count=4, optimizer_writes_before=step-1,
                optimizer_writes_after=step, optimizer_writes_this_attempt=1, opportunities_processed=1,
                cumulative_aborted_attempts=0)
            self.attempts.append(attempt)
            self.updates.append(dict(run_id=self.run_id, step=step, attempt_id=aid, attempt_index=step,
                optimizer_writes_before=step-1, optimizer_writes_after=step, actual_optimizer_writes=1,
                probe_version=f'{self.run_id}:{((step-1)//50)*50}:probe'))
            for rank in range(4):
                row = dict(run_id=self.run_id, slurm_job_id=self.job, world_size=4, rank=rank,
                    logical_update=step, attempt_id=aid, attempt_index=step, writes_after=step,
                    scheduler_last_epoch_after=step, scheduler_advances=1)
                self.schedulers[rank] += [dict(row, kind=kind) for kind in ('BEGIN', 'COMMIT_PREPARED', 'COMPLETE')]
                context = dict(logical_update=step, attempt_id=aid, attempt_index=step)
                common = dict(run_id=self.run_id, rank=rank, context=context)
                self.workers[rank] += [dict(common, kind='OPTIMIZER_WRITE', write_count=step), dict(common, kind='OPPORTUNITY'),
                    dict(run_id=self.run_id, rank=rank, kind='OUTER_BUDGET', logical_update=step,
                         attempt_id=aid, attempt_index=step, cumulative_writes=step)]
        self.ledger('logs/actual-updates.jsonl', self.updates)
        self.ledger('logs/actual-attempts.jsonl', self.attempts)
        for rank in range(4):
            self.ledger(f'logs/scheduler-rank-{rank}.jsonl', self.schedulers[rank])
            self.ledger(f'logs/optimizer-rank-{rank}.jsonl', self.workers[rank])
        for step in core.STEPS:
            self.make_checkpoint(step)

    def measure(self, split, step):
        identity = self.identities[split]; count = identity['rows']
        em = [float(source == 'nq') for source in identity['data_sources']]
        f1 = [1. if correct else .25 for correct in em]
        return dict(run_id=self.run_id, split=split, step=step, final=split == 'final',
            question_ids=identity['question_ids'], data_sources=identity['data_sources'], em=em, f1=f1,
            counts=identity['source_counts'], correct=sum(em), weighted_em=sum(em)/count,
            weighted_f1=math.fsum(f1)/count, validation_sha256=identity['sha256'])

    def load_canonical(self, base, split):
        return copy.deepcopy(self.identities[split])

    def make_checkpoint(self, step):
        relative = f'checkpoints/search-g1-common8/actor/global_step_{step}'
        cp = self.base/relative; proofs = []
        for rank in range(4):
            path = self.put(relative+f'/training_state/rank-{rank}.pt', b'opaque optimizer/RNG fixture')
            proofs.append(dict(rank=rank, run_id=self.run_id, sha256=core.digest(path.read_bytes())))
        self.put(relative+'/training_state/driver.pt', b'opaque saved driver/RNG fixture')
        self.put(relative+'/model-00001-of-00003.safetensors', b'opaque actor shard fixture1')
        self.put(relative+'/model-00002-of-00003.safetensors', b'opaque actor shard fixture2')
        self.put(relative+'/model-00003-of-00003.safetensors', b'opaque actor shard fixture3')
        state = dict(completed_outer=step, question_probe=None, worker_files=proofs)
        self.put(relative+'/training_state/manifest.json', state)
        hashes = {str(p.relative_to(cp)): core.digest(p.read_bytes()) for p in cp.rglob('*') if p.is_file()}
        complete = dict(run_id=self.run_id, step=step, checkpoint=str(cp), finite_all_saved_tensors=True, files_sha256=hashes)
        self.put(relative+'/complete.json', complete)
        files = {'checkpoint/'+str(p.relative_to(cp)):dict(bytes=p.stat().st_size, sha256=core.digest(p.read_bytes()), path=str(p))
                 for p in cp.rglob('*') if p.is_file()}
        files['metadata/run-manifest.json'] = dict(bytes=self.run_path.stat().st_size,
            sha256=core.digest(self.run_path.read_bytes()), path=str(self.run_path))
        for name, sha in self.run.get('source_sha256', {}).items():
            p = self.base/'repo'/name
            files['source/'+name] = dict(path=str(p), bytes=p.stat().st_size, sha256=sha)
        snapshot = dict(base=str(self.base), source_commit=self.run['source_commit'], run_id=self.run_id,
            step=step, training_world_size=4, question_probe_saved=False, files=files,
            completed_update=self.updates[step-1], total_bytes=sum(row['bytes'] for row in files.values()))
        folder = f'preparation/checkpoint-archives/global_step_{step}'
        snap_path = self.put(folder+'/source-snapshot.json', snapshot)
        accepted = dict(status=core.ARCHIVE_STATUS, actual_cuda_restore_accepted=False,
            fresh_destination_sha_verified=True, destination_fsynced=True, fresh_source_after_archive_sha_verified=True,
            snapshot_sha256=core.digest(snap_path.read_bytes()), file_count=len(files),
            **{key:snapshot[key] for key in ('base', 'run_id', 'source_commit', 'step', 'total_bytes')})
        self.put(folder+'/complete.json', accepted)

    def verify(self, **kwargs):
        with patch.object(core, 'selection_evidence', return_value=OPAQUE_SELECTION_FIXTURE_SCOPE):
            return core.verify_final(self.base, load_canonical=self.load_canonical, **kwargs)

    def test_optional_restore_proof_is_consumed_without_starting_gpu_or_marking_goal_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            calls=[]
            def accepted(path,**kwargs):
                calls.append((path,kwargs))
                self.assertEqual(kwargs['checkpoint_summary']['step'],200)
                self.assertEqual(kwargs['progress']['cumulative_actual_optimizer_writes_per_rank'],200)
                return {'actual_cuda_restore_accepted':True,'actual_restore_performed_by_this_cpu_reader':False}
            stub=SimpleNamespace(verify_actual_final_restore=accepted)
            with patch.dict('sys.modules',{'reproduction.easy2560_joint4.final_restore_acceptance':stub}):
                report=self.verify(final_restore_proof='/fixture/completed-gpu-proof.json')
            self.assertEqual(len(calls),1)
            self.assertTrue(report['actual_cuda_restore_accepted'])
            self.assertFalse(report['actual_restore_performed'])
            self.assertEqual((report['gpu_allocations'],report['model_reloads'],report['new_training_updates']),(0,0,0))
            self.assertFalse(report['overall_goal_complete'])

    def test_native_full51713_preserves_repeated_ids_and_outputs_both_macro_scores(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            original_import = builtins.__import__
            def no_model_import(name, *args, **kwargs):
                self.assertNotIn(name.split('.')[0], ('torch', 'ray', 'vllm', 'transformers'))
                return original_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=no_model_import):
                result = self.verify()
            final = result['evaluations']['final51713']
            self.assertEqual(final['question_count'], 51713)
            self.assertEqual(len(set(self.final['question_ids'])), 1)
            self.assertFalse(final['deduplicated'])
            self.assertAlmostEqual(final['macro_em'], 1/7)
            self.assertAlmostEqual(final['macro_f1'], (1+6*.25)/7)
            self.assertNotEqual(final['weighted_em'], final['macro_em'])
            self.assertEqual(set(result['evaluations']['fixed512']), {'5','50','100','150','200'})
            self.assertTrue(result['progress']['no_update201'])
            self.assertFalse(result['actual_cuda_restore_accepted']); self.assertFalse(result['overall_goal_complete'])
            self.assertEqual(result['gpu_allocations'], 0); self.assertEqual(result['model_reloads'], 0)

    def test_micro16_source_requires_explicit_commit_and_published_closed_maps(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
            for requested in (None, self.commit[:7], 'f'*40):
                with self.subTest(requested=requested), self.assertRaises(ValueError):
                    self.verify(**({} if requested is None else dict(training_commit=requested)))
            result = self.verify(training_commit=self.commit)
            self.assertTrue(result['cpu_final_accepted'])
            self.assertEqual(result['source_commit'], self.commit)
            binding = result['source_publication_binding']
            self.assertEqual(binding['policy'], 'explicit_caller_commit_publication_runtime_run_map')
            self.assertEqual(binding['requested_training_commit'], self.commit)
            self.assertFalse(binding['automatic_head_or_ancestor_allow'])
            self.assertFalse(binding['git_ancestry_verified_here'])
            self.assertFalse(result['actual_restore_performed'])
            self.assertFalse(result['actual_cuda_restore_accepted'])

    def test_explicit_source_binding_rejects_missing_or_different_publication_and_maps(self):
        for broken in ('missing_publication', 'not_fresh', 'wrong_commit', 'runtime_files',
                       'runtime_repo', 'run_map', 'archive_map'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as td:
                self.fixture(td, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
                publication_path = self.base/'source'/self.commit/'source-publication.json'
                publication = json.loads(publication_path.read_bytes())
                runtime_path = self.base/'preparation/run-source.json'
                runtime = json.loads(runtime_path.read_bytes())
                if broken == 'missing_publication': publication_path.unlink()
                elif broken in ('not_fresh', 'wrong_commit'):
                    if broken == 'not_fresh': publication['fresh_source_all_files_sha_verified'] = False
                    else: publication['source_commit'] = core.REVIEWED_COMMIT
                    publication_path.write_bytes(core.encoded(publication))
                elif broken in ('runtime_files', 'runtime_repo'):
                    if broken == 'runtime_files': runtime['files']['reviewed.py']['sha256'] = '0'*64
                    else: runtime['runtime_repo'] = '/another-run/repo'
                    runtime_path.write_bytes(core.encoded(runtime))
                elif broken == 'run_map':
                    self.run['source_sha256']['reviewed.py'] = '0'*64
                    self.put('logs/run-manifest.json', self.run)
                else:
                    path = self.base/'preparation/checkpoint-archives/global_step_50/source-snapshot.json'
                    snapshot = json.loads(path.read_bytes())
                    snapshot['files']['source/reviewed.py']['sha256'] = '0'*64
                    path.write_bytes(core.encoded(snapshot))
                    receipt_path = path.parent/'complete.json'
                    receipt = json.loads(receipt_path.read_bytes())
                    receipt['snapshot_sha256'] = core.digest(path.read_bytes())
                    receipt_path.write_bytes(core.encoded(receipt))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.verify(training_commit=self.commit)

    def test_new_source_retains_early_missing200_and_fresh_only_checks(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
            self.ledger('logs/actual-updates.jsonl', self.updates[:-1])
            with patch.object(core, 'published_source_binding', side_effect=AssertionError('must fail early')):
                with self.assertRaisesRegex(ValueError, 'MISSING200'):
                    self.verify(training_commit=self.commit)
            self.run['full_state_restore'] = {'completed_outer': 5, 'old_job': '730080'}
            self.put('logs/run-manifest.json', self.run)
            with self.assertRaisesRegex(ValueError, 'CROSS_JOB_AUTHENTICATED_CHAIN_NOT_IMPLEMENTED'):
                self.verify(training_commit=self.commit)

    def test_missing200_is_rejected_before_any_evaluation_or_checkpoint_reader(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td); self.ledger('logs/actual-updates.jsonl', self.updates[:-1])
            with patch.object(core, 'evaluations', side_effect=AssertionError('must fail early')):
                with self.assertRaisesRegex(ValueError, 'MISSING200'):
                    self.verify()

    def test_each_fixed_milestone_is_required_once_and_stage64_cannot_replace_it(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for step in core.STEPS:
                with self.subTest(step=step):
                    changed = [dict(row, split='stage') if row['step'] == step and not row['final'] else row for row in self.validations]
                    self.ledger('logs/validation-identities.jsonl', changed)
                    with self.assertRaisesRegex(ValueError, 'MISSING_OR_DUPLICATE_FIXED512_'+str(step)):
                        core.evaluations(self.base, self.run, self.load_canonical)
            self.ledger('logs/validation-identities.jsonl', self.validations+[self.validations[0]])
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_FIXED512_5'):
                core.evaluations(self.base, self.run, self.load_canonical)

    def test_four_gpu_padding_or_uid_dedup_cannot_pass_native_final(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for size in (1, 51712, 51716, 51720):
                with self.subTest(size=size):
                    bad = copy.deepcopy(self.final)
                    for key in ('question_ids', 'data_sources', 'em', 'f1'):
                        bad[key] = (bad[key]+bad[key][:7])[:size]
                    self.put('logs/final-identity.json', bad)
                    self.ledger('logs/validation-identities.jsonl', self.validations[:-1]+[bad])
                    with self.assertRaisesRegex(ValueError, 'OCCURRENCES_DIFFER|SCORE_COUNT_DIFFERS'):
                        core.evaluations(self.base, self.run, self.load_canonical)

    def test_ordered_sources_and_nonfinite_or_inconsistent_f1_fail(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for variant in ('source_order', 'f1', 'macro_cannot_hide_weighted'):
                with self.subTest(variant=variant):
                    bad = copy.deepcopy(self.final)
                    if variant == 'source_order':
                        bad['data_sources'][0], bad['data_sources'][-1] = bad['data_sources'][-1], bad['data_sources'][0]
                    elif variant == 'f1': bad['f1'][0] = .5
                    else: bad['weighted_f1'] = 0.
                    with self.assertRaises(ValueError):
                        core.scores(bad, self.identities['final'], self.run, 200, split='final', final=True)
            with self.assertRaisesRegex(ValueError, 'NONFINITE_JSON'):
                core.decode(b'{"f1":NaN}')

    def test_fixed5_scalar_artifact_is_cross_checked(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td); self.after['rows'][0]['em'] = 0
            self.put('logs/formal-inline-development-after.json', self.after)
            with self.assertRaisesRegex(ValueError, 'FIXED5_SCALAR_ARTIFACT_SCORES_DIFFER'):
                core.evaluations(self.base, self.run, self.load_canonical)

    def test_cp1_without_historical_qa_uses_current_formal5_with_all_score_checks(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            chain = dict(boundary=1, historical={})
            result = core.evaluations(self.base, self.run, self.load_canonical, chain=chain)
            self.assertEqual(set(result['fixed512']), {'5', '50', '100', '150', '200'})
            self.assertEqual(result['final51713']['question_count'], 51713)
            self.assertEqual(result['fixed5_artifact']['path'], str(self.base/'logs/formal-inline-development-after.json'))
            self.assertNotIn('authenticated_historical', result['fixed5_artifact'])
            for change, expected in (('identity', 'FIXED5_SCALAR_ARTIFACT_IDENTITY_DIFFERS'),
                                     ('score', 'FIXED5_SCALAR_ARTIFACT_SCORES_DIFFER')):
                with self.subTest(change=change):
                    bad = copy.deepcopy(self.after)
                    if change == 'identity': bad['run_id'] = 'foreign-run'
                    else: bad['rows'][0]['em'] = 0
                    self.put('logs/formal-inline-development-after.json', bad)
                    with self.assertRaisesRegex(ValueError, expected):
                        core.evaluations(self.base, self.run, self.load_canonical, chain=chain)
            self.put('logs/formal-inline-development-after.json', self.after)
            for rows in (self.validations[1:], [self.validations[0]] + self.validations):
                self.ledger('logs/validation-identities.jsonl', rows)
                with self.assertRaisesRegex(ValueError, 'MISSING_OR_DUPLICATE_FIXED512_5'):
                    core.evaluations(self.base, self.run, self.load_canonical, chain=chain)

    def test_cp5_still_requires_historical_validation_and_uses_historical_qa5(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            self.ledger('logs/validation-identities.jsonl', self.validations[1:])
            chain = dict(boundary=5, cutoff_attempt=5, source_base=self.base/'old',
                source_run=self.run, source_job='old-job', historical={
                    'logs/validation-identities.jsonl': json.dumps(self.validations[0]).encode()+b'\n',
                    'logs/formal-inline-development-after.json': core.encoded(self.after)})
            # A bad current QA5 cannot replace the authenticated historical one.
            self.put('logs/formal-inline-development-after.json', dict(self.after, run_id='foreign-current'))
            result = core.evaluations(self.base, self.run, self.load_canonical, chain=chain)
            self.assertTrue(result['fixed5_artifact']['authenticated_historical'])
            self.assertEqual(result['fixed5_artifact']['path'], str(chain['source_base']/'logs/formal-inline-development-after.json'))
            self.ledger('logs/validation-identities.jsonl', self.validations)
            with self.assertRaisesRegex(ValueError, 'MISSING_OR_DUPLICATE_FIXED512_5'):
                core.evaluations(self.base, self.run, self.load_canonical, chain=chain)
            self.ledger('logs/validation-identities.jsonl', self.validations[1:])
            missing = dict(chain, historical={'logs/formal-inline-development-after.json': core.encoded(self.after)})
            with self.assertRaises(KeyError):
                core.evaluations(self.base, self.run, self.load_canonical, chain=missing)

    def test_extra201_driver_scheduler_worker_or_completed_row_fails(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            changes = [('logs/actual-updates.jsonl', self.updates, dict(self.updates[-1], step=201)),
                       ('logs/actual-attempts.jsonl', self.attempts, dict(self.attempts[-1], logical_update=201,
                            attempt_id=f'{201:032x}', attempt_index=201)),
                       ('logs/scheduler-rank-3.jsonl', self.schedulers[3], dict(self.schedulers[3][-1], logical_update=201,
                            attempt_id=f'{201:032x}', attempt_index=201)),
                       ('logs/optimizer-rank-2.jsonl', self.workers[2], dict(self.workers[2][-1], logical_update=201,
                            attempt_id=f'{201:032x}', attempt_index=201))]
            for name, rows, extra in changes:
                with self.subTest(name=name):
                    self.ledger(name, rows+[extra])
                    with self.assertRaisesRegex(ValueError, 'EXTRA_COMPLETED|POST200'):
                        core.no201(self.base, self.run)
                    self.ledger(name, rows)

    def test_partial_final_ledgers_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            cases = [('logs/actual-updates.jsonl', self.updates), ('logs/actual-attempts.jsonl', self.attempts),
                     ('logs/optimizer-rank-0.jsonl', self.workers[0]), ('logs/scheduler-rank-1.jsonl', self.schedulers[1])]
            for name, rows in cases:
                with self.subTest(name=name):
                    self.ledger(name, rows, b'{"partial":')
                    with self.assertRaisesRegex(ValueError, 'PARTIAL_OR_OVERSIZED'):
                        core.no201(self.base, self.run)
                    self.ledger(name, rows)

    def test_worker_write201_counter_cannot_hide_under_logical200(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            bad = copy.deepcopy(self.workers[0])
            next(row for row in reversed(bad) if row['kind'] == 'OPTIMIZER_WRITE')['write_count'] = 201
            self.ledger('logs/optimizer-rank-0.jsonl', bad)
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_WORKER_ATTEMPT'):
                core.no201(self.base, self.run)

    def test_real_guard_retry_before200_is_allowed_without_inventing_an_update(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for update in self.updates:
                update['attempt_index'] += 1; update['attempt_id'] = f'{update["attempt_index"]:032x}'
            for attempt in self.attempts:
                attempt['attempt_index'] += 1; attempt['attempt_id'] = f'{attempt["attempt_index"]:032x}'
                attempt['cumulative_aborted_attempts'] = 1
            retry = dict(self.attempts[0], attempt_id=f'{1:032x}', attempt_index=1,
                completed_outer=0, complete_outer=False, scheduler_advances=0,
                optimizer_writes_before=0, optimizer_writes_after=0, optimizer_writes_this_attempt=0)
            self.ledger('logs/actual-updates.jsonl', self.updates)
            self.ledger('logs/actual-attempts.jsonl', [retry]+self.attempts)
            for rank in range(4):
                for row in self.schedulers[rank]+self.workers[rank]:
                    if 'attempt_index' not in row:
                        row['context'] = dict(row['context'])
                    context = row if 'attempt_index' in row else row['context']
                    context['attempt_index'] += 1; context['attempt_id'] = f'{context["attempt_index"]:032x}'
                sched = dict(self.schedulers[rank][0], attempt_id=f'{1:032x}', attempt_index=1,
                    writes_after=0, scheduler_last_epoch_after=0, scheduler_advances=0)
                self.ledger(f'logs/scheduler-rank-{rank}.jsonl',
                    [dict(sched, kind='BEGIN'), dict(sched, kind='ABORTED_ATTEMPT')]+self.schedulers[rank])
                context = dict(attempt_id=f'{1:032x}', attempt_index=1, logical_update=1)
                retry_worker = [dict(run_id=self.run_id, rank=rank, context=context, kind='OPPORTUNITY'),
                    dict(run_id=self.run_id, rank=rank, logical_update=1, attempt_id=f'{1:032x}',
                         attempt_index=1, kind='ABORTED_ATTEMPT', cumulative_writes=0)]
                self.ledger(f'logs/optimizer-rank-{rank}.jsonl', retry_worker+self.workers[rank])
            result = core.no201(self.base, self.run)
            self.assertEqual(result['actual_completed_updates'], 200)
            self.assertEqual(result['actual_attempt_count'], 201)
            self.assertTrue(result['no_update201'])

    def test_accepted_archive_metadata_suffices_after_all_hpc_tensors_are_gone(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td); archive_root = self.base/'remote-HK-archive'
            for step in core.STEPS:
                cp = self.base/f'checkpoints/search-g1-common8/actor/global_step_{step}'
                target = archive_root/f'global_step_{step}'
                (target/'checkpoint/training_state').mkdir(parents=True)
                shutil.copyfile(cp/'complete.json', target/'checkpoint/complete.json')
                shutil.copyfile(cp/'training_state/manifest.json', target/'checkpoint/training_state/manifest.json')
                shutil.rmtree(cp)
            result = self.verify(archive_root=archive_root)
            self.assertTrue(result['cpu_final_accepted'])
            for proof in result['checkpoints'].values():
                self.assertFalse(proof['local_tensor_presence_required'])
                self.assertFalse(proof['checkpoint_large_files_rehashed_by_this_consumer'])

    def test_real_sealed_sidecar_verify_supplies_retired_source_metadata(self):
        from reproduction.easy2560_joint4 import checkpoint_evidence as evidence
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for step in core.STEPS:
                cp = self.base/f'checkpoints/search-g1-common8/actor/global_step_{step}'
                source = self.base/f'preparation/checkpoint-archives/global_step_{step}'
                folder = f'preparation/checkpoint-evidence/global_step_{step}'
                copies = {'identity/source-snapshot.json': source/'source-snapshot.json',
                          'identity/accepted-archive.json': source/'complete.json',
                          'identity/checkpoint-complete.json': cp/'complete.json',
                          'identity/training-state-manifest.json': cp/'training_state/manifest.json'}
                files = {}
                for name, path in copies.items():
                    target = self.put(folder+'/'+name, path.read_bytes())
                    files[name] = dict(bytes=target.stat().st_size, sha256=core.digest(target.read_bytes()))
                accepted = json.loads((source/'complete.json').read_bytes())
                core_archive = dict(status=core.ARCHIVE_STATUS,
                    source_snapshot=dict(relative_path='identity/source-snapshot.json', sha256=files['identity/source-snapshot.json']['sha256']),
                    source_acceptance=dict(relative_path='identity/accepted-archive.json', sha256=files['identity/accepted-archive.json']['sha256']))
                identity = dict(source_commit=core.REVIEWED_COMMIT, run_id=self.run_id, source_job=self.job, step=step)
                manifest = dict(identity, role='easy2560_checkpoint_evidence_supplement',
                    actual_cuda_restore_accepted=False, post_callback_rng_captured=False, single_original_job_prefix=True,
                    inherited_history_assembled=False, source_base=str(self.base), source_checkpoint=str(cp),
                    files=files, core_archive=core_archive, scope='synthetic small receipt fixture, no recovery claim')
                path = self.put(folder+'/supplement.json', manifest)
                self.put(folder+'/complete.json', dict(identity, status=evidence.STATUS,
                    supplement_sha256=core.digest(path.read_bytes()), actual_cuda_restore_accepted=False, post_callback_rng_captured=False))
                shutil.rmtree(cp); shutil.rmtree(source)
            result = self.verify()
            self.assertTrue(result['cpu_final_accepted'])
            self.assertTrue(all(row['sealed_supplement']['single_original_job_prefix'] for row in result['checkpoints'].values()))
            self.assertFalse(result['actual_restore_performed'])

    def test_explicit_published_source_survives_retirement_via_verified_sealed_evidence(self):
        from reproduction.easy2560_joint4 import checkpoint_evidence as evidence
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
            publication = self.base/'source'/self.commit/'source-publication.json'
            runtime = self.base/'preparation/run-source.json'
            for step in core.STEPS:
                cp = self.base/f'checkpoints/search-g1-common8/actor/global_step_{step}'
                source = self.base/f'preparation/checkpoint-archives/global_step_{step}'
                folder = f'preparation/checkpoint-evidence/global_step_{step}'
                copies = {'identity/source-snapshot.json': source/'source-snapshot.json',
                          'identity/accepted-archive.json': source/'complete.json',
                          'identity/checkpoint-complete.json': cp/'complete.json',
                          'identity/training-state-manifest.json': cp/'training_state/manifest.json',
                          'identity/run-manifest.json': self.run_path,
                          'identity/source-publication.json': publication,
                          'identity/run-source.json': runtime}
                files = {}
                for name, path in copies.items():
                    target = self.put(folder+'/'+name, path.read_bytes())
                    files[name] = dict(bytes=target.stat().st_size, sha256=core.digest(target.read_bytes()))
                source_binding = evidence.published_source_binding(self.base, self.run, self.commit,
                    json.loads(publication.read_bytes()), json.loads(runtime.read_bytes()),
                    json.loads((source/'source-snapshot.json').read_bytes()))
                core_archive = dict(status=core.ARCHIVE_STATUS,
                    source_snapshot=dict(relative_path='identity/source-snapshot.json', sha256=files['identity/source-snapshot.json']['sha256']),
                    source_acceptance=dict(relative_path='identity/accepted-archive.json', sha256=files['identity/accepted-archive.json']['sha256']))
                identity = dict(source_commit=self.commit, run_id=self.run_id, source_job=self.job, step=step)
                manifest = dict(identity, role='easy2560_checkpoint_evidence_supplement',
                    actual_cuda_restore_accepted=False, post_callback_rng_captured=False, single_original_job_prefix=True,
                    inherited_history_assembled=False, source_base=str(self.base), source_checkpoint=str(cp),
                    source_publication_binding=source_binding, files=files, core_archive=core_archive,
                    scope='synthetic sealed published source fixture, no recovery claim')
                path = self.put(folder+'/supplement.json', manifest)
                self.put(folder+'/complete.json', dict(identity, status=evidence.STATUS,
                    supplement_sha256=core.digest(path.read_bytes()), actual_cuda_restore_accepted=False, post_callback_rng_captured=False))
                shutil.rmtree(cp); shutil.rmtree(source)
            publication.unlink(); runtime.unlink()
            result = self.verify(training_commit=self.commit)
            self.assertTrue(result['cpu_final_accepted'])
            self.assertIn('/checkpoint-evidence/global_step_200/identity/',
                          result['source_publication_binding']['publication']['path'])
            self.assertTrue(all(row['sealed_supplement']['source_commit'] == self.commit
                                for row in result['checkpoints'].values()))
            self.assertFalse(result['actual_cuda_restore_accepted'])

    def test_archive_completed_update_must_match_the_actual_attempt_not_only_step(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            source = self.base/'preparation/checkpoint-archives/global_step_200'
            snapshot = json.loads((source/'source-snapshot.json').read_bytes())
            snapshot['completed_update']['attempt_id'] = '0'*32
            (source/'source-snapshot.json').write_bytes(core.encoded(snapshot))
            receipt = json.loads((source/'complete.json').read_bytes())
            receipt['snapshot_sha256'] = core.digest((source/'source-snapshot.json').read_bytes())
            (source/'complete.json').write_bytes(core.encoded(receipt))
            with self.assertRaisesRegex(ValueError, 'ARCHIVE_COMPLETED_OUTER_DIFFERS'):
                core.checkpoint_evidence(self.base, self.run, 200)

    def test_native_physical_fallbacks_do_not_change_logical_four_or_add_a_head(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            for physical in (32, 16, 8, 4):
                self.run['actor_micro_batch_per_rank'] = physical
                # Run identity changes here are synthetic: rebuild matching
                # archive metadata rather than editing a live training run.
                self.put('logs/run-manifest.json', self.run)
                for step in core.STEPS:
                    cp = self.base/f'checkpoints/search-g1-common8/actor/global_step_{step}'
                    shutil.rmtree(cp); self.make_checkpoint(step)
                self.assertTrue(self.verify()['cpu_final_accepted'])
            self.run['actor_logical_micro_batch_per_rank'] = 8
            self.put('logs/run-manifest.json', self.run)
            with self.assertRaisesRegex(ValueError, 'REVIEWED_EASY_NATIVE4_RUN_REQUIRED'):
                self.verify()

    def test_wrong_archive_receipt_or_missing_native_rank_never_passes(self):
        with tempfile.TemporaryDirectory() as td:
            self.fixture(td)
            path = self.base/'preparation/checkpoint-archives/global_step_200/complete.json'
            original = path.read_bytes(); bad = json.loads(original); bad['snapshot_sha256'] = '0'*64
            path.write_bytes(core.encoded(bad))
            with self.assertRaisesRegex(ValueError, 'ARCHIVE_SHA_RECEIPT_DIFFERS'):
                core.checkpoint_evidence(self.base, self.run, 200)
            path.write_bytes(original)
            cp = self.base/'checkpoints/search-g1-common8/actor/global_step_200'
            state = json.loads((cp/'training_state/manifest.json').read_bytes()); state['worker_files'].pop()
            (cp/'training_state/manifest.json').write_bytes(core.encoded(state))
            with self.assertRaisesRegex(ValueError, 'NATIVE4_STATE_DIFFERS'):
                core.checkpoint_evidence(self.base, self.run, 200)


if __name__ == '__main__':
    unittest.main()
