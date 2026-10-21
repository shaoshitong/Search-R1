"""Stdlib-only synthetic filesystem tests; no model or recovery acceptance."""
import copy
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from reproduction.easy2560_joint4 import checkpoint_evidence as core

PUBLISHED_MICRO16_COMMIT = '8a5bac94b8230a99b81977f5290589e3d3708dfe'


class EvidenceTests(unittest.TestCase):
    def test_actual_shared_root_schema_accepts_mou_directory_with_jxu_execution(self):
        # Actual source observation: executor14090, base owner21930/gid21919/
        # mode0770; run and eligible JSON owner14090/gid21919/mode0660.
        path = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017')
        for owner in (21930, 14090):
            st = SimpleNamespace(st_uid=owner, st_gid=21919, st_mode=core.stat.S_IFDIR | 0o770)
            with self.subTest(owner=owner), patch.object(core.Path, 'resolve', lambda p, *a, **k: p), \
                    patch.object(core.Path, 'stat', return_value=st), \
                    patch.object(core.os, 'getuid', return_value=14090), \
                    patch.object(core.os, 'geteuid', return_value=14090), \
                    patch.object(core.os, 'access', return_value=True) as access:
                self.assertEqual(core.owned_base(path), path)
                access.assert_called_once_with(path, os.W_OK | os.X_OK)

    def test_shared_directory_exception_does_not_relax_run_file_or_executor_ownership(self):
        path = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017')
        for uid, euid, owner, group, writable in (
                (21930, 21930, 21930, 21919, True), (14090, 0, 21930, 21919, True),
                (14090, 14090, 42, 21919, True), (14090, 14090, 21930, 42, True),
                (14090, 14090, 21930, 21919, False)):
            st = SimpleNamespace(st_uid=owner, st_gid=group, st_mode=core.stat.S_IFDIR | 0o770)
            with self.subTest(uid=uid, euid=euid, owner=owner, group=group, writable=writable), \
                    patch.object(core.Path, 'resolve', lambda p, *a, **k: p), \
                    patch.object(core.Path, 'stat', return_value=st), \
                    patch.object(core.os, 'getuid', return_value=uid), \
                    patch.object(core.os, 'geteuid', return_value=euid), \
                    patch.object(core.os, 'access', return_value=writable):
                with self.assertRaises(ValueError): core.owned_base(path)
        file_path = path/'logs/run-manifest.json'
        for owner, accepted in ((14090, True), (21930, False)):
            st = SimpleNamespace(st_uid=owner, st_gid=21919, st_mode=core.stat.S_IFREG | 0o660)
            with self.subTest(file_owner=owner), patch.object(core.Path, 'resolve', lambda p, *a, **k: p), \
                    patch.object(core.Path, 'is_symlink', return_value=False), \
                    patch.object(core.Path, 'stat', return_value=st), \
                    patch.object(core.os, 'getuid', return_value=14090):
                if accepted: self.assertEqual(core.regular(file_path), file_path)
                else:
                    with self.assertRaisesRegex(ValueError, 'REGULAR_OWNED_SOURCE_REQUIRED'):
                        core.regular(file_path)

    def write(self, relative, value):
        path = self.base/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(core.encoded(value) if not isinstance(value, bytes) else value)
        return path

    def lines(self, relative, values, suffix=b''):
        return self.write(relative, b''.join(core.canonical(r)+b'\n' for r in values)+suffix)

    def fixture(self, folder, step=5, *, abort_first=False, fixed5=False, training_commit=None, physical=32):
        self.base = Path(folder).resolve(); self.step = step
        self.commit = training_commit or core.REVIEWED_COMMIT
        self.run_id = 'a'*32; self.job = '730080'
        self.cp = self.base/f'checkpoints/search-g1-common8/actor/global_step_{step}'
        self.code = self.write('repo/reviewed.py', b'fixture source only\n')
        self.data = self.write('data/manifest.json', {'fixture': True})
        self.pool = self.write('external/eligible.json', {'fixture_pool': True})
        round_ = 0 if step <= 50 else step-50
        probe_dir = self.base/f'artifacts/round_{round_}-attempt-fixture'
        heads = {}
        for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
            path = self.write(str((probe_dir/'heads'/name).relative_to(self.base)), b'fixture head')
            heads[name] = core.digest(path.read_bytes())
        report = self.write(str((probe_dir/'heads/calibration-report.json').relative_to(self.base)), {'fixture': True})
        self.probe = dict(run_id=self.run_id, version=f'{self.run_id}:{round_}:fixture', round=round_,
            encoder='/remote/official', encoder_sha256={'model.safetensors': 'b'*64},
            commit=str(probe_dir/'heads/commit_probe.pkl'), boundary=str(probe_dir/'heads/boundary_probe.pkl'),
            head_sha256=heads, calibration_report_sha256=core.digest(report.read_bytes()))
        self.probe_path = self.write(str((probe_dir/'version-manifest.json').relative_to(self.base)), self.probe)
        self.run = dict(profile='common8', source_commit=self.commit, run_id=self.run_id,
            execution_site='hpc3', execution_backend='slurm',
            easy2560_selection=True, training_world_size=4, data_selection_mode='mixed_first',
            actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4,
            selection_candidate_questions=2560, selection_selected_questions=512, selection_fixed_candidate_budget=True,
            all_correct_virtual_zero=False, selection_eligible_pool_path=str(self.pool),
            source_sha256={'reviewed.py': core.digest(self.code.read_bytes())},
            input_sha256={'manifest.json': core.digest(self.data.read_bytes())})
        run_path = self.write('logs/run-manifest.json', self.run)
        if self.commit != core.REVIEWED_COMMIT:
            published = dict(source_commit=self.commit, fresh_source_all_files_sha_verified=True,
                files={'reviewed.py':dict(bytes=self.code.stat().st_size,
                                          sha256=core.digest(self.code.read_bytes()))})
            self.write('source/'+self.commit+'/source-publication.json', published)
            self.write('preparation/run-source.json', dict(source_commit=self.commit,
                files=published['files'], runtime_repo=str(self.base/'repo')))
        self.write('logs/native-allocation-admission.json', dict(accepted=True, run_id=self.run_id, slurm_job_id=self.job))
        cfg = dict(data=dict(selection_mode='mixed_first'), algorithm=dict(all_correct_virtual_zero=False),
                   actor_rollout_ref=dict(actor=dict(ppo_logical_micro_batch_size=4)))
        self.write('logs/resolved-training-config.json', dict(run_id=self.run_id, resolved_config=cfg,
                    resolved_config_sha256=core.digest(core.canonical(cfg))))
        self.write('logs/formal-inline-development-initial-skipped.json', dict(
                    run_id=self.run_id, current_run_evaluation_performed=False))
        attempts = []; updates = []; scheduler = [[] for _ in range(4)]; worker = [[] for _ in range(4)]
        completed = writes = 0
        for index in range(1, step + int(abort_first) + 1):
            success = not(abort_first and index == 1)
            aid = f'{index:032x}'; outer = completed+1; before = writes
            writes += int(success); completed += int(success)
            attempt = dict(attempt_id=aid, attempt_index=index, logical_update=outer,
                completed_outer=completed, optimizer_writes_before=before, optimizer_writes_after=writes,
                optimizer_writes_this_attempt=int(success), opportunities_processed=1, complete_outer=success,
                cumulative_aborted_attempts=int(abort_first), all_ranks_agreed=True, real_rank_count=4,
                run_id=self.run_id, slurm_job_id=self.job)
            attempts.append(attempt)
            raw_trace = b'private selected token fixture '+aid.encode()
            self.write('artifacts/behavior-traces/attempt-'+aid+'.json', raw_trace)
            gate = dict(run_id=self.run_id, slurm_job_id=self.job, source_sha256=self.run['source_sha256'],
                attempt_id=aid, attempt_index=index, logical_update=outer, accepted=True,
                trace_sha256=core.digest(raw_trace), private_trace_sha256=core.digest(raw_trace))
            gate_path = self.write('logs/behavior-probability-attempt-'+aid+'.json', gate)
            if success:
                update = dict(step=completed, run_id=self.run_id, attempt_id=aid, attempt_index=index,
                    optimizer_writes_after=writes, probe_version=self.probe['version'],
                    behavior_probability_gate_sha256=core.digest(gate_path.read_bytes()),
                    behavior_trace_sha256=core.digest(raw_trace))
                updates.append(update)
                self.write('artifacts/mixed-selection/'+aid+'.json', dict(
                    step=completed, source_commit=self.commit,
                    policy_snapshot_id=f'{self.run_id}:{completed}:policy-{aid}',
                    fixed_candidate_budget=True, attempted_candidates=2560, selected_groups=512))
            for rank in range(4):
                common = dict(run_id=self.run_id, slurm_job_id=self.job, world_size=4, rank=rank,
                    attempt_id=aid, attempt_index=index, logical_update=outer, writes_after=writes)
                scheduler[rank] += [dict(common, kind=k) for k in
                    (('BEGIN','COMMIT_PREPARED','COMPLETE') if success else ('BEGIN','ABORTED_ATTEMPT'))]
                context = dict(attempt_id=aid, attempt_index=index)
                w = dict(run_id=self.run_id, rank=rank, context=context)
                worker[rank].append(dict(w, kind='POLICY_MONITOR',
                    objective_weight_definition='equal four-row logical micro masked_mean; weighted global numerator/denominator',
                    monitor_identity=dict(selection_indices=list(range(8)))))
                if success: worker[rank].append(dict(w, kind='OPTIMIZER_WRITE'))
                worker[rank].append(dict(w, kind='OPPORTUNITY'))
                worker[rank].append(dict(w, kind='OUTER_BUDGET' if success else 'ABORTED_ATTEMPT', cumulative_writes=writes))
        self.attempts = attempts; self.updates = updates; self.schedulers = scheduler; self.workers = worker
        self.lines('logs/actual-updates.jsonl', updates, b'{"step":999')
        self.lines('logs/actual-attempts.jsonl', attempts, b'{"attempt_index":')
        for rank in range(4):
            self.lines(f'logs/scheduler-rank-{rank}.jsonl', scheduler[rank], b'{bad later')
            self.lines(f'logs/optimizer-rank-{rank}.jsonl', worker[rank], b'{bad later')
        validation = [dict(step=step, split='stage' if step == 5 else 'validation')]
        if fixed5: validation.append(dict(step=5, split='validation'))
        self.lines('logs/validation-identities.jsonl', validation, b'{later partial')
        self.driver = dict(schema_version=1, completed_outer=step, run_id=self.run_id,
            source_commit=self.commit, question_probe=None, selection_mode='mixed_first',
            easy2560_selection=True, probe_version=self.probe['version'], probe_round=round_,
            reward_refresh_pending=step in (50,100,150), rng=dict(torch_cpu=b'fixture RNG'),
            candidate_stream=dict(candidate_order=[2,0,1], candidate_cursor=3, epoch=1,
                consumed_source_rows=step*2560+17, pending_offset=1, pending_source=['a','b'],
                prefix20_pool_binding=dict(eligible_pool_sha256=core.digest(self.pool.read_bytes()))),
            attempt_progress=dict(completed_outer=step, cumulative_writes=writes,
                attempt_index=len(attempts), aborted_attempts=int(abort_first), full_budget_outer=0))
        self.save_driver()
        proofs = []
        for rank in range(4):
            p = self.write(str((self.cp/f'training_state/rank-{rank}.pt').relative_to(self.base)), b'not reloaded optimizer bytes')
            proofs.append(dict(rank=rank, run_id=self.run_id, actual_optimizer_writes=writes,
                path=str(p), bytes=p.stat().st_size, sha256=core.digest(p.read_bytes())))
        self.state = dict(completed_outer=step, question_probe=None, worker_files=proofs,
            driver_file=dict(path=str(self.cp/'training_state/driver.pt')))
        self.state_path = self.write(str((self.cp/'training_state/manifest.json').relative_to(self.base)), self.state)
        files = {str(p.relative_to(self.cp)):core.digest(p.read_bytes()) for p in self.cp.rglob('*') if p.is_file()}
        self.complete = dict(run_id=self.run_id, step=step, finite_all_saved_tensors=True,
            actual_optimizer_writes_per_rank=writes, files_sha256=files)
        self.complete_path = self.write(str((self.cp/'complete.json').relative_to(self.base)), self.complete)
        snapshot_files = {}
        def add(name, p):
            snapshot_files[name] = dict(path=str(p), bytes=p.stat().st_size, sha256=core.digest(p.read_bytes()))
        for p in self.cp.rglob('*'):
            if p.is_file(): add('checkpoint/'+str(p.relative_to(self.cp)), p)
        add('metadata/run-manifest.json', run_path); add('data/manifest.json', self.data)
        add('source/reviewed.py', self.code)
        for p in probe_dir.rglob('*'):
            if p.is_file(): add('grounding/'+probe_dir.name+'/'+str(p.relative_to(probe_dir)), p)
        self.snapshot = dict(schema_version=1, role='easy2560_full_checkpoint_source_snapshot', base=str(self.base),
            source_commit=self.commit, run_id=self.run_id, step=step, training_world_size=4,
            files=snapshot_files, completed_update=updates[-1], total_bytes=sum(r['bytes'] for r in snapshot_files.values()))
        self.archive_root = f'preparation/checkpoint-archives/global_step_{step}'
        self.snapshot_path = self.write(self.archive_root+'/source-snapshot.json', self.snapshot)
        self.accepted = dict(status=core.ARCHIVE_STATUS, actual_cuda_restore_accepted=False,
            fresh_source_after_archive_sha_verified=True, fresh_destination_sha_verified=True, destination_fsynced=True,
            snapshot_sha256=core.digest(self.snapshot_path.read_bytes()), file_count=len(snapshot_files),
            **{k:self.snapshot[k] for k in ('base','run_id','source_commit','step','total_bytes')})
        self.accepted_path = self.write(self.archive_root+'/complete.json', self.accepted)

    def save_driver(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw,'w') as z:
            z.writestr('driver/data.pkl', pickle.dumps(self.driver, protocol=2)); z.writestr('driver/byteorder', b'little')
        self.write(str((self.cp/'training_state/driver.pt').relative_to(self.base)), raw.getvalue())

    def seal(self):
        with patch.object(core, 'owned_base', side_effect=lambda p: Path(p).resolve()):
            return core.seal(self.base, self.commit, self.run_id, self.job, self.step)

    def test_pre5_chain_can_have_no_validation_prefix_but_cp5_cannot(self):
        with tempfile.TemporaryDirectory() as folder:
            self.base = Path(folder).resolve()
            def observe(step, chain):
                captured = {}
                bundle = SimpleNamespace(put=lambda name, raw, record: captured.setdefault(name, raw))
                result = core._validation_observation(self.base, step, bundle, chain=chain)
                self.assertIn('segments/current/logs/validation-identities.jsonl', captured)
                return result
            self.lines('logs/validation-identities.jsonl', [dict(step=5, split='validation')])
            chain = dict(boundary=1, historical={})
            result = observe(5, chain)
            self.assertTrue(result['fixed512_seen_at_observation'])
            self.assertFalse(result['final51713_seen_at_observation'])
            self.lines('logs/validation-identities.jsonl', [dict(step=1, split='validation'),
                                                          dict(step=5, split='validation')])
            with self.assertRaisesRegex(ValueError, 'CURRENT_VALIDATION_REPLAYS_HISTORICAL_BOUNDARY'):
                observe(5, chain)
            self.lines('logs/validation-identities.jsonl', [dict(step=50, split='validation')])
            with self.assertRaises(KeyError):
                observe(50, dict(boundary=5, historical={}))
            historical = {'logs/validation-identities.jsonl': b'{"step":5,"split":"validation"}\n'}
            self.assertTrue(observe(50, dict(boundary=5, historical=historical))['fixed512_seen_at_observation'])

    def test_exact_prefixes_and_refs_are_sealed_without_old_receipt_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, abort_first=True)
            original = {p: p.read_bytes() for p in (self.complete_path, self.snapshot_path, self.accepted_path)}
            result = self.seal(); target = Path(result['directory'])
            self.assertEqual(result['status'], core.STATUS)
            self.assertFalse(result['actual_cuda_restore_accepted']); self.assertFalse(result['post_callback_rng_captured'])
            for path, raw in original.items(): self.assertEqual(path.read_bytes(), raw)
            manifest = json.loads((target/'supplement.json').read_bytes())
            self.assertTrue(manifest['single_original_job_prefix']); self.assertFalse(manifest['inherited_history_assembled'])
            self.assertEqual(manifest['copied_selected_trace_bytes'], 0)
            self.assertEqual(len(json.loads((target/'selected-large-references.json').read_bytes())), 6)
            self.assertEqual((target/'prefix/actual-updates.jsonl').read_bytes().count(b'\n'), 5)
            self.assertNotIn(b'999', (target/'prefix/actual-updates.jsonl').read_bytes())
            summary = json.loads((target/'saved-driver-summary.json').read_bytes())
            self.assertEqual(summary['stream']['consumed_source_rows'], 12817)
            self.assertEqual(summary['stream']['pending_offset'], 1)
            self.assertEqual(self.seal(), result)
            self.assertTrue(all(Path(r['path']).is_relative_to(target) for r in result['files_to_archive']))

    def test_all_fixed_callback_boundaries_are_truthful_after_observed_callback(self):
        for step in core.STEPS:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder, step=step, fixed5=True)
                result = self.seal(); root = Path(result['directory'])
                phase = json.loads((root/'checkpoint-callback-boundary.json').read_bytes())
                self.assertEqual(phase['checkpoint_saved_after'], 'stage64' if step == 5 else 'fixed512')
                self.assertEqual(phase['fixed512_callback_pending_at_checkpoint'], step == 5)
                self.assertEqual(phase['grounding_refresh_pending_at_checkpoint'], step in (50,100,150))
                self.assertEqual(phase['final51713_pending_at_checkpoint'], step == 200)
                self.assertFalse(phase['post_callback_rng_captured'])

    def test_explicit_published_micro16_source_seals_with_independent_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
            result = self.seal()
            binding = result['source_publication_binding']
            self.assertEqual(result['source_commit'], self.commit)
            self.assertEqual(binding['requested_training_commit'], self.commit)
            self.assertEqual(binding['policy'], 'explicit_caller_commit_publication_runtime_run_map')
            self.assertFalse(binding['automatic_head_or_ancestor_allow'])
            self.assertFalse(binding['git_ancestry_verified_here'])
            self.assertTrue(result['single_original_job_prefix'])
            self.assertFalse(result['inherited_history_assembled'])
            self.assertFalse(result['actual_cuda_restore_accepted'])
            target = Path(result['directory'])
            self.assertTrue((target/'identity/source-publication.json').is_file())
            self.assertTrue((target/'identity/run-source.json').is_file())
            # Verification authenticates sealed small evidence even after the
            # original published tree and tensor bytes have been retired.
            (self.base/'source'/self.commit/'source-publication.json').unlink()
            (self.base/'preparation/run-source.json').unlink()
            for path in self.cp.glob('training_state/*.pt'): path.unlink()
            self.assertEqual(core.verify(target), result)

    def test_non93_source_needs_explicit_publication_runtime_and_archived_map(self):
        for broken in ('missing_publication', 'not_fresh', 'wrong_commit', 'runtime_files',
                       'runtime_repo', 'run_map', 'archive_map', 'archive_bytes'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
                publication_path = self.base/'source'/self.commit/'source-publication.json'
                publication = json.loads(publication_path.read_bytes())
                runtime_path = self.base/'preparation/run-source.json'
                runtime = json.loads(runtime_path.read_bytes())
                if broken == 'missing_publication': publication_path.unlink()
                elif broken == 'not_fresh':
                    publication['fresh_source_all_files_sha_verified'] = False
                    publication_path.write_bytes(core.encoded(publication))
                elif broken == 'wrong_commit':
                    publication['source_commit'] = core.REVIEWED_COMMIT
                    publication_path.write_bytes(core.encoded(publication))
                elif broken in ('runtime_files', 'runtime_repo'):
                    if broken == 'runtime_files': runtime['files']['reviewed.py']['sha256'] = '0'*64
                    else: runtime['runtime_repo'] = '/another-run/repo'
                    runtime_path.write_bytes(core.encoded(runtime))
                elif broken == 'run_map':
                    run = copy.deepcopy(self.run); run['source_sha256']['reviewed.py'] = '0'*64
                    self.write('logs/run-manifest.json', run)
                    self.snapshot['files']['metadata/run-manifest.json']['sha256'] = core.digest(
                        (self.base/'logs/run-manifest.json').read_bytes())
                elif broken == 'archive_map': self.snapshot['files']['source/reviewed.py']['sha256'] = '0'*64
                else: self.snapshot['files']['source/reviewed.py']['bytes'] += 1
                if broken in ('run_map', 'archive_map', 'archive_bytes'):
                    self.snapshot_path.write_bytes(core.encoded(self.snapshot))
                    self.accepted['snapshot_sha256'] = core.digest(self.snapshot_path.read_bytes())
                    self.accepted_path.write_bytes(core.encoded(self.accepted))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.seal()
                self.assertFalse((self.base/'preparation/checkpoint-evidence/global_step_5').exists())

    def test_requested_commit_is_not_inferred_from_run_or_current_git_head(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, training_commit=PUBLISHED_MICRO16_COMMIT, physical=16)
            with patch.object(core, 'owned_base', side_effect=lambda p: Path(p).resolve()):
                for requested in (core.REVIEWED_COMMIT, self.commit[:7], 'a'*40):
                    with self.subTest(requested=requested), self.assertRaises(ValueError):
                        core.seal(self.base, requested, self.run_id, self.job, self.step)

    def test_failed_seal_publishes_no_partial_canonical_bundle(self):
        for broken in ('missing_rank', 'changed_driver', 'cross_job', 'missing_completed', 'wrong_step', 'bad_trace', 'wrong_pool'):
            with self.subTest(broken=broken), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder)
                if broken == 'missing_rank': (self.base/'logs/scheduler-rank-3.jsonl').unlink()
                if broken == 'changed_driver': (self.cp/'training_state/driver.pt').write_bytes(b'changed')
                if broken == 'cross_job':
                    rows=copy.deepcopy(self.attempts); rows[-1]['slurm_job_id']='another-job'
                    self.lines('logs/actual-attempts.jsonl', rows)
                if broken == 'missing_completed': self.lines('logs/actual-updates.jsonl', self.updates[:-1])
                if broken == 'wrong_step': self.complete_path.write_bytes(core.encoded(dict(self.complete,step=50)))
                if broken == 'bad_trace': (self.base/'artifacts/behavior-traces'/('attempt-'+self.attempts[0]['attempt_id']+'.json')).write_bytes(b'changed')
                if broken == 'wrong_pool': self.pool.write_bytes(b'changed pool')
                with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
                    self.seal()
                root = self.base/'preparation/checkpoint-evidence'
                self.assertFalse((root/'global_step_5').exists())
                self.assertFalse(list(root.glob('.partial-*')))

    def test_cross_execution_prefix_requires_implemented_chain_not_relabelling(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            self.write('logs/run-manifest.json', dict(self.run,full_state_restore={'completed_outer': 50}))
            with self.assertRaisesRegex(ValueError,'CROSS_JOB_AUTHENTICATED_PREFIX_NOT_IMPLEMENTED'):
                self.seal()

    def test_tampered_sealed_file_is_not_overwritten_on_reentry(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            result=self.seal(); p=Path(result['directory'])/'prefix/actual-updates.jsonl'
            p.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SOURCE_SHA_DIFFERS'): self.seal()
            self.assertEqual(p.read_bytes(),b'corrupt')

    def test_reentry_allows_future_append_but_rejects_changed_source_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            result = self.seal()
            path = self.base/'logs/actual-updates.jsonl'
            with path.open('ab') as f: f.write(b' harmless unparsed future tail')
            self.assertEqual(self.seal(), result)
            path.write_bytes(b'X' + path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, 'SOURCE_AFTER_COPY_SHA_DIFFERS'):
                self.seal()
            # Pure verification of the sealed sidecar remains independent of
            # a later modified/retired source tensor or source ledger.
            self.assertEqual(core.verify(result['directory']), result)

    def test_large_checkpoint_files_are_not_required_for_sealed_verify(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            result = self.seal()
            for p in self.cp.glob('training_state/*.pt'): p.unlink()
            self.assertEqual(core.verify(result['directory']), result)
            self.assertEqual(result['source_base'], str(self.base))
            self.assertEqual(result['core_archive']['status'], core.ARCHIVE_STATUS)
            self.assertFalse(result['core_archive']['remote_location']['current_remote_read_performed'])

    def test_foreign_probe_or_wrong_selection_cannot_replace_saved_identity(self):
        for kind in ('probe', 'selection'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder)
                if kind == 'probe':
                    self.probe_path.write_bytes(core.encoded(dict(self.probe, version='future-live-version')))
                else:
                    aid = self.updates[0]['attempt_id']
                    self.write('artifacts/mixed-selection/'+aid+'.json', dict(
                        step=999, source_commit=core.REVIEWED_COMMIT, policy_snapshot_id='wrong',
                        fixed_candidate_budget=True, attempted_candidates=2560, selected_groups=512))
                with self.assertRaises(ValueError): self.seal()

    def test_import_and_cli_help_use_stdlib_only(self):
        code = '''import sys,importlib.abc
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','numpy','ray','pyarrow','omegaconf','safetensors'): raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Deny())
from reproduction.easy2560_joint4 import checkpoint_evidence
from reproduction.prefix20_joint4.cpu_audit import safe_pt
assert 'torch' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__': unittest.main()
