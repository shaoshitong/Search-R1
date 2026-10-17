"""Real file retirement safety tests using tiny CPU fixtures, never checkpoints."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import retire_candidate_raw as core


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name).resolve()
        self.namespace = patch.object(core, 'owned_base', side_effect=lambda value: Path(value).resolve())
        self.namespace.start(); self.run_id = 'a'*32; self.job = '730080'
        self.run = dict(run_id=self.run_id, source_commit=core.REVIEWED_TRAINING_COMMIT,
            source_sha256={'actual-source.py': 'b'*64}, easy2560_selection=True, profile='common8',
            training_world_size=4, execution_site='hpc3', execution_backend='slurm',
            data_selection_mode='mixed_first', actor_logical_micro_batch_per_rank=4,
            selection_candidate_questions=2560, selection_selected_questions=512,
            selection_fixed_candidate_budget=True, slurm_job_id=self.job)
        self.write('logs/run-manifest.json', self.run)
        self.write('logs/native-allocation-admission.json',
                   dict(accepted=True, run_id=self.run_id, slurm_job_id=self.job))
        self.rows = []; self.pools = {}
        for step in range(1, 5): self.add_outer(step)
        self.ledger()
        # An active pool and a failed pool have no completed update authority.
        self.write('artifacts/mixed-candidate-pools/'+'e'*32+'-pool-0/0.pt', b'active')
        self.write('artifacts/mixed-candidate-pools/'+'f'*32+'-pool-0/0.pt', b'failed')
        self.write('artifacts/mixed-candidate-pools/'+'f'*32+'-pool-0/selection-failure.json', {'status': 'failure'})
        self.write('checkpoints/actor/global_step_1/model.safetensors', b'never-delete-checkpoint')
        self.write('artifacts/round_0/heads/probe.pkl', b'never-delete-probe')

    def tearDown(self): self.namespace.stop(); self.temp.cleanup()
    def write(self, relative, value):
        path = self.base/relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode())
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def ledger(self, suffix=b''):
        self.write('logs/actual-updates.jsonl',
            b''.join(json.dumps(row, sort_keys=True).encode()+b'\n' for row in self.rows)+suffix)

    def add_outer(self, step, *, full=True):
        attempt = format(step, '032x'); snapshot = self.run_id+':'+str(step)+':policy-'+attempt
        relative = 'artifacts/mixed-candidate-pools/'+attempt+'-pool-0'; self.pools[step] = relative
        groups = [dict(ordinal=i, trajectory_ids=[attempt+'-'+str(i)+'-'+str(j) for j in range(5)],
            strict_correctness=[1, 0, 1, 0, 0], category='mixed', selected=i<512) for i in range(2560)]
        selected_ids = [value for group in groups[:512] for value in group['trajectory_ids']]
        trace_sha = self.write('artifacts/behavior-traces/attempt-'+attempt+'.json',
            dict(role='cpu-retained-selected-trace-fixture', trajectory_ids=selected_ids))
        gate = dict(accepted=True, run_id=self.run_id, source_sha256=self.run['source_sha256'],
            slurm_job_id=self.job, logical_update=step, attempt_id=attempt, attempt_index=step,
            policy_snapshot_id=snapshot, trajectory_count=2560, trajectory_ids=selected_ids,
            row_tensor_sha256={value: 'b'*64 for value in selected_ids},
            trace_sha256=trace_sha, private_trace_sha256=trace_sha)
        gate_sha = self.write('logs/behavior-probability-attempt-'+attempt+'.json', gate)
        self.write('artifacts/mixed-selection/'+attempt+'.json',
            dict(step=step, source_commit=self.run['source_commit'], policy_snapshot_id=snapshot,
                fixed_candidate_budget=True, attempted_candidates=2560, responses_per_group=5,
                selected_groups=512, pool_retry=0, pool_path=str(self.base/relative), groups=groups))
        self.write(relative+'/progress.json', dict(step=step, complete_ordinals=list(range(2560)),
            policy_snapshot_id=snapshot, reserved=list(range(2560))))
        for name in core.RAW_NAMES: self.write(relative+'/'+name, ('raw-'+str(step)+'-'+name).encode())
        for start in (0, 512, 1024, 1536, 2048):
            for suffix in ('-before-generation.pt', '-after-generation.pt'):
                self.write(relative+'/'+str(start)+suffix, b'preserved-driver-rng-and-stream')
            for suffix in ('-worker-rng', '-after-generation-worker-rng'):
                for rank in range(4):
                    self.write(relative+'/'+str(start)+suffix+'/rank-'+str(rank)+'.pt', b'preserved-worker-rng')
        row = dict(run_id=self.run_id, step=step, attempt_id=attempt, attempt_index=step,
            behavior_probability_gate_sha256=gate_sha, behavior_trace_sha256=trace_sha,
            update_status='complete' if full else 'partial', full_budget_complete=full,
            actual_optimizer_writes=20 if full else 10, opportunities_processed=20,
            optimizer_writes_before=(step-1)*20, optimizer_writes_after=step*20 if full else step*20-10)
        self.rows.append(row)

    def plan(self):
        return core.build_plan(self.base, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job)
    def apply(self, plan):
        path = core.save_plan(plan)
        return core.apply_plan(path, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job), path

    def test_only_five_files_in_older_successes_retired_and_all_recovery_evidence_retained(self):
        before = {str(p.relative_to(self.base)): p.read_bytes() for p in self.base.rglob('*') if p.is_file()}
        plan = self.plan()
        self.assertEqual([pool['step'] for pool in plan['pools']], [1, 2])
        expected = {self.pools[step]+'/'+name for step in (1, 2) for name in core.RAW_NAMES}
        result, path = self.apply(plan)
        self.assertEqual(result['status'], core.STATUS)
        self.assertEqual(len(result['actions']), 10)
        for relative, value in before.items():
            file = self.base/relative
            if relative in expected: self.assertFalse(file.exists())
            else: self.assertEqual(file.read_bytes(), value, relative)
        log = [json.loads(line) for line in (path.parent/'actions.jsonl').read_text().splitlines()]
        self.assertEqual([row['stage'] for row in log], ['PREPARED', 'UNLINKED']*10)
        self.assertEqual(core.apply_plan(path, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job), result)
        self.assertFalse(result['filesystem_quota_free_bytes_measured'])

    def test_partial_or_failed_pool_and_missing_trace_have_no_retirement_authority(self):
        self.rows[0].update(update_status='partial', full_budget_complete=False)
        self.ledger()
        attempt = self.rows[1]['attempt_id']
        (self.base/'artifacts/behavior-traces'/('attempt-'+attempt+'.json')).unlink()
        plan = self.plan()
        self.assertEqual(plan['pools'], [])
        self.assertEqual(len(plan['skipped']), 2)
        for step in (1, 2): self.assertTrue((self.base/self.pools[step]/'0.pt').exists())

    def test_append_during_training_is_safe_but_rewritten_completed_prefix_rejected(self):
        plan = self.plan(); self.add_outer(5); self.ledger(suffix=b'{"step":6')
        result, _ = self.apply(plan)
        self.assertEqual(result['pool_count'], 2)
        self.assertTrue((self.base/self.pools[4]/'0.pt').exists())
        plan = self.plan(); self.rows[0]['probe_version'] = 'rewritten'
        self.ledger()
        with self.assertRaisesRegex(ValueError, 'LEDGER_PREFIX_CHANGED'): self.apply(plan)
        self.assertTrue((self.base/self.pools[3]/'0.pt').exists())

    def test_symlink_hardlink_and_foreign_namespace_never_authorize_raw_retirement(self):
        raw = self.base/self.pools[1]/'0.pt'; raw.unlink()
        target = self.base/'checkpoints/actor/global_step_1/model.safetensors'
        raw.symlink_to(target)
        other = self.base/self.pools[2]/'0.pt'
        os.link(other, self.base/'hardlink-copy')
        plan = self.plan()
        self.assertEqual(plan['pools'], [])
        self.assertEqual(target.read_bytes(), b'never-delete-checkpoint')
        with self.assertRaisesRegex(ValueError, 'SOURCE_DEPENDENCY_REVIEW_REQUIRED'):
            core.build_plan(self.base, self.run_id, 'c'*40, self.job)
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_RUN_OR_SOURCE_DIFFERS'):
            core.build_plan(self.base, 'other', core.REVIEWED_TRAINING_COMMIT, self.job)

    def test_changed_selected_trace_or_raw_file_after_plan_blocks_before_unlink(self):
        for kind in ('trace', 'raw'):
            plan = self.plan(); path = core.save_plan(plan)
            relative = ('artifacts/behavior-traces/attempt-'+self.rows[1]['attempt_id']+'.json'
                        if kind == 'trace' else self.pools[2]+'/512.pt')
            target = self.base/relative; previous = target.read_bytes(); target.write_bytes(b'changed')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                core.apply_plan(path, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job)
            self.assertTrue((self.base/self.pools[1]/'0.pt').exists())
            target.write_bytes(previous)

    def test_plan_cannot_target_checkpoint_or_reduce_two_outer_retention(self):
        with self.assertRaisesRegex(ValueError, 'KEEP_AT_LEAST_TWO_COMPLETED'):
            core.build_plan(self.base, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job, 1)
        plan = self.plan()
        plan['pools'][0]['files'][0]['relative_path'] = 'checkpoints/actor/global_step_1/model.safetensors'
        with self.assertRaisesRegex(ValueError, 'TARGETS_OUTSIDE_FIVE_RAW_FILES'): self.apply(plan)
        self.assertTrue((self.base/'checkpoints/actor/global_step_1/model.safetensors').exists())

    def test_failed_or_insufficient_success_retry_pool_is_preserved(self):
        self.write(self.pools[1]+'/selection-failure.json', {'status': 'failure'})
        plan = self.plan()
        self.assertEqual([pool['step'] for pool in plan['pools']], [2])
        self.apply(plan)
        self.assertTrue((self.base/self.pools[1]/'0.pt').exists())

    def test_crash_after_prepared_unlink_can_resume_without_guessing_extra_freed_bytes(self):
        plan = self.plan(); path = core.save_plan(plan)
        record = plan['pools'][0]['files'][0]
        journal = path.parent/'actions.jsonl'
        journal.write_text(json.dumps(dict(stage='PREPARED', relative_path=record['relative_path'], file=record))+'\n')
        (self.base/record['relative_path']).unlink()
        result = core.apply_plan(path, self.run_id, core.REVIEWED_TRAINING_COMMIT, self.job)
        self.assertEqual(result['actions'][0]['stage'], 'ABSENT_AFTER_PREPARED')
        self.assertEqual(result['unconfirmed_prior_unlink_bytes'], record['bytes'])
        self.assertEqual(len(result['actions']), 10)

    def test_all_absent_skips_large_trace_hash_without_claiming_history_or_freed_bytes(self):
        for step in (1, 2):
            for name in core.RAW_NAMES: (self.base/self.pools[step]/name).unlink()
        original = core.read_file
        def cheap_only(base, relative, **kwargs):
            self.assertNotIn('behavior-traces', str(relative))
            self.assertNotIn('generation.pt', str(relative))
            self.assertNotIn('worker-rng', str(relative))
            return original(base, relative, **kwargs)
        with patch.object(core, 'read_file', side_effect=cheap_only): plan = self.plan()
        self.assertEqual(plan['pools'], []); self.assertEqual(plan['removable_logical_bytes'], 0)
        self.assertTrue(all(row['reason'] == 'all_five_raw_absent_no_new_retirement_or_free_byte_claim'
                            for row in plan['skipped']))

    def test_partial_missing_set_is_abnormal_and_preserves_other_four_without_trace_hash(self):
        (self.base/self.pools[1]/'512.pt').unlink()
        original = core.pool_evidence
        def guard(base, run, update, job):
            self.assertNotEqual(update['step'], 1)
            return original(base, run, update, job)
        with patch.object(core, 'pool_evidence', side_effect=guard): plan = self.plan()
        self.assertEqual([pool['step'] for pool in plan['pools']], [2])
        self.assertIn('PARTIAL_RAW_SET_PRESERVED', plan['skipped'][0]['reason'])
        for name in ('0.pt', '1024.pt', '1536.pt', '2048.pt'):
            self.assertTrue((self.base/self.pools[1]/name).exists())


if __name__ == '__main__': unittest.main()
