"""Stdlib scalar-ledger fixtures; no model, CUDA or production run is accessed."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import strict_final_write_audit as core


COMMIT = 'e082a23b5a65b302e2016aa0e5ab43b10b543d4f'


class StrictWriteAuditTests(unittest.TestCase):
    def fixture(self, folder, *, last_writes=20, abort_first=False):
        base = Path(folder).resolve()
        run = dict(run_id='a'*32, source_commit=COMMIT,
                   easy2560_selection=True, training_world_size=4,
                   source_sha256={'fixture.py': 'b'*64})
        def put(relative, value):
            path = base/relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value)+'\n')
        def ledger(relative, values):
            path = base/relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(value)+'\n' for value in values))
        put('logs/run-manifest.json', run)
        put('logs/native-allocation-admission.json', dict(run_id=run['run_id'], accepted=True, slurm_job_id='745371'))
        published = dict(source_commit=COMMIT, fresh_source_all_files_sha_verified=True,
                         files={'fixture.py': dict(bytes=0, sha256='b'*64)})
        put('source/'+COMMIT+'/source-publication.json', published)
        put('preparation/run-source.json', dict(source_commit=COMMIT, files=published['files'], runtime_repo=str(base/'repo')))
        updates, attempts, schedules, workers = [], [], [[] for _ in range(4)], [[] for _ in range(4)]
        if abort_first:
            aborted = dict(run_id=run['run_id'], slurm_job_id='745371', attempt_id='%032x' % 1,
                attempt_index=1, all_ranks_agreed=True, real_rank_count=4, logical_update=1,
                optimizer_writes_this_attempt=0, opportunities_processed=1,
                optimizer_writes_before=0, optimizer_writes_after=0, complete_outer=False,
                scheduler_advances=0, completed_outer=0, cumulative_aborted_attempts=1)
            attempts.append(aborted)
            for rank in range(4):
                common = dict(run_id=run['run_id'], slurm_job_id='745371', attempt_id=aborted['attempt_id'],
                    attempt_index=1, logical_update=1, rank=rank)
                scheduler = dict(common, world_size=4, writes_after=0, scheduler_last_epoch_after=0, scheduler_advances=0)
                schedules[rank].extend(dict(scheduler, kind=kind) for kind in ('BEGIN', 'ABORTED_ATTEMPT'))
                workers[rank].append(dict(common, kind='ABORTED_ATTEMPT', cumulative_writes=0))
        before = 0
        for step in range(1, 201):
            count = last_writes if step == 200 else 20
            after = before + count; index = step+int(abort_first); aid = '%032x' % index
            common = dict(run_id=run['run_id'], slurm_job_id='745371', attempt_id=aid, attempt_index=index)
            attempts.append(dict(common, all_ranks_agreed=True, real_rank_count=4,
                logical_update=step, optimizer_writes_this_attempt=count, opportunities_processed=count,
                optimizer_writes_before=before, optimizer_writes_after=after, complete_outer=True,
                scheduler_advances=1, completed_outer=step, cumulative_aborted_attempts=int(abort_first)))
            updates.append(dict(common, step=step, optimizer_writes_before=before,
                                optimizer_writes_after=after, actual_optimizer_writes=count))
            for rank in range(4):
                scheduler = dict(common, world_size=4, rank=rank, logical_update=step,
                    writes_after=after, scheduler_last_epoch_after=step, scheduler_advances=1)
                schedules[rank].extend(dict(scheduler, kind=kind) for kind in ('BEGIN', 'COMMIT_PREPARED', 'COMPLETE'))
                context = dict(logical_update=step, attempt_id=aid, attempt_index=index)
                workers[rank].extend(dict(run_id=run['run_id'], rank=rank, context=context,
                    kind='OPTIMIZER_WRITE', write_count=write) for write in range(before+1, after+1))
                workers[rank].append(dict(common, rank=rank, logical_update=step,
                    kind='OUTER_BUDGET', cumulative_writes=after))
            before = after
        ledger('logs/actual-updates.jsonl', updates)
        ledger('logs/actual-attempts.jsonl', attempts)
        for rank in range(4):
            ledger('logs/scheduler-rank-%d.jsonl' % rank, schedules[rank])
            ledger('logs/optimizer-rank-%d.jsonl' % rank, workers[rank])
        return base, run

    def snapshot(self, base):
        return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in base.rglob('*') if p.is_file()}

    def proof(self):
        roles = dict(scheduler=dict(final_completed_outer=200, cumulative_actual_writes=4000,
                                    terminal_attempt_index=200),
                     optimizer=dict(final_completed_outer=200, cumulative_actual_writes=4000,
                                    terminal_attempt_index=200))
        return dict(actual_completed_updates=200, rank_count=4, no_update201=True,
                    no_post200_driver_scheduler_worker_attempts=True, actual_attempt_count=200,
                    cumulative_actual_optimizer_writes_per_rank=4000,
                    ranks={str(rank): copy.deepcopy(roles) for rank in range(4)})

    def test_actual_native4_ledgers_exact4000_pass_without_mutation_or_extra_call_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            base, _ = self.fixture(folder); before = self.snapshot(base)
            result = core.verify_strict_final_writes(base, training_commit=COMMIT)
            self.assertEqual(result['status'], core.STATUS)
            self.assertEqual(result['retained_optimizer_writes_per_rank'], 4000)
            self.assertTrue(result['strict_retained_write_audit_accepted'])
            self.assertFalse(result['zero_extra_historical_adam_calls_proven'])
            self.assertFalse(result['zero_historical_rollbacks_proven'])
            self.assertFalse(result['actual_cuda_restore_accepted'])
            self.assertEqual(before, self.snapshot(base))

    def test_valid_partial3999_is_rejected_after_authoritative_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            base, run = self.fixture(folder, last_writes=19)
            proof = core.final_cpu_acceptance.no201(base, run)
            self.assertEqual(proof['cumulative_actual_optimizer_writes_per_rank'], 3999)
            with self.assertRaisesRegex(ValueError, 'EXACT4000'):
                core.verify_strict_final_writes(base, training_commit=COMMIT)

    def test_zero_retained_write_failed_attempt_is_not_counted_as_completed_outer(self):
        with tempfile.TemporaryDirectory() as folder:
            base, _ = self.fixture(folder, abort_first=True)
            result = core.verify_strict_final_writes(base, training_commit=COMMIT)
            self.assertEqual(result['actual_attempt_count'], 201)
            self.assertEqual(result['actual_completed_updates'], 200)
            self.assertEqual(result['retained_optimizer_writes_per_rank'], 4000)
            self.assertFalse(result['zero_extra_historical_adam_calls_proven'])

    def test_nonreviewed_original_source_publication_must_be_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            base, _ = self.fixture(folder)
            path = base/'source'/COMMIT/'source-publication.json'
            record = json.loads(path.read_text()); record['fresh_source_all_files_sha_verified'] = False
            path.write_text(json.dumps(record)+'\n')
            with patch.object(core.checkpoint_evidence, 'authenticated_restore_chain') as chain, \
                 patch.object(core.final_cpu_acceptance, 'no201') as no201:
                with self.assertRaisesRegex(ValueError, 'PUBLISHED_SOURCE_REQUIRED'):
                    core.verify_strict_final_writes(base, training_commit=COMMIT)
                chain.assert_not_called(); no201.assert_not_called()

    def test4000_never_passes_if_real_base_validation_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            base, _ = self.fixture(folder)
            path = base/'logs/optimizer-rank-3.jsonl'
            rows = path.read_text().splitlines(); rows.pop(0)
            path.write_text('\n'.join(rows)+'\n')
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE_WORKER_ATTEMPT'):
                core.verify_strict_final_writes(base, training_commit=COMMIT)

    def test_authenticated_chain_is_passed_to_no201_after_source_binding(self):
        run = dict(run_id='a'*32, source_commit=COMMIT, easy2560_selection=True, training_world_size=4)
        chain = {'public': {'source_job': '745022', 'completed_outer': 1}}
        order = []
        def source(base, supplied, commit):
            order.append('source'); self.assertIs(supplied, run); self.assertEqual(commit, COMMIT)
            return {'original_training_source_bound': True}
        def history(base, supplied):
            order.append('chain'); self.assertIs(supplied, run); return chain
        def no201(base, supplied, *, chain):
            order.append('no201'); self.assertEqual(chain['public']['source_job'], '745022')
            proof = self.proof(); proof['authenticated_history'] = chain['public']; return proof
        with patch.object(core.final_cpu_acceptance, 'small', return_value=(run, {})), \
             patch.object(core.final_cpu_acceptance, 'published_source_binding', side_effect=source), \
             patch.object(core.checkpoint_evidence, 'authenticated_restore_chain', side_effect=history), \
             patch.object(core.final_cpu_acceptance, 'no201', side_effect=no201):
            result = core.verify_strict_final_writes('/fixture', training_commit=COMMIT)
        self.assertEqual(order, ['source', 'chain', 'no201'])
        self.assertEqual(result['proof']['authenticated_history'], chain['public'])

    def test_dependency_exception_cannot_emit_false_pass_or_continue(self):
        run = dict(run_id='a'*32, source_commit=COMMIT, easy2560_selection=True, training_world_size=4)
        for failing in ('source', 'chain', 'no201'):
            with self.subTest(failing=failing), \
                 patch.object(core.final_cpu_acceptance, 'small', return_value=(run, {})), \
                 patch.object(core.final_cpu_acceptance, 'published_source_binding',
                    side_effect=ValueError('source rejected') if failing == 'source' else None) as source, \
                 patch.object(core.checkpoint_evidence, 'authenticated_restore_chain',
                    side_effect=ValueError('chain rejected') if failing == 'chain' else None) as chain, \
                 patch.object(core.final_cpu_acceptance, 'no201',
                    side_effect=ValueError('no201 rejected') if failing == 'no201' else None,
                    return_value=self.proof()) as no201, contextlib.redirect_stdout(io.StringIO()) as output:
                code = core.main(['--base', '/fixture', '--training-commit', COMMIT])
                result = json.loads(output.getvalue())
                self.assertEqual(code, 1); self.assertFalse(result['strict_retained_write_audit_accepted'])
                self.assertEqual(result['status'], core.REJECTED)
                if failing == 'source': chain.assert_not_called(); no201.assert_not_called()
                if failing == 'chain': no201.assert_not_called()

    def test_missing_bool_wrong_world_or_inconsistent_rank_proofs_reject(self):
        run = dict(run_id='a'*32, source_commit=COMMIT, easy2560_selection=True, training_world_size=4)
        mutations = [
            lambda p: p.pop('cumulative_actual_optimizer_writes_per_rank'),
            lambda p: p.update(cumulative_actual_optimizer_writes_per_rank=True),
            lambda p: p.update(cumulative_actual_optimizer_writes_per_rank=4000.0),
            lambda p: p.update(actual_completed_updates=True),
            lambda p: p.update(actual_completed_updates=199),
            lambda p: p.update(rank_count=True),
            lambda p: p.update(rank_count=8),
            lambda p: p.update(no_update201=1),
            lambda p: p.update(actual_attempt_count=True),
            lambda p: p['ranks'].pop('3'),
            lambda p: p['ranks']['2']['optimizer'].update(cumulative_actual_writes=3999),
            lambda p: p['ranks']['1']['scheduler'].update(terminal_attempt_index=True),
        ]
        for mutate in mutations:
            proof = self.proof(); mutate(proof)
            with self.subTest(proof=proof), \
                 patch.object(core.final_cpu_acceptance, 'small', return_value=(run, {})), \
                 patch.object(core.final_cpu_acceptance, 'published_source_binding', return_value={}), \
                 patch.object(core.checkpoint_evidence, 'authenticated_restore_chain', return_value=None), \
                 patch.object(core.final_cpu_acceptance, 'no201', return_value=proof):
                with self.assertRaises(ValueError): core.verify_strict_final_writes('/fixture', training_commit=COMMIT)

    def test_explicit_original_source_and_native4_required_before_dependencies(self):
        valid = dict(run_id='a'*32, source_commit=COMMIT, easy2560_selection=True, training_world_size=4)
        cases = [(valid, None), (valid, True), (valid, 'f'*40)]
        cases += [(dict(valid, training_world_size=value), COMMIT) for value in (None, True, 4.0, 8)]
        for run, commit in cases:
            with self.subTest(run=run, commit=commit), \
                 patch.object(core.final_cpu_acceptance, 'small', return_value=(run, {})), \
                 patch.object(core.final_cpu_acceptance, 'published_source_binding') as source, \
                 patch.object(core.checkpoint_evidence, 'authenticated_restore_chain') as chain, \
                 patch.object(core.final_cpu_acceptance, 'no201') as no201:
                with self.assertRaises(ValueError): core.verify_strict_final_writes('/fixture', training_commit=commit)
                source.assert_not_called(); chain.assert_not_called(); no201.assert_not_called()

    def test_cli_success_is_one_json_report_and_requires_explicit_commit(self):
        with patch.object(core, 'verify_strict_final_writes', return_value={'status': core.STATUS}) as verify, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            code = core.main(['--base', '/fixture', '--training-commit', COMMIT])
            self.assertEqual(code, 0); self.assertEqual(json.loads(output.getvalue()), {'status': core.STATUS})
            verify.assert_called_once_with('/fixture', training_commit=COMMIT)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            core.main(['--base', '/fixture'])
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
