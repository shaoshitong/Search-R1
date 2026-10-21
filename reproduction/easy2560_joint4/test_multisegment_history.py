"""CPU provenance tests for CP5 -> CP50 -> an actual new execution segment.

Opaque fixture bytes are never presented as an actual CUDA reload. These tests
exercise the posthoc consumer's exact ledger routing and stale-suffix rejection.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reproduction.easy2560_joint4 import checkpoint_evidence as evidence
from reproduction.easy2560_joint4 import final_cpu_acceptance as final
from reproduction.easy2560_joint4.test_authenticated_history import bind_fixture, ledger_rows, record, put, NEW
from reproduction.easy2560_joint4.test_final_cpu_acceptance import FinalAcceptanceTests

THIRD = 'e' * 40


def bounded(path, terminal):
    chunks = []
    for raw in path.read_bytes().splitlines(keepends=True):
        row = json.loads(raw); chunks.append(raw)
        if terminal(row): break
    return record(path, str(path.relative_to(path.parents[1])), b''.join(chunks))


def three_segments(root, source_outer=50, tensor_count=435):
    old, middle, _ = bind_fixture(root, final_mode=True)
    current = FinalAcceptanceTests('runTest'); current.fixture(root/'third', training_commit=THIRD, physical=8)
    ledger_rows(current, False, '730400')
    cp = middle.base/('checkpoints/search-g1-common8/actor/global_step_%d' % source_outer)
    if source_outer != 50:
        shutil.copytree(middle.base/'checkpoints/search-g1-common8/actor/global_step_50',cp)
    prior = evidence.authenticated_restore_chain(middle.base, middle.run)
    history = {}
    for rel in ['logs/actual-updates.jsonl', 'logs/actual-attempts.jsonl'] + [
            'logs/%s-rank-%d.jsonl' % (role, rank) for role in ('scheduler', 'optimizer') for rank in range(4)]:
        path = middle.base/rel
        terminal = (lambda row: row.get('step') == source_outer) if rel.endswith('actual-updates.jsonl') else (
            (lambda row: row.get('attempt_index') == source_outer) if rel.endswith('actual-attempts.jsonl') else
            (lambda row: evidence.attempt_identity(row)[0] == source_outer and row['kind'] in ('COMPLETE','OUTER_BUDGET')))
        history[rel] = bounded(path, terminal)
    rel = 'logs/validation-identities.jsonl'
    if source_outer >= 50:
        history[rel] = bounded(middle.base/rel, lambda row: row['step'] == source_outer)
    for rel in ['logs/native-allocation-admission.json', 'preparation/run-source.json',
                'source/'+NEW+'/source-publication.json']:
        history[rel] = record(middle.base/rel, rel)
    updates = final.chained_lines(middle.base, 'logs/actual-updates.jsonl', prior)[:source_outer]
    attempts = final.chained_lines(middle.base, 'logs/actual-attempts.jsonl', prior)[:source_outer]
    progress = dict(completed_outer=source_outer, cumulative_writes=95+source_outer, attempt_index=source_outer, aborted_attempts=0, full_budget_outer=5)
    files = [record(cp/('training_state/rank-%d.pt' % rank)) for rank in range(4)]
    receipt = dict(schema_version=2, status=evidence.RESTORE_STATUS, kind='easy2560_native4',
        hf_saved_tensor_count=tensor_count,
        source_base=str(middle.base), checkpoint=str(cp), step=source_outer, world_size=4, run_id=middle.run_id,
        source_job=middle.job, source_commit=NEW, actual_optimizer_writes_per_rank=95+source_outer,
        attempt_progress=progress, source_run_manifest=record(middle.base/'logs/run-manifest.json'),
        files=files, driver_file=record(cp/'training_state/driver.pt'), driver_state=dict(source_consumed=source_outer*2560),
        reward_probe_version='round-fixture', reward_probe_round=(source_outer//50)*50, post_callback_rng_captured=True,
        authenticated_updates=updates, authenticated_attempts=attempts, historical_files=history,
        historical_artifacts=dict(prefix_files=list(history.values())))
    rp = put(current.base/'restore50-receipt.json', receipt); sha = evidence.digest(rp.read_bytes())
    current.run['full_state_restore'] = dict(kind='easy2560_native4', source_receipt=str(rp), source_receipt_sha256=sha,
        source_base=str(middle.base), source_checkpoint=str(cp), completed_outer=source_outer,
        source_commit=NEW, source_job=middle.job, actual_optimizer_writes_per_rank=95+source_outer)
    put(current.base/'logs/run-manifest.json', current.run)
    uuids = ['GPU-third-%d' % rank for rank in range(4)]
    put(current.base/'logs/native-allocation-admission.json', dict(accepted=True, run_id=current.run_id,
        slurm_job_id=current.job, actual_cuda_runtime_uuids=uuids))
    ranks = []
    for rank in range(4):
        ranks.append(dict(rank=rank, run_id=current.run_id, source_run_id=middle.run_id,
            source_receipt_sha256=sha, source_job=middle.job, source_commit=NEW, source_step=source_outer,
            source_checkpoint=str(cp), source_state_full_sha256=files[rank]['full_read_sha256'],
            slurm_job_id=current.job, actual_cuda_uuid=uuids[rank], world_size=4,
            actual_optimizer_writes=95+source_outer, scheduler_last_epoch=source_outer, last_attempt_index=source_outer,
            actual_model_collective_readback_equal=True, actual_adam_scheduler_readback_equal=True,
            actual_rng_readback_equal=True, model_hf_exact_tensors_checked_on_rank0=tensor_count if rank==0 else 0,
            process_identity=dict(pid=2000+rank,start_ticks=19000+rank)))
    put(current.base/'logs/full-state-restored-workers.json', dict(run_id=current.run_id, source_commit=THIRD,
        slurm_job_id=current.job, source_receipt_sha256=sha, actual_world_size=4, ranks=ranks))
    put(current.base/'logs/full-state-restored-driver.json', dict(run_id=current.run_id, source_step=source_outer,
        source_job=middle.job, source_commit=NEW, slurm_job_id=current.job, source_receipt_sha256=sha,
        source_driver_full_sha256=receipt['driver_file']['full_read_sha256'],
        actual_candidate_source_cursor=source_outer*2560, attempt_progress=progress,
        actual_probe_version=receipt['reward_probe_version'], actual_probe_round=receipt['reward_probe_round'],
        actual_driver_rng_readback_equal=True, actual_candidate_stream_readback_equal=True,
        post_callback_rng_captured=True, same_job_continuation=False))
    current.ledger('logs/actual-updates.jsonl', current.updates[source_outer:])
    current.ledger('logs/actual-attempts.jsonl', current.attempts[source_outer:])
    for rank in range(4):
        for role, rows in [('scheduler',current.schedulers[rank]),('optimizer',current.workers[rank])]:
            current.ledger('logs/%s-rank-%d.jsonl' % (role,rank),
                           [r for r in rows if evidence.attempt_identity(r)[0] > source_outer])
    current.ledger('logs/validation-identities.jsonl', [r for r in current.validations if r['step'] > source_outer])
    for step in final.STEPS:
        path=current.base/('checkpoints/search-g1-common8/actor/global_step_%d/complete.json' % step)
        path.unlink()
        current.make_checkpoint(step)
    return old, middle, current


class MultisegmentHistoryTests(unittest.TestCase):
    def test_generic_restore_chain_accepts339_and_rejects_incomplete_model_readback(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, current = three_segments(Path(td), tensor_count=339)
            self.assertEqual(evidence.authenticated_restore_chain(current.base,current.run)['boundary'], 50)
            path = current.base/'logs/full-state-restored-workers.json'
            workers = json.loads(path.read_bytes())
            workers['ranks'][0]['model_hf_exact_tensors_checked_on_rank0'] = 338
            put(path, workers)
            with self.assertRaisesRegex(ValueError, 'RESTORED_RANK0_FULL_MODEL'):
                evidence.authenticated_restore_chain(current.base,current.run)

    def test_three_sources_keep_exact_identities_and_ignore_unaccepted_middle_suffix(self):
        with tempfile.TemporaryDirectory() as td:
            old, middle, current = three_segments(Path(td))
            chain = final.restore_chain(current.base,current.run)
            progress = final.no201(current.base,current.run,chain)
            self.assertTrue(progress['no_update201'])
            calls=[]
            verifier=SimpleNamespace(verify_attempt=lambda base,run,update,expected_job:
                calls.append((str(base),run['source_commit'],update['step'],expected_job)) or dict(step=update['step']))
            with patch.dict('sys.modules', {'reproduction.easy2560_joint4.selection_evidence':verifier}):
                result=final.selection_evidence(current.base,current.run,chain)
            self.assertEqual([(s['first_outer'],s['last_outer'],s['slurm_job_id']) for s in result['source_segments']],
                             [(1,5,old.job),(6,50,middle.job),(51,200,current.job)])
            self.assertEqual({r[3] for r in calls[:5]}, {old.job})
            self.assertEqual({r[3] for r in calls[5:50]}, {middle.job})
            self.assertEqual({r[3] for r in calls[50:]}, {current.job})
            self.assertEqual(final.evaluations(current.base,current.run,current.load_canonical,chain)['fixed512']['50']['question_count'],512)
            # Native model/Adam receipt SHA/readbacks remain meaningful after a
            # superseded non-milestone CP payload is removed. Reader needs no tensor.
            shutil.rmtree(middle.base/'checkpoints/search-g1-common8/actor/global_step_50')
            self.assertEqual(final.restore_chain(current.base,current.run)['boundary'],50)

    def test_nested_small_sidecar_revalidates_offline_without_ancestor_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            old,middle,current=three_segments(Path(td))
            sealed=Path(td).resolve()/'sealed';sealed.mkdir();bundle=evidence.Bundle(sealed)
            first=evidence.authenticated_restore_chain(current.base,current.run,bundle=bundle)
            self.assertIn('chain/parent/historical/logs/actual-updates.jsonl',bundle.files)
            self.assertIn('chain/historical/logs/actual-updates.jsonl',bundle.files)
            shutil.rmtree(old.base);shutil.rmtree(middle.base)
            second=evidence.authenticated_restore_chain(current.base,current.run,sealed_directory=sealed)
            self.assertEqual(first['public'],second['public'])
            path=sealed/'chain/parent/historical/logs/actual-updates.jsonl'
            path.write_bytes(path.read_bytes().replace(b'"step":1',b'"step":2',1))
            with self.assertRaisesRegex(ValueError,'SEALED_HISTORICAL_FILE_DIFFERS'):
                evidence.authenticated_restore_chain(current.base,current.run,sealed_directory=sealed)

    def test_nonmilestone17_source_can_have_no_new_qa_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            old,middle,current=three_segments(Path(td),source_outer=17)
            chain=final.restore_chain(current.base,current.run)
            self.assertEqual(chain['boundary'],17)
            self.assertTrue(final.no201(current.base,current.run,chain)['no_update201'])
            self.assertEqual([row['step'] for row in evidence._historical_rows(chain,'logs/validation-identities.jsonl')],[5])

    def test_complete_final_cpu_accepts_cp50_resume_and_keeps_each_milestone_origin(self):
        with tempfile.TemporaryDirectory() as td:
            old,middle,current=three_segments(Path(td))
            with patch.object(final,'selection_evidence',return_value=dict(fixture_scope='dispatch separately tested')):
                result=final.verify_final(current.base,training_commit=THIRD,load_canonical=current.load_canonical)
            self.assertTrue(result['progress']['no_update201'])
            self.assertEqual(result['authenticated_history']['completed_outer'],50)
            self.assertEqual(result['checkpoints']['5']['source_base'],str(old.base))
            self.assertEqual(result['checkpoints']['50']['source_base'],str(middle.base))
            self.assertEqual(result['checkpoints']['100']['source_checkpoint'],
                             str(current.base/'checkpoints/search-g1-common8/actor/global_step_100'))

    def test_wrong_restore_rng_model_or_current_suffix_identity_rejected(self):
        for mutation in ('rng','model','suffix_job','post201'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                old,middle,current=three_segments(Path(td))
                if mutation in ('rng','model'):
                    p=current.base/'logs/full-state-restored-workers.json';value=json.loads(p.read_bytes())
                    value['ranks'][2]['actual_rng_readback_equal' if mutation=='rng' else 'actual_model_collective_readback_equal']=False
                    put(p,value)
                    with self.assertRaisesRegex(ValueError,'RESTORED_RANK'):final.restore_chain(current.base,current.run)
                else:
                    if mutation=='suffix_job':
                        p=current.base/'logs/actual-attempts.jsonl'; rows=list(final.lines(p));rows[0]['slurm_job_id']=middle.job
                    else:
                        p=current.base/'logs/actual-updates.jsonl';rows=list(final.lines(p));rows.append(dict(rows[-1],step=201))
                    put(p,b''.join(evidence.canonical(r)+b'\n' for r in rows))
                    with self.assertRaises(ValueError):final.no201(current.base,current.run)


if __name__ == '__main__': unittest.main()
