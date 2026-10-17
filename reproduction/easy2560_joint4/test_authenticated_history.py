"""Opaque CPU fixtures authenticate two real source identities, never reload a model."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction.easy2560_joint4 import checkpoint_evidence as evidence
from reproduction.easy2560_joint4 import final_cpu_acceptance as final
from reproduction.easy2560_joint4 import test_checkpoint_evidence as seal_fixture
from reproduction.easy2560_joint4 import test_final_cpu_acceptance as final_fixture

OLD = '8a5bac94b8230a99b81977f5290589e3d3708dfe'
NEW = 'f'*40


def record(path, relative=None, raw=None):
    value = path.read_bytes() if raw is None else raw
    st = path.stat()
    result = dict(path=str(path), bytes=len(value), full_read_sha256=evidence.digest(value),
                  device=st.st_dev, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
    if relative is not None: result['relative_path'] = relative
    if raw is not None: result['prefix_only'] = True
    return result


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else evidence.encoded(value))
    return path


def ledger_rows(fixture, is_seal, job):
    """First5 actually have100 writes; later partial-safe outers each commit1."""
    writes = 0
    for a, u in zip(fixture.attempts, fixture.updates):
        before = writes; count = 20 if a['completed_outer'] <= 5 else 1; writes += count
        a.update(slurm_job_id=job, optimizer_writes_before=before, optimizer_writes_after=writes,
                 optimizer_writes_this_attempt=count, opportunities_processed=count, scheduler_advances=1)
        u.update(optimizer_writes_before=before, optimizer_writes_after=writes,
                 actual_optimizer_writes=count)
    for rank in range(4):
        for row in fixture.schedulers[rank]:
            step = row['logical_update']; a = fixture.attempts[step-1]
            row.update(slurm_job_id=job, writes_after=a['optimizer_writes_after'],
                       scheduler_last_epoch_after=step, scheduler_advances=1)
        rebuilt = []
        for a in fixture.attempts:
            aid = a['attempt_id']; old = [r for r in fixture.workers[rank]
                if (r.get('context') or r).get('attempt_id') == aid]
            for row in old:
                row = copy.deepcopy(row)
                if 'context' in row: row['context']['logical_update'] = a['logical_update']
                if row['kind'] == 'OPTIMIZER_WRITE':
                    rebuilt += [dict(row, write_count=n) for n in
                        range(a['optimizer_writes_before']+1, a['optimizer_writes_after']+1)]
                else:
                    if row['kind'] == 'OUTER_BUDGET': row['cumulative_writes'] = a['optimizer_writes_after']
                    rebuilt.append(row)
        fixture.workers[rank] = rebuilt
    fixture.job = job
    if is_seal:
        for a,u in zip(fixture.attempts,fixture.updates):
            path=fixture.base/('logs/behavior-probability-attempt-'+a['attempt_id']+'.json')
            gate=json.loads(path.read_bytes());gate['slurm_job_id']=job;put(path,gate)
            u['behavior_probability_gate_sha256']=evidence.digest(path.read_bytes())
        fixture.driver['attempt_progress'].update(cumulative_writes=writes, full_budget_outer=5)
        for row in fixture.state['worker_files']: row['actual_optimizer_writes'] = writes
    return writes


def save_seal(f):
    f.run['checkpoint_dir'] = str(f.base/'checkpoints/search-g1-common8')
    f.write('logs/run-manifest.json', f.run)
    f.save_driver()
    f.write(str(f.state_path.relative_to(f.base)), f.state)
    f.complete.update(checkpoint=str(f.cp), actual_optimizer_writes_per_rank=f.driver['attempt_progress']['cumulative_writes'])
    f.complete['files_sha256'] = {str(p.relative_to(f.cp)):evidence.digest(p.read_bytes())
        for p in f.cp.rglob('*') if p.is_file() and p.name != 'complete.json'}
    f.write(str(f.complete_path.relative_to(f.base)), f.complete)
    for name, row in f.snapshot['files'].items():
        path = Path(row['path']); row.update(bytes=path.stat().st_size, sha256=evidence.digest(path.read_bytes()))
    for path in f.cp.rglob('*'):
        if path.is_file(): f.snapshot['files']['checkpoint/'+str(path.relative_to(f.cp))] = dict(
            path=str(path), bytes=path.stat().st_size, sha256=evidence.digest(path.read_bytes()))
    f.snapshot.update(completed_update=f.updates[-1], question_probe_saved=False,
                      total_bytes=sum(r['bytes'] for r in f.snapshot['files'].values()))
    f.write(str(f.snapshot_path.relative_to(f.base)), f.snapshot)
    f.accepted.update(snapshot_sha256=evidence.digest(f.snapshot_path.read_bytes()),
                      total_bytes=f.snapshot['total_bytes'], file_count=len(f.snapshot['files']))
    f.write(str(f.accepted_path.relative_to(f.base)), f.accepted)


def bind_fixture(root, *, final_mode=False):
    old = seal_fixture.EvidenceTests('runTest'); old.fixture(root/'old', fixed5=True, training_commit=OLD, physical=16)
    ledger_rows(old, True, '730226')
    old.write(str((old.cp/'model-00001-of-00001.safetensors').relative_to(old.base)), b'opaque fixture HF model')
    old.driver['candidate_stream']['consumed_source_rows'] = 12800
    save_seal(old)
    new = final_fixture.FinalAcceptanceTests('runTest') if final_mode else seal_fixture.EvidenceTests('runTest')
    if final_mode: new.fixture(root/'new', training_commit=NEW, physical=16)
    else: new.fixture(root/'new', step=50, training_commit=NEW, physical=16)
    ledger_rows(new, not final_mode, '730300')
    uuids = ['GPU-fixture-%d' % rank for rank in range(4)]
    allocation = dict(accepted=True, run_id=new.run_id, slurm_job_id=new.job, actual_cuda_runtime_uuids=uuids)
    put(new.base/'logs/native-allocation-admission.json', allocation)
    put(old.base/'logs/native-allocation-admission.json', dict(accepted=True, run_id=old.run_id, slurm_job_id=old.job))
    if final_mode:
        qa5 = new.after; validations = new.validations[:1]
    else:
        qa5 = dict(run_id=old.run_id, step=5, role='formal_inline_development_evaluation', rows=[{}]*512)
        validations = [dict(run_id=old.run_id, step=5, split='stage'), dict(run_id=old.run_id, step=5, split='validation')]
    put(old.base/'logs/formal-inline-development-after.json', qa5)
    old.lines('logs/validation-identities.jsonl', validations)
    historical = []
    data = {'logs/actual-updates.jsonl':old.updates, 'logs/actual-attempts.jsonl':old.attempts,
            'logs/validation-identities.jsonl':validations}
    for rank in range(4):
        data['logs/scheduler-rank-%d.jsonl'%rank] = old.schedulers[rank]
        data['logs/optimizer-rank-%d.jsonl'%rank] = old.workers[rank]
    for rel, rows in data.items():
        raw = b''.join(evidence.canonical(r)+b'\n' for r in rows)
        old.lines(rel, rows, b'{later unaccepted tail')
        historical.append(record(old.base/rel, rel, raw))
    for rel in ['logs/native-allocation-admission.json','logs/formal-inline-development-after.json']:
        historical.append(record(old.base/rel, rel))
    for rel, path in [('checkpoint/complete.json',old.cp/'complete.json'),
                      ('checkpoint/training_state/manifest.json',old.cp/'training_state/manifest.json')]:
        historical.append(record(path, rel))
    files = [record(old.cp/f'training_state/rank-{rank}.pt') for rank in range(4)]
    receipt = dict(status=evidence.RESTORE_STATUS, kind='easy2560_native4', source_base=str(old.base),
        checkpoint=str(old.cp), step=5, world_size=4, run_id=old.run_id, source_job=old.job, source_commit=OLD,
        actual_optimizer_writes_per_rank=100, attempt_progress=old.driver['attempt_progress'],
        source_run_manifest=record(old.base/'logs/run-manifest.json'), files=files,
        driver_file=record(old.cp/'training_state/driver.pt'), driver_state=dict(source_consumed=12800),
        reward_probe_version=old.probe['version'], reward_probe_round=0,
        reward_probe_manifest=record(old.probe_path),
        historical_artifacts=dict(prefix_files=historical))
    rp = put(new.base/'restore-receipt.json', receipt); sha = evidence.digest(rp.read_bytes())
    new.run['full_state_restore'] = dict(kind='easy2560_native4', source_receipt=str(rp), source_receipt_sha256=sha,
        source_base=str(old.base), source_checkpoint=str(old.cp), completed_outer=5,
        source_commit=OLD, source_job=old.job, actual_optimizer_writes_per_rank=100)
    new.run['checkpoint_dir'] = str(new.base/'checkpoints/search-g1-common8')
    put(new.base/'logs/run-manifest.json', new.run)
    ranks = []
    for rank in range(4):
        ranks.append(dict(rank=rank, run_id=new.run_id, source_run_id=old.run_id,
            source_receipt_sha256=sha, source_job=old.job, source_commit=OLD, source_step=5,
            source_checkpoint=str(old.cp), source_state_full_sha256=files[rank]['full_read_sha256'],
            slurm_job_id=new.job, actual_cuda_uuid=uuids[rank], world_size=4,
            actual_optimizer_writes=100, scheduler_last_epoch=5, last_attempt_index=5,
            actual_model_collective_readback_equal=True, actual_adam_scheduler_readback_equal=True,
            actual_rng_readback_equal=True, model_hf_exact_tensors_checked_on_rank0=435 if rank==0 else 0,
            process_identity=dict(pid=1000+rank,start_ticks=9000+rank)))
    put(new.base/'logs/full-state-restored-workers.json', dict(run_id=new.run_id, source_commit=NEW,
        slurm_job_id=new.job, source_receipt_sha256=sha, actual_world_size=4, ranks=ranks))
    put(new.base/'logs/full-state-restored-driver.json', dict(run_id=new.run_id, source_step=5,
        source_job=old.job, source_commit=OLD, slurm_job_id=new.job, source_receipt_sha256=sha,
        source_driver_full_sha256=receipt['driver_file']['full_read_sha256'],
        actual_candidate_source_cursor=12800, attempt_progress=old.driver['attempt_progress'],
        actual_probe_version=old.probe['version'], actual_probe_round=0,
        actual_driver_rng_readback_equal=True, actual_candidate_stream_readback_equal=True,
        post_callback_rng_captured=False, same_job_continuation=False))
    for rel, values in [('logs/actual-updates.jsonl',new.updates),('logs/actual-attempts.jsonl',new.attempts)]:
        put(new.base/rel, b''.join(evidence.canonical(r)+b'\n' for r in values if (r.get('step') or r['completed_outer'])>5))
    for rank in range(4):
        for role, values in [('scheduler',new.schedulers[rank]),('optimizer',new.workers[rank])]:
            rows = [r for r in values if evidence.attempt_identity(r)[0]>5]
            put(new.base/f'logs/{role}-rank-{rank}.jsonl', b''.join(evidence.canonical(r)+b'\n' for r in rows))
    if final_mode:
        put(new.base/'logs/validation-identities.jsonl',b''.join(evidence.canonical(r)+b'\n' for r in new.validations if r['step']>5))
        for step in final.STEPS:
            (new.base/f'checkpoints/search-g1-common8/actor/global_step_{step}/complete.json').unlink()
            new.make_checkpoint(step)
    else:
        for a in new.attempts[5:]:
            p=new.base/('logs/behavior-probability-attempt-'+a['attempt_id']+'.json')
            gate=json.loads(p.read_bytes());gate['slurm_job_id']=new.job;put(p,gate)
            new.updates[a['completed_outer']-1]['behavior_probability_gate_sha256']=evidence.digest(p.read_bytes())
        put(new.base/'logs/actual-updates.jsonl',b''.join(evidence.canonical(r)+b'\n' for r in new.updates[5:]))
        save_seal(new)
    return old, new, receipt


class AuthenticatedHistoryTests(unittest.TestCase):
    def test_cross_source50_seal_and_offline_verify_keep_two_job_identities(self):
        with tempfile.TemporaryDirectory() as td:
            old,new,_=bind_fixture(Path(td)); result=new.seal()
            self.assertFalse(result['single_original_job_prefix']);self.assertTrue(result['inherited_history_assembled'])
            self.assertEqual(result['authenticated_history']['source_job'],'730226')
            manifest=json.loads((Path(result['directory'])/'supplement.json').read_bytes())
            self.assertIn('chain/historical/logs/optimizer-rank-0.jsonl',manifest['files'])
            self.assertIn('segments/current/logs/optimizer-rank-0.jsonl',manifest['files'])
            refs=json.loads((Path(result['directory'])/'selected-large-references.json').read_bytes())
            self.assertEqual(refs[0]['source_commit'],OLD);self.assertEqual(refs[-1]['source_commit'],NEW)
            shutil.rmtree(old.base)
            self.assertEqual(evidence.verify(result['directory'])['authenticated_history'],result['authenticated_history'])

    def test_full200_final_reads_original_QA5_CP5_and_suffix_no201(self):
        with tempfile.TemporaryDirectory() as td:
            old,new,_=bind_fixture(Path(td),final_mode=True)
            result=new.verify(training_commit=NEW)
            self.assertEqual(result['progress']['cumulative_actual_optimizer_writes_per_rank'],295)
            self.assertTrue(result['progress']['no_update201']);self.assertFalse(result['single_original_job_prefix'])
            self.assertEqual(result['checkpoints']['5']['source_base'],str(old.base))
            self.assertEqual(result['evaluations']['fixed5_artifact']['path'],str(old.base/'logs/formal-inline-development-after.json'))
            self.assertEqual(result['evaluations']['final51713']['question_count'],51713)

    def test_original_archive_QA5_and_false_single_job_labels_cannot_be_substituted(self):
        for mutation in ('archive','QA5','single_job_label'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                old,new,_=bind_fixture(Path(td))
                if mutation=='single_job_label':
                    sealed=new.seal();folder=Path(sealed['directory'])
                    p=folder/'supplement.json';m=json.loads(p.read_bytes())
                    m.update(single_original_job_prefix=True,inherited_history_assembled=False,authenticated_history=None)
                    put(p,m);q=folder/'complete.json';marker=json.loads(q.read_bytes())
                    marker['supplement_sha256']=evidence.digest(p.read_bytes());put(q,marker)
                    with self.assertRaisesRegex(ValueError,'HISTORY_LABEL'):evidence.verify(folder)
                elif mutation=='QA5':
                    put(old.base/'logs/formal-inline-development-after.json',dict(run_id=old.run_id,step=5,rows=[]))
                    with self.assertRaises(ValueError):evidence.authenticated_restore_chain(new.base,new.run)
                else:
                    p=old.base/'preparation/checkpoint-archives/global_step_5/complete.json'
                    d=json.loads(p.read_bytes());d['source_commit']=NEW;put(p,d)
                    with self.assertRaisesRegex(ValueError,'CP5_ARCHIVE'):evidence.authenticated_restore_chain(new.base,new.run)

    def test_changed_old_prefix_receipt_sha_wrong_uuid_and_rank0_model_readback_fail(self):
        for mutation in ('prefix','receipt','uuid','rank0','rng','source_state','driver_cursor'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                old,new,_=bind_fixture(Path(td))
                if mutation=='prefix':
                    p=old.base/'logs/optimizer-rank-0.jsonl';p.write_bytes(p.read_bytes().replace(b'POLICY_MONITOR',b'POLICY_M0NITOR',1))
                elif mutation=='receipt':
                    p=new.base/'restore-receipt.json';p.write_bytes(p.read_bytes()+b' ')
                elif mutation=='driver_cursor':
                    p=new.base/'logs/full-state-restored-driver.json';d=json.loads(p.read_bytes());d['actual_candidate_source_cursor']=10240;put(p,d)
                else:
                    p=new.base/'logs/full-state-restored-workers.json';d=json.loads(p.read_bytes())
                    if mutation=='uuid':d['ranks'][1]['actual_cuda_uuid']='GPU-foreign'
                    if mutation=='rank0':d['ranks'][0]['model_hf_exact_tensors_checked_on_rank0']=434
                    if mutation=='rng':d['ranks'][2]['actual_rng_readback_equal']=False
                    if mutation=='source_state':d['ranks'][3]['source_state_full_sha256']='0'*64
                    put(p,d)
                with self.assertRaises(ValueError):evidence.authenticated_restore_chain(new.base,new.run)

    def test_relabelled_current_prefix_and_missing200_or_extra201_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            old,new,_=bind_fixture(Path(td),final_mode=True)
            path=new.base/'logs/actual-updates.jsonl';rows=list(final.lines(path))
            put(path,b''.join(evidence.canonical(r)+b'\n' for r in rows[:-1]))
            with patch.object(final,'restore_chain',side_effect=AssertionError('missing200 must fail first')):
                with self.assertRaisesRegex(ValueError,'MISSING200'):new.verify(training_commit=NEW)
            extra=dict(rows[-1],step=201)
            put(path,b''.join(evidence.canonical(r)+b'\n' for r in rows+[extra]))
            with self.assertRaisesRegex(ValueError,'MISSING200'):new.verify(training_commit=NEW)
            put(path,b''.join(evidence.canonical(r)+b'\n' for r in rows))
            p=new.base/'logs/scheduler-rank-0.jsonl';events=list(final.lines(p));events[0]['slurm_job_id']=old.job
            put(p,b''.join(evidence.canonical(r)+b'\n' for r in events))
            with self.assertRaisesRegex(ValueError,'SCHEDULER_NATIVE4'):new.verify(training_commit=NEW)


if __name__=='__main__':unittest.main()
