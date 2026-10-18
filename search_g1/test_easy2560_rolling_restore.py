"""CPU provenance regressions for retained full-state arbitrary-outer recovery.

The synthetic files do not claim CUDA acceptance. Production uses the native
rank validator and actual worker restore; these tests exercise saved boundaries,
lineage flattening, source selection, and failures without GPU allocation.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from search_g1 import easy2560_restore as module
from search_g1.test_easy2560_restore import Fixture, write


def equal_fixture(a, b):
    if a != b: raise AssertionError('Fixture tree differs')


def extend(f, outer, *, first=1, job='730226', micro=16):
    """Make a real synthetic suffix with global, unrestarted counters."""
    destination = f.cp.parent / ('global_step_' + str(outer))
    if destination != f.cp:
        f.cp.rename(destination); f.cp = destination; f.state = destination/'training_state'
    f.run['actor_micro_batch_per_rank'] = micro
    f.run['budget'].update(ppo_micro_batch_size=micro*4, per_rank_actor_micro_batch=micro,
        ppo_mini_batch_size=128, per_rank_actor_logical_micro_batch=4,
        objective_reduction='original four-row micro masked_mean')
    write(f.base/'logs/run-manifest.json', f.run)
    write(f.base/'logs/native-allocation-admission.json', dict(accepted=True, run_id=f.run_id,
        training_world_size=4, slurm_job_id=job))
    f.ids = [f'{n:032x}' for n in range(1, outer+1)]
    f.progress = dict(completed_outer=outer, cumulative_writes=outer*20, attempt_index=outer,
        aborted_attempts=0, full_budget_outer=outer)
    for rank, saved in enumerate(f.rank_states):
        saved.update(source_commit=f.run['source_commit'], completed_outer=outer,
            actual_optimizer_writes=outer*20, last_attempt_index=outer, used_attempt_ids=f.ids,
            scheduler=dict(last_epoch=outer, _step_count=outer+1))
        write(f.state/f'rank-{rank}.pt', saved)
    f.driver.update(source_commit=f.run['source_commit'], completed_outer=outer,
        attempt_progress=copy.deepcopy(f.progress), reward_refresh_pending=outer in (50,100,150))
    f.driver['candidate_stream']['consumed_source_rows'] = outer*2560
    write(f.state/'driver.pt', f.driver)
    refresh(f)
    updates, attempts = [], []
    for step in range(first, outer+1):
        aid = f.ids[step-1]
        gate = f.base/'logs'/('behavior-probability-attempt-'+aid+'.json')
        write(gate, dict(run_id=f.run_id, attempt_id=aid, logical_update=step))
        write(f.base/'artifacts/mixed-selection'/(aid+'.json'), dict(step=step))
        updates.append(dict(step=step, run_id=f.run_id, attempt_id=aid, attempt_index=step,
            optimizer_writes_before=(step-1)*20, optimizer_writes_after=step*20,
            actual_optimizer_writes=20, behavior_probability_gate_sha256=module.digest(gate)))
        attempts.append(dict(attempt_index=step, attempt_id=aid, run_id=f.run_id, slurm_job_id=job))
    f.lines('logs/actual-updates.jsonl', updates); f.lines('logs/actual-attempts.jsonl', attempts)
    for role in ('optimizer', 'scheduler'):
        for rank in range(4):
            f.lines(f'logs/{role}-rank-{rank}.jsonl', [dict(rank=rank, run_id=f.run_id,
                context=dict(attempt_index=step)) for step in range(first,outer+1)])
    if outer < 5:
        (f.base/'logs/formal-inline-development-after.json').unlink()
        (f.base/'logs/validation-identities.jsonl').unlink()


def refresh(f):
    outer = f.driver['completed_outer']; writes = f.progress['cumulative_writes']
    rows = [dict(path=str(f.state/f'rank-{rank}.pt'), rank=rank,
        actual_optimizer_writes=writes, run_id=f.run_id, sha256=module.digest(f.state/f'rank-{rank}.pt'))
        for rank in range(4)]
    write(f.state/'manifest.json', dict(completed_outer=outer, question_probe=None,
        worker_files=rows, driver_file=dict(path=str(f.state/'driver.pt'),sha256=module.digest(f.state/'driver.pt'))))
    write(f.cp/'complete.json', dict(step=outer, run_id=f.run_id, checkpoint=str(f.cp),
        finite_all_saved_tensors=True, validated_tensor_count=1, actual_optimizer_writes_per_rank=writes,
        files_sha256={str(p.relative_to(f.cp)):module.digest(p) for p in f.cp.rglob('*')
                      if p.is_file() and p.name!='complete.json'}))


def boundary(f):
    outer = f.driver['completed_outer']; folder = f.base/'preparation/rolling-checkpoints'/f'global_step_{outer}'/'boundary'
    folder.mkdir(parents=True)
    driver = copy.deepcopy(f.driver)
    driver.update(callbacks_complete=True, reward_refresh_pending=False, rng=dict(torch_cpu=[90],numpy=[91],python=[92]))
    adopted_round=min(outer//50,3)*50
    if adopted_round:
        old = next((f.base/'artifacts').glob('round_*/version-manifest.json'))
        probe = module._json(old)
        probe.update(round=adopted_round, version=f.run_id+':'+str(adopted_round)+':adopted')
        write(f.base/'artifacts'/f'round_{adopted_round}-adopted/version-manifest.json', probe)
        driver.update(probe_round=adopted_round, probe_version=probe['version'])
    rows = []
    for rank, value in enumerate(f.rank_states):
        saved = copy.deepcopy(value); saved.pop('optimizer'); saved.pop('scheduler')
        saved['rng'] = dict(fixture_cpu_rng=['post-callback',rank])
        path=folder/f'rank-{rank}.pt';write(path,saved)
        rows.append(dict(path=str(path),rank=rank,sha256=module.digest(path),bytes=path.stat().st_size))
    p=folder/'driver.pt';write(p,driver)
    value=dict(schema_version=1,run_id=f.run_id,source_commit=f.run['source_commit'],completed_outer=outer,
        callbacks_complete=True,worker_files=rows,driver_file=dict(path=str(p),sha256=module.digest(p),bytes=p.stat().st_size),
        probe_version=driver['probe_version'],probe_round=driver['probe_round'],reward_refresh_pending=False)
    write(folder/'manifest.json',value)
    return folder, driver


def create(f):
    with patch.object(module,'_load_cpu',side_effect=lambda path:json.loads(Path(path).read_text())), \
         patch.object(module,'_validate_rank',side_effect=Fixture.rank_check), \
         patch('search_g1.full_state_restore.equal_tree',side_effect=equal_fixture):
        return module.create_receipt(f.base,f.cp,f.output)


def restore_edge(f, parent):
    receipt=module._json(parent.output); outer=receipt['step']; sha=module.digest(parent.output)
    f.run['full_state_restore']=dict(kind=module.KIND, source_receipt=str(parent.output), source_receipt_sha256=sha,
        source_base=str(parent.base),source_job=receipt['source_job'],source_commit=receipt['source_commit'],
        completed_outer=outer,attempt_progress=receipt['attempt_progress'])
    write(f.base/'logs/run-manifest.json',f.run)
    rows=[dict(rank=rank,source_step=outer,source_run_id=f.run_id,source_receipt_sha256=sha,
        actual_adam_scheduler_readback_equal=True,actual_rng_readback_equal=True,
        actual_model_collective_readback_equal=True,actual_optimizer_writes=receipt['attempt_progress']['cumulative_writes'],
        scheduler_last_epoch=outer,last_attempt_index=receipt['attempt_progress']['attempt_index'],
        model_hf_exact_tensors_checked_on_rank0=1 if rank==0 else 0) for rank in range(4)]
    write(f.base/'logs/full-state-restored-workers.json',dict(ranks=rows,source_receipt_sha256=sha,source_step=outer))
    write(f.base/'logs/full-state-restored-driver.json',dict(completed_outer=outer,
        actual_driver_rng_readback_equal=True,actual_candidate_stream_readback_equal=True,
        attempt_progress=receipt['attempt_progress'],actual_candidate_source_cursor=receipt['driver_state']['source_consumed']))


class RollingRestoreTests(unittest.TestCase):
    def test_arbitrary_outer_post_callback_rng_and_no_invented_QA_before5(self):
        for outer in (1,6,49,199):
            with self.subTest(outer=outer), tempfile.TemporaryDirectory() as td:
                f=Fixture(td);extend(f,outer,micro=8);folder,saved=boundary(f);create(f)
                _,receipt,_,_,_=module.source(f.output)
                self.assertEqual(receipt['step'],outer);self.assertTrue(receipt['post_callback_rng_captured'])
                self.assertEqual(receipt['driver_file']['path'],str(folder/'driver.pt'))
                target=copy.deepcopy(f.run);target.update(run_id='new',source_commit='9'*40)
                module.bind(target,f.output)
                self.assertEqual(target['full_state_restore']['completed_outer'],outer)
                self.assertEqual(target['full_state_restore']['actual_optimizer_writes_per_rank'],outer*20)
                with patch.object(module,'_load_cpu',side_effect=lambda path:module._json(path)):
                    rng=module.boundary_worker_rng(receipt,2,f.rank_states[2])
                self.assertEqual(rng,dict(fixture_cpu_rng=['post-callback',2]))
                self.assertNotEqual(rng,f.rank_states[2]['rng'])
                if outer<5:self.assertIsNone(receipt['historical_artifacts']['development']['after5'])

    def test_milestone_uses_adopted_probe_and_actual_post_refit_driver(self):
        with tempfile.TemporaryDirectory() as td:
            f=Fixture(td);extend(f,50,micro=8);folder,saved=boundary(f);create(f)
            _,receipt,_,_,_=module.source(f.output)
            self.assertFalse(receipt['reward_refresh_pending'])
            self.assertEqual(receipt['reward_probe_round'],50)
            self.assertEqual(receipt['driver_state']['probe_version'],saved['probe_version'])
            self.assertTrue(receipt['driver_state']['callbacks_complete'])
            target=copy.deepcopy(f.run);module.bind(target,f.output)
            self.assertEqual(target['initial_probe_version'],saved['probe_version'])
            destination=Path(td)/'new';write(destination/'logs/run-manifest.json',target)
            installed=module.install_restored_reward_probe(destination,target)
            self.assertEqual(installed['round'],50);self.assertFalse(installed['new_calibration_generated'])
            self.assertFalse(Path(receipt['reward_probe_manifest']['path']).name=='live')

    def test_existing_pre_refit50_complete_checkpoint_can_resume_pending_refit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);parent=Fixture(root/'original');extend(parent,5);parent.create()
            f=Fixture(root/'current');f.run['source_commit']='9'*40
            extend(f,50,first=6,job='734689',micro=8);restore_edge(f,parent)
            # Existing 4cc has no post-callback sidecar at milestone50. It is
            # still a real full recovery point, with its refit returned pending.
            create(f);_,receipt,_,_,_=module.source(f.output)
            self.assertEqual(receipt['step'],50)
            self.assertEqual(receipt['reward_probe_round'],0)
            self.assertTrue(receipt['reward_refresh_pending'])
            self.assertFalse(receipt['post_callback_rng_captured'])
            target=copy.deepcopy(f.run);target.pop('full_state_restore')
            module.bind(target,f.output)
            self.assertEqual(target['full_state_restore']['completed_outer'],50)
            self.assertTrue(target['full_state_restore']['reward_refresh_pending'])
            self.assertEqual(target['full_state_restore']['actual_optimizer_writes_per_rank'],1000)

    def test_post_callback_driver_restore_reads_boundary_payload(self):
        import os
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            f=Fixture(td);extend(f,50,micro=8);folder,saved=boundary(f);create(f)
            target=copy.deepcopy(f.run);module.bind(target,f.output)
            destination=Path(td)/'new';write(destination/'logs/run-manifest.json',target)
            module.install_restored_reward_probe(destination,target)
            seen=[]
            def restore(trainer,actual,run,**kwargs):
                seen.append(actual)
                return dict(completed_outer=50,actual_driver_rng_readback_equal=True,
                    actual_candidate_stream_readback_equal=True,actual_candidate_source_cursor=50*2560,
                    attempt_progress=actual['attempt_progress'],actual_probe_version=actual['probe_version'],actual_probe_round=50)
            fake=SimpleNamespace(restore_easy2560_driver=restore)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(destination)), \
                 patch.dict('sys.modules',{'search_g1.training_state':fake}), \
                 patch.object(module,'_load_cpu',side_effect=lambda path:module._json(path)):
                proof=module.restore_driver(SimpleNamespace(),f.output)
            self.assertEqual(seen,[saved])
            self.assertNotEqual(seen[0]['rng'],f.driver['rng'])
            self.assertTrue(proof['post_callback_rng_captured']);self.assertEqual(proof['source_step'],50)

    def test_chained_current_full_save_survives_deletion_of_parent_tensor_directory(self):
        for first,second,third in ((5,6,7),(6,7,8)):
            with self.subTest(lineage=(first,second,third)), tempfile.TemporaryDirectory() as td:
                root=Path(td)
                parent=Fixture(root/'first');extend(parent,first)
                if first==5:parent.create()  # Existing pre-QA CP5 stays compatible.
                else:boundary(parent);create(parent)
                child=Fixture(root/'second');child.run['source_commit']='9'*40
                extend(child,second,first=second,job='734689',micro=8);restore_edge(child,parent);boundary(child)
                shutil.rmtree(parent.cp)
                create(child)
                _,receipt,_,_,_=module.source(child.output)
                self.assertEqual([r['step'] for r in receipt['authenticated_updates']],list(range(1,second+1)))
                self.assertEqual(len(receipt['lineage']),1)
                self.assertFalse(any(Path(row['path']).is_relative_to(parent.cp)
                    for row in receipt['historical_artifacts']['prefix_files']))
                newest=Fixture(root/'third');newest.run['source_commit']='a'*40
                extend(newest,third,first=third,job='735000',micro=4);restore_edge(newest,child);boundary(newest)
                shutil.rmtree(child.cp)
                create(newest);_,last,_,_,_=module.source(newest.output)
                self.assertEqual([r['step'] for r in last['authenticated_updates']],list(range(1,third+1)))
                self.assertEqual([r['completed_outer'] for r in last['lineage']],[first,second])
                target=copy.deepcopy(newest.run);target.pop('full_state_restore')
                module.bind(target,newest.output)
                self.assertEqual(target['full_state_restore']['completed_outer'],third)

    def test_partial_or_corrupt_boundary_is_not_a_restart_source(self):
        for bad in ('driver','worker','callbacks'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as td:
                f=Fixture(td);extend(f,7,micro=8);folder,_=boundary(f)
                if bad=='callbacks':
                    value=module._json(folder/'manifest.json');value['callbacks_complete']=False;write(folder/'manifest.json',value)
                else:(folder/('driver.pt' if bad=='driver' else 'rank-2.pt')).write_bytes(b'incomplete')
                with self.assertRaises(ValueError):create(f)
                self.assertFalse(f.output.exists())

    def test_missing_actual_restore_readback_and_changed_parent_receipt_fail_chain(self):
        for bad in ('data','receipt','rank'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as td:
                root=Path(td);parent=Fixture(root/'one');extend(parent,5);parent.create()
                child=Fixture(root/'two');extend(child,6,first=6,job='734689');restore_edge(child,parent);boundary(child)
                if bad=='receipt':parent.output.write_text(parent.output.read_text()+' ')
                elif bad=='data':
                    p=child.base/'logs/full-state-restored-driver.json';value=module._json(p)
                    value['actual_candidate_stream_readback_equal']=False;write(p,value)
                else:
                    p=child.base/'logs/full-state-restored-workers.json';value=module._json(p);value['ranks'].pop();write(p,value)
                with self.assertRaises(ValueError):create(child)

    def test_latest_selection_uses_verified_pointer_and_never_searches_max_directory(self):
        with tempfile.TemporaryDirectory() as td:
            f=Fixture(td);extend(f,6);boundary(f)
            with patch('search_g1.easy2560_rolling_checkpoint.latest_checkpoint',return_value={'checkpoint':str(f.cp)}) as latest, \
                 patch.object(module,'_load_cpu',side_effect=lambda path:module._json(path)), \
                 patch.object(module,'_validate_rank',side_effect=Fixture.rank_check), \
                 patch('search_g1.full_state_restore.equal_tree',side_effect=equal_fixture):
                module.create_receipt(f.base,'latest',f.output)
            latest.assert_called_once_with(f.base,verify=True)
            self.assertEqual(module._json(f.output)['step'],6)

    def test_capture_boundary_freezes_probe_JSON_and_marks_final_evaluation_pending(self):
        from types import SimpleNamespace
        import os
        from search_g1.easy2560_restore_runtime import capture_boundary
        with tempfile.TemporaryDirectory() as td:
            f=Fixture(td);extend(f,50)
            folder,saved=boundary(f)
            shutil.rmtree(folder)
            saved['callbacks_complete']=True
            probe=module._json(next((f.base/'artifacts').glob('round_50*/version-manifest.json')))
            write(f.base/'artifacts/live/manifest.json',probe)
            def atomic(path,value):
                write(path,value)
                return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=module.digest(path))
            def capture(directory,checkpoint=False):
                self.assertFalse(checkpoint)
                records=[]
                for rank,value in enumerate(f.rank_states):
                    state=copy.deepcopy(value);state.pop('optimizer');state.pop('scheduler')
                    records.append(dict(atomic(Path(directory)/f'rank-{rank}.pt',state),rank=rank))
                return records
            def driver(trainer,run,*,callbacks_complete=False):
                self.assertTrue(callbacks_complete);return copy.deepcopy(saved)
            trainer=SimpleNamespace(global_steps=50,actor_rollout_wg=SimpleNamespace(capture_mixed_state=capture))
            fake=SimpleNamespace(atomic_save=atomic,capture_easy2560_driver=driver)
            with patch.dict(os.environ,SEARCH_G1_BASE=str(f.base)), \
                 patch.dict('sys.modules',{'search_g1.training_state':fake}):
                path=Path(capture_boundary(trainer,f.cp))
            value=module._json(path)
            snapshot=Path(value['reward_probe_manifest']['path'])
            self.assertEqual(module._json(snapshot),probe)
            write(f.base/'artifacts/live/manifest.json',dict(probe,version='later-round100',round=100))
            self.assertEqual(module._json(snapshot),probe)
            self.assertEqual(module.digest(snapshot),value['reward_probe_manifest']['sha256'])
            self.assertTrue(value['callbacks_complete']);self.assertFalse(value['reward_refresh_pending'])
            self.assertEqual(value['probe_round'],50)

    def test_completed200_checkpoint_cannot_bind_training_outer201(self):
        with tempfile.TemporaryDirectory() as td:
            f=Fixture(td);extend(f,199);boundary(f);create(f)
            source=list(module.source(f.output));source[1]=dict(source[1],step=200)
            with patch.object(module,'source',return_value=tuple(source)),self.assertRaisesRegex(ValueError,'PRECEDE_OUTER200'):
                module.bind(copy.deepcopy(f.run),f.output)


if __name__=='__main__':unittest.main()
