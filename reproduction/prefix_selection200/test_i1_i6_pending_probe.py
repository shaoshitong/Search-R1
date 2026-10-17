"""Constructed CPU fault boundaries; not a real four-GPU restore qualification."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pwd
import tempfile
import unittest
from unittest.mock import patch

from . import i1_i6_pending_probe as recovery
from . import launch
from .adaptive_candidates import ensure,policy
from .candidate_schedule import schedule_metadata
from .pool import initial_state,reserve_for_outer,state_sha
from .storage import actual_actor_boundary
from . import i1_i6_joint_resume as joint_recovery


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True));return value


def actor(outer,state):
    new=7;actual=7*outer
    ranks=[dict(rank=r,actual_adam_steps=actual,new_adam_steps=new,
        optimizer_writes_before=actual-new,optimizer_writes_after=actual,
        optimizer_writes_this_attempt=new,opportunities_processed=20,opportunities=20,
        skipped_update_opportunities=13,complete_outer=True,full_budget_complete=False,
        automatic_rollback_performed=False,opportunity_statuses=['written']*7+['skipped_guard']*13,
        written_with_post_guard=0) for r in range(4)]
    return dict(run_id='constructed',completed_outer=outer,scheduler_completed_outer=outer,
        actor_version='actor%d'%outer,sampling_actor_version='actor%d'%(outer-1),
        grounding_state=state,grounding_probe_version=state['version'],
        actual_adam_steps_per_rank=[actual]*4,new_adam_steps_per_rank=[new]*4,
        skipped_update_opportunities=[13]*4,opportunities_processed=20,complete_outer=True,
        full_budget_complete=False,status='complete',automatic_rollback_performed=False,rank_receipts=ranks)


def boundary(base,n=10):
    c=dict(base=str(base),run_id='constructed',adaptive_candidates=policy(30000),
        candidate_schedule=dict(effective_from_outer=1,legacy_candidates=2560,
            ordinary_candidates=5120,update_candidates=2560,probe_update_period=10,
            probe_update_at_final=False,selected_questions=512))
    current=initial_state(30000);previous=None
    for outer in range(1,n+1):
        root=base/'outers'/('outer-%06d'%outer);root.mkdir(parents=True)
        ensure(c,outer);before=copy.deepcopy(current)
        ids,cycles,current=reserve_for_outer(current,c,outer)
        if outer<n:
            write(root/'complete.json',dict(run_id=c['run_id'],outer=outer,
                rollout=dict(run_id=c['run_id'],outer=outer,questions=512,trajectories=2560,
                    actual_selected_mixed_questions=256,actual_selected_mixed_fraction=.5)))
        else:
            previous=before
            reservation=dict(outer=n,before_sha256=state_sha(before),after_sha256=state_sha(current),
                indices=ids,cycles=cycles,after=copy.deepcopy(current),**schedule_metadata(c,n))
    state=dict(version='grounding%d'%((n-1)//50*50),round=(n-1)//50*50)
    old=actor(n-1,state);new=actor(n,state)
    before=recovery.probe_boundary(n-1)
    probe=dict(run_id=c['run_id'],completed_outer=before['last_update_outer'],
        actual_adam_steps_per_rank=[before['actual_adam_per_rank']]*4)
    snapshot=base/'prior-pool.json';write(snapshot,previous)
    joint=dict(run_id=c['run_id'],outer=n-1,actor=old,probe=probe,grounding=state,
        pool_snapshot=str(snapshot),pool_sha256=recovery.sha(snapshot))
    return c,joint,new,probe,current,reservation


class PendingBoundaryTests(unittest.TestCase):
    def test_unstable_metadata_bytes_are_discarded_and_a_stable_retry_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'receipt.json';path.write_bytes(b'old rejected bytes')
            original=joint_recovery._signature;calls=[]
            def change_after_first_read(info):
                calls.append(1)
                if len(calls)==2:
                    path.write_bytes(b'new stable bytes');return ('changed',)
                return original(info)
            with patch.object(joint_recovery,'_signature',side_effect=change_after_first_read):
                self.assertEqual(joint_recovery._raw(path),b'new stable bytes')
            self.assertGreaterEqual(len(calls),4)
            with patch.object(joint_recovery,'_signature',side_effect=range(6)):
                with self.assertRaisesRegex(ValueError,'three bounded attempts'):joint_recovery._raw(path)

    def test_cross_node_recovery_requires_fresh_owned_terminal_slurm_not_local_proc(self):
        config=dict(execution_backend='slurm_hpc3',slurm_job='123',slurm_node='old-node')
        terminal=dict(host='old-node',owned_births=[dict(uid=os.getuid())])
        user=pwd.getpwuid(os.getuid()).pw_name
        with patch.object(recovery.socket,'gethostname',return_value='new-node'):
            for state in ('FAILED','CANCELLED by 21930','TIMEOUT'):
                with patch.object(recovery.subprocess,'check_output',return_value='123|%s|%s|old-node\n'%(user,state)):
                    self.assertEqual(recovery._writer_absence(config,terminal),[])
            for row in ('123|%s|RUNNING|old-node\n'%user,'123|other|FAILED|old-node\n',
                        '123|%s|FAILED|wrong-node\n'%user,''):
                with patch.object(recovery.subprocess,'check_output',return_value=row):
                    with self.assertRaises(RuntimeError):recovery._writer_absence(config,terminal)
            with patch.object(recovery.subprocess,'check_output',side_effect=TimeoutError('observation unavailable')):
                with self.assertRaises(TimeoutError):recovery._writer_absence(config,terminal)

    def test_guarded_actor_and_dynamic_pool_are_retained_at_later_p10_boundaries(self):
        for n in (10,20,50):
            with self.subTest(outer=n),tempfile.TemporaryDirectory() as tmp:
                args=boundary(Path(tmp).resolve(),n)
                self.assertTrue(actual_actor_boundary(args[2],n))
                self.assertEqual(recovery.validate_boundary(*args),n)
                self.assertEqual(args[2]['actual_adam_steps_per_rank'],[7*n]*4)
                self.assertNotEqual(args[4]['consumed_candidates'],n*2560)
                self.assertEqual(len(args[5]['indices']),2560)
                self.assertEqual(len(set(args[5]['indices'])),2560)

    def test_reject_lost_guarded_writes_or_inconsistent_rank_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=boundary(Path(tmp).resolve())
            for index,change in ((2,dict(actual_adam_steps_per_rank=[200]*4)),
                    (2,dict(scheduler_completed_outer=9)),
                    (2,dict(actual_adam_steps_per_rank=[70,69,70,70])),
                    (3,dict(actual_adam_steps_per_rank=[1]*4)),
                    (4,dict(consumed_candidates=25600))):
                args=copy.deepcopy(original);args[index].update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):recovery.validate_boundary(*args)

    def test_reject_changed_candidate_order_or_adaptive_feedback_even_same_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=boundary(Path(tmp).resolve())
            args=copy.deepcopy(original);args[5]['indices'][0],args[5]['indices'][1]=args[5]['indices'][1],args[5]['indices'][0]
            with self.assertRaises(ValueError):recovery.validate_boundary(*args)
            args=copy.deepcopy(original);args[5]['candidate_budget']['previous_joint_sha256']='f'*64
            with self.assertRaises(ValueError):recovery.validate_boundary(*args)
            args=copy.deepcopy(original);args[2]['grounding_state']=dict(version='other')
            with self.assertRaises(ValueError):recovery.validate_boundary(*args)

    def test_final200_and_ordinary_rounds_cannot_claim_pending_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=boundary(Path(tmp).resolve())
            for n in (11,200):
                args=copy.deepcopy(original);args[2]['completed_outer']=n
                with self.subTest(outer=n),self.assertRaises(ValueError):recovery.validate_boundary(*args)

    def test_unsaved_probe_writes_and_unknown_inflight_are_never_rolled_back(self):
        a=dict(new_adam_steps_per_rank=[7]*4)
        t=dict(termination_cause='FAULT_EXIT',uncommitted_after_joint=dict(
            actor_actual_writes_per_rank=[7]*4,probe_actual_writes_per_rank=[0]*4,
            optimizer_work_in_flight=False,all_updates_accounted=True))
        with patch.object(recovery,'_writer_barrier') as barrier:
            recovery._terminal({},t,{},10,a);self.assertTrue(barrier.called)
            for change in (dict(probe_actual_writes_per_rank=[1]*4),dict(optimizer_work_in_flight=True),
                           dict(actor_actual_writes_per_rank=[20]*4),dict(all_updates_accounted=False)):
                bad=copy.deepcopy(t);bad['uncommitted_after_joint'].update(change)
                with self.subTest(change=change),self.assertRaises(ValueError):recovery._terminal({},bad,{},10,a)

    def test_modern_pending_path_reaches_import_verifier_before_cuda(self):
        class ReachedVerifier(Exception):pass
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(resume_pending_probe=dict(adapter=recovery.ADAPTER))
            with patch.object(launch,'config',return_value=c),patch.object(launch,'sha',return_value='sealed'), \
                 patch.object(launch,'read',return_value=dict(config_sha256='sealed')), \
                 patch.object(recovery,'validate_imported_pending',side_effect=ReachedVerifier):
                with self.assertRaises(ReachedVerifier):launch.run(Path(tmp))
            c['resume_pending_probe']['adapter']='legacy_hardcoded_counts'
            with patch.object(launch,'config',return_value=c),patch.object(launch,'sha',return_value='sealed'), \
                 patch.object(launch,'read',return_value=dict(config_sha256='sealed')):
                with self.assertRaisesRegex(ValueError,'compatible recovery adapter'):launch.run(Path(tmp))

    def test_import_keeps_original_bytes_and_inode_and_does_not_copy_or_reset_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source=root/'source';base=root/'target';source.mkdir();(base/'preparation').mkdir(parents=True)
            config=dict(base=str(base),execution_id='new')
            write(base/'preparation/launch-config.json',config)
            origin=source/'state.json';write(origin,dict(actor_adam=[70]*4,probe_adam=[0]*4,stage='probe_pending'))
            stat=origin.stat();row=dict(source=str(origin),target=str(base/'actor/latest.json'),
                dev=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=recovery.sha(origin))
            result=dict(adapter=recovery.ADAPTER,assets=[row],pending_outer=10,completed_joint_outer=9,
                qualification_proof_sha256='a'*64,source_joint_sha256='b'*64,source_binding={},execution_id='new')
            with patch.object(recovery,'plan',return_value=result):
                receipt=recovery.import_pending(base)
                self.assertEqual(receipt['status'],'I1_I6_SAVED_ACTOR_PENDING_PROBE_IMPORTED')
                target=base/'actor/latest.json'
                self.assertEqual(target.read_bytes(),origin.read_bytes());self.assertEqual(target.stat().st_ino,origin.stat().st_ino)
                self.assertEqual(recovery.validate_imported_pending(base,config),9)
                self.assertEqual(recovery.import_pending(base),receipt)
                pending=base/'outers/outer-000010/probe-update';pending.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError,'already begun'):recovery.validate_imported_pending(base,config)

    def test_completed_pending_probe_commits_guarded_actor_and_dynamic_clock_without_replay(self):
        from .test_grounding_protocol import GroundingLifecycleTests
        from . import grounding
        from .i1_i6_joint_resume import source_envelope
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp).resolve();c,joint,a,probe,pool,reservation=boundary(base)
            fixture=GroundingLifecycleTests();ground,_=fixture.fixture(base)
            ground['run_id']=c['run_id'];c.update(ground)
            state=fixture.adopted(base,c,grounding.initialize(base,c))
            a.update(grounding_state=state,grounding_probe_version=state['version'])
            c.update(source_commit='c'*40,execution_id='new',source_compatibility='/data/constructed-proof',
                source_compatibility_sha256='d'*64,resume_pending_probe=dict(source_commit='a'*40,
                    source_execution_id='old',source_base='/data/constructed-parent',
                    completed_joint_outer=9,pending_outer=10,qualification_proof_sha256='e'*64))
            root=base/'outers/outer-000010'
            write(base/'actor/latest.json',a);write(base/'question-pool/state.json',pool)
            write(root/'probe-score/selection.json',dict(predicted=True))
            selection=root/'probe-score/actual-em-selection.json';write(selection,dict(selected=True))
            write(root/'full/complete.json',dict(run_id=c['run_id'],outer=10,questions=512,trajectories=2560))
            probe=dict(probe,completed_outer=10,actual_adam_steps_per_rank=[50]*4)
            write(base/'probe/latest.json',probe);(base/'probe/latest.pt').write_bytes(b'constructed opaque state fixture, not a model')
            metrics=root/'probe-update/metrics.json';write(metrics,dict(run_id=c['run_id'],outer=10))
            complete=root/'probe-update/complete.json'
            write(complete,dict(status='UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE',run_id=c['run_id'],outer=10,
                new_adam_steps_per_rank=50,metrics_sha256=recovery.sha(metrics),checkpoint=probe))
            envelope=dict(science_source_commit='a'*40,authorized_candidate_delta=None)
            with patch('reproduction.prefix_selection200.i1_i6_joint_resume.verify_source_compatibility',return_value=envelope):
                result=launch.commit_outer(base,c,10,root,selection,{},recovery=dict(pending_outer=10),grounding_state=state)
            self.assertEqual(result['actor']['actual_adam_steps_per_rank'],[70]*4)
            self.assertEqual(result['probe']['actual_adam_steps_per_rank'],[50]*4)
            self.assertEqual(result['candidate_schedule'],schedule_metadata(c,10))
            self.assertEqual(result['source_lineage']['repaired_pending_outer'],10)
            self.assertEqual(result['pending_probe_recovery']['imported_actor_updates_reexecuted'],0)
            self.assertEqual(result['grounding'],state)
            self.assertEqual(result['runtime_source_commit'],'c'*40)

    def test_small_actual_adam_rng_next_update_matches_after_retained_state_import(self):
        import torch
        rng=torch.get_rng_state()
        try:
            torch.manual_seed(456)
            p=torch.nn.Parameter(torch.tensor([.7,-.4]));adam=torch.optim.Adam([p],lr=5e-6)
            for _ in range(7):
                (p*torch.randn(2)).sum().backward();adam.step();adam.zero_grad()
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();source=root/'source';base=root/'target';source.mkdir();(base/'preparation').mkdir(parents=True)
                config=dict(base=str(base),execution_id='new');write(base/'preparation/launch-config.json',config)
                origin=source/'state.pt'
                torch.save(dict(model=p.detach().clone(),optimizer=adam.state_dict(),rng=torch.get_rng_state()),origin)
                info=origin.stat();row=dict(source=str(origin),target=str(base/'probe/latest.pt'),dev=info.st_dev,
                    inode=info.st_ino,bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=recovery.sha(origin))
                planned=dict(adapter=recovery.ADAPTER,assets=[row],pending_outer=10,completed_joint_outer=9,
                    qualification_proof_sha256='a'*64,source_joint_sha256='b'*64,source_binding={},execution_id='new')
                with patch.object(recovery,'plan',return_value=planned):recovery.import_pending(base)
                sample=torch.randn(2);(p*sample).sum().backward();adam.step();adam.zero_grad()
                saved=torch.load(base/'probe/latest.pt',map_location='cpu',weights_only=False)
                q=torch.nn.Parameter(saved['model']);resumed=torch.optim.Adam([q],lr=5e-6);resumed.load_state_dict(saved['optimizer'])
                torch.set_rng_state(saved['rng']);sample2=torch.randn(2)
                torch.testing.assert_close(sample,sample2,rtol=0,atol=0)
                (q*sample2).sum().backward();resumed.step()
                torch.testing.assert_close(p,q,rtol=0,atol=0)
                for key in ('step','exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(adam.state[p][key],resumed.state[q][key],rtol=0,atol=0)
                self.assertEqual(int(resumed.state[q]['step']),8)
        finally:torch.set_rng_state(rng)


if __name__=='__main__':unittest.main()
