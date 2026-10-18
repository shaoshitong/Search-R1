"""Small CPU metadata tests; no model, payload hash, remote action or GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .b_progressive_resume import LATEST,boundary,qualified_boundary,pending_pool,import_complete,validate_imported,sha,recovery_environment
from .pool import reserve_for_outer,state_sha


class LatestProgressiveResumeTests(unittest.TestCase):
    def config(self,n,latest=True):
        spec=dict(completed_outer=n)
        if latest:spec['recovery_mode']=LATEST
        return dict(run_id='same-logical-run',resume_joint=spec)

    def test_initial_human_rollback_is_not_a_latest_state_fallback(self):
        self.assertEqual(boundary(self.config(50,False)),(50,False))
        for n in (51,100,150,200):self.assertEqual(boundary(self.config(n)),(n,True))
        for n,latest in ((51,False),(49,True),(50,True),(201,True)):
            with self.assertRaises(ValueError):boundary(self.config(n,latest))

    def test_recovery_process_uses_completed_head_not_next_stage(self):
        for n,length in ((100,20),(101,40),(150,40),(151,80),(200,80)):
            with patch('reproduction.prefix_selection200.b_progressive_resume.read',return_value={'outer':n}):
                self.assertEqual(recovery_environment('/source'),{'SEARCH_G1_B_PREFIX_TOKENS':str(length)})

    def test_latest_actor_skips_and_probe_missing_class_are_actual_clocks(self):
        c=self.config(150);j=dict(run_id=c['run_id'],outer=150,
            actor={'actual_adam_steps_per_rank':[2997]*4},probe={'actual_adam_steps_per_rank':[2020]*4})
        proof=dict(status='AB_JOINT_FULL_SHA_CPU_PASS',outer=150,actual_cpu_deserialization=True,
            run_id=c['run_id'],actor_actual_adam_steps_per_rank=[2997]*4,probe_actual_adam_steps_per_rank=[2020]*4)
        with patch('reproduction.prefix_selection200.b_progressive_resume.read',return_value=j):
            self.assertEqual(qualified_boundary(c,'/source',proof,j),(150,True))
        with patch('reproduction.prefix_selection200.b_progressive_resume.read',return_value=dict(j,outer=151)):
            with self.assertRaises(ValueError):qualified_boundary(c,'/source',proof,j)
        bad=copy.deepcopy(proof);bad['probe_actual_adam_steps_per_rank']=[2000]*4
        with self.assertRaises(ValueError):qualified_boundary(c,'/source',bad,j)

    def test_reserved_next_questions_and_cursor_survive_without_trajectory_replay(self):
        from .candidate_schedule import consumed_candidates_through
        with tempfile.TemporaryDirectory()as directory:
            source=Path(directory);root=source/'outers/outer-000101';root.mkdir(parents=True)
            (source/'question-pool').mkdir()
            c={};consumed=consumed_candidates_through(c,100);size=12000
            before=dict(next_outer=101,consumed_candidates=consumed,seed=1729,cycle=consumed//size,
                cursor=consumed%size,order=list(range(size)),candidate_questions=2560)
            indices,cycles,after=reserve_for_outer(before,c,101)
            def put(path,value):path.write_text(json.dumps(value))
            snapshot=source/'completed-pool.json';put(snapshot,before)
            put(source/'question-pool/state.json',after)
            reservation=dict(before_sha256=state_sha(before),after_sha256=state_sha(after),after=after,indices=indices,cycles=cycles)
            put(root/'pool-reservation.json',reservation);put(root/'selection.json',{'authentic':'retained metadata'})
            result=pending_pool(source,c,dict(pool_snapshot=str(snapshot)),100)
            self.assertEqual(result['state'],after);self.assertEqual(len(result['files']),2)
            self.assertTrue(all(isinstance(p,str)for p in result['files']))
            changed=dict(after,cursor=after['cursor']+1);put(source/'question-pool/state.json',changed)
            with self.assertRaises(ValueError):pending_pool(source,c,dict(pool_snapshot=str(snapshot)),100)

    def test_import_retains_actual150_state_and_pending151_without_old50_reset(self):
        from .candidate_schedule import consumed_candidates_through
        with tempfile.TemporaryDirectory()as directory:
            source=Path(directory)/'source';target=Path(directory)/'new';source.mkdir();target.mkdir()
            for p in (source/'preparation',source/'question-pool',source/'probe',target/'preparation'):p.mkdir(parents=True)
            def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            n=150;c=self.config(n);c['resume_joint'].update(source_base=str(source));put(source/'preparation/launch-config.json',dict(run_id=c['run_id']))
            consumed=consumed_candidates_through({},n);size=12000
            before=dict(next_outer=n+1,consumed_candidates=consumed,seed=1729,cycle=consumed//size,cursor=consumed%size,order=list(range(size)),candidate_questions=2560)
            snapshot=source/'pool-at150.json';put(snapshot,before);indices,cycles,after=reserve_for_outer(before,{},n+1)
            put(source/'question-pool/state.json',after);put(source/'question-pool/binding.json',{'same':'binding'})
            pending=source/'outers/outer-000151'
            put(pending/'pool-reservation.json',dict(before_sha256=state_sha(before),after_sha256=state_sha(after),after=after,indices=indices,cycles=cycles))
            put(pending/'questions.json',{'authentic_questions':'already reserved'})
            tensor=source/'probe150.payload';tensor.write_bytes(b'unchanged-raw-ema-adam-rng-40-token-state')
            j=dict(run_id=c['run_id'],outer=n,actor={'actual_adam_steps_per_rank':[2997]*4},probe={'actual_adam_steps_per_rank':[2020]*4},probe_snapshot=str(tensor),pool_snapshot=str(snapshot))
            for i in range(1,n+1):put(source/'outers'/('outer-%06d'%i)/'complete.json',j if i==n else dict(outer=i))
            put(source/'latest-joint.json',j)
            proof=source/'qualified150.json';put(proof,dict(status='AB_JOINT_FULL_SHA_CPU_PASS',outer=n,actual_cpu_deserialization=True,run_id=c['run_id'],actor_actual_adam_steps_per_rank=[2997]*4,probe_actual_adam_steps_per_rank=[2020]*4,files=[]))
            c['resume_joint'].update(qualification_proof=str(proof),qualification_proof_sha256=sha(proof),source_joint_sha256=sha(source/'outers/outer-000150/complete.json'))
            put(target/'preparation/launch-config.json',c)
            with patch('reproduction.prefix_selection200.b_progressive_resume.verify',return_value={}),patch('reproduction.prefix_selection200.b_progressive_resume.stopped'):
                result=import_complete(target);self.assertEqual(validate_imported(target,c),150)
                self.assertEqual(import_complete(target)['source_joint_sha256'],result['source_joint_sha256'])
            self.assertEqual(result['imported_actor_adam'],[2997]*4);self.assertEqual(result['imported_probe_adam'],[2020]*4)
            self.assertEqual(json.loads((target/'question-pool/state.json').read_text()),after)
            self.assertEqual((target/'probe/latest.pt').stat().st_ino,tensor.stat().st_ino)
            self.assertEqual((target/'probe/latest.pt').read_bytes(),b'unchanged-raw-ema-adam-rng-40-token-state')
            self.assertEqual(json.loads((pending/'questions.json').read_text()),json.loads((target/'outers/outer-000151/questions.json').read_text()))
            self.assertFalse((target/'outers/outer-000151/prefix').exists())


if __name__=='__main__':unittest.main()
