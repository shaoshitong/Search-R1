"""Frozen-A recovery metadata tests; no actual weights or remote/GPU action."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import ab_frozen_latest_resume as frozen
from .b_progressive_resume import sha
from .pool import reserve_for_outer, state_sha


class FrozenLatestRecoveryTests(unittest.TestCase):
    def fixture(self, source, n=100):
        c = dict(run_id='same-frozen-A', selection_probe_experiment='warmup_frozen',
                 resume_joint=dict(recovery_mode=frozen.MODE, completed_outer=n))
        joint = dict(run_id=c['run_id'], outer=n, actor={'actual_adam_steps_per_rank':[20*n-3]*4},
                     probe={'actual_adam_steps_per_rank':[0]*4})
        proof = dict(status='AB_JOINT_FULL_SHA_CPU_PASS', run_id=c['run_id'], outer=n,
                     actual_cpu_deserialization=True, actor_actual_adam_steps_per_rank=[20*n-3]*4,
                     probe_actual_adam_steps_per_rank=[0]*4, files=[])
        (source/'latest-joint.json').write_text(json.dumps(joint))
        return c, joint, proof

    def test_actual_skipped_actor_steps_not_assumed20n(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); c,j,p=self.fixture(root)
            self.assertEqual(frozen.qualify(c,root,p,j),100)
            bad=copy.deepcopy(p);bad['actor_actual_adam_steps_per_rank']=[2000]*4
            with self.assertRaises(ValueError):frozen.qualify(c,root,bad,j)

    def test_cannot_fallback_to_last_audited_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,j,p=self.fixture(root)
            (root/'latest-joint.json').write_text(json.dumps(dict(j,outer=101)))
            with self.assertRaises(ValueError):frozen.qualify(c,root,p,j)

    def test_cannot_import_balanced_probe_or_nonzero_probe_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c,j,p=self.fixture(root)
            c['selection_probe_experiment']='balanced_online_ema'
            with self.assertRaises(ValueError):frozen.qualify(c,root,p,j)
            c['selection_probe_experiment']='warmup_frozen';p['probe_actual_adam_steps_per_rank']=[10]*4
            with self.assertRaises(ValueError):frozen.qualify(c,root,p,j)

    def test_import_keeps_frozen_head_inode_actual_actor_and_reserved_questions(self):
        from .candidate_schedule import consumed_candidates_through
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';target=Path(directory)/'new'
            source.mkdir();target.mkdir();c,j,p=self.fixture(source)
            def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            (target/'preparation').mkdir(); n=100
            before=dict(next_outer=101,consumed_candidates=consumed_candidates_through({},100),
                        seed=1729,cycle=0,cursor=0,order=list(range(12000)),candidate_questions=2560)
            consumed=before['consumed_candidates'];before.update(cycle=consumed//12000,cursor=consumed%12000)
            indices,cycles,after=reserve_for_outer(before,{},101)
            snapshot=source/'pool-at100.json';put(snapshot,before)
            tensor=source/'frozen-probe.payload';tensor.write_bytes(b'frozen-head-no-optimizer')
            j.update(pool_snapshot=str(snapshot),probe_snapshot=str(tensor))
            for i in range(1,101):put(source/'outers'/('outer-%06d'%i)/'complete.json',j if i==n else {'outer':i})
            put(source/'latest-joint.json',j);put(source/'preparation/launch-config.json',dict(run_id=c['run_id'],selection_probe_experiment='warmup_frozen'))
            put(source/'question-pool/state.json',after);put(source/'question-pool/binding.json',{'same':'binding'})
            nextroot=source/'outers/outer-000101'
            put(nextroot/'pool-reservation.json',dict(before_sha256=state_sha(before),after_sha256=state_sha(after),after=after,indices=indices,cycles=cycles))
            put(nextroot/'questions.json',{'reserved':'authentic questions'})
            proof=source/'cpu-proof100.json';put(proof,p)
            c['resume_joint'].update(source_base=str(source),source_config_sha256=sha(source/'preparation/launch-config.json'),qualification_proof=str(proof),qualification_proof_sha256=sha(proof),source_joint_sha256=sha(source/'outers/outer-000100/complete.json'))
            put(target/'preparation/launch-config.json',c)
            with patch.object(frozen,'verify',return_value={}),patch.object(frozen,'stopped'):
                result=frozen.import_complete(target)
                self.assertEqual(result['status'],'A_LATEST_FULL_JOINT_IMPORTED')
                self.assertEqual(frozen.validate_imported(target,c),100)
                self.assertEqual(frozen.import_complete(target)['source_joint_sha256'],result['source_joint_sha256'])
            self.assertEqual(result['imported_actor_adam'],[1997]*4)
            self.assertEqual(result['imported_probe_adam'],[0]*4)
            self.assertEqual((target/'probe/latest.pt').stat().st_ino,tensor.stat().st_ino)
            self.assertEqual(json.loads((target/'question-pool/state.json').read_text()),after)
            self.assertTrue((target/'outers/outer-000101/questions.json').exists())
            self.assertFalse((target/'outers/outer-000101/prefix').exists())

    def test_original_i1_entry_dispatches_frozen_route_only_for_explicit_mode(self):
        from . import i1_i6_joint_resume as legacy
        c=dict(selection_probe_experiment='warmup_frozen',resume_joint={'recovery_mode':frozen.MODE})
        with patch.object(frozen,'verify',return_value={'exact':'source'}) as verify:
            self.assertEqual(legacy.verify_source_compatibility(c),{'exact':'source'})
            verify.assert_called_once_with(c)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(frozen,'plan',return_value={'actual':'latest'}) as plan:
                self.assertEqual(legacy.plan(str(Path(directory).resolve()),c),{'actual':'latest'})
                plan.assert_called_once()


if __name__=='__main__':unittest.main()
