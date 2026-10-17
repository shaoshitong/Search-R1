"""One CPU-only controller fixture: old full boundary80, then scheduled ordinary81.

Only tiny receipt/head byte fixtures are created; no tensor/model is loaded and
no inference, retrieval, optimizer step or remote operation is performed.
"""
import json
from pathlib import Path
import tempfile
import unittest

from . import grounding
from .launch import commit_outer
from .pool import initial_state, reserve_indices, reserve_for_outer
from .test_candidate_schedule import configured
from .test_grounding_protocol import actor as actor_fixture
from reproduction.prefix20_online.launch import sha


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True))
    return value


def fitted_state(base, config):
    root=base/'fixture-grounding';root.mkdir()
    heads={}
    for name in ('commit_probe.pkl','boundary_probe.pkl'):
        path=root/name;path.write_bytes(('CPU receipt fixture '+name).encode());heads[name]=sha(path)
    questions=['question%d'%index for index in range(300)]
    config.update(grounding_refresh_enabled=True,grounding_refit_rounds=[50,100,150],
        grounding_calibration_questions='/data/fixed-calibration300.parquet',
        grounding_calibration_sha256='calibration',grounding_calibration_question_ids=questions,
        grounding_refit_visible_device=6,grounding_probe_version='run:50:fixture',
        grounding_manifest=str(root/'initial-manifest.json'),grounding_manifest_sha256='initial-fit-sha')
    manifest=write(root/'round50.json',dict(run_id='run',round=50,version='run:50:fixture',
        encoder='/data/actual-actor50',encoder_sha256={'model.safetensors':'actor50-hash'},
        source_actor_version='actor50',commit=str(root/'commit_probe.pkl'),boundary=str(root/'boundary_probe.pkl'),
        head_sha256=heads,label_proof=dict(question_count=300,question_ids=questions,
            interaction_protocol=grounding.INTERACTION_CONTRACT),calibration_report=dict(unique_question_count=300)))
    state=grounding._manifest_state(config,root/'round50.json')
    replicas=[dict(rank=rank,ok=True,round=50,version=state['version'],encoder=state['encoder'])for rank in range(4)]
    return grounding.adopt(base,config,state,replicas)


def phase_files(base, config, outer, state, pool):
    root=base/'outers'/('outer-%06d'%outer)
    current=actor_fixture(outer,state,actual=19*outer,new=19)
    write(base/'actor/latest.json',current)
    probe=write(base/'probe/latest.json',dict(run_id='run',completed_outer=80,actual_adam_steps_per_rank=[400]*4))
    if not (base/'probe/latest.pt').exists():(base/'probe/latest.pt').write_bytes(b'CPU sealed probe receipt fixture')
    write(base/'question-pool/state.json',pool)
    selection=root/'probe-score/selection.json';write(selection,dict(outer=outer,selected_questions=512))
    write(root/'full/complete.json',dict(run_id='run',outer=outer,questions=512,trajectories=2560))
    if outer==80:
        metrics=root/'probe-update/metrics.json';write(metrics,dict(run_id='run',outer=80,unselected_training_questions=2560))
        write(root/'probe-update/complete.json',dict(status='UNSELECTED2560G5_PERIODIC_PROBE_50_ADAM_COMPLETE',
            outer=80,run_id='run',new_adam_steps_per_rank=50,metrics_sha256=sha(metrics),checkpoint=probe))
    else:
        write(root/'probe-frozen.json',dict(status='PROBE_FROZEN_NO_OPTIMIZER_UPDATE',outer=outer,
            new_adam_steps_per_rank=0,actual_adam_steps_per_rank=[400]*4))
    return root,selection


class SegmentedPoolJointCommit(unittest.TestCase):
    def test_commit_keeps_actual_old80_pool_and_accepts_only_scheduled5120_suffix(self):
        for wrong_legacy_suffix in (False,True):
            with self.subTest(wrong_legacy_suffix=wrong_legacy_suffix),tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);config=dict(configured(80),run_id='run',source_commit='source',execution_id='new')
                state=fitted_state(base,config)
                # Construct the genuine legacy pool clock by its original reserve operation.
                pool=initial_state(17011,1729)
                for outer in range(1,81):_,_,pool=reserve_indices(pool,outer)
                self.assertEqual(pool['consumed_candidates'],80*2560)
                root80,selection80=phase_files(base,config,80,state,pool)
                old=commit_outer(base,config,80,root80,selection80,{},grounding_state=state)
                old_snapshot=Path(old['pool_snapshot']);old_bytes=old_snapshot.read_bytes()
                old_joint_bytes=(base/'latest-joint.json').read_bytes()
                if wrong_legacy_suffix:
                    _,_,after=reserve_indices(pool,81,2560)
                else:
                    _,_,after=reserve_for_outer(pool,config,81)
                root81,selection81=phase_files(base,config,81,state,after)
                if wrong_legacy_suffix:
                    with self.assertRaisesRegex(RuntimeError,'actual reserved candidate pool'):
                        commit_outer(base,config,81,root81,selection81,{},grounding_state=state)
                    self.assertFalse((root81/'complete.json').exists())
                    self.assertEqual((base/'latest-joint.json').read_bytes(),old_joint_bytes)
                else:
                    committed=commit_outer(base,config,81,root81,selection81,{},grounding_state=state)
                    saved=json.loads(Path(committed['pool_snapshot']).read_text())
                    self.assertEqual(saved['consumed_candidates'],80*2560+5120)
                    self.assertEqual(saved['next_outer'],82)
                    self.assertEqual(saved,after)
                    self.assertFalse(committed['periodic_probe_update'])
                    self.assertEqual(committed['candidate_schedule']['scheduled_candidate_questions'],5120)
                    self.assertEqual(json.loads((base/'latest-joint.json').read_text())['outer'],81)
                self.assertEqual(old_snapshot.read_bytes(),old_bytes)
                self.assertEqual(json.loads(old_bytes),pool)


if __name__=='__main__':unittest.main()
