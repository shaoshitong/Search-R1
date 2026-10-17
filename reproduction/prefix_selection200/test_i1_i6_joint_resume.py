"""Small constructed CPU metadata/state tests; no native-model acceptance evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import i1_i6_joint_resume as resume
from . import i1_i6_actor_resume as wrapper
from . import i1_i6_checkpoint_cpu as qualify
from .actor import _state_binding, learning_rate
from .test_storage_guard_boundary import actor_receipt
from . import test_grounding_protocol as _grounding_fixture
from . import grounding


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(json.dumps(value,sort_keys=True).encode())


def producer(path,raw=b'constructed placeholder, not a native tensor'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def fixture(root, *, qualified=False):
    root=root.resolve();source=root/'source';target=root/'target';source.mkdir();target.mkdir()
    for base in (source,target):
        path=base/'repo/reproduction/prefix_selection200';path.mkdir(parents=True)
        (path/'actor.py').write_text('def unchanged_actor(): return 1\n')
        (path/'probe.py').write_text('def update(): return 50\ndef load_online(): return 0\ndef score(): return '+('2560' if base==source else '5120')+'\n')
    (target/'repo/reproduction/prefix_selection200/candidate_schedule.py').write_text('ORDINARY=5120\n')
    human=target/'preparation/human-intent.json';write(human,dict(actual_user_authorization='constructed fixture variable pool'))
    schedule=dict(effective_from_outer=3,legacy_candidates=2560,ordinary_candidates=5120,update_candidates=2560,
        probe_update_period=10,probe_update_at_final=False,selected_questions=512)
    delta=dict(candidate_schedule=schedule,human_intent_sha256=resume.sha(human),
        only_authorized_algorithm_delta='ordinary_candidate_pool_5120_periodic2560')
    parent=resume.source_manifest(source/'repo','a'*40)
    current=resume.source_manifest(target/'repo','b'*40,science_source_commit='a'*40)
    proof=resume.compatibility_proof(parent,current,authorized_candidate_delta=delta)
    proofpath=target/'preparation/source-compatibility.json';write(proofpath,proof)
    helper=_grounding_fixture.GroundingLifecycleTests();config,_=helper.fixture(source)
    config.update(base=str(source),run_id='run',execution_id='parent-execution',source_commit='a'*40,
        official_model_dir='/data/official',nonmixed_advantage_mode='zero_anchor_all_correct_only',outers=200,
        world_size=4,candidate_questions=2560,seed=1729,canonical_train='/data/train',canonical_train_sha256='d'*64)
    state=helper.adopted(source,config,grounding.initialize(source,config))
    pool=dict(order=list(range(10000)),cycle=0,cursor=5120,consumed_candidates=5120,next_outer=3,seed=1729,candidate_questions=2560)
    probe_blob=producer(source/'probe/latest.pt');write(source/'probe/initialization-complete.json',dict(run_id='run'))
    probe=dict(run_id='run',completed_outer=0,actual_adam_steps_per_rank=[0]*4,bytes=probe_blob['bytes'],sha256=probe_blob['sha256'])
    payloads=[probe_blob]
    for outer in (1,2):
        actor=actor_receipt(outer,outer-1,1);actor['run_id']='run'
        directory=source/'actor/outers'/('outer-%04d-attempt-%s'%(outer,'c'*32));hf=directory/'huggingface'
        weight=producer(hf/'model.safetensors');payloads.append(weight)
        ranks=[producer(directory/('rank-%d.pt'%rank)) for rank in range(4)];payloads.extend(ranks)
        driver=producer(directory/'driver.pt');payloads.append(driver)
        actor.update(actor_version='actor%d'%outer,grounding_state=state,grounding_probe_version=state['version'],
            binding=_state_binding(dict(config,grounding_state=state)),rank_states=ranks,driver_state=driver,
            hf_files={'model.safetensors':weight},hf_model_dir=str(hf),model_master_dtype='torch.float32')
        folder=source/'outers'/('outer-%06d'%outer);folder.mkdir(parents=True)
        savedpool=dict(pool,cursor=2560*outer,consumed_candidates=2560*outer,next_outer=outer+1)
        write(folder/'pool-state-at-completion.json',savedpool);write(folder/'grounding-state-at-completion.json',state)
        os.link(source/'probe/latest.pt',folder/'probe-state-at-completion.pt')
        joint=dict(run_id='run',source_commit='a'*40,outer=outer,actor=actor,probe=probe,grounding=state,
            grounding_snapshot=str(folder/'grounding-state-at-completion.json'),grounding_snapshot_sha256=resume.sha(folder/'grounding-state-at-completion.json'),
            pool_snapshot=str(folder/'pool-state-at-completion.json'),pool_sha256=resume.sha(folder/'pool-state-at-completion.json'),
            probe_snapshot=str(folder/'probe-state-at-completion.pt'))
        write(folder/'complete.json',joint)
    write(source/'latest-joint.json',joint);write(source/'actor/latest.json',actor);write(source/'probe/latest.json',probe)
    write(source/'question-pool/state.json',pool);write(source/'question-pool/binding.json',dict(candidate_questions=2560))
    birth=dict(pid=123456,start_ticks=42,sid=123456,uid=0)
    ledger=dict(run_id='run',jobs={'owned':dict(state='STOPPED',ledger={'123456':birth})})
    write(source/'logs/process-ledger.json',ledger);write(source/'preparation/launch-config.json',config)
    terminal=dict(run_id='run',execution_id=config['execution_id'],source_commit=config['source_commit'],base=str(source),
        termination_cause='AUTHORIZED_BOUNDARY_HANDOFF',pipeline_alive=False,owned_writers_gone=True,actual_births_present=[],
        owned_births=[birth],process_ledger_sha256=resume.sha(source/'logs/process-ledger.json'),human_intent_sha256=resume.sha(human),
        stopped_after_complete_outer=2,uncommitted_after_joint=dict(actor_actual_writes_per_rank=[0]*4,
            probe_actual_writes_per_rank=[0]*4,optimizer_work_in_flight=False,all_updates_accounted=True))
    termpath=root/'terminal.json';write(termpath,terminal)
    current=dict(config,base=str(target),execution_id='new-execution',source_commit='b'*40,science_source_commit='a'*40,
        source_compatibility=str(proofpath),source_compatibility_sha256=resume.sha(proofpath),candidate_schedule=schedule,
        candidate_schedule_authorization=str(human),candidate_schedule_authorization_sha256=resume.sha(human),
        resume_joint=dict(adapter=resume.ADAPTER,source_base=str(source),source_commit=config['source_commit'],
            source_execution_id=config['execution_id'],source_config_sha256=resume.sha(source/'preparation/launch-config.json'),
            completed_outer=2,source_joint_sha256=resume.sha(source/'latest-joint.json'),terminal_proof=str(termpath),terminal_proof_sha256=resume.sha(termpath)))
    if qualified:
        # Synthetic external qualifier envelope exercises transport/schema only;
        # these dummy blobs are never called actual tensor qualification evidence.
        files=[]
        for path in sorted(source.rglob('*')):
            if not path.is_file():continue
            info=path.stat();files.append(dict(path=str(path),bytes=info.st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dev=info.st_dev,inode=info.st_ino,mtime_ns=info.st_mtime_ns))
        q=dict(status=resume.QUALIFIED,run_id='run',source_execution_id=config['execution_id'],source_commit=config['source_commit'],
            completed_outer=2,source_joint_sha256=resume.sha(source/'latest-joint.json'),actual_adam_steps_per_rank=[2]*4,
            actor_rank_optimizer_rng_scheduler_verified=4,actor_driver_rng_verified=True,probe_model_optimizer_rank_rng_verified=True,
            pool_exact_permutation_verified=True,payload_full_sha_read_once=True,verified_files=files,constructed_fixture_only=True)
        qpath=root/'qualification.json';write(qpath,q)
        current['resume_joint'].update(qualification_proof=str(qpath),qualification_proof_sha256=resume.sha(qpath))
    write(target/'preparation/launch-config.json',current)
    return source,target,current,proof,terminal


class RecoveryMetadataTests(unittest.TestCase):
    def test_unknown_state_cannot_import_and_source_math_delta_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target,c,proof,terminal=fixture(Path(temporary))
            with patch.object(resume,'_native_writers_present',return_value=[]):
                result=resume.plan(target,c)
                self.assertEqual(result['status'],'NEEDS_GENUINE_FULL_STATE_QUALIFICATION')
                self.assertEqual(result['consumed_candidates_at_boundary'],5120)
                self.assertEqual(result['actual_adam_steps_per_rank'],[2]*4)
                with self.assertRaises(RuntimeError):resume.import_complete(target)
            changed=copy.deepcopy(proof['target_publication']);changed['files_sha256']['reproduction/prefix_selection200/actor.py']='f'*64
            with self.assertRaises(ValueError):resume.compatibility_proof(proof['parent_publication'],changed,authorized_candidate_delta=proof['authorized_candidate_delta'])
            changed=copy.deepcopy(proof['target_publication']);changed['critical_ast_sha256']['update']='f'*64
            with self.assertRaises(ValueError):resume.compatibility_proof(proof['parent_publication'],changed,authorized_candidate_delta=proof['authorized_candidate_delta'])
            self.assertFalse((target/'actor/latest.json').exists())

    def test_constructed_qualified_import_preserves_raw_inodes_history_and_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target,c,proof,terminal=fixture(Path(temporary),qualified=True)
            def json_or_code_only(path):
                self.assertNotIn(Path(path).suffix,('.pt','.pkl','.safetensors'))
                return original_sha(path)
            original_sha=resume.sha
            with patch.object(resume,'_native_writers_present',return_value=[]),patch.object(resume,'sha',side_effect=json_or_code_only):
                result=resume.import_complete(target)
                self.assertEqual(resume.validate_imported_boundary(target,c),2)
                again=resume.import_complete(target)
            self.assertEqual(result['status'],'I1_I6_FULL_JOINT_IMPORTED');self.assertEqual(again['completed_outer'],2)
            self.assertEqual((target/'probe/latest.pt').stat().st_ino,(source/'probe/latest.pt').stat().st_ino)
            self.assertEqual((target/'actor/latest.json').read_bytes(),(source/'actor/latest.json').read_bytes())
            self.assertEqual(resume.read(target/'question-pool/state.json'),resume.read(source/'question-pool/state.json'))
            self.assertEqual(resume.read(target/'outers/outer-000001/complete.json')['source_commit'],'a'*40)
            self.assertFalse(result['cuda_restore_verified']);self.assertEqual(result['tensor_payloads_read'],0)

    def test_live_writers_partial_unknown_and_wrong_handoff_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target,c,proof,terminal=fixture(Path(temporary))
            with patch.object(resume,'_native_writers_present',return_value=[{'pid':123456}]):
                with self.assertRaises(RuntimeError):resume.plan(target,c)
            path=Path(c['resume_joint']['terminal_proof'])
            for mutation in (dict(uncommitted_after_joint=dict(terminal['uncommitted_after_joint'],actor_actual_writes_per_rank=[1]*4)),dict(human_intent_sha256='e'*64)):
                changed=dict(terminal,**mutation);write(path,changed);c['resume_joint']['terminal_proof_sha256']=resume.sha(path)
                with patch.object(resume,'_native_writers_present',return_value=[]):
                    with self.assertRaises(ValueError):resume.plan(target,c)

    def test_wrapper_only_maps_provenance_and_keeps_optimizer_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            source,target,c,proof,terminal=fixture(Path(temporary));c['grounding_state']=resume.read(source/'actor/latest.json')['grounding_state']
            captured=[];sentinel=object()
            fake=SimpleNamespace(_state_binding=_state_binding,atomic_json=lambda path,value,**kw:captured.append(value))
            state=SimpleNamespace(atomic_save=lambda path,value:captured.append(value))
            with wrapper.binding_context(c,actor_module=fake,state_module=state):
                bound=fake._state_binding(c)
                self.assertEqual(bound['source_commit'],'a'*40);self.assertEqual(c['source_commit'],'b'*40)
                state.atomic_save('not-written',dict(run_id='run',optimizer=sentinel,binding=bound))
            self.assertIs(captured[0]['optimizer'],sentinel)
            self.assertEqual(captured[0]['runtime_source_commit'],'b'*40)
            self.assertEqual(captured[0]['science_source_commit'],'a'*40)


class GenuineQualifierSmallStateTests(unittest.TestCase):
    def test_actual_probe_producer_recipe_initial_and_periodic_rng_boundary(self):
        import numpy as np
        import torch
        from . import probe
        from . import prefix5_model
        from .contract import PROBE_LR
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'small-source.pt'
            torch.save(dict(binding=dict(feature_manifest_sha256='c'*64,feature_spec_sha256='d'*64)),source)
            imported=dict(source_epoch=4,source_model_kind='ema',source_warmup_run='warmup',
                warmup_model_weights_only=True,warmup_optimizer_imported=False,warmup_rng_imported=False)
            args=SimpleNamespace(state_dir=root/'probe',source_checkpoint=source,source_epoch=4,
                source_model_kind='ema',source_sha256=resume.sha(source),run_id='run',
                source_run='warmup',test_manifest_sha='c'*64)
            def small_warmup(*args,**kwargs):
                return {'small':torch.ones(2)},{'rope':torch.ones(2)},imported
            with patch.object(probe,'remote',side_effect=Path),patch.object(probe,'warmup_model_weights',side_effect=small_warmup):
                probe.initialize(args)
            saved=torch.load(root/'probe/latest.pt',map_location='cpu',weights_only=False)
            receipt=resume.read(root/'probe/latest.json')
            c=dict(run_id='run',probe_source_sha256=args.source_sha256,probe_source_epoch=4,
                probe_source_run='warmup',fixed_test_manifest_sha256='c'*64)
            with patch.object(prefix5_model,'PARAMETER_SHAPES',{'small':(2,)}),patch.object(prefix5_model,'NATIVE_BUFFER_LAYOUTS',[{'rope':(2,)}]):
                self.assertEqual(qualify.probe_state(saved,receipt,c,9,{})['actual_adam_steps_per_rank'],[0]*4)
                parameter=torch.nn.Parameter(saved['model']['small'].clone())
                optimizer=torch.optim.Adam([parameter],lr=PROBE_LR,weight_decay=0.)
                for _ in range(50):
                    optimizer.zero_grad();parameter.square().sum().backward();optimizer.step()
                saved.update(completed_outer=10,actual_adam_steps_per_rank=[50]*4,model={'small':parameter.detach().clone()},
                    optimizer=optimizer.state_dict(),actor_version='sampling-before10',
                    scored_features_manifest_sha256='e'*64,own_em_sha256='f'*64,
                    rank_rng=[dict(rank=rank,actual_adam_steps=50,python=random.Random(rank).getstate(),
                        numpy=np.random.RandomState(rank).get_state(),torch_cpu=torch.get_rng_state(),
                        torch_cuda=torch.tensor([1,2,3],dtype=torch.uint8)) for rank in range(4)])
                periodic=dict(receipt,completed_outer=10,actual_adam_steps_per_rank=[50]*4)
                history={10:dict(probe=periodic,rollout=dict(actor_version='sampling-before10'),
                    probe_metrics=dict(scored_features_manifest_sha256='e'*64,own_em_sha256='f'*64))}
                self.assertEqual(qualify.probe_state(saved,periodic,c,11,history)['actual_adam_steps_per_rank'],[50]*4)
                saved['rank_rng'][3]['actual_adam_steps']=49
                with self.assertRaises(ValueError):qualify.probe_state(saved,periodic,c,11,history)

    def test_tiny_real_cpu_adam_rng_and_scheduler_allow_actual_skip_counts(self):
        import numpy as np
        import torch
        config=dict(run_id='run',source_commit='a'*40,official_model_dir='/data/official',nonmixed_advantage_mode='zero_anchor_all_correct_only')
        grounding_state=dict(version='ground0',lifecycle={'same':'fixture'})
        actor=dict(binding=_state_binding(dict(config,grounding_state=grounding_state)),actual_adam_steps_per_rank=[3]*4,
            grounding_state=grounding_state,grounding_probe_version='ground0',original_rollout_sha256='f'*64)
        rng=dict(python=random.Random(1).getstate(),numpy=np.random.RandomState(1).get_state(),
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.tensor([1,2,3],dtype=torch.uint8))
        optimizer=dict(param_groups=[dict(params=[0],lr=learning_rate(2),weight_decay=.01,betas=(.9,.999))],
            state={0:dict(step=torch.tensor(3.),exp_avg=torch.zeros(2),exp_avg_sq=torch.ones(2))})
        saved=dict(run_id='run',rank=0,completed_outer=2,binding=actor['binding'],actual_adam_steps=3,scheduler_completed_outer=2,
            grounding_state=grounding_state,grounding_probe_version='ground0',original_rollout_sha256='f'*64,
            model_master_dtype='torch.float32',optimizer_layout='FSDP_FULL_SHARD_same_Qwen2DecoderLayer_wrap_world4',
            optimizer_parameter_layout=[[dict(dtype='torch.float32',shape=[2],numel=2)]],optimizer=optimizer,
            scheduler=dict(last_epoch=2,_step_count=3,base_lrs=[1e-6],_last_lr=[learning_rate(2)]),rng=rng)
        self.assertEqual(qualify.actor_state(saved,actor,config,2,0)['actual_adam_steps'],3)
        saved['optimizer']['state'][0]['exp_avg'][0]=float('nan')
        with self.assertRaises(ValueError):qualify.actor_state(saved,actor,config,2,0)


if __name__=='__main__':unittest.main()
