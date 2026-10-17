import copy
import unittest

from .checkpoint_cpu_once import validate_probe_clock, validate_selection
from .diversity import select_groups, summarize
from .test_diversity import fixture


class ActualMilestoneContractTests(unittest.TestCase):
    def test_diversity_recomputed_from_all1024_original_labels(self):
        selection, labels = fixture([0]*256+[1]*256+[2]*256+[3]*256)
        for tid, row in labels.items(): row['trajectory_id'] = tid
        chosen, all_groups = select_groups(selection, labels)
        hashes = {key: {'sha256': key} for key in ('selection', 'rollout', 'actor_selection')}
        selected = dict(selected=chosen, continued1024=summarize(all_groups), actor512=summarize(chosen),
            prefix_selection_sha256='selection', full_rollout_sha256='rollout',
            outcome_selection='population_strict_EM_variance_K_times_5minusK_over25',
            tie_break='frozen_prefix_q_descending_then_original_candidate_ordinal',
            probe_uses_all_continued1024=True, uncontinued_candidate_questions=4096,
            all_candidate_accuracy_available=False)
        actor_rollout = dict(actor_selection_metrics=selected, actor_selection_sha256='actor_selection',
            source_full_rollout_sha256='rollout', questions=512, trajectories=2560,
            actor_version='actor4', raw_native_records_preserved=True)
        args = (selection, list(labels.values()), selected, {'actor_version': 'actor4'},
                actor_rollout, {'actor_selection': selected}, hashes)
        self.assertTrue(validate_selection(*args)['deterministic_population_variance_top512_recomputed'])
        selected['selected'] = list(reversed(chosen))
        with self.assertRaises(ValueError): validate_selection(*args)

    def test_twenty_synchronous_steps_true_mean_and_prediction_clock(self):
        ranks = [[] for _ in range(8)]; globals_ = []; extra = 0
        for step in range(1, 21):
            n = 52 if step%5 == 0 else 51
            counts = [6]*8
            for j in range(n-48): counts[(extra+j)%8] += 1
            extra = (extra+n-48)%8
            for rank, count in enumerate(counts):
                ranks[rank].append(dict(rank=rank, outer=5, step=step, actual_adam_steps=80+step,
                    actual_learning_rate=1e-6, local_questions=count, local_trajectories=count*5,
                    global_questions=n, trajectory_bce=.75, group_bce=.625, total_loss=1.375,
                    backward_loss=1.375*8*count/n))
            globals_.append(dict(outer=5, step=step, actual_adam_steps_per_rank=80+step,
                actual_learning_rate=1e-6, global_questions=n, global_trajectories=n*5,
                loss_prediction_clock='before_this_step_adam_update', trajectory_bce=.75,
                group_bce=.625, total_loss=1.375))
        metrics = dict(global_step_loss_curve={'points': globals_},
            diagnostic_semantics='same_post_update_head_complete_selected1024G5_and_fixed512G5',
            train_online_trajectory_bce=.75, train_online_group_bce=.625, train_online_total_loss=1.375)
        self.assertEqual(validate_probe_clock(metrics, ranks, globals_, 5)['trajectory_denominator'], 5120)
        wrong = copy.deepcopy(ranks); wrong[0][-1]['actual_adam_steps'] = 99
        with self.assertRaises(ValueError): validate_probe_clock(metrics, wrong, globals_, 5)
        wrong = copy.deepcopy(metrics); wrong['train_online_total_loss'] *= 2
        with self.assertRaises(ValueError): validate_probe_clock(wrong, ranks, globals_, 5)


class CurrentRuntimeAcceptanceTests(unittest.TestCase):
    def test_rank_rollout_is_actor512_and_never_full1024_receipt(self):
        from .checkpoint_cpu_once import validate_actor_rollout_binding
        path='/mnt/fixture/actor-selection/complete.json'
        saved=dict(original_rollout_path=path,original_rollout_sha256='actor512')
        validate_actor_rollout_binding(saved,path,{'sha256':'actor512'})
        with self.assertRaises(ValueError): validate_actor_rollout_binding(saved,path,{'sha256':'full1024'})
        with self.assertRaises(ValueError): validate_actor_rollout_binding(saved,'/mnt/fixture/full/complete.json',{'sha256':'actor512'})

    def test_grounding_clock_binds_original_attempt_not_launch_or_live(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from .actor import _state_binding, REWARD_ENV
        from .checkpoint_i1i6_cpu import audit_attempt_grounding
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve(); (base/'grounding').mkdir()
            initial = base/'initial.json'
            initial.write_text(json.dumps(dict(round=0,version='round0')))
            config = dict(base=str(base), run_id='fixture', source_commit='frozen-source',
                official_model_dir='/mnt/fixture/official', grounding_probe_version='round0',
                grounding_initial_probe_version='round0', grounding_refit_rounds=[0,50,100,150],
                grounding_manifest=str(initial), grounding_manifest_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
                grounding_calibration_sha256='calibration', interaction_contract='fixture-contract')
            manifests = {0:initial}
            for n in (0,50,100,150):
                version = 'round%d'%n
                if n:
                    manifest=base/('manifest%d.json'%n)
                    heads={key:base/('%s%d.pkl'%(key,n)) for key in ('commit','boundary')}
                    for path in heads.values(): path.write_bytes(b'small head fixture')
                    manifest.write_text(json.dumps(dict(run_id='fixture',round=n,version=version,
                        actor_version='fixture-actor%04d'%n,label_proof={'interaction_contract':'fixture-contract'},
                        calibration_data_sha256='calibration', **{k:str(v) for k,v in heads.items()},
                        head_sha256={k+'_probe.pkl':hashlib.sha256(v.read_bytes()).hexdigest() for k,v in heads.items()})))
                    manifests[n]=manifest
                    (base/'grounding'/('round-%03d-complete.json'%n)).write_text(json.dumps(dict(
                        run_id='fixture',round=n,version=version,actual_both_heads_fitted=True,
                        actor_version='fixture-actor%04d'%n,manifest=str(manifest),
                        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
                (base/'grounding'/('adopted-round-%03d.json'%n)).write_text(json.dumps(dict(
                    round=n,version=version,manifest=str(manifests[n]))))
            for outer,n in ((5,0),(50,0),(51,50),(100,50),(101,100),(150,100),(151,150),(200,150)):
                root=base/'outers'/('outer-%06d'%outer);root.mkdir(parents=True)
                attempt=dict(config,actor_attempt_id='ab12',grounding_probe_version='round%d'%n,
                    grounding_active_manifest=str(manifests[n]))
                attempt_path=root/'actor-attempt-ab12-config.json'
                attempt_path.write_text(json.dumps(attempt))
                actor=dict(state_path=str(base/'actor/outers'/('outer-%04d-attempt-ab12'%outer)),
                    actual_global_micro=128, actual_local_micro=16, gradient_accumulation=1,
                    binding=_state_binding(config),reward=dict(version='round%d'%n,trajectories=2560,
                    actual_reward_world_size=8,prefix_correctness_head_used_for_reward=False,
                    ranks=[dict(rank=r,version='round%d'%n,reward_environment=REWARD_ENV,
                        prefix_correctness_head_used_for_reward=False,score_mean=.5) for r in range(8)]))
                proof,_=audit_attempt_grounding(base,config,actor,root,outer)
                self.assertEqual(proof['round'],n)
                wrong=copy.deepcopy(actor);wrong['reward']['ranks'][0]['version']='future'
                with self.assertRaises(ValueError): audit_attempt_grounding(base,config,wrong,root,outer)
                if n:
                    attempt['grounding_probe_version']='round0';attempt_path.write_text(json.dumps(attempt))
                    with self.assertRaises(ValueError): audit_attempt_grounding(base,config,actor,root,outer)

    def test_all_twenty_actor_writes_and_monitors_must_be_finite(self):
        from .checkpoint_i1i6_cpu import validate_pair_writes, OBJECTIVE
        writes=[dict(objective=OBJECTIVE,local_trajectories=16,global_mini=128,grad_norm=.1,
            metrics=[dict(logical_pair=p,policy_tokens=2,pair_policy_tokens=2,gradient_weight=1/8,
                loss=.5,clip_fraction=0.,old_policy_kl=0.) for p in range(8)]) for _ in range(20)]
        monitors=[dict(event='POLICY_MONITOR',stage=stage,optimizer_writes=count,report={'invalid_count':0})
            for n in range(1,21) for stage,count in (('before_minibatch',n-1),('after_minibatch',n))]
        validate_pair_writes(writes,monitors,[1]*2560,1729,50,0)
        for index in (0,9,19):
            wrong=copy.deepcopy(writes);wrong[index]['grad_norm']=float('nan')
            with self.assertRaises(ValueError): validate_pair_writes(wrong,monitors,[1]*2560,1729,50,0)
        wrong=copy.deepcopy(monitors);wrong[0]['report']['invalid_count']=1
        with self.assertRaises(ValueError): validate_pair_writes(writes,wrong,[1]*2560,1729,50,0)

    def test_driver_binds_execution_base_and_actual_reward_with_inherited_state(self):
        import hashlib
        import json
        from pathlib import Path
        from .checkpoint_i1i6_cpu import validate_driver_binding
        base=Path('/mnt/current');config={'run_id':'fixture'}
        previous={'run_id':'fixture','completed_outer':99}
        driver=dict(driver_metadata_binding=dict(source='current_execution_base',outer=100,
            run_id='fixture',score_manifest_sha256='score'),candidate_pool_state={'path':str(base/'question-pool/state.json')},
            prefix_probe_receipt=dict(path=str(base/'probe/latest.json'),value=previous,
                sha256=hashlib.sha256(json.dumps(previous,sort_keys=True,allow_nan=False).encode()).hexdigest()),
            driver_state={'legacy_alias':'/mnt/old'},grounding_probe_version='round50')
        attempt=dict(driver_state=driver['driver_state'],grounding_probe_version='round50')
        validate_driver_binding(driver,base,config,attempt,100,'score')
        wrong=copy.deepcopy(driver);wrong['candidate_pool_state']['path']='/mnt/old/question-pool/state.json'
        with self.assertRaises(ValueError): validate_driver_binding(wrong,base,config,attempt,100,'score')
        wrong=copy.deepcopy(driver);wrong['grounding_probe_version']='round0'
        with self.assertRaises(ValueError): validate_driver_binding(wrong,base,config,attempt,100,'score')

    def test_once_failure_cannot_be_overwritten(self):
        import json
        from pathlib import Path
        import tempfile
        from .checkpoint_cpu_once import _once
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proof.json'
            _once(path,{'status':'FAILED'})
            with self.assertRaises(FileExistsError): _once(path,{'status':'PASS'})
            self.assertEqual(json.loads(path.read_text()),{'status':'FAILED'})


if __name__ == '__main__': unittest.main()
