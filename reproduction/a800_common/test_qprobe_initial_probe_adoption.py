"""CPU provenance checks; these do not claim actual calibration/model admission."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reproduction.a800_common import qprobe_adoption_source_delta as sources
from reproduction.common8.source_files import runtime_files


class Initial0Micro4BudgetTests(unittest.TestCase):
    def fixture(self):
        from reproduction.common8.formal_contract import formal_mathematics
        previous=dict(profile='common8',training_world_size=4,run_id='6de29bcb7be741cfa1bfacdf9e34efc9',
            source_commit='f820ba6cb2db4f5cbe2ffc370cf515c3fe22d43c')
        current=dict(profile='common8',execution_backend='ssh_local',training_world_size=4,completed_updates=0,
            target_updates=200,actor_micro_batch_per_rank=4,precision_profile='bf16_policy_fp32_master')
        budget=json.loads((Path(__file__).resolve().parents[2]/'reproduction/common8/budget.json').read_text())
        previous['budget']=dict(budget,**formal_mathematics(previous))
        current['budget']=dict(budget,**formal_mathematics(current))
        return previous,current

    def test_only_exact_fresh_micro4_and_existing_precision_transition_are_accepted(self):
        from reproduction.a800_common.adopt_initial_probes import budget_changes_allowed
        previous,current=self.fixture();changes=budget_changes_allowed(current,previous)
        self.assertEqual(set(changes),{'compute_dtype','rollout_dtype','allow_tf32','per_rank_actor_micro_batch',
                                      'ppo_micro_batch_size','objective_reduction'})
        for patch in ({'training_world_size':8},{'completed_updates':1},{'actor_micro_batch_per_rank':4.0},
                      {'full_state_restore':{'checkpoint':'prior'}}):
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                budget_changes_allowed(dict(current,**patch),previous)
        for key,value in (('ppo_micro_batch_size',32),('per_rank_actor_micro_batch',2),
                          ('objective_reduction','other reduction'),('lr_warmup_steps',58),('per_rank_log_prob_micro_batch',4)):
            changed=copy.deepcopy(current);changed['budget'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):budget_changes_allowed(changed,previous)


class QProbeInitial0SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.previous = sources.git_source_map(cls.repo, sources.ORIGINAL_SOURCE_COMMIT)
        cls.delta = sources.generate(cls.repo)
        cls.delta_raw = json.dumps(cls.delta, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
        cls.current = {str(p.relative_to(cls.repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in runtime_files(cls.repo) if str(p.relative_to(cls.repo)) != sources.DELTA_PATH}
        cls.current[sources.DELTA_PATH] = hashlib.sha256(cls.delta_raw).hexdigest()

    def setUp(self):
        self.previous = dict(self.previous)
        self.current = dict(self.current)
        self.delta = copy.deepcopy(self.delta)
        scientific = dict(initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',
            reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',
            initial_model_sha256={'model.safetensors': 'a'*64},
            model_revision=sources.MODEL_REVISION,
            input_sha256={'train.parquet': 'b'*64, 'calibration-300.parquet': 'c'*64},
            calibration_question_ids=['reserved-'+str(i) for i in range(300)],
            retrieval_asset_paths={'e5_Flat.index': '/data/search-r1/assets/wiki/e5_Flat.index'},
            retrieval_encoder='/data/search-r1/assets/e5-base-v2')
        self.old = dict(scientific, profile='common8', execution_backend='ssh_local',
            training_world_size=4, completed_updates=0, target_updates=200,
            run_id=sources.ORIGINAL_RUN_ID, execution_id='d'*32,
            source_commit=sources.ORIGINAL_SOURCE_COMMIT, source_sha256=self.previous)
        self.run = dict(copy.deepcopy(scientific), profile='common8', execution_backend='ssh_local',
            training_world_size=4, completed_updates=0, target_updates=200,
            run_id='e'*32, execution_id='f'*32, source_commit='a'*40,
            source_sha256=self.current, precision_profile=sources.PRECISION_PROFILE,
            initial_probe_adoption_source_base=sources.ORIGINAL_BASE,
            mixed_first_selection=False, retriever_version='1'*64,
            question_probe=dict(candidate_questions=5120, selected_questions=512,
                responses_per_question=5, exploration_questions=0,
                bootstrap_max_groups_per_outer=5120,
                teacher_warmup_cache='/data/shaoshitong/new-qprobe/teacher-warmup-cache'))

    def validate(self):
        return sources.validate(self.delta, self.previous, self.current, self.run, self.old)

    def rebound_delta(self):
        """Even regenerated pairs may not authorize an unapproved core edit."""
        target = {k:v for k,v in self.current.items() if k != sources.DELTA_PATH}
        self.delta['target_source_map_sha256_without_delta'] = sources.document_sha(target)
        self.delta['changes'] = {name: dict(old_sha256=self.previous.get(name), new_sha256=target.get(name))
            for name in sorted(set(self.previous)|set(target)) if self.previous.get(name) != target.get(name)}

    def test_exact_original_to_qprobe_delta_passes(self):
        changes = self.validate()
        self.assertEqual(changes, self.delta['changes'])
        self.assertIn('verl/trainer/ppo/ray_trainer.py', changes)
        self.assertIn('search_g1/qprobe_features.py', changes)
        self.assertNotIn('ts_probe_train.py', changes)

    def test_changed_scorer_and_binary_feature_or_head_fitting_reject(self):
        for name in ('search_g1/signals.py', 'verl/utils/reward_score/qa_em.py',
                     'ts_probe_dump.py', 'ts_probe_train.py', 'probe_server.py'):
            with self.subTest(name=name):
                saved = self.current[name]
                self.current[name] = '9'*64
                self.rebound_delta()
                with self.assertRaisesRegex(ValueError, 'UNAPPROVED_INTEGRATION_EDIT|FROZEN_G1'):
                    self.validate()
                self.current[name] = saved
                self.rebound_delta()

    def test_changed_core_rejects_even_with_updated_pairs(self):
        self.current['reproduction/common8/behavior_runtime.py'] = '9'*64
        self.rebound_delta()
        with self.assertRaisesRegex(ValueError, 'UNAPPROVED_INTEGRATION_EDIT'):
            self.validate()

    def test_changed_qprobe_feature_definition_rejects(self):
        for name in ('search_g1/qprobe_features.py', 'search_g1/qprobe_prepare.py'):
            with self.subTest(name=name):
                saved = self.current[name]
                self.current[name] = '9'*64
                self.rebound_delta()
                with self.assertRaisesRegex(ValueError, 'UNAPPROVED_INTEGRATION_EDIT'):
                    self.validate()
                self.current[name] = saved
                self.rebound_delta()

    def test_explicit_question_head_loss_edits_require_exact_complete_source_delta(self):
        self.run['question_probe'].update(loss_mode='sigma3_std_aux6_v1',mean_loss_weight=0.0,
                                          std_loss_weight=3.0,count_aux_loss_weight=0.1)
        for name in ('search_g1/qprobe_model.py','reproduction/common8/qprobe_configuration.py',
                     'reproduction/common8/qprobe_bind.py','verl/trainer/config/ppo_trainer.yaml'):
            with self.subTest(name=name):
                saved=self.current[name];self.current[name]='9'*64
                with self.assertRaisesRegex(ValueError,'TARGET_SOURCE_MAP_DIFFERS'):self.validate()
                self.rebound_delta();self.assertEqual(self.validate(),self.delta['changes'])
                self.current[name]=saved;self.rebound_delta()

    def test_changed_allowed_integration_bytes_need_a_new_exact_delta(self):
        self.current['search_g1/qprobe_warmup.py'] = '9'*64
        with self.assertRaisesRegex(ValueError, 'TARGET_SOURCE_MAP_DIFFERS'):
            self.validate()

    def test_unknown_source_addition_rejects_even_with_updated_pairs(self):
        self.current['search_g1/new_unreviewed_core.py'] = '9'*64
        self.rebound_delta()
        with self.assertRaisesRegex(ValueError, 'UNAPPROVED_INTEGRATION_EDIT'):
            self.validate()

    def test_missing_changed_pair_and_wrong_origin_reject(self):
        self.delta['changes'].pop('verl/workers/actor/dp_actor.py')
        with self.assertRaisesRegex(ValueError, 'CHANGED_SOURCE_PAIRS_DIFFER'):
            self.validate()
        self.delta = copy.deepcopy(type(self).delta)
        self.delta['original_run_id'] = 'wrong-run'
        with self.assertRaisesRegex(ValueError, 'UNRECOGNIZED_DELTA'):
            self.validate()

    def test_rewritten_baseline_cannot_authorize_a_core_edit(self):
        self.current['verl/workers/actor/dp_actor.py'] = '9'*64
        self.delta['approved_base_source_sha256']['verl/workers/actor/dp_actor.py'] = '9'*64
        self.delta['approved_base_map_sha256'] = sources.document_sha(self.delta['approved_base_source_sha256'])
        self.rebound_delta()
        with self.assertRaisesRegex(ValueError, 'APPROVED_BASELINE_DIFFERS'):
            self.validate()

    def test_changed_model_or_data_identity_rejects(self):
        for name in ('initial_model_sha256', 'input_sha256', 'calibration_question_ids'):
            with self.subTest(name=name):
                before = copy.deepcopy(self.run[name])
                self.run[name] = {'changed': '9'*64}
                with self.assertRaisesRegex(ValueError, 'SCIENTIFIC_IDENTITY_CHANGED'):
                    self.validate()
                self.run[name] = before

    def test_nonzero_counter_or_inherited_actor_or_head_rejects(self):
        self.run['completed_updates'] = 1
        with self.assertRaisesRegex(ValueError, 'FRESH_FOUR_RANK_ACTOR'):
            self.validate()
        self.run['completed_updates'] = 0
        self.run['full_state_restore'] = {'checkpoint': 'prior-actor'}
        with self.assertRaisesRegex(ValueError, 'FRESH_FOUR_RANK_ACTOR'):
            self.validate()
        self.run.pop('full_state_restore')
        self.run['question_probe']['initial_head_checkpoint'] = '/data/prior/head.pt'
        with self.assertRaisesRegex(ValueError, 'MUST_NOT_INHERIT'):
            self.validate()

    def test_target_delta_bytes_and_helpers_are_checked_by_adoption_gate(self):
        from reproduction.a800_common import adopt_initial_probes as adoption
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            doc = base/'repo'/sources.DELTA_PATH
            doc.parent.mkdir(parents=True)
            doc.write_bytes(self.delta_raw)
            module = base/'repo/reproduction/a800_common/adopt_initial_probes.py'
            with patch.object(adoption, '__file__', str(module)):
                self.assertTrue(adoption.source_changed_allowed(
                    'reproduction/a800_common/adopt_initial_probes.py',
                    self.previous, self.current, self.run, self.old))
                doc.write_bytes(self.delta_raw+b' ')
                with self.assertRaisesRegex(ValueError, 'differs from the target source snapshot'):
                    adoption.source_changed_allowed('reproduction/a800_common/adopt_initial_probes.py',
                        self.previous, self.current, self.run, self.old)


if __name__ == '__main__':
    unittest.main()
