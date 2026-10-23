"""CPU coverage for default compatibility and explicit native4 rollout wiring."""
import ast
import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from search_g1.easy7b_infra import flags, profile_flags, LEAN_PROFILES, FA3_PROFILES
from reproduction.easy2560_joint4.prepare_training import selection_settings

ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / 'verl/workers/rollout/vllm_rollout/vllm_rollout.py'
NEW = 'fa3-lean-search-stop-v1'


class SearchStopWiringTests(unittest.TestCase):
    def fixture(self, root, profile):
        from reproduction.common8.test_configuration import ConfigurationTests
        from reproduction.common8.formal_contract import formal_mathematics
        from reproduction.common8.model_identity import SEVEN
        from reproduction.easy2560_joint4.test_common_contracts import easy_run
        from reproduction.easy2560_joint4.control import MODEL
        base, run = ConfigurationTests().fixture(root)
        run.update(easy_run(8), model_revision=SEVEN[1], initial_model=MODEL, reference_model=MODEL)
        run.update(selection_settings('/remote/predecessor', 8, infra_profile=profile))
        run['budget'].update(model_id=SEVEN[0], model_revision=SEVEN[1], world_size=4)
        run['budget'].update(formal_mathematics(run))
        (base / 'logs/run-manifest.json').write_text(json.dumps(run))
        return base, run

    def test_new_profile_adds_only_search_flag_and_keeps_old_profile_exact(self):
        old = profile_flags('fa3-lean-v1'); new = profile_flags(NEW)
        self.assertEqual(old, dict(selected_policy_logits=False, exact_retrieval_cache=True,
            flash_attention_3_actor=True, flash_attention_3_rollout=True,
            flash_attention_3_compact_gqa=True, reuse_snapshot_weights=True))
        self.assertTrue(new.pop('search_action_boundary_stop')); self.assertEqual(new, old)
        a = selection_settings('/remote/predecessor', 8, infra_profile='fa3-lean-v1')
        b = selection_settings('/remote/predecessor', 8, restoring=True, infra_profile=NEW)
        a.pop('infra_optimizations'); b.pop('infra_optimizations')
        self.assertEqual(a.pop('infra_profile'), 'fa3-lean-v1')
        self.assertEqual(b.pop('infra_profile'), NEW); self.assertEqual(a, b)
        self.assertIn(NEW, LEAN_PROFILES); self.assertIn(NEW, FA3_PROFILES)

    def test_real_recipe_arguments_add_one_flag_and_keep_native4_200_batch(self):
        from reproduction.common8.configuration import arguments
        with tempfile.TemporaryDirectory() as td:
            base, old = self.fixture(td, 'fa3-lean-v1')
            _, old_args = arguments(base)
            new = copy.deepcopy(old)
            new.update(selection_settings('/remote/predecessor', 8, infra_profile=NEW))
            (base / 'logs/run-manifest.json').write_text(json.dumps(new))
            _, new_args = arguments(base)
            toggle = '+actor_rollout_ref.rollout.search_action_boundary_stop=true'
            self.assertEqual([value for value in new_args if value != toggle], old_args)
            self.assertEqual(new_args.count(toggle), 1)
            for value in ('trainer.n_gpus_per_node=4', 'trainer.total_training_steps=201',
                    'actor_rollout_ref.actor.ppo_mini_batch_size=128',
                    'actor_rollout_ref.actor.ppo_micro_batch_size=32',
                    'actor_rollout_ref.actor.ppo_logical_micro_batch_size=4',
                    'data.selection_candidate_questions=2560', 'data.selection_selected_questions=512'):
                self.assertIn(value, new_args)
            self.assertEqual(new['target_updates'], 200)
            self.assertTrue(flags(new)['search_action_boundary_stop'])
            self.assertNotIn('search_action_boundary_stop', flags(old))

    def test_launch_command_preserves_four_gpu_and_other_arguments(self):
        from reproduction.easy2560_joint4.control import BASE, launch_command
        old = launch_command(BASE, 'a' * 40, 8, infra_profile='fa3-lean-v1')
        new = launch_command(BASE, 'a' * 40, 8, infra_profile=NEW)
        self.assertEqual(old[:-1], new[:-1]); self.assertEqual(new[-1], NEW)
        self.assertIn('--gres=gpu:4', new)

    def test_actual_constructor_installs_only_for_explicit_true(self):
        tree = ast.parse(ROLLOUT.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'vLLMRollout')
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        node = next(node for node in init.body if isinstance(node, ast.If) and
            'search_action_boundary_stop' in ast.unparse(node.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), str(ROLLOUT), 'exec')
        engine, tokenizer = object(), object()
        for value in (False, None, 1, True):
            obj = SimpleNamespace(inference_engine=SimpleNamespace(llm_engine=engine), _search_g1_action_stop=None)
            with patch('search_g1.search_action_stop.install_search_action_stop', return_value='installed') as install:
                exec(code, dict(self=obj, config={'search_action_boundary_stop': value}, tokenizer=tokenizer))
                if value is True:
                    install.assert_called_once_with(engine, tokenizer)
                    self.assertEqual(obj._search_g1_action_stop, 'installed')
                else:
                    install.assert_not_called(); self.assertIsNone(obj._search_g1_action_stop)

    def test_actual_phase_report_adds_only_enabled_counter_receipt(self):
        tree = ast.parse(ROLLOUT.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'vLLMRollout')
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'end_generation_phase')
        scope = {'copy': copy}; exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROLLOUT), 'exec'), scope)
        for enabled in (False, True):
            counter = SimpleNamespace(report=lambda: {'confirmed_search_stops': 42}) if enabled else None
            obj = SimpleNamespace(_generation_phase={'service': SimpleNamespace(close=lambda abort: {'completed_requests': 64})},
                _generation_phase_setup_handle={'phase_id': 'p'}, generation_scheduler_limits={'max_num_seqs': 64},
                _search_g1_flash3_functions=None, _search_g1_action_stop=counter,
                _generation_cache_setup_started=False,
                inference_engine=SimpleNamespace(llm_engine=SimpleNamespace(_common8_stop_capture=object())))
            result = scope['end_generation_phase'](obj, {'phase_id': 'p'})
            self.assertEqual(result['completed_requests'], 64)
            self.assertEqual(result.get('search_action_stop'), {'confirmed_search_stops': 42} if enabled else None)


class TransitionProfileTests(unittest.TestCase):
    def test_source_binding_default_old_profile_and_explicit_new_restore_globals(self):
        from reproduction.operations.easy7b_20261021 import fa3_lean_transition as lean
        original = lean.transition.PROFILE
        with tempfile.TemporaryDirectory() as td:
            source = Path(td); (source / 'logs').mkdir()
            (source / 'logs/run-manifest.json').write_text(json.dumps({'run_id': 'run'}))
            for kwargs, expected in (({}, 'fa3-lean-v1'), ({'infra_profile': NEW}, NEW)):
                with patch.object(lean.transition.control, 'own_base', side_effect=Path):
                    with lean.source_binding(source_base=source, source_job='1', source_commit='a' * 40,
                            authorization={}, **kwargs) as bound:
                        self.assertEqual(bound.PROFILE, expected)
                self.assertEqual(lean.transition.PROFILE, original)

    def test_cpu_srun_command_only_appends_profile_when_explicit(self):
        from reproduction.operations.easy7b_20261021 import launch_lean_transition_on_retrieval as launch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); authorization = root / 'authorization.json'; authorization.write_text('{}')
            with patch.object(launch, 'own_base', side_effect=Path):
                old = launch.command(root, 'a' * 40, authorization)
                new = launch.command(root, 'a' * 40, authorization, infra_profile=NEW)
            self.assertNotIn('--infra-profile', old)
            self.assertEqual(new[:-2], old); self.assertEqual(new[-2:], ['--infra-profile', NEW])
            self.assertIn('--gres=none', new); self.assertIn('--cpus-per-task=4', new)

    def test_detached_supervisor_and_srun_receive_same_explicit_profile(self):
        from reproduction.operations.easy7b_20261021 import launch_lean_transition_on_retrieval as launch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); authorization = root / 'authorization.json'; authorization.write_text('{}')
            with patch.object(launch, 'own_base', side_effect=Path), patch.object(
                    launch.subprocess, 'Popen', return_value=SimpleNamespace(pid=123)) as spawn, patch.object(
                    launch, 'process_identity', return_value={'pid': 123}):
                launch.launch(root, 'a' * 40, authorization, infra_profile=NEW)
            runner = spawn.call_args.args[0]
            self.assertEqual(runner[-2:], ['--infra-profile', NEW])
            claim = json.loads((root / 'preparation/lean-transition-retrieval-launch-claim.json').read_text())
            self.assertEqual(claim['srun_command'][-2:], ['--infra-profile', NEW])

    def test_both_existing_clis_pass_explicit_profile_without_running_actions(self):
        from reproduction.operations.easy7b_20261021 import fa3_lean_transition as lean
        from reproduction.operations.easy7b_20261021 import launch_lean_transition_on_retrieval as launch
        with tempfile.TemporaryDirectory() as td:
            authorization = Path(td) / 'authorization.json'; authorization.write_text('{}')
            bound = SimpleNamespace(prepare=Mock(return_value={'status': 'mock-only'}))
            with patch.object(lean, 'source_binding', return_value=nullcontext(bound)) as binding, redirect_stdout(io.StringIO()):
                lean.main(['prepare', '--commit', 'a' * 40, '--authorization-json', str(authorization),
                           '--infra-profile', NEW])
            self.assertEqual(binding.call_args.kwargs['infra_profile'], NEW)
            with patch.object(launch, 'launch', return_value={}) as invoke, redirect_stdout(io.StringIO()):
                launch.main(['launch', '--commit', 'a' * 40, '--infra-profile', NEW])
            self.assertEqual(invoke.call_args.kwargs['infra_profile'], NEW)


if __name__ == '__main__':
    unittest.main()
