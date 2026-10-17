"""Reproduce the native P10 diagnostic/main-pool boundary without CUDA."""
import ast
import copy
import json
import re
from pathlib import Path
import tempfile
import unittest

from reproduction.prefix_selection200.adaptive_candidates import policy
from reproduction.prefix_selection200.candidate_schedule import candidate_count


ROOT = Path(__file__).resolve().parents[2]


def function(module, name, namespace):
    namespace.setdefault('__name__', 'reproduction.prefix_selection200._fixture')
    namespace.setdefault('__package__', 'reproduction.prefix_selection200')
    tree = ast.parse((ROOT / module).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), module, 'exec'), namespace)
    return namespace[name]


class DiagnosticScopeTests(unittest.TestCase):
    def test_driver_json_seal_is_not_compared_to_tensor_payload_digest(self):
        def need(ok,message):
            if not ok:raise ValueError(message)
        check=function('reproduction/prefix_selection200/i1_i6_checkpoint_cpu.py',
                       'selection_probe_state',dict(need=need,WORLD=4,
                       probe_boundary=lambda outer:dict(last_update_outer=0,actual_adam_per_rank=0)))
        selected=dict(sha256='a'*64,value=dict(run_id='r',completed_outer=0,
                      actual_adam_steps_per_rank=[0]*4,sha256='b'*64))
        score=dict(probe_before_checkpoint_sha256='b'*64)
        check(selected,score,dict(run_id='r'),9)
        bad=copy.deepcopy(selected);bad['value']['sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'selection head'):
            check(bad,score,dict(run_id='r'),9)
        bad=copy.deepcopy(selected);bad['value']['actual_adam_steps_per_rank']=[50]*4
        with self.assertRaisesRegex(ValueError,'selection head'):
            check(bad,score,dict(run_id='r'),9)

    def test_qualification_does_not_require_pruned_historical_probe_aliases(self):
        inputs = function('reproduction/prefix_selection200/i1_i6_checkpoint_cpu.py',
                          'qualification_inputs', dict(Path=Path))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve()
            names=['official/config.json','terminal.json','actor/hf/config.json',
                   'actor/rank0.pt','actor/driver.pt','old/grounding.json','old/pool.json',
                   'grounding/manifest.json','grounding/commit.pkl','grounding/boundary.pkl',
                   'grounding/adoption.json']
            for name in names:
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            old=dict(probe_snapshot=str(base/'old/pruned-probe.pt'),
                grounding_snapshot=str(base/'old/grounding.json'),pool_snapshot=str(base/'old/pool.json'),
                grounding={key:str(base/'grounding'/name) for key,name in
                    [('manifest_path','manifest.json'),('commit','commit.pkl'),
                     ('boundary','boundary.pkl'),('adoption_receipt','adoption.json')]})
            actor=dict(hf_model_dir=str(base/'actor/hf'),hf_files={'config.json':{}},
                rank_states=[dict(path=str(base/'actor/rank0.pt'))],driver_state=dict(path=str(base/'actor/driver.pt')))
            paths=inputs(dict(official_model_dir=str(base/'official')),actor,{1:old},base/'terminal.json')
            self.assertNotIn(Path(old['probe_snapshot']),paths)
            self.assertTrue(all(path.resolve(strict=True)==path for path in paths))
            # Required current actor assets are never silently dropped.
            (base/'actor/rank0.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                [path.resolve(strict=True) for path in paths]

    def test_rotating_storage_accepts_only_owned_hpc3_recovery_siblings(self):
        base=Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
        admits=function('reproduction/prefix_selection200/storage.py','hpc3_namespace',
                        dict(HPC3_BASE=base,re=re))
        for name in (base.name,base.name+'-r2',base.name+'-r19'):
            self.assertTrue(admits(base.with_name(name)))
        for name in (base.name+'-other',base.name+'-r0',base.name+'-r2-foreign',base.name+'-r'):
            self.assertFalse(admits(base.with_name(name)))
        self.assertFalse(admits(Path('/tmp')/(base.name+'-r2')))

    def test_fixed512_never_resolves_the_diagnostic_directory_as_training_history(self):
        isolate = function('reproduction/prefix_selection200/launch.py',
                           'diagnostic_sampling_config', {})
        count = function('reproduction/prefix_selection200/rollout.py', 'question_count',
                         dict(candidate_count=candidate_count, PERIODIC_CANDIDATES=2560,
                              SELECTED_QUESTIONS=512))
        with tempfile.TemporaryDirectory() as directory:
            original = dict(base=directory, run_id='r', adaptive_candidates=policy(168508),
                candidate_schedule=dict(effective_from_outer=1, legacy_candidates=2560,
                    ordinary_candidates=5120, update_candidates=2560, probe_update_period=10,
                    probe_update_at_final=False, selected_questions=512),
                diagnostic_only=True, rollout_candidate_questions=512,
                rollout_continuation_questions=512, actor_version='actor0009')
            before = copy.deepcopy(original)
            with self.assertRaises(FileNotFoundError):
                count(original, 'prefix', 10)
            fixed = isolate(original)
            for phase in ('prefix', 'continue'):
                self.assertEqual(count(fixed, phase, 10), 512)
            self.assertEqual(original, before)
            self.assertEqual(fixed['actor_version'], 'actor0009')

    def test_bound_sampling_budget_is_not_carried_into_fixed_diagnostic(self):
        isolate = function('reproduction/prefix_selection200/launch.py',
                           'diagnostic_sampling_config', {})
        original = dict(candidate_budget={'questions': 2560}, adaptive_candidates={},
                        candidate_schedule={}, sampling_attempt_id='retry1', run_id='r')
        fixed = isolate(original)
        self.assertFalse(set(fixed) & {'candidate_budget', 'adaptive_candidates', 'candidate_schedule'})
        self.assertEqual(fixed['sampling_attempt_id'], 'retry1')
        self.assertIn('candidate_budget', original)

    def test_recovery_plan_uses_source_feedback_before_target_history_is_imported(self):
        from reproduction.prefix_selection200.adaptive_candidates import ensure
        pending = function('reproduction/prefix_selection200/i1_i6_joint_resume.py',
                           'pending_candidate_questions', {})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            root = source / 'outers/outer-000001'
            root.mkdir(parents=True)
            original = dict(base=str(source), run_id='r', adaptive_candidates=policy(168508))
            ensure(original, 1)
            joint = dict(run_id='r', outer=1, rollout=dict(run_id='r', outer=1,
                questions=512, trajectories=2560, actual_selected_mixed_questions=256,
                actual_selected_mixed_fraction=.5))
            (root / 'complete.json').write_text(json.dumps(joint))
            target = dict(original, base=str(Path(directory) / 'not-imported-yet'))
            with self.assertRaises(FileNotFoundError):
                candidate_count(target, 2)
            self.assertEqual(pending(target, original, 2), 10240)


if __name__ == '__main__':
    unittest.main()
