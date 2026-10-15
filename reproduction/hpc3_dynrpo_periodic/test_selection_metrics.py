"""Exercise the real collector/trainer metric boundary without model imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]


def production_audit(namespace):
    """Run collect's actual audit construction after completed real-rollout labels.

    The fixture substitutes only the rollout/model inputs. Keeping these AST
    statements from production makes this catch a producer/consumer mismatch
    without importing CUDA, Ray, or the optional DataProto test dependencies.
    """
    path = ROOT / 'search_g1/qprobe_runtime.py'
    tree = ast.parse(path.read_text())
    collect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    start = next(i for i, n in enumerate(collect.body)
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == 'categories')
    end = next(i for i, n in enumerate(collect.body[start:], start)
               if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == 'audit' and n.value.func.attr == 'update')
    module = ast.Module(body=collect.body[start:end + 1], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['audit']


def trainer_metrics(audit, token_binary):
    path = ROOT / 'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Compare)
                 and isinstance(n.test.left, ast.Attribute)
                 and n.test.left.attr == 'qprobe_runtime'
                 and any(isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name)
                         and x.value.id == 'mixed_audit' for x in ast.walk(n)))
    namespace = dict(self=SimpleNamespace(qprobe_runtime=SimpleNamespace(token_binary=token_binary)),
                     mixed_audit=audit, metrics={})
    exec(compile(ast.Module(body=block.body, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['metrics']


class SelectionMetricBoundaryTests(unittest.TestCase):
    def test_periodic_binary_and_legacy_audits_reach_actual_trainer_metrics(self):
        for token_binary, exploration, metric in (
                (True, 0, 'selected_predicted_binary_mixed'),
                (False, 64, 'selected_predicted_diversity')):
            with self.subTest(token_binary=token_binary):
                questions = [SimpleNamespace(content_hash='content-' + str(i)) for i in range(5120)]
                members = [SimpleNamespace(ordinal=i, qid='q' + str(i),
                    selection_source=('predicted_binary_mixed' if token_binary else 'predicted_diversity'))
                    for i in range(512)]
                tensor = SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 12345))
                measured = dict(all=dict(questions=512, binary_accuracy=.625,
                    binary_confusion=dict(tp=120, tn=200, fp=112, fn=80), is_diagnostic=True))
                audit = production_audit(dict(questions=questions, members=members,
                    vectors=[[int(j < i % 6) for j in range(5)] for i in range(512)],
                    runtime=SimpleNamespace(token_binary=token_binary,
                        head_kind='qwen_native_periodic_lastblock_binary_v1', feature_spec_sha='feature'),
                    exploration_questions=exploration,
                    output=SimpleNamespace(batch={'sampling_mask': tensor},
                        meta_info={'actual_retrieval_stats': [0] * 2560}),
                    manager=SimpleNamespace(timing_raw={'generate': 2.}),
                    time=SimpleNamespace(monotonic=lambda: 3.), started=1.,
                    outer=1, tag='attempt-1', identity=dict(policy_snapshot_id='actor:0',
                        trajectory_ids=['trajectory-' + str(i) for i in range(2560)]),
                    stream=SimpleNamespace(consumed=5120),
                    run=dict(source_commit='commit', model_revision='revision',
                             budget={'common_manifest_sha256': 'data'}),
                    trainer=SimpleNamespace(reward_fn=SimpleNamespace(expected_probe_version='reward')),
                    prediction_version='head:400', measured=measured, contracts={}))
                self.assertEqual(audit[metric], 512 - exploration)
                self.assertNotIn('selected_predicted_diversity' if token_binary
                                 else 'selected_predicted_binary_mixed', audit)
                metrics = trainer_metrics(audit, token_binary)
                self.assertEqual(metrics['selection/' + metric], 512 - exploration)
                self.assertEqual(metrics['selection/selected_all_wrong'] +
                    metrics['selection/selected_all_correct'] + audit['selected_mixed'], 512)
                self.assertEqual(metrics['selection/selected_random_exploration'], exploration)
                self.assertEqual(metrics['question_probe/prequential/all/questions'], 512)
                self.assertEqual(metrics['question_probe/prequential/all/binary_accuracy'], .625)
                self.assertNotIn('question_probe/prequential/all/binary_confusion', metrics)
                self.assertNotIn('question_probe/prequential/all/is_diagnostic', metrics)


if __name__ == '__main__':
    unittest.main()
