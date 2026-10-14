"""CPU-only binding tests: reject wrong CP/config before Ray/GPU startup."""
import copy
import tempfile
import unittest
import sys
import types
import hashlib
import json
from unittest.mock import patch
from pathlib import Path
from reproduction.common8.final_resume_runner import file_sha, validate_binding, final_config, resolved_document, finalize_result


class Document(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def document(value):
    return Document({key: document(item) for key, item in value.items()}) if isinstance(value, dict) else copy.deepcopy(value)


class BindingTests(unittest.TestCase):
    def test_original_resolved_launch_wrapper_is_run_and_hash_bound(self):
        config = {'max_turns': 4}
        proof = dict(run_id='same', resolved_config=config,
            resolved_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
        self.assertEqual(resolved_document(proof, 'same'), config)
        with self.assertRaises(ValueError): resolved_document(proof, 'other')
        proof['resolved_config']['max_turns'] = 2
        with self.assertRaises(ValueError): resolved_document(proof, 'same')

    def test_journal_interface_imports(self):
        from reproduction.common8.final_resume_journal import make_binding, load_prefix, append_batch, aggregate_final
        self.assertTrue(all(callable(value) for value in (make_binding, load_prefix, append_batch, aggregate_final)))

    def test_final_config_preserves_scientific_recipe_and_never_mutates_source(self):
        original = dict(actor_rollout_ref=dict(model=dict(path='/official', actor_checkpoint_path='/cp5'),
            rollout=dict(gpu_memory_utilization=.1, tensor_model_parallel_size=1, n=1, temperature=1.,
                         top_p=1., response_length=4096), actor=dict(optim=dict(lr=1e-6))),
            trainer=dict(n_gpus_per_node=8, nnodes=1, full_state_restore_receipt='/old', weight_continuation_step=5),
            data=dict(val_batch_size=64, max_prompt_length=4096, max_obs_length=500),
            max_turns=4, retriever=dict(topk=3, url='http://127.0.0.1:18010'),
            do_search=True, algorithm=dict(adv_estimator='grpo'))
        before = copy.deepcopy(original)
        fake = types.SimpleNamespace(OmegaConf=types.SimpleNamespace(create=document, set_struct=lambda *args: None),
                                     open_dict=None)
        with patch.dict(sys.modules, {'omegaconf': fake}):
            result = final_config(original, self.checkpoint)
        self.assertEqual(original, before)
        self.assertNotIn('full_state_restore_receipt', result.trainer)
        self.assertNotIn('actor_checkpoint_path', result.actor_rollout_ref.model)
        self.assertEqual(result.data.max_prompt_length, 4096)
        self.assertEqual(result.data.max_obs_length, 500)
        self.assertEqual(result.actor_rollout_ref.actor.optim.lr, 1e-6)
        self.assertEqual(result.actor_rollout_ref.rollout.temperature, 1.)
        self.assertEqual(result.actor_rollout_ref.rollout.response_length, 4096)
        self.assertEqual(result.retriever, before['retriever'])
        self.assertEqual(result.data.val_batch_size, 256)
        self.assertEqual(result.actor_rollout_ref.rollout.gpu_memory_utilization, .35)

    def test_single_role_uses_direct_ray_constructor_with_required_arguments(self):
        import ast
        import inspect
        from reproduction.common8.final_resume_runner import run_final
        tree = ast.parse(inspect.getsource(run_final))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'workers' for t in node.targets)]
        self.assertEqual(len(assignments), 1)
        calls = []
        class RequiredWorker:
            def __init__(self, config, role):
                calls.append((config, role))
        def with_args(cls, **kwargs):
            return lambda: cls(**kwargs)
        def group(resource_pool, ray_cls_with_init):
            return ray_cls_with_init()
        scope = dict(RayWorkerGroup=group, RayClassWithInitArgs=with_args,
            worker_cls=RequiredWorker, config=types.SimpleNamespace(actor_rollout_ref='cp200'),
            pools=types.SimpleNamespace(get_resource_pool=lambda role: 'eight'),
            Role=types.SimpleNamespace(ActorRollout='rollout'))
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<actual-ray-construction>', 'exec'), scope)
        self.assertEqual(calls, [('cp200', 'actor_rollout')])
        self.assertIsInstance(scope['workers'], RequiredWorker)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.checkpoint = self.root/'cp200'; self.checkpoint.mkdir()
        self.generation = self.checkpoint/'generation_config.json'
        self.generation.write_text('{"eos_token_id":151645}')
        self.manifest = self.root/'checkpoint-manifest.json'; self.manifest.write_text('{"step":200}')
        self.config = self.root/'resolved.json'; self.config.write_text('{"max_turns":4}')
        self.binding = dict(role='final_only_cp200_resume', checkpoint_step=200,
            checkpoint=str(self.checkpoint), checkpoint_manifest=str(self.manifest),
            checkpoint_manifest_sha256=file_sha(self.manifest),
            generation_config_sha256=file_sha(self.generation), resolved_config_sha256=file_sha(self.config),
            validation_sha256='a'*64, run_id='scientific-run', source_commit='b'*40,
            source_sha256={'runner.py':'c'*64}, journal_binding={'step':200})

    def test_bound_artifacts_accept_without_gpu_imports(self):
        validate_binding(self.binding, self.config)

    def test_complete_result_reentry_is_idempotent_and_rejects_changes(self):
        path = self.root/'final-result.json'
        value = dict(qa_count=51713, correct=27000)
        finalize_result(path, value)
        finalize_result(path, value)
        with self.assertRaises(ValueError):
            finalize_result(path, dict(qa_count=51713, correct=27001))

    def test_previous_checkpoint_rejected(self):
        wrong = dict(self.binding, checkpoint_step=5)
        with self.assertRaises(ValueError): validate_binding(wrong, self.config)

    def test_changed_config_rejected(self):
        self.config.write_text('{"max_turns":2}')
        with self.assertRaises(ValueError): validate_binding(self.binding, self.config)

    def test_changed_generation_policy_rejected(self):
        self.generation.write_text('{"eos_token_id":0}')
        with self.assertRaises(ValueError): validate_binding(self.binding, self.config)

    def test_changed_checkpoint_receipt_rejected(self):
        self.manifest.write_text('{"step":5}')
        with self.assertRaises(ValueError): validate_binding(self.binding, self.config)

    def test_incomplete_identity_rejected(self):
        wrong = dict(self.binding); wrong.pop('validation_sha256')
        with self.assertRaises(ValueError): validate_binding(wrong, self.config)


if __name__ == '__main__': unittest.main()
