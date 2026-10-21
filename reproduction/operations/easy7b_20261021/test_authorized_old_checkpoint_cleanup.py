"""Destructive-scope regressions: never confuse trajectories or 7B assets with old CPs."""
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('old_cleanup',HERE/'apply_authorized_old_checkpoint_cleanup.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Scope(unittest.TestCase):
 def test_actual_checkpoint_layouts(self):
  for rel in [
   'dynrpo-prefix5-thread5-20261017-r4/actor/outers/outer-0200-attempt-acde1234/rank-0.pt',
   'dynrpo-prefix5-thread5-20261017-r4/probe/head-outer-0190.pt',
   'dynrpo-prefix5-thread5-20261017-r4/outers/outer-000190/probe-update/epoch-003.pt',
   'cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015/models/main/model-00001-of-00002.safetensors',
   'dynrpo-prefix5-thread5-20261017-r2/actor/outers/outer-0200-attempt-acde1234/huggingface/model-00001.safetensors',
   'topd-20261008/runtime/logs/topd-hpc3-707690/evidence/checkpoints/actor/global_step_200/resume-rank0.pt',
   'easy2560-corollout-joint4-micro8-after731063-20261018/checkpoints/search-g1-common8/actor/global_step_200/training_state/driver.pt',
   'easy2560-corollout-joint4-micro8-after731063-20261018/preparation/rolling-checkpoints/global_step_200/boundary/driver.pt',
   'cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011/checkpoints/global_step_200/reranker_actor_rollout/model_world_size_4_rank_0.pt',
  ]:self.assertTrue(m.allowed(m.ROOT/rel),rel)
 def test_keep_every_protected_experiment_and_shared_base(self):
  for rel in [
   'dynrpo-qwen25-7b-probe-warmup-20261019/checkpoints/global_step_200/model.safetensors',
   'qprobe-teacher-warmup200-sigma3-20261015/checkpoints/global_step_50/model.safetensors',
   'easy2560-corollout-joint4-qwen7b-20261021/checkpoints/global_step_1/model.safetensors',
   'easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021/checkpoints/global_step_4/model.safetensors',
   'topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725/model.safetensors',
  ]:self.assertFalse(m.allowed(m.ROOT/rel),rel)
 def test_keep_raw_trajectory_even_with_step_name(self):
  for rel in [
   'dynrpo-prefix5-thread5-20261017-r4/outers/outer-0200-attempt-acde1234/raw.pt',
   'topd-20261008/outputs/global_step_200/trajectories.pt',
   'easy2560-corollout-joint4-20261017/final-eval/global_step_200/data.pt',
   'cosearch-continue300-x203-20261008/checkpoints/global_step_200/trajectories/results.pt',
   'topd-20261008/runtime/logs/x/checkpoints/global_step_200/eval-rollout.pt',
   'topd-20261008/runtime/logs/x/checkpoints/global_step_200/complete.json',
  ]:self.assertFalse(m.allowed(m.ROOT/rel),rel)

if __name__=='__main__':unittest.main()
