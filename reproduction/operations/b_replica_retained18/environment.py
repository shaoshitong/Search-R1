"""Inspect the prepared eight-card environment without launching any work.

No allocation, model load, CUDA initialization, checkpoint import or training
is performed. An experiment assignment must be supplied separately by the user.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def inspect(config_path):
    from reproduction.prefix_selection200.contract import WORLD
    from reproduction.prefix_selection200.launch import validate_config, _require_training_assignment
    from reproduction.prefix_selection200.actor import ActorBudget, VIRTUAL_PER_RANK, rank_plan
    from reproduction.prefix_selection200.ab_policy import balanced_plan
    from reproduction.prefix_selection200.probe_plan import _natural_plan, rank_question_slice, ddp_mean_scale
    from reproduction.prefix_selection200.b_progressive import prefix, continuation_questions
    path = Path(config_path); c = json.loads(path.read_text())
    validate_config(c, Path(c['base']))
    if WORLD != 8 or c['physical_gpu_indices'] != list(range(8)):
        raise ValueError('Explicit eight-card environment is required')
    if not (c.get('preparation_only') and c.get('experiment_assignment_pending') and c.get('training_authorized') is False):
        raise ValueError('This inspection is scoped to the unassigned preparation-only environment')
    try:
        _require_training_assignment(c)
    except ValueError:
        launch_blocked = True
    else:
        raise ValueError('Preparation-only environment must reject training')
    budget = ActorBudget(c['actor_global_micro_batch_size'])
    actor = rank_plan(1729, 1)
    assert len(actor) == 8 and {len(rank) for rank in actor} == {20}
    assert sorted(i for rank in actor for batch in rank for i in batch) == list(range(2560))
    assert {len(batch) for rank in actor for batch in rank} == {16}
    assert VIRTUAL_PER_RANK == 1 and budget.local_micro == 2 and budget.accumulation == 8
    probe = []
    for n, updates in ((512, 10), (2560, 50)):
        plan = _natural_plan(1729, 1, n, updates)
        seen = []
        for batch in plan:
            weighted_rank_means = []
            for rank in range(WORLD):
                ids = rank_question_slice(batch, rank)
                seen.extend(ids)
                weighted_rank_means.append(math.fsum(i + .125 for i in ids) / len(ids)
                    * ddp_mean_scale(len(ids), batch['global_questions']))
            actual = math.fsum(weighted_rank_means) / WORLD
            expected = math.fsum(i + .125 for i in batch['question_indices']) / batch['global_questions']
            assert math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-10)
        assert sorted(seen) == list(range(n))
        _, sample = balanced_plan([0] * (n//4) + [1] * (3*n//4), 1729, 1)
        assert sample['sampled_counts'] == [n//2, n//2]
        probe.append(dict(questions=n,updates=updates,questions_per_rank=n//WORLD,
            global_group_mean_preserved=True,balanced_classes=sample['sampled_counts']))
    packages = {name: importlib.util.find_spec(name) is not None
        for name in ('torch', 'transformers', 'vllm', 'faiss', 'flash_attn', 'datasets', 'safetensors')}
    if not all(packages.values()):
        raise ValueError('Missing retained runtime package: ' + repr(packages))
    for key in ('model_dir','official_model_dir','encoder_model_dir','canonical_train',
                'source_question_pool','probe_source_checkpoint','fixed_test_features',
                'retrieval_index','retrieval_corpus','grounding_manifest'):
        if not Path(c[key]).exists():
            raise ValueError('Retained remote asset missing: ' + key)
    import torch
    assert not torch.cuda.is_initialized(), 'Environment inspection must not initialize CUDA'
    return dict(status='EIGHT_GPU_ENVIRONMENT_PREPARED_NOT_STARTED',checked_unix=time.time(),
        source_commit=c['source_commit'],base=c['base'],world_size=WORLD,
        cards=c['physical_gpu_indices'],physical_gpu_uuids=c['physical_gpu_uuids'],
        actor=dict(global_micro=budget.global_micro,local_micro=budget.local_micro,
            accumulation=budget.accumulation,global_mini=128,opportunities_per_rank=20,
            virtual_blocks_per_rank=VIRTUAL_PER_RANK),probe=probe,packages=packages,
        latest_B_available=[dict(first_outer=a,last_outer=b,prefix=prefix(a),
            candidate_questions=5120,ordinary_continued=continuation_questions(a),actor_groups=512)
            for a,b in ((51,100),(101,150),(151,200))],
        training_blocked=launch_blocked,experiment_assignment_pending=True,
        checkpoint_imported=False,cuda_initialized=torch.cuda.is_initialized(),
        training_started=False,python=sys.executable)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.config),allow_nan=False))


if __name__=='__main__':main()
