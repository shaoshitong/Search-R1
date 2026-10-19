"""CPU-only verification of deployed source, dependencies and retained assets.

Does not submit, import an online checkpoint, initialize CUDA, generate rollouts,
or construct optimizers. Physical GPU UUID binding is deliberately deferred.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect(base):
    base = Path(base).resolve()
    c = json.loads((base / 'preparation/prepared-config.json').read_text())
    assert c['base'] == str(base)
    assert c['preparation_only'] and c['training_authorized'] is False
    assert not c['initialization_assignment_pending']
    assert c['planned_initialization'] == 'official_actor_and_epoch5_ema_model_only'
    assert c['physical_gpu_uuids'] == c['auxiliary_gpu_uuids'] == []
    assert c['world_size'] == 8
    forbidden = ['actor', 'probe', 'outers', 'latest-joint.json',
                 'preparation/submit-intent.json', 'preparation/submit-receipt.json']
    assert not any((base / p).exists() for p in forbidden)
    publication = json.loads((base / 'preparation/source-publication.json').read_text())
    for row in publication['files']:
        p = base / 'repo' / row['path']
        assert sha(p) == row['sha256'], str(p)

    packages = {p: importlib.metadata.version(p) for p in
                ('torch', 'transformers', 'vllm', 'datasets', 'safetensors', 'numpy')}
    reference = json.loads((base / 'repo/reproduction/operations/'
                           'reference_b_hpc3_prepared/reference-warmup-ema-fingerprint.json').read_text())
    assert packages == reference['packages'], (packages, reference['packages'])
    modules = {}
    for name in ('torch', 'transformers', 'vllm', 'datasets', 'safetensors',
                 'faiss', 'flash_attn', 'sklearn', 'joblib'):
        modules[name] = getattr(importlib.import_module(name), '__file__', None)
    import torch
    assert not torch.cuda.is_initialized()
    from reproduction.prefix_selection200.contract import WORLD
    from reproduction.prefix_selection200.actor import rank_plan, ActorBudget
    from reproduction.prefix_selection200.ab_policy import balanced_plan, updates
    from reproduction.prefix_selection200.b_progressive import prefix, continuation_questions
    from reproduction.prefix_selection200.prefix_diversity import select
    from reproduction.prefix_selection200.launch import _require_training_assignment
    assert WORLD == 8
    plan = rank_plan(1729, 1)
    assert len(plan) == 8 and all(len(r) == 20 for r in plan)
    assert sorted(i for r in plan for batch in r for i in batch) == list(range(2560))
    budget = ActorBudget(c['actor_global_micro_batch_size'])
    assert budget.local_micro == 1 and budget.accumulation == 16
    for n in (512, 2560):
        _, receipt = balanced_plan([0] * (n // 4) + [1] * (3 * n // 4), 1729, 65)
        assert receipt['sampled_counts'] == [n // 2] * 2
    assert updates(65) == 10 and updates(70) == 50
    assert [prefix(n) for n in (50, 51, 100, 101, 150, 151, 200)] == [5, 20, 20, 40, 40, 80, 80]
    assert [continuation_questions(n) for n in (51, 101, 151)] == [768, 1024, 1536]
    candidates = [dict(policy_token_ids=[i, i + 1], trajectory_id=str(i),
                       response_index=i, prefix_sha256=str(i)) for i in range(10)]
    chosen, proof = select(candidates)
    assert len(chosen) == proof['unique_retained_prefixes'] == 5
    assert not proof['probe_outputs_used'] and not proof['outcome_labels_used']
    try:
        _require_training_assignment(c)
    except ValueError:
        training_blocked = True
    else:
        raise AssertionError('Preparation must refuse execution')

    for key, path in c['asset_paths'].items():
        assert Path(path).exists(), (key, path)
    for key, expected in c['small_asset_sha256'].items():
        assert sha(c['asset_paths'][key]) == expected, key
    p = Path(c['probe_source_checkpoint'])
    assert sha(p) == c['probe_source_sha256']
    saved = torch.load(p, map_location='cpu', weights_only=False)
    from reproduction.prefix_selection200.prefix5_model import warmup_model_weights
    model, buffers, warmup_proof = warmup_model_weights(saved, epoch=5, model_kind='ema')
    def tensor(t):
        t = t.detach().cpu().contiguous()
        return dict(shape=list(t.shape), dtype=str(t.dtype),
                    sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
    assert {k: tensor(v) for k, v in model.items()} == reference['ema_model']
    assert {k: tensor(v) for k, v in buffers.items()} == reference['ema_nonpersistent_buffers']
    assert not torch.cuda.is_initialized()
    return dict(status='REFERENCE_B_HPC3_ENVIRONMENT_CPU_VERIFIED_NOT_STARTED',
                observed_unix=time.time(), base=str(base), python=sys.executable,
                deployment_commit=publication['commit'], source_files=len(publication['files']),
                reference_runtime='bc8b16728b689ccbb97457e06f2ef0a65cb39fa7',
                world_size=8, packages=packages, imported_modules=modules,
                warmup=warmup_proof, ema_tensor_count=len(model), native_buffers=len(buffers),
                source_warmup_file_sha256=reference['file_sha256'],
                local_warmup_file_sha256=c['probe_source_sha256'],
                ema_and_buffers_semantically_identical=True,
                asset_paths=c['asset_paths'], small_asset_sha256=c['small_asset_sha256'],
                actor_opportunities_per_rank=20, actor_local_micro=budget.local_micro,
                actor_accumulation=budget.accumulation, ordinary_probe_updates=10,
                periodic_probe_updates=50, training_blocked=training_blocked,
                training_started=False, optimizer_constructed=False,
                checkpoint_imported=False, cuda_initialized=False,
                gpu_runtime_validated=False, physical_uuid_binding_pending=True,
                initialization_assignment_pending=False,
                scope='CPU imports, mathematical plans, source and small-asset SHA, exact EMA14/buffers2; no GPU/Slurm/DDP/model-forward validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.base), ensure_ascii=False))


if __name__ == '__main__':
    main()
