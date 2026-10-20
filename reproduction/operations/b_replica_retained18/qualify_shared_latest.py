"""Read a stopped native four/eight-rank joint once on CPU before shared-node recovery."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--outer', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-repo')
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    joint = json.loads((source / 'latest-joint.json').read_text())
    cleanup = json.loads((source / 'logs/cleanup-status.json').read_text())
    if joint['outer'] != args.outer or not 51 <= args.outer < 200:
        raise ValueError('Reconcile the actual latest joint before qualification')
    if not cleanup['cleaned'] or not cleanup['owned_births_all_gone']:
        raise ValueError('Original writer cleanup must be confirmed')
    probe = joint['probe']
    world=len(joint['actor']['actual_adam_steps_per_rank'])
    if world not in (4,8):raise ValueError('Retain actual native rank count')
    os.environ.update(CUDA_VISIBLE_DEVICES='', SEARCH_G1_PHYSICAL_WORLD_SIZE=str(world))
    # The JSON producer receipt does not contain the dynamic head binding.
    # Read the retained raw/EMA checkpoint on CPU to select its real dimension.
    import torch
    saved_probe = torch.load(joint['probe_snapshot'], map_location='cpu', weights_only=False)
    prefix = saved_probe['binding']['prefix_tokens']
    for key in ('model', 'ema_model'):
        if tuple(saved_probe[key]['output.weight'].shape) != (1, (273 + prefix) * 2048):
            raise ValueError('Saved probe binding and raw/EMA readout shape differ')
    del saved_probe
    if prefix not in (20, 40, 80):
        raise ValueError('Retain the actual saved progressive probe dimension')
    if len(joint['actor']['rank_states']) != world:
        raise ValueError('Retain every actual native rank state')
    os.environ['SEARCH_G1_B_PREFIX_TOKENS'] = str(prefix)
    # Resolve scientific imports from the retained frozen source, not another
    # experiment or a globally installed package.
    sys.path.insert(0, args.runtime_repo or str(source / 'repo'))
    from reproduction.prefix_selection200.diverse_native_resume import qualify
    output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(event='CPU_QUALIFICATION_BEGIN', observed_unix=time.time(),
        source=str(source), outer=args.outer, prefix_tokens=prefix,
        actor_actual_steps=joint['actor']['actual_adam_steps_per_rank'],
        probe_actual_steps=probe['actual_adam_steps_per_rank'],
        optimizer_reset=False, new_optimizer_steps=0)), flush=True)
    result = qualify(source, output)
    print(json.dumps(dict(event='CPU_QUALIFICATION_COMPLETE', observed_unix=time.time(),
        proof=str(output), outer=result['outer'], actor=result['actor_adam'],
        probe=result['probe_adam'], files=len(result['files']))), flush=True)


if __name__ == '__main__':
    main()
