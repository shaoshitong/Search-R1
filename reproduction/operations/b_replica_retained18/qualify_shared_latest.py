"""Read a stopped eight-rank joint once on CPU before shared-node recovery."""
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
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    joint = json.loads((source / 'latest-joint.json').read_text())
    cleanup = json.loads((source / 'logs/cleanup-status.json').read_text())
    if joint['outer'] != args.outer or not 51 <= args.outer < 200:
        raise ValueError('Reconcile the actual latest joint before qualification')
    if not cleanup['cleaned'] or not cleanup['owned_births_all_gone']:
        raise ValueError('Original writer cleanup must be confirmed')
    probe = joint['probe']
    prefix = probe['binding']['prefix_tokens']
    if prefix not in (20, 40, 80):
        raise ValueError('Retain the actual saved progressive probe dimension')
    if len(joint['actor']['actual_adam_steps_per_rank']) != 8:
        raise ValueError('This recovery retains eight native rank states')
    os.environ.update(CUDA_VISIBLE_DEVICES='', SEARCH_G1_PHYSICAL_WORLD_SIZE='8',
                      SEARCH_G1_B_PREFIX_TOKENS=str(prefix))
    # Resolve scientific imports from the retained frozen source, not another
    # experiment or a globally installed package.
    sys.path.insert(0, str(source / 'repo'))
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
