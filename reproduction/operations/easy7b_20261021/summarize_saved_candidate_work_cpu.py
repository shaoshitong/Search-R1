"""Aggregate existing remote candidate tokens, without loading a model or CUDA."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU only; never download candidate tensors to Mac.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    rows = []
    for pool in a.pool:
        total = Counter()
        chunks = []
        for path in sorted((q for q in pool.glob('*.pt') if q.stem.isdecimal()),
                           key=lambda q: int(q.stem)):
            began = time.monotonic()
            record = torch.load(path, map_location='cpu', weights_only=False)
            turns = record['output'].non_tensor_batch['behavior_turns']
            counts = Counter(trajectories=len(turns))
            for trajectory in turns:
                for turn in trajectory:
                    counts['actions'] += 1
                    counts['raw_generated_tokens'] += turn['raw_generated_token_count']
                    counts['retained_policy_tokens'] += turn['kept_token_count']
                    counts['request_tokens'] += len(turn['actual_request_token_ids'])
                    counts['raw_500_token_actions'] += int(turn['raw_generated_token_count'] == 500)
            total.update(counts)
            chunks.append(dict(ordinal=int(path.stem), bytes=path.stat().st_size,
                               seconds=time.monotonic()-began, counts=dict(counts)))
            del record, turns, turn, trajectory
            gc.collect()
        rows.append(dict(pool=str(pool), totals=dict(total), chunks=chunks))
        print(json.dumps(dict(stage='POOL_AGGREGATED', pool=str(pool), totals=dict(total))), flush=True)
    result = dict(epoch=time.time(), host=platform.node(), model_loaded=False,
                  cuda_allocated=False, production_modified=False, pools=rows)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
