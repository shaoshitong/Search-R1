"""Compare the implemented native decoder with every stored cut in an outer.

Only existing remote candidate tensors and tokenizer files are read. No model,
CUDA, retrieval request, production process or saved token is changed.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if sys.platform == 'darwin':
        raise SystemExit('Remote CPU only; do not download raw candidates to Mac.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    from transformers import AutoTokenizer
    from reproduction.common8.generation_behavior import prefix_text_decoders
    from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True)
    visible, executor = prefix_text_decoders(tokenizer)
    rows = []
    for path in sorted((q for q in a.pool.glob('*.pt') if q.stem.isdecimal()),
                       key=lambda q: int(q.stem)):
        record = torch.load(path, map_location='cpu', weights_only=False)
        actions = mismatches = 0
        began = time.monotonic()
        for trajectory in record['output'].non_tensor_batch['behavior_turns']:
            for turn in trajectory:
                cut = raw_prefix_to_first_closing_tag(turn['raw_generated_token_ids'], visible,
                    executor_decode=executor, eos_token_id=turn['eos_token_id'],
                    pad_token_id=turn['pad_token_id'], special_token_ids=turn['special_token_ids'],
                    effective_stop_token_ids=turn['effective_stop_provenance']['effective_all_stop_token_ids'])
                actions += 1
                mismatches += int(cut != turn['prefix_cut'] or
                    cut['kept_token_count'] != turn['kept_token_count'] or
                    cut['kept_token_ids'] != turn['raw_generated_token_ids'][:turn['kept_token_count']])
        row = dict(ordinal=int(path.stem), actions=actions, mismatches=mismatches,
                   seconds=time.monotonic()-began)
        rows.append(row)
        print(json.dumps(dict(stage='COMPLETE_SAVED_CHUNK', **row)), flush=True)
        del record, trajectory, turn, cut
        gc.collect()
    result = dict(epoch=time.time(), pool=str(a.pool), tokenizer_class=type(tokenizer).__name__,
        tokenizer_module=type(tokenizer).__module__, chunks=rows,
        actions=sum(r['actions'] for r in rows), mismatches=sum(r['mismatches'] for r in rows),
        implemented_production_helper_executed=True, production_modified=False,
        model_loaded=False, cuda_allocated=False)
    result['all_saved_cuts_and_kept_tokens_equal'] = bool(rows) and result['mismatches'] == 0
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not result['all_saved_cuts_and_kept_tokens_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
