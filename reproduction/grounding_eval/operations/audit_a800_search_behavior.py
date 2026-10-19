"""Read actual executed normal actions versus retained/raw marker counts, CPU only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--normal', required=True)
    p.add_argument('--source-base', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); base = Path(a.source_base); normal = Path(a.normal)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(base / 'repo'))
    from verl.utils import hf_tokenizer
    tok = hf_tokenizer(str(base / 'checkpoints/search-g1-common8/actor/global_step_200'))
    counters = {k: Counter() for k in ('physical_calls', 'behavior_turns', 'kept_open_markers',
        'kept_paired_markers', 'raw_open_markers', 'actions', 'action_sequences', 'max_turn_index')}
    mismatches = []; examples = []; sums = Counter(); records = 0
    for file in sorted((normal / 'records').glob('*.json')):
        r = json.loads(file.read_text()); records += 1; sequence = []; turns = []
        observed = 0; searches = 0; raw_markers = 0
        for t in r['behavior_turns']:
            ids = t['raw_generated_token_ids']; kept = ids[:t['kept_token_count']]
            text = tok.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            match = re.search(r'<(search|answer)>(.*?)</\1>', text, re.S)
            action = match.group(1) if match else 'invalid'
            sequence.append(action); counters['actions'][action] += 1
            observed += bool(t['observation_token_ids'])
            searches += action == 'search' and t['turn_index'] < 4
            raw_markers += raw.count('<search>')
            turns.append(dict(turn_index=t['turn_index'], action=action,
                actual_retrieved_tokens=len(t['observation_token_ids']),
                input_tokens=len(t['actual_request_token_ids']), raw_tokens=len(ids),
                kept_tokens=len(kept), kept_text=text[:300],
                raw_tail=raw[len(text):][:250] if raw.startswith(text) else None))
        kept = ''.join(r['model_spans'])
        opened = kept.count('<search>'); paired = len(re.findall(r'<search>.*?</search>', kept, re.S))
        counters['physical_calls'][r['actual_retrieval_calls']] += 1
        counters['behavior_turns'][len(turns)] += 1
        counters['kept_open_markers'][opened] += 1
        counters['kept_paired_markers'][paired] += 1
        counters['raw_open_markers'][raw_markers] += 1
        counters['action_sequences'][','.join(sequence)] += 1
        counters['max_turn_index'][max(t['turn_index'] for t in r['behavior_turns'])] += 1
        sums.update(physical_calls=r['actual_retrieval_calls'], actual_observed_searches=observed,
            reconstructed_nonfinal_search_actions=searches, kept_open_search_markers=opened,
            kept_paired_search_markers=paired, raw_open_search_markers=raw_markers)
        if r['actual_retrieval_calls'] != observed or observed != searches or paired != r['model_search_markers']:
            mismatches.append(dict(qid=r['qid'], stored=r['actual_retrieval_calls'], observed=observed, reconstructed=searches))
        if len(turns) != 2 or r['actual_retrieval_calls'] != 1:
            if len(examples) < 12:
                examples.append(dict(qid=r['qid'], evaluation_ordinal=r['evaluation_ordinal'],
                    question=r['question'][:160], original_answer=r['original_answer'],
                    calls=r['actual_retrieval_calls'], turns=turns))
    binding = json.loads((normal / 'binding.json').read_text())
    resolved = json.loads((base / 'logs/resolved-training-config.json').read_text())
    report = dict(at=time.time(), records=records, normal_config=binding['normal_config'],
        counters={k: dict(v) for k, v in counters.items()}, sums=dict(sums),
        per_question={k: v / records for k, v in sums.items()},
        mismatches=mismatches, examples=examples,
        source_run_id=binding['source_run_id'], source_commit=binding['source_commit'],
        source_max_turns=resolved['resolved_config']['max_turns'],
        source_script_sha256={name: hashlib.sha256((base / 'repo' / name).read_bytes()).hexdigest()
            for name in ('reproduction/common8/generation_behavior.py', 'search_r1/llm_agent/generation.py')})
    with Path(a.output).open('x') as f:
        json.dump(report, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k: v for k, v in report.items() if k != 'examples'}), flush=True)


if __name__ == '__main__':
    main()
