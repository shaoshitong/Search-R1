"""Isolated CPU comparison; patch strings supplied by the local controller."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import types

from transformers import AutoTokenizer
from reproduction.dynrpo7b.recipe import MODEL
from reproduction.common8.generation_behavior import prefix_text_decoders
import reproduction.common8.generation_behavior as behavior
from reproduction.common8.behavior_trace import raw_prefix_to_first_closing_tag as raw_cut
from search_g1.test_prefix20_generation import fixture

if 'VALIDATE_SOURCE' not in globals():
    root = Path(__file__).resolve().parents[3]
    text = (root / 'reproduction/common8/generation_behavior.py').read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'validated_raw_turn')
    VALIDATE_SOURCE = ast.get_source_segment(text, node)
    GENERATION_SOURCE = (root / 'reproduction/dynrpo7b/generation.py').read_text()

start = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
visible, executor = prefix_text_decoders(tokenizer)
hfv = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
hfe = lambda ids: tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
A = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
fixture_manifest = json.loads((A / 'online-mixed-layer-prefix-20261022-v1/fixture-complete.json').read_text())
reference = fixture_manifest['source_full'][-1]
assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256']
turns = []
with gzip.open(reference['path'], 'rt') as stream:
    for line in stream:
        turns.extend(json.loads(line)['behavior_turns'])
# Choose long actions before consulting any correctness labels, to exercise
# real80/160 token boundaries as well as genuine short tails.
turns = sorted(turns, key=lambda t: len(t['raw_generated_token_ids']), reverse=True)[:40]
special = sorted(set(map(int, tokenizer.all_special_ids)))
cases = equal = 0
for row in turns:
    full = row['raw_generated_token_ids']
    stops = row['actual_sampling_chunks'][-1]['sampling']['effective_all_stop_token_ids']
    for budget in (80, 160):
        for raw in (full[:budget], full[:min(7, len(full))], full):
            kwargs = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                          special_token_ids=special, effective_stop_token_ids=stops)
            old = raw_cut(raw, hfv, executor_decode=hfe, **kwargs)
            new = raw_cut(raw, visible, executor_decode=executor, **kwargs)
            cases += 1
            assert old == new
            equal += 1

# Only this isolated CPU process executes the new definitions. No remote file
# or the already running trainer's process/source is changed.
exec(VALIDATE_SOURCE, behavior.__dict__)
module = types.ModuleType('cpu_dynrpo_fastpath')
sys.modules[module.__name__] = module
exec(GENERATION_SOURCE, module.__dict__)
protocol = []
for budget in (80, 160):
    def scripted(request, rank):
        return ([40] * (budget - 1) + [50], 'length') if request['max_tokens'] == budget else ([51, 99], 'stop')
    old, states, identity, services, tools = fixture(scripted)
    collector = module.PrefixCollector(old.manager, identity, states, prefix_tokens=budget,
        service_factory=old.service_factory, tool_runner=old.tool_runner)
    scans, hashes = [], []
    scan, digest = module.raw_prefix_to_first_closing_tag, module.document_sha256
    module.raw_prefix_to_first_closing_tag = lambda *a, **k: (scans.append(1) or scan(*a, **k))
    module.document_sha256 = lambda value: (hashes.append(1) or digest(value))
    collector.phase(states, identity, prefix_only=True)
    assert len(scans) == 0 and states[0].status == 'paused'
    collector.phase(states, identity, prefix_only=False)
    assert len(scans) == 1 and states[0].status == 'complete' and len(tools) == 1 and len(hashes) == 2
    module.raw_prefix_to_first_closing_tag, module.document_sha256 = scan, digest
    tok = collector.manager.tokenizer
    expected = raw_cut([40] * (budget - 1) + [50, 51, 99],
        lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        eos_token_id=99, pad_token_id=0, special_token_ids=collector.special,
        executor_decode=lambda ids: tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        effective_stop_token_ids=[99])
    assert states[0].turns[0]['prefix_cut'] == expected
    protocol.append(dict(prefix_budget=budget, first_chunk_extra_scans=0,
        combined_chunk_extra_scans=1, phase_source_hash_calls=2,
        closing_tag_crosses_chunk_boundary=True, all_cut_fields_equal=True))

print(json.dumps(dict(status='CPU_PREFIX_FASTPATH_ALL_CUT_FIELDS_EQUAL', GPU_used=False,
    model_forward=False, remote_running_source_modified=False, real_raw_reference=reference,
    real_turns=len(turns), real_turns_at_least80=sum(len(t['raw_generated_token_ids']) >= 80 for t in turns),
    real_turns_at_least160=sum(len(t['raw_generated_token_ids']) >= 160 for t in turns),
    real_cases=cases, all_fields_equal=equal, protocol=protocol,
    seconds=time.monotonic() - start), sort_keys=True))
