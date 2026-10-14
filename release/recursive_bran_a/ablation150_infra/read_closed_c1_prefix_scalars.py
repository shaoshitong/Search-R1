"""Read existing formal C1 scalar prefixes; no writes, model, GPU or sampling."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
GROUP = BASE / 'C1-root-only'
RUN = '260d1debb6214d75b3d1a5954d88f524'
CONFIG = 'b2a47bdbd8d721579e983062eb3ee942cc1e4b6aa021e7c49861e3763cdb10de'
SCIENCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
ANSI = re.compile(r'\x1b\[[0-9;]*m')
STEP = re.compile(r'\bstep:(\d+) - ')


def read_prefix():
    receipt_path = GROUP / 'receipts/configuration.json'
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt['run_id'] == RUN and receipt['config_sha256'] == CONFIG
    assert receipt['source_commit'] == SCIENCE and receipt['group'] == 'C1-root-only'
    configuration = Path(receipt['config'])
    assert configuration.resolve().is_relative_to(GROUP.resolve())
    assert hashlib.sha256(configuration.read_bytes()).hexdigest() == CONFIG
    closed = {}
    recovered = False
    for output in sorted((GROUP / 'output').glob('attempt-*')):
        assert not output.is_symlink()
        recovered = recovered or (output / 'zero-write-recovered.json').is_file()
        path = output / 'completed-outers.jsonl'
        if not path.is_file():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue
                value = json.loads(line)
                assert value['run_id'] == RUN and value['config_sha256'] == CONFIG
                assert value['source_commit'] == SCIENCE and value['group'] == 'C1-root-only'
                outer = value['completed_outer']
                assert type(outer) is int and 1 <= outer <= 150
                assert value['next_outer'] == outer + 1
                assert outer not in closed or closed[outer] == value
                closed[outer] = value
    entry = json.loads((GROUP / 'receipts/current-entry.json').read_text())
    recovered = recovered or bool(entry.get('zero_write_manifest'))
    assert recovered and entry['run_id'] == RUN and entry['config_sha256'] == CONFIG
    choices = [n for n in (5, 50, 100, 150) if set(range(1, n + 1)).issubset(closed)]
    assert choices, 'No closed formal scalar prefix'
    milestone = max(choices)
    assert closed[milestone]['metrics']['val/qa_count/all'] == 512
    metrics, hashes = {}, {}
    for log in sorted((GROUP / 'logs').glob('training*.log')):
        assert not log.is_symlink()
        with log.open(errors='replace') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    continue
                clean = ANSI.sub('', line)
                match = STEP.search(clean)
                if not match or not 1 <= int(match[1]) <= milestone:
                    continue
                row = {}
                for item in clean[match.end():].split(' - '):
                    key, sep, value = item.rpartition(':')
                    if sep:
                        try:
                            row[key] = float(value)
                        except ValueError:
                            pass
                if 'timing_s/step' not in row:
                    continue
                assert all(math.isfinite(v) for v in row.values())
                outer = int(match[1])
                assert outer not in metrics or metrics[outer] == row
                counters = closed[outer]['optimizer_calls_per_rank']
                assert len(counters) == 8 and len(set(counters.values())) == 1
                assert row['training/actual_adam_calls_per_rank'] == next(iter(counters.values()))
                metrics[outer] = row
                hashes[outer] = hashlib.sha256(line.encode()).hexdigest()
    assert set(metrics) == set(range(1, milestone + 1))
    return dict(protocol='ablation150_closed_prefix_native_scalar_v1', time=time.time(),
        run_id=RUN, config_sha256=CONFIG, scientific_source_commit=SCIENCE,
        configuration_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        recovered_outer1=True, completed_prefix_outer=milestone,
        outer1_metrics=metrics[1], prefix_metrics=metrics,
        native_line_SHA_by_outer=hashes, status='COMPLETE_EXISTING_NATIVE_SCALAR_PREFIX',
        Node_writes=False, training_action='NONE', additional_sampling_or_training=False,
        scope='Existing scalar records and completed outer counters only; no tensor reload, GPU billing or new EM')


if __name__ == '__main__':
    try:
        result = read_prefix()
    except Exception as error:
        result = dict(protocol='ablation150_closed_prefix_native_scalar_v1', status='INCOMPLETE',
            reason=type(error).__name__ + ': ' + str(error), time=time.time(),
            additional_sampling_or_training=False, training_action='NONE', Node_writes=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
