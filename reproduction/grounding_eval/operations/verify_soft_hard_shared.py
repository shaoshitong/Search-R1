"""Fresh shared-byte and frozen-v4 reclassification acceptance, CPU/read-only data.

Writes only its own new acceptance receipt. Does not imply paper-protocol parity.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    expected = json.load(sys.stdin)
    root = Path('/data/user/jxu957/experiments/grounding-eval-strict-20261019')
    mirror = root / 'a800-mirror'
    runtime = root / 'runtime-e8c5bcb7'
    destination = root / 'soft-to-hard-shared-independent-verification.json'
    if destination.exists():
        raise RuntimeError('A prior receipt exists; inspect it instead of overwriting')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    observed = []
    for e in expected['files']:
        path = mirror / e['path']
        assert path.is_file(), e['path']
        item = dict(path=e['path'], bytes=path.stat().st_size, sha256=sha(path))
        assert item == e, (e, item)
        observed.append(item)
        print('VERIFY_STAGE:' + e['path'], flush=True)
    source_files = {}
    for rel, wanted in expected['frozen_code_sha256'].items():
        measured = sha(runtime / rel)
        assert measured == wanted, rel
        source_files[rel] = measured
    result_root = mirror / 'soft-to-hard-cf-full-v4/results'
    results, inventory = [], []
    for path in sorted(result_root.glob('*.json')):
        inventory.append(dict(path=str(path.relative_to(mirror)), bytes=path.stat().st_size, sha256=sha(path)))
        results.append(json.loads(path.read_text()))
        if len(results) % 256 == 0:
            print('VERIFY_RESULTS:' + str(len(results)), flush=True)
    assert len(results) == expected['results_count'] == 1637
    tree = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert tree == expected['result_tree_sha256']
    sys.path.insert(0, str(runtime))
    from reproduction.grounding_eval.core import aggregate, normalize, read_jsonl, CF_CONFIG
    from search_r1.qa_protocol import extract_final_answer
    assert CF_CONFIG['replacement_version'] == 'typed-question-guided-v4-punctuation-aliases'
    prepared = list(read_jsonl(mirror / 'soft-to-hard-cf-full-v4/prepared.jsonl'))
    assert len(prepared) == 3610
    for ordinal, row in enumerate(prepared):
        assert row['evaluation_ordinal'] == ordinal and row['data_source'] == 'nq'
        answer = extract_final_answer(row['model_spans'])
        em = int(bool(answer and normalize(answer) in {normalize(g) for g in row['gold']}))
        assert row['original_answer'] == answer and row['original_em'] == em
    for row in results:
        assert row['status'] == 'CLASSIFIED'
        assert extract_final_answer([row['cf_response']]) == row['cf_answer']
    summary = aggregate(prepared, results)
    saved = json.loads((mirror / 'soft-to-hard-cf-full-v4/verified-summary.json').read_text())
    for key in ('status', 'counts', 'coverage', 'TC', 'D_Inv', 'TC_denominator', 'D_Inv_denominator'):
        assert summary[key] == saved[key], key
    for key in ('normal', 'counterfactual'):
        assert summary['token_usage'][key] == saved['token_usage'][key], key
    assert summary['status'] == 'complete'
    report = dict(checked_at=time.time(), state='SHARED_COPY_FRESH_SHA_AND_RECLASSIFICATION_VERIFIED',
        acceptance_scope='Frozen v4 execution/data consistency; not paper-protocol equivalence',
        paper_protocol_alignment_claimed=False, files=observed,
        results_count=len(results), expected_results=1637, missing=[], result_tree_sha256=tree,
        frozen_code_sha256=source_files, prepared_original_answers_reparsed=3610,
        actual_cf_responses_reparsed=1637, independent_reclassification=True,
        counts=summary['counts'], coverage=summary['coverage'], TC=summary['TC'], D_Inv=summary['D_Inv'],
        normal_token_usage=summary['token_usage']['normal'])
    with destination.open('x') as f:
        json.dump(report, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print('VERIFY_RESULT:' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
