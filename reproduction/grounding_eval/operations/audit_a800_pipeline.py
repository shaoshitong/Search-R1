"""Small final file/hash/count audit of one completed A800 pipeline; no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task-root', required=True); p.add_argument('--prefix', required=True)
    args = p.parse_args(); base = Path(args.task_root)
    if args.prefix not in ('search-g1', 'soft-to-hard'):
        raise ValueError('Only the authorized two A800 methods are supported')
    normal = base / (args.prefix + '-normal')
    cf = base / (args.prefix + '-cf-full-v4')
    pipe = base / (args.prefix + '-pipeline-v4')
    summary = json.loads((cf / 'verified-summary.json').read_text())
    assert summary['status'] == 'complete' and summary['coverage']['pending_or_error'] == 0
    assert summary['counts']['normal_questions'] == 3610 and summary['independent_reclassification']
    assert json.loads((normal / 'independent-content-token-audit.json').read_text())['exact_question_gold_occurrence_match']
    binding = json.loads((pipe / 'binding.json').read_text())
    complete = json.loads((pipe / 'complete.json').read_text())
    assert complete['exit_code'] == 0
    exits = {p.name: json.loads(p.read_text())['exit_code'] for p in pipe.glob('*-exit.json')}
    assert len(exits) == 5 and all(v == 0 for v in exits.values())
    ns = list((normal / 'records').glob('*.json'))
    assert len(ns) == 3610
    paths = [normal / n for n in ('normal.jsonl', 'normal-config.json', 'binding.json',
        'normal-complete.json', 'independent-content-token-audit.json')]
    paths += [cf / n for n in ('manifest.json', 'prepared.jsonl', 'verified-summary.json')]
    paths += [pipe / 'complete.json']
    files = [dict(path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]
    rs = [dict(path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=sha(p))
          for p in sorted((cf / 'results').glob('*.json'))]
    assert len(rs) == summary['result_file_count'] == summary['coverage']['classified']
    result_tree_sha = hashlib.sha256(json.dumps(rs, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    report = dict(at=time.time(), method=binding['method'], source_commit=binding['source_commit'],
        normal_records=len(ns), normal_bytes=sum(p.stat().st_size for p in ns),
        results_count=len(rs), results_bytes=sum(r['bytes'] for r in rs),
        result_tree_sha256=result_tree_sha,
        result_tree_definition='SHA256(canonical JSON sorted array {path taskrelative,bytes,sha256}; files sorted by filename)',
        files=files, phase_exit_codes=exits, summary=summary)
    dest = pipe / 'final-file-audit.json'
    with dest.open('x') as f:
        json.dump(report, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
