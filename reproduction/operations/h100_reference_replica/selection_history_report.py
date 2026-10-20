"""Merge published and sealed actual selection counts, without remote actions.

Uses only completed original observations. Never fits missing observations,
loads tensors, edits training, or interprets mixing as classification accuracy.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .follow_h10011_ab_normal_boundary import ROOT

REPORT = Path(__file__).resolve().parents[2] / 'reports/h100-prefix5-ab200-20261018'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    baseline = REPORT / 'progressive-through-a181-b153/PER-ITERATION.json'
    sources = [baseline] + [ROOT / 'work' / name for name in (
        'prefix5-ab-h10011-normal-follow-actual-facts.json.closed-source8-fault-20261020',
        'prefix5-ab-h10011-normal-follow-actual-facts.json.closed-b175-oom-20261020')]
    compact = read(ROOT / 'work/prefix5-ab-latest-compact.json')
    # The live reader replaces its cumulative small file. Retain exactly the
    # version used here so this publication's source digest remains verifiable.
    live = ROOT / 'work/prefix5-ab-h10011-normal-follow-actual-facts.json'
    snapshot = live.read_bytes()
    if json.loads(snapshot)['last_observation'] != compact['original']:
        raise ValueError('Reader advanced between files; reread after its commit')
    args.output.mkdir(parents=True, exist_ok=True)
    sealed = args.output / 'NEW-OBSERVED-FACTS.json'
    sealed.write_bytes(snapshot)
    sources.append(sealed)
    limits = {arm: value['complete_outer'] for arm, value in compact['arms'].items()}
    published = read(baseline)
    rows = {arm: {row['outer']: copy.deepcopy(row) for row in published['current'][arm]['rows']}
            for arm in ('A', 'B')}
    for source in sources[1:]:
        for arm, facts in read(source)['facts'].items():
            for fact in facts:
                n, metrics = fact['outer'], fact['selection_metrics']
                if n > limits[arm]:
                    raise ValueError('Observation exceeds latest complete boundary')
                if n in rows[arm]:
                    old = rows[arm][n]['selected']
                    if any(old[key] != metrics[key] for key in ('mixed', 'all_wrong', 'all_correct')):
                        raise ValueError('Conflicting original selected counts')
                    if not math.isclose(old['mean_selection_q'], metrics['mean_selection_q'], abs_tol=1e-12):
                        raise ValueError('Conflicting original selection score')
                rows[arm][n] = dict(outer=n, periodic=n < 200 and n % 10 == 0,
                    prefix_tokens=5 if arm == 'A' or n <= 50 else 20 if n <= 100 else 40 if n <= 150 else 80,
                    selected=metrics, actual_complete_source=str(source))
    for arm, limit in limits.items():
        if sorted(rows[arm]) != list(range(1, limit + 1)):
            raise ValueError('Actual history is incomplete; do not fill or interpolate')
        for row in rows[arm].values():
            metrics = row['selected']
            if (metrics['questions'] != 512 or sum(metrics[k] for k in ('mixed', 'all_wrong', 'all_correct')) != 512
                    or metrics['mixed_fraction'] != metrics['mixed'] / 512):
                raise ValueError('Authentic G5 strict-EM count denominator differs')
    windows = []
    for lo, hi in [(6, 9)] + [(n + 1, n + 9) for n in range(10, 190, 10)] + [(191, 200)]:
        arms = {}
        for arm, limit in limits.items():
            ids = [n for n in range(lo, min(hi, limit) + 1) if not rows[arm][n]['periodic']]
            if ids:
                arms[arm] = dict(actual_rounds=ids, complete_window=ids[-1] == hi,
                    mean_mixed_count=statistics.mean(rows[arm][n]['selected']['mixed'] for n in ids),
                    mean_mixed_fraction=statistics.mean(rows[arm][n]['selected']['mixed_fraction'] for n in ids))
        if arms:
            windows.append(dict(start=lo, end=hi, arms=arms))
    result = dict(observed_utc=compact['observed_utc'], complete_outer=limits,
        current={a: dict(rows=[v[n] for n in sorted(v)]) for a, v in rows.items()},
        source_files=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        ordinary_window_means=windows, unobserved_values_fitted=False, B_old_stopped_51_58_excluded=True,
        ordinary_means_exclude_periodic=True, metric_is_probe_classification_accuracy=False,
        A_B_post50_differences=published['A_B_post50_differences'],
        A_complete_including_final_QA=True, B_final_QA_complete=False, whole_goal_complete=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'PER-ITERATION.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'PER-ITERATION.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['outer', 'periodic', 'A_prefix', 'A_mixed_count', 'A_mixed_fraction',
            'A_all_wrong', 'A_all_correct', 'A_mean_q', 'B_prefix', 'B_mixed_count', 'B_mixed_fraction',
            'B_all_wrong', 'B_all_correct', 'B_mean_q'])
        for n in range(1, max(limits.values()) + 1):
            cells = [n, n < 200 and n % 10 == 0]
            for arm in ('A', 'B'):
                row = rows[arm].get(n, {})
                cells.append(row.get('prefix_tokens', ''))
                cells.extend(row.get('selected', {}).get(k, '')
                    for k in ('mixed', 'mixed_fraction', 'all_wrong', 'all_correct', 'mean_selection_q'))
            writer.writerow(cells)
    lines = ['# Actual selected mixing: A%d / B%d' % (limits['A'], limits['B']), '',
        'Each observed value is from an actually completed outer, strict-EM G5 group counts, denominator512. '
        'Empty B cells mean not completed, never fitted/interpolated. The stopped old B51–58 is excluded. '
        'This is selected group prevalence, not probe classification accuracy or QA EM/F1.', '',
        'A uses frozen warmup/prefix5. Human-authorized B changed after50: prefix20/admit768, '
        'after100 prefix40/admit1024, after150 prefix80/admit1536, with outcome-assisted final512 reranking. '
        'This is not an isolated EMA comparison. Outcome-assisted periodic10…190 rounds are excluded from ordinary means. '
        'Round200 is ordinary. A historical16 actor skipped opportunities remain disclosed separately.', '',
        '| Ordinary rounds | A mean mixed /512 | B mean mixed /512 |', '|---|---:|---:|']
    for window in windows:
        cells = []
        for arm in ('A', 'B'):
            value = window['arms'].get(arm)
            cells.append('%.2f / %.2f%%%s' % (value['mean_mixed_count'], 100 * value['mean_mixed_fraction'],
                '' if value['complete_window'] else ' (partial)') if value else 'pending')
        lines.append('| %d–%d | %s | %s |' % (window['start'], window['end'], *cells))
    lines += ['', 'A200 and final51713 are accepted. B training, final QA and true native exit/resource closure '
        'remain pending; no whole-goal completion is asserted. See PER-ITERATION.csv for every actual row.']
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(args.output), complete_outer=limits,
        missing_values_fitted=False, training_modified=False)))


if __name__ == '__main__':
    main()
