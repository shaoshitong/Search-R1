"""Plot committed combined dev aggregates locally; no model, node or inference reads."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess


MILESTONES = (5, 50, 100, 150)
REUSE = {
    'budget-10': 'A-original',
    'position-uniform': 'A-original',
    'generation-4': 'A-original',
    'generation-1': 'C1-root-only',
}
SERIES = {
    'A-original': '#64748b',
    'C1-root-only': '#2563eb',
    'C2-random-final': '#ea580c',
    'budget-5': '#0f766e',
    'budget-20': '#7c3aed',
    'position-early': '#dc2626',
    'position-late': '#db2777',
    'generation-2': '#65a30d',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True,
                          check=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve()
    repo = source.parents[3]
    report = args.report.resolve()
    relative = report.relative_to(repo).as_posix()
    source_relative = source.relative_to(repo).as_posix()
    source_commit = git(repo, 'log', '-1', '--format=%H', '--', source_relative).decode().strip()
    report_commit = git(repo, 'log', '-1', '--format=%H', '--', relative).decode().strip()
    require(git(repo, 'show', source_commit + ':' + source_relative) == source.read_bytes(),
            'Renderer must match its committed source')
    raw = {}
    for name in ('aggregate.json', 'curve.csv'):
        path = report / name
        require(not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
                'Expected a small committed aggregate')
        raw[name] = path.read_bytes()
        require(raw[name] == git(repo, 'show', report_commit + ':' + relative + '/' + name),
                'Input differs from committed report bytes')
    aggregate = json.loads(raw['aggregate.json'])
    require(aggregate['protocol'] == 'recursive_ablation150_combined_closed_aggregates_v1'
            and aggregate['scientific_source_commit'] == '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
            and aggregate['goal_complete'] is False, 'Unexpected report protocol or science source')
    conditions = aggregate['conditions']
    require('A-original' in conditions and set(conditions) <= set(SERIES) | set(REUSE),
            'Unexpected condition labels')
    points = {}
    for label, condition in conditions.items():
        for key, point in condition['milestones'].items():
            outer = int(key)
            require(str(outer) == key and outer in MILESTONES and point['outer'] == outer
                    and point['dev_status'] == 'COMPLETE_NATIVE_AGGREGATE',
                    'Expected a complete scheduled dev point')
            dev = point['dev512']
            for dataset, count in (('all', 512), ('nq', 256), ('hotpotqa', 256)):
                value = dev[dataset]
                require(type(value['hits']) is int and value['count'] == count
                        and 0 <= value['hits'] <= count
                        and math.isfinite(value['em'])
                        and abs(value['em'] - value['hits'] / count) < 1e-12,
                        'Dataset hits/count/EM differ')
            require(dev['all']['hits'] == dev['nq']['hits'] + dev['hotpotqa']['hits'],
                    'Dataset totals differ')
            points[label, outer] = dev
    rows = list(csv.DictReader(io.StringIO(raw['curve.csv'].decode())))
    csv_points = {}
    for row in rows:
        key = row['condition'], int(row['outer'])
        require(key not in csv_points and key in points, 'Duplicate or unknown CSV point')
        dev = points[key]
        require(int(row['hits']) == dev['all']['hits']
                and int(row['nq_hits']) == dev['nq']['hits']
                and int(row['hotpot_hits']) == dev['hotpotqa']['hits']
                and abs(float(row['em']) - dev['all']['em']) < 1e-12,
                'CSV point differs from accepted aggregate')
        csv_points[key] = row
    require(set(csv_points) == set(points), 'CSV/aggregate coverage differs')
    for alias, original in REUSE.items():
        for (label, outer), dev in points.items():
            if label == alias:
                require((original, outer) in points and all(
                    dev[dataset][field] == points[original, outer][dataset][field]
                    for dataset in ('all', 'nq', 'hotpotqa')
                    for field in ('count', 'hits', 'em')),
                        'Reused condition differs from its original')
    output = args.output.resolve()
    require(not output.exists(), 'Use a fresh figure directory')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(os.environ.get('TMPDIR', '/tmp'))
                                            / 'search-r1-aggregate-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    plotted = {}
    for label, color in SERIES.items():
        steps = sorted(outer for condition, outer in points if condition == label)
        if not steps:
            continue
        plotted[label] = steps
        for ax, dataset in zip(axes, ('all', 'nq', 'hotpotqa')):
            ax.plot(steps, [points[label, outer][dataset]['em'] * 100 for outer in steps],
                    'o--', color=color, label=label, linewidth=1.5, markersize=5)
    for ax, title in zip(axes, ('Fixed dev512 EM', 'NQ EM / 256 questions',
                                'HotpotQA EM / 256 questions')):
        ax.set(title=title, xlabel='Completed outer', ylabel='EM (%)', ylim=(0, 100))
        ax.set_xticks(MILESTONES)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Recursive Tree-GRPO: accepted aggregate evaluation points', fontsize=14)
    fig.supxlabel(
        'Markers are measured checkpoints; connecting lines do not measure intermediate EM. Single seed.\n'
        'Reused labels: budget-10 / position-uniform / generation-4 = A-original; generation-1 = C1-root-only.\n'
        'Missing points stay missing. Curves alone do not establish statistical significance or causality.',
        fontsize=8)
    output.mkdir(parents=True)
    fig.savefig(output / 'aggregate-em.png', dpi=150,
                metadata={'Software': 'Search-R1 accepted aggregate renderer'})
    fig.savefig(output / 'aggregate-em.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    provenance = dict(protocol='recursive_ablation150_combined_aggregate_figure_v1',
        source_commit=source_commit, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        report_commit=report_commit, plotted_unique_conditions=plotted, reused_labels=REUSE,
        accepted_label_points=aggregate['accepted_label_points'],
        planned_label_points=aggregate['planned_label_points'],
        missing_label_points=aggregate['missing_label_points'],
        input_sha256={name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        output_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ('aggregate-em.png', 'aggregate-em.pdf')},
        additional_training=False, additional_evaluation=False, remote_actions=False,
        raw_rollout_or_checkpoint_reads=False, complete_goal_claim=False)
    (output / 'figure-provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), files=3, plotted_unique_conditions=plotted,
                          additional_evaluation=False)), flush=True)


if __name__ == '__main__':
    main()
