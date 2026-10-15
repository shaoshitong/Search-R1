"""Render small, already published aggregate CSVs; no experiment or model reads."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value, optional=False):
    if optional and value == '':
        return math.nan
    result = float(value)
    require(math.isfinite(result), 'Nonfinite recorded aggregate')
    return result


def git_bytes(repo, commit, path):
    return subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=repo,
                          capture_output=True, check=True).stdout


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
    def last_commit(path):
        return subprocess.run(['git', 'log', '-1', '--format=%H', '--', path],
            cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    source_commit, report_commit = last_commit(source_relative), last_commit(relative)
    require(git_bytes(repo, source_commit, source_relative) == source.read_bytes(),
            'Renderer has uncommitted changes')
    raw = {}
    for name in ('summary.json', 'curve.csv', 'native-scalar-history.csv'):
        path = report / name
        require(not path.is_symlink() and path.stat().st_size <= 1024 * 1024,
                'Expected only a small published aggregate')
        raw[name] = path.read_bytes()
        require(raw[name] == git_bytes(repo, report_commit, relative + '/' + name),
                'Report differs from its committed bytes')
    summary = json.loads(raw['summary.json'])
    curve = list(csv.DictReader(io.StringIO(raw['curve.csv'].decode())))
    history = list(csv.DictReader(io.StringIO(raw['native-scalar-history.csv'].decode())))
    outer = summary['milestone']
    require(outer in (5, 50, 100, 150), 'Not a scheduled milestone')
    labels = sorted({row['condition'] for row in curve})
    require(len(labels) == 2 and 'A-original' in labels, 'Expected one condition and A reuse')
    label = next(item for item in labels if item != 'A-original')
    expected_steps = [n for n in (5, 50, 100, 150) if n <= outer]
    groups = {item: sorted((row for row in curve if row['condition'] == item),
                          key=lambda row: int(row['outer'])) for item in labels}
    for item, rows in groups.items():
        require([int(row['outer']) for row in rows] == expected_steps,
                'Missing or duplicate formal points')
        for row in rows:
            hits, count = number(row['hits']), number(row['count'])
            nq, hotpot = number(row['nq_hits']), number(row['hotpot_hits'])
            require(count == 512 and hits.is_integer() and nq.is_integer() and hotpot.is_integer()
                    and 0 <= nq <= 256 and 0 <= hotpot <= 256 and hits == nq + hotpot
                    and abs(number(row['em']) - hits / count) < 1e-12,
                    'Aggregate EM/count differs')
        current = summary['dev512'][item]
        require(int(rows[-1]['hits']) == current['all']['hits']
                and int(rows[-1]['nq_hits']) == current['nq']['hits']
                and int(rows[-1]['hotpot_hits']) == current['hotpotqa']['hits'],
                'Current figure point differs from summary')
    require([int(row['outer']) for row in history] == list(range(1, outer + 1)),
            'Native history coverage differs')
    for row in history:
        for key, value in row.items():
            if value != '':
                number(value)
    for row in groups[label]:
        require(abs(number(history[int(row['outer']) - 1]['dev_em_if_milestone'])
                    - number(row['em'])) < 1e-12, 'History/formal EM differs')
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {'A-original': '#64748b', label: '#2563eb'}
    for item, rows in groups.items():
        xs = [int(row['outer']) for row in rows]
        axes[0, 0].plot(xs, [number(row['em']) * 100 for row in rows],
                        'o--', color=colors[item], label=item, linewidth=1.5)
        for dataset, style in (('nq_hits', 'o--'), ('hotpot_hits', 's:')):
            axes[0, 1].plot(xs, [number(row[dataset]) / 256 * 100 for row in rows],
                style, color=colors[item], label=item + (' / NQ' if dataset == 'nq_hits' else ' / Hotpot'))
    for ax, title in zip(axes[0], ('Fixed dev512 EM', 'Fixed dataset EM (256 each)')):
        ax.set(title=title, xlabel='Completed outer', ylabel='EM (%)', ylim=(0, 100))
        ax.set_xticks(expected_steps)
        ax.legend(fontsize=8, loc='upper left')
    xs = list(range(1, outer + 1))
    axes[1, 0].plot(xs, [number(row['pg_loss'], optional=True) for row in history],
                    color=colors[label], linewidth=1.3)
    axes[1, 0].set(title='Recorded PPO loss / ' + label,
                   xlabel='Completed outer', ylabel='Native pg_loss')
    for field, title, color in (
        ('native_gen_seconds', 'Generation + composition', '#2563eb'),
        ('native_update_seconds', 'Actor update', '#ea580c'),
        ('native_step_seconds', 'Native step (includes scheduled eval)', '#64748b'),
        ('rolling_save_seconds_outside_native_step', 'Rolling save outside step', '#16a34a')):
        axes[1, 1].plot(xs, [number(row[field], optional=True) / 60 for row in history],
                        label=title, color=color, linewidth=1.2)
    axes[1, 1].set(title='Recorded wall clocks / ' + label,
                   xlabel='Completed outer', ylabel='Minutes')
    axes[1, 1].legend(fontsize=7, loc='upper left')
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle(f'{label} vs A: accepted aggregates through outer {outer}', fontsize=14)
    fig.supxlabel('EM markers are measured checkpoints; connecting lines do not measure intermediate EM.\n'
                  'Single seed. Missing clocks stay missing; original pre-fault sampling is excluded from native clocks.',
                  fontsize=8)
    output.mkdir(parents=True)
    fig.savefig(output / 'curves.png', dpi=150, metadata={'Software': 'Search-R1 aggregate renderer'})
    fig.savefig(output / 'curves.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    provenance = dict(protocol='ablation150_published_aggregate_figure_v1', milestone=outer,
        condition=label, source_commit=source_commit,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), report_commit=report_commit,
        input_sha256={name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
        output_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                       for name in ('curves.png', 'curves.pdf')},
        additional_training=False, additional_evaluation=False, remote_actions=False,
        raw_rollout_or_checkpoint_reads=False, complete_goal_claim=False)
    (output / 'figure-provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'milestone': outer, 'files': 3,
                      'additional_evaluation': False}), flush=True)


if __name__ == '__main__':
    main()
