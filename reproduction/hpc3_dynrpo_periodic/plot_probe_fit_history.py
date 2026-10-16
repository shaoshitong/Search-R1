"""Plot numeric probe fitting audits; no weights, tensors, or raw rollouts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source0 = args.input_dir / 'probe0-full-fit-history.json'
    source25 = args.input_dir / 'refit25-fit-quality-latest.json'
    initial, continued = [json.loads(p.read_text()) for p in (source0, source25)]
    curves = {0: initial['epoch_history'], 25: continued['summaries']}
    assert [r['epoch'] for r in curves[0]] == list(range(21))
    assert [r['epoch'] for r in curves[25]] == list(range(1, 21))
    assert [r['actual_head_adam_updates'] for r in curves[0]] == list(range(0, 401, 20))
    assert [r['actual_head_adam_updates'] for r in curves[25]] == list(range(420, 801, 20))
    epochs, batches = [], []
    for outer, rows in curves.items():
        for row in rows:
            assert row['training_groups'] == 5120
            assert abs(row['total_loss'] - row['binary_bce'] - row['regression_contribution']) < 1e-5
            epochs.append(dict(outer=outer, fit_epoch=row['epoch'],
                cumulative_adam_steps=row['actual_head_adam_updates'],
                total_loss=row['total_loss'], binary_bce=row['binary_bce'],
                binary_accuracy_percent=100 * row['binary_accuracy'],
                std_mse_raw=row['regression_mse_raw'], std_mae=row['sample_std_mae'],
                top512_mixed_percent=100 * row['top_k']['mixed_precision']))
        mini = initial['minibatches'] if outer == 0 else continued['minibatches']
        assert len(mini) == 400
        for index, row in enumerate(mini, 1):
            step = row['head_adam_steps_after'] if outer == 0 else row['adam_steps']
            loss = row['pre_step_loss'] if outer == 0 else row['loss']
            assert step == (0 if outer == 0 else 400) + index
            assert row['epoch'] == (index - 1) // 20 + 1
            assert math.isfinite(loss) and loss > 0
            batches.append(dict(outer=outer, fit_epoch=row['epoch'],
                minibatch_index=row['minibatch_index'] if outer == 0 else row['batch'],
                relative_adam_update=index, cumulative_adam_step=step,
                actual_pre_step_minibatch_total_loss=loss))
    for row in epochs:
        values = [b['actual_pre_step_minibatch_total_loss'] for b in batches
                  if b['outer'] == row['outer'] and b['fit_epoch'] == row['fit_epoch']]
        row['mean_actual_training_minibatch_loss'] = math.fsum(values) / len(values) if values else None
    export_csv(args.output_dir / 'probe-fit-epoch-metrics.csv', epochs)
    export_csv(args.output_dir / 'probe-fit-minibatches.csv', batches)
    wide = []
    for epoch in range(21):
        row = dict(fit_epoch=epoch)
        for outer in curves:
            metric = next((r for r in epochs if r['outer'] == outer and r['fit_epoch'] == epoch), {})
            for field in ('total_loss', 'mean_actual_training_minibatch_loss',
                          'binary_accuracy_percent', 'binary_bce', 'std_mse_raw'):
                row['outer' + str(outer) + '_' + field] = metric.get(field)
        wide.append(row)
    export_csv(args.output_dir / 'probe-fit-epoch-comparison.csv', wide)
    colors = {0: '#2368b3', 25: '#d55e00'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2,
        'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    specifications = [
        ('total_loss', 'Total loss after each epoch', 'Loss (log scale)', True),
        ('binary_bce', 'Classification BCE after each epoch', 'BCE (log scale)', True),
        ('binary_accuracy', 'Mixed / non-mixed classification accuracy', 'Accuracy (%)', False),
        ('regression_mse_raw', 'Sample-std regression: no improvement after epoch 1', 'Raw MSE', False)]
    for ax, (key, title, ylabel, logarithmic) in zip(axes.flat, specifications):
        for outer, rows in curves.items():
            values = [r[key] * (100 if key == 'binary_accuracy' else 1) for r in rows]
            ax.plot([r['epoch'] for r in rows], values, color=colors[outer],
                marker='o', markersize=3.5, linewidth=1.8, label='Outer ' + str(outer))
        ax.set(title=title, xlabel='Fit epoch', ylabel=ylabel, xlim=(-.3, 20.5))
        ax.set_xticks([0, 1, 5, 10, 15, 20])
        if logarithmic:
            ax.set_yscale('log')
        if key == 'binary_accuracy':
            ax.set_ylim(20, 103)
        ax.legend(frameon=False, loc='best')
    axes[0, 0].annotate('Outer 0: epoch-1 spike', xy=(1, curves[0][1]['total_loss']),
        xytext=(4, 7), arrowprops=dict(arrowstyle='->', color=colors[0]), color=colors[0], fontsize=9)
    fig.suptitle('DynRPO probe fitting: outer 0 versus outer 25', fontsize=15, y=.985)
    fig.text(.5, .945, "Epoch-end metrics on each phase's own 5,120-question training pool; not held-out evaluation.",
        ha='center', fontsize=10)
    fig.text(.5, .015, 'Outer 0 includes the recorded pre-fit baseline (epoch 0). Outer 25 has no recorded pre-fit whole-pool baseline.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .92))
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir / ('probe-fit-epoch-curves.' + suffix), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for outer in curves:
        rows = [r for r in batches if r['outer'] == outer]
        values = [r['actual_pre_step_minibatch_total_loss'] for r in rows]
        ax.plot(range(1, 401), values, color=colors[outer], alpha=.24, linewidth=.8,
            label='Outer ' + str(outer) + ': all 400 observed mini-batches')
        mean = [math.fsum(values[i - 19:i + 1]) / 20 for i in range(19, 400)]
        ax.plot(range(20, 401), mean, color=colors[outer], linewidth=1.8,
            label='Outer ' + str(outer) + ': trailing 20-update mean')
    ax.set(xlabel='Relative Adam update within this fitting phase', ylabel='Pre-step mini-batch total loss (log)',
        title='Actual mini-batch loss during fitting; 20 Adam updates per epoch', xlim=(1, 400))
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=9)
    fig.text(.5, .02, 'These are observed training-batch losses, distinct from whole-pool epoch-end losses in the other figure.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output_dir / ('probe-fit-minibatch-loss.' + suffix), dpi=180)
    plt.close(fig)
    manifest = dict(scope='Two different fitting pools / frozen student snapshots; not independent holdout.',
        epoch0_source_sha256=initial['source_sha256'], epoch25_state_sha256=continued['snapshot_sha256'],
        source_audit_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source0, source25)},
        epoch_metric_rows=len(epochs), actual_minibatch_rows=len(batches), outer25_pre_fit_whole_pool_metrics_recorded=False,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir()
               if p.is_file() and p.name != 'manifest.json'})
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
