"""Render the aggregate-only probe diagnostic; no training data or models."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    d = json.loads(args.input.read_text())
    rows, windows, heads = d['per_outer'], d['windows'], d['training_head_snapshots']
    assert d['cutoff'] == 108 and [r['step'] for r in rows] == list(range(1, 109))
    colors = dict(train='#215f9a', new='#cb6727', random='#258376', top='#8a52a2')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.3), layout='constrained')
    x = [r['step'] for r in rows]
    centers = [(w['first'] + w['last']) / 2 for w in windows]
    ax[0, 0].plot(x, [r['ce'] for r in rows], label='Replay CE, after fit', color=colors['train'])
    ax[0, 0].plot(x, [r['loss'] for r in rows], label='Replay total loss', color=colors['train'], ls=':', alpha=.8)
    ax[0, 0].plot(centers, [w['slices']['all']['count_nll'] for w in windows],
                  'o-', label='New selected groups: pre-fit NLL', color=colors['new'], ms=4)
    ax[0, 0].axvline(16, color='#777777', lw=.8, ls='--')
    ax[0, 0].set(title='A. Loss: changing replay and future groups', xlabel='Formal outer', ylabel='Loss / NLL')
    ax[0, 0].legend(fontsize=8)
    for name, label, color in [('random_exploration', 'Random 64', colors['random']),
                               ('predicted_diversity', 'Top 448', colors['top']),
                               ('all', 'All selected 512', colors['train'])]:
        ax[0, 1].plot(centers, [100*w['slices'][name]['actual_mixed_fraction'] for w in windows],
                      'o-', label=label, color=color, ms=4)
    ax[0, 1].set(title='B. Actual mixed prevalence (K = 1..4)', xlabel='Formal outer (window centers)', ylabel='Mixed groups (%)', ylim=(0, 60))
    ax[0, 1].legend(fontsize=8)
    h100 = next(h for h in heads if h['outer'] == 100)
    c = np.array(h100['training_unweighted']['count_confusion'])
    norm = c / c.sum(1, keepdims=True)
    im = ax[1, 0].imshow(norm, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    for i in range(6):
        for j in range(6):
            ax[1, 0].text(j, i, f'{100*norm[i,j]:.1f}', ha='center', va='center',
                          color='white' if norm[i,j] > .5 else '#243447', fontsize=8)
    ax[1, 0].set(title='C. Head 100: training count confusion (%)', xlabel='Predicted K (argmax)', ylabel='Actual K', xticks=range(6), yticks=range(6))
    fig.colorbar(im, ax=ax[1, 0], shrink=.8, label='Within true class')
    xs, ys = [], []
    for h in heads:
        source = h['training_per_selection_source'].get('random_exploration')
        if source:
            xs.append(h['outer']); ys.append(source['unweighted']['std_ranking_auc_unweighted'])
    ax[1, 1].plot(xs, ys, 'o-', label='Replay random groups: after fit', color=colors['train'], ms=5)
    ax[1, 1].plot(centers, [w['slices']['random_exploration']['std_ranking_auc_unweighted'] for w in windows],
                  'o-', label='New random groups: pre-fit', color=colors['random'], ms=4)
    ax[1, 1].axhline(.5, color='#888888', lw=.8, ls='--', label='Chance')
    ax[1, 1].set(title='D. Ranking mixed groups by expected std', xlabel='Formal outer', ylabel='ROC AUC', ylim=(.45, .9))
    ax[1, 1].legend(fontsize=8)
    fig.suptitle('Question-diversity probe: existing records through outer 108', fontsize=14)
    fig.supxlabel('Replay and pre-fit groups differ in policy, time and selection; these are not a held-out question split.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)


if __name__ == '__main__':
    main()
