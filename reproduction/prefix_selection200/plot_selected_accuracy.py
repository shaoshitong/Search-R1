"""Plot audited small selected-group and periodic probe metadata, without model data."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot(source, out):
    data = json.loads(Path(source).read_text())
    assert not data['missing']
    rows, periods = data['rows'], data['periodic']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 160})
    x = [r['outer'] for r in rows]
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    fig.suptitle('Selected 512 groups: selection-time predictions vs actual strict-EM outcomes', fontsize=13)
    axs[0].plot(x, [r['balanced_accuracy'] for r in rows], color='#086788', lw=2.3,
                label='Balanced accuracy (threshold 0.5)')
    axs[0].plot(x, [r['accuracy'] for r in rows], color='#d47b22', lw=1.8,
                label='Ordinary accuracy (= actual mixed fraction here)')
    axs[0].set(ylabel='Classification accuracy', ylim=(0, 1))
    axs[0].legend(loc='upper right')
    axs[0].text(.02, .04, 'Every selected group was predicted mixed: mixed recall 100%, non-mixed recall 0%.',
                transform=axs[0].transAxes, fontsize=9)
    axs[1].plot(x, [r['mean_predicted_mixed_probability'] for r in rows], color='#8e479b',
                label='Mean predicted mixed probability', lw=2)
    axs[1].plot(x, [r['actual_mixed_fraction'] for r in rows], color='#26996f',
                label='Actual mixed fraction', lw=2)
    axs[1].set(ylabel='Probability / fraction', xlabel='Completed actor outer', ylim=(0, 1))
    axs[1].legend(loc='upper right')
    for ax in axs:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        for p in periods:
            ax.axvline(p['outer'], ls=':', color='gray', alpha=.5)
    fig.savefig(out/'selected512-accuracy.png')
    fig.savefig(out/'selected512-accuracy.pdf')
    plt.close(fig)

    colors = {'total_loss':'#086788', 'trajectory_bce':'#d47b22', 'group_bce':'#8e479b'}
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    fig.suptitle('Probe fitting: all 100 actual synchronous Adam updates per period (LR 1e-5)', fontsize=13)
    for ax, p in zip(axs.flat, periods):
        curve = p['loss_rows']
        for key, color in colors.items():
            ys = np.array([r[key] for r in curve])
            ax.plot(range(1, len(ys)+1), ys, color=color, alpha=.15, lw=.7)
            ax.plot(range(10, len(ys)+1), np.convolve(ys, np.ones(10)/10, mode='valid'),
                    color=color, label=key.replace('_',' '), lw=1.8)
        ax.set(title='Outer %d' % p['outer'], xlabel='Adam update within this period', ylabel='BCE loss')
        ax.grid(alpha=.2)
    axs.flat[0].legend(fontsize=8)
    fig.supxlabel('Faint: raw batch losses before each update. Bold: trailing 10-step mean. Batches differ; this is not a fixed-set loss curve.', fontsize=9)
    fig.savefig(out/'probe-adam-loss-curves.png')
    fig.savefig(out/'probe-adam-loss-curves.pdf')
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle('Paired same-data evaluation before and after each probe fit', fontsize=13)
    for col, split in enumerate(('train','test')):
        for key, color in colors.items():
            for p in periods:
                xx = [p['outer']-.9, p['outer']+.9]
                yy = [p['before_update'][split][key], p['after_update'][split][key]]
                axs[0,col].plot(xx, yy, color=color, marker='o', markersize=4,
                                label=key.replace('_',' ') if p is periods[0] else None)
        for p in periods:
            axs[1,col].plot([p['outer']-.9,p['outer']+.9],
                [p['before_update'][split]['balanced_accuracy'],p['after_update'][split]['balanced_accuracy']],
                '-o', color='#26996f')
        axs[0,col].set(title='Full training pool (5120 groups)' if split=='train' else 'Held-out diagnostic (512 groups)',
                       ylabel='BCE loss')
        axs[0,col].legend(fontsize=8)
        axs[1,col].axhline(.5,color='gray',ls='--',lw=1)
        axs[1,col].set(ylabel='Mixed balanced accuracy', ylim=(.48,.73), xlabel='Probe update outer')
        axs[1,col].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axs[:,col]:
            ax.set_xticks([p['outer'] for p in periods])
            ax.grid(alpha=.2)
    fig.supxlabel('Each short segment: left = before, right = after. Actors and generated outcomes differ between periods; no cross-period connecting line.', fontsize=9)
    fig.savefig(out/'probe-paired-fit-evaluation.png')
    fig.savefig(out/'probe-paired-fit-evaluation.pdf')
    plt.close(fig)

    with (out/'selected512-all-outers.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summaries=[]
    for p in periods:
        row={'outer':p['outer']}
        for split in ('train','test'):
            for when,key in [('before','before_update'),('after','after_update')]:
                for metric in ('total_loss','trajectory_bce','group_bce','balanced_accuracy'):
                    row[split+'_'+when+'_'+metric]=p[key][split][metric]
        summaries.append(row)
    with (out/'probe-paired-fit-evaluation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    return dict(complete_outer=data['complete_outer'],latest=rows[-1],paired=summaries)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output')
    args=parser.parse_args()
    print(json.dumps(plot(args.source,args.output),allow_nan=False))
