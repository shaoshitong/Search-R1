"""Plot only small aggregate audit summaries; no trajectory/tensor data locally."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(directory):
    source=json.loads((directory/'probe-mixed-score-analysis-summary-20261018.json').read_text())
    identity=json.loads((directory/'probe-mixed-prefix-identity-summary-20261018.json').read_text())
    rounds=[row['outer'] for row in source['diagnostics']]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'q':'#2166ac','best_opposite_pair':'#b2182b','probability_variance':'#777777'}
    for key,label in [('q','Original q'),('best_opposite_pair','max(p) [1 - min(p)]'),
                      ('probability_variance','Variance of p')]:
        rows=[row for row in source['per_round'] if row['statistic']==key]
        axes[0,0].plot(rounds,[r['auroc'] for r in rows],'-o',ms=4,label=label,color=colors[key])
    axes[0,0].axhline(.5,ls=':',color='black',lw=1)
    axes[0,0].set(title='A. Ranking true Mixed: complete 2,560-question pools',ylabel='AUROC',ylim=(.42,.85))
    axes[0,0].legend(frameon=False,fontsize=9)
    diag=source['diagnostics']
    for key,label,color in [('top512_mean_q','Mean predicted q in top512','#b2182b'),
                             ('top512_actual_mixed','Observed Mixed in q-ranked top512','#2166ac'),
                             ('actual_mixed','Complete-pool Mixed prevalence','#777777')]:
        axes[0,1].plot(rounds,[100*r[key] for r in diag],'-o',ms=4,label=label,color=color)
    axes[0,1].set(title='B. Confidence and actual selection yield',ylabel='Percent',ylim=(0,100))
    axes[0,1].legend(frameon=False,fontsize=8.5)
    rows=identity['rows']
    axes[1,0].plot(rounds,[100*r['same_prefix5_groups']/r['groups'] for r in rows],'-o',ms=4,
                   label='Five identical Prefix5 token sequences',color='#2166ac')
    axes[1,0].plot(rounds,[100*r['same_five_p_groups']/r['groups'] for r in rows],'--',lw=2,
                   label='Five identical predicted probabilities',color='#b2182b')
    axes[1,0].set(title='C. Loss of within-question prefix variation',ylabel='Questions (%)',ylim=(0,105))
    axes[1,0].legend(frameon=False,fontsize=8.5)
    for r in diag[-2:]:
        bins=r['q_calibration_bins']
        axes[1,1].plot([b['mean_prediction'] for b in bins],[b['observed_mixed'] for b in bins],'-o',ms=4,
                       label=f"Outer {r['outer']}")
    axes[1,1].plot([0,1],[0,1],':',color='black',label='Perfect calibration')
    axes[1,1].set(title='D. Held-out q calibration (equal-count bins)',xlabel='Mean predicted q',
                  ylabel='Observed Mixed fraction',xlim=(0,1),ylim=(0,1))
    axes[1,1].legend(frameon=False,fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Outer iteration')
    fig.suptitle('Saved Prefix5 probe predictions: 28,160 questions / 140,800 trajectories\n'
                 'Pre-update predictions; top512 here uses q ranking, not periodic actual-label selection',fontsize=12)
    for extension in ['png','pdf']:
        path=directory/('probe-mixed-score-analysis-20261018.'+extension)
        fig.savefig(path,dpi=180)
        print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
