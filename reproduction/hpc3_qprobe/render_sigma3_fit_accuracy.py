"""Static scientific figures from aggregate-only post-fit diagnostics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    data=json.loads(a.input.read_text());rows=data['per_outer_post_fit']
    assert data['status'] in ('VERIFIED_OFFLINE_RECONSTRUCTION','VERSION_LOSS_VERIFIED_RECONSTRUCTION_TAIL_AFTER_FULL_STATE_ANCHOR')
    assert [r['outer'] for r in rows]==list(range(data['cutoff']+1))
    x=np.array([r['outer'] for r in rows]);active=x>0
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    blue,orange,gray='#176ba0','#c85b20','#737b83'
    fig,axes=plt.subplots(1,3,figsize=(14.7,4.6),layout='constrained')
    for ax,key,title,unit,baseline in zip(axes,
        ('binary_probability_accuracy','count_accuracy','variance_rmse'),
        ('Binary accuracy: mixed vs unmixed','Six-class accuracy: K = 0..5','Sample variance fit: RMSE'),
        ('Accuracy (%)','Accuracy (%)','Variance RMSE (lower is better)'),
        ('binary_majority_baseline','count_majority_baseline',None)):
        factor=100 if baseline else 1
        current=np.array([r['current_cohort'][key] for r in rows])*factor
        replay=np.array([r['replay_unweighted'][key] for r in rows])*factor
        ax.plot(x[active],current[active],color=blue,lw=1.6,label='Current 512, after fit')
        ax.plot(x,replay,color=orange,lw=1.7,label='Entire replay, after fit')
        if baseline:
            ax.plot(x[active],[r['current_cohort'][baseline]*100 for r in rows if r['outer']>0],
                color=blue,alpha=.45,lw=1,ls=':',label='Current 512: majority baseline')
            ax.plot(x,[r['replay_unweighted'][baseline]*100 for r in rows],
                color=orange,alpha=.5,lw=1,ls=':',label='Replay: majority baseline')
        for y,color in ((current[-1],blue),(replay[-1],orange)):
            ax.scatter(x[-1],y,color=color,s=22,zorder=4)
        ax.axvline(16,color=gray,lw=.8,ls='--',alpha=.45)
        anchor=data.get('last_full_state_anchor',data['cutoff'])
        if anchor<data['cutoff']:ax.axvspan(anchor+.5,data['cutoff']+.5,color=gray,alpha=.08)
        ax.set(title=title,xlabel='Outer iteration',ylabel=unit)
        ax.grid(axis='y',alpha=.18)
        ax.legend(fontsize=8,loc='best',frameon=False)
    fig.suptitle('Question Diversity Probe (sigma3): post-fit training diagnostics',fontsize=15)
    footer='Warmup 0 uses 5,120 groups; replay drops warmup at outer 16. Data and actor policy change each round.'
    if data.get('last_full_state_anchor',data['cutoff'])<data['cutoff']:
        footer+=' Shaded tail: model-version/loss verified; last complete state anchor = '+str(data['last_full_state_anchor'])+'.'
    fig.supxlabel(footer,fontsize=9)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.7,4.6),layout='constrained')
    for ax,key,title in zip(axes[:2],('sigma_rmse','variance_r2'),('Standard deviation fit: RMSE','Variance fit: R squared')):
        ax.plot(x[active],[r['current_cohort'][key] for r in rows if r['outer']>0],color=blue,label='Current 512, after fit')
        ax.plot(x,[r['replay_unweighted'][key] for r in rows],color=orange,label='Entire replay, after fit')
        ax.set(title=title,xlabel='Outer iteration',ylabel='RMSE' if key=='sigma_rmse' else 'R squared')
        if key=='variance_r2':ax.axhline(0,color=gray,lw=.8,ls=':')
        ax.legend(fontsize=9,frameon=False);ax.grid(axis='y',alpha=.18)
    for key,label,color in (('replay_weighted_loss','Total loss',blue),('sigma_ce','CE3',orange),('std_mse','Normalized sigma MSE','#31836a')):
        rs=[r for r in rows if 'logged_fit' in r]
        axes[2].plot([r['outer'] for r in rs],[r['logged_fit'][key] for r in rs],label=label,color=color)
    axes[2].set(title='Actual logged post-fit replay loss',xlabel='Outer iteration',ylabel='Age-weighted loss')
    axes[2].legend(fontsize=9,frameon=False);axes[2].grid(axis='y',alpha=.18)
    fig.suptitle('Fitting diagnostics: accuracy is not variance quality',fontsize=15)
    fig.supxlabel('Variance prediction is E[sigma squared], not the square of E[sigma]. No recall metrics.',fontsize=9)
    other=a.output.with_name(a.output.stem+'-fit-detail.png')
    fig.savefig(other,dpi=170);fig.savefig(other.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':main()
