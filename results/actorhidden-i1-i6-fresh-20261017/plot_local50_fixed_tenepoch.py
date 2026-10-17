"""Scientific figures from small aggregate metrics, never raw predictions."""
import argparse
import json
from pathlib import Path


def render(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(Path(summary).read_text())
    epochs = [v['epoch'] for v in data['epochs']]
    colors = dict(train='#1763a4', test='#d45b20')
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    specifications = [('trajectory_accuracy','Trajectory classification accuracy',True),
                      ('balanced_accuracy','Mixed-group balanced accuracy',True),
                      ('accuracy','Mixed-group ordinary accuracy',True),
                      ('trajectory_bce','Trajectory BCE',False),
                      ('group_bce','Mixed-group BCE',False),
                      ('total_loss','Total joint loss (weight 1 + 1)',False)]
    for ax,(key,title,percent) in zip(axes.flat,specifications):
        for split in ('train','test'):
            values = [v['metrics'][split][key] * (100 if percent else 1) for v in data['epochs']]
            ax.plot(epochs,values,'o-',color=colors[split],label=split,markersize=3)
        if key=='balanced_accuracy':ax.axhline(50,color='grey',linestyle=':',label='chance')
        if key in ('accuracy','trajectory_accuracy'):
            first=data['epochs'][0]['metrics']['test']
            p=(first['mixed_count']/first['count'])if key=='accuracy' else first['trajectory_correct_prevalence']
            ax.axhline(100*max(p,1-p),color=colors['test'],linestyle=':',label='test majority')
        ax.set(title=title,xlabel='Probe epoch (0 = before fitting)',ylabel='Percent'if percent else'Loss')
        ax.set_xticks(epochs);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Frozen actor50 | fixed 5120 G5 train / 512 G5 test | LR 1e-5\nPost-update evaluation on identical trajectories; no actor updates',fontsize=13)
    output=Path(output)
    fig.savefig(output.with_suffix('.png'),dpi=160)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for split in ('train','test'):
        for label,style in ((0,'--'),(1,'-')):
            values=[v['splits'][split]['p_given_em_'+str(label)]['mean']for v in data['epochs']]
            axes[0].plot(epochs,values,style,color=colors[split],label=split+' EM='+str(label))
        values=[v['splits'][split]['by_correct_count']['5']['mean_p']for v in data['epochs']]
        axes[1].plot(epochs,values,'o-',color=colors[split],label=split+' all-correct groups')
        values=[v['splits'][split]['group_calibration']['auc']for v in data['epochs']]
        axes[2].plot(epochs,values,'o-',color=colors[split],label=split)
    axes[0].set(title='Mean predicted correctness by actual EM',ylabel='Mean p')
    axes[1].axhline(.1669358413,color='grey',linestyle=':',label='Synthetic bad stationary point')
    axes[1].set(title='All-correct groups: mean predicted p',ylabel='Mean p (true EM = 1)')
    axes[2].axhline(.5,color='grey',linestyle=':');axes[2].set(title='Mixed-group ranking AUROC',ylabel='AUROC')
    for ax in axes:
        ax.set_xlabel('Probe epoch');ax.set_xticks(epochs);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output.parent/(output.name+'-probabilities.png'),dpi=160)
    plt.close(fig)


def render_comparison(full, policy, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full=json.loads(Path(full).read_text());policy=json.loads(Path(policy).read_text())
    assert len(full['epochs'])==len(policy['epochs'])==11
    for a,b in zip(full['epochs'],policy['epochs']):
        for name in ('actor_version','fixed_train_scan_sha256','fixed_test_scan_sha256','learning_rate'):
            assert a['metrics'][name]==b['metrics'][name]
        assert b['metrics']['policy_only_readout'] is True
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    fields=[('trajectory_accuracy','Trajectory accuracy',100),('balanced_accuracy','Mixed balanced accuracy',100),
            ('accuracy','Mixed ordinary accuracy',100),('trajectory_bce','Trajectory BCE',1),
            ('group_bce','Mixed-group BCE',1),('total_loss','Total joint loss',1)]
    for ax,(key,title,scale)in zip(axes.flat,fields):
        for data,color,label in ((full,'#1763a4','Q+20 readout'),(policy,'#d45b20','20-only readout')):
            for split,style in (('train','-'),('test','--')):
                ax.plot(range(11),[r['metrics'][split][key]*scale for r in data['epochs']],style,
                        marker='o',markersize=3,color=color,label=label+' '+split)
        if key=='balanced_accuracy':ax.axhline(50,color='grey',linestyle=':')
        ax.set(title=title,xlabel='Probe epoch',ylabel='Percent'if scale==100 else'Loss');ax.set_xticks(range(11));ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Matched fixed actor50 / fixed G5 data / same source probe / LR 1e-5\nQuestion remains in context and head attention; only direct Linear readout differs')
    output=Path(output);fig.savefig(output.with_suffix('.png'),dpi=160);fig.savefig(output.with_suffix('.pdf'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary');p.add_argument('output');a=p.parse_args();render(a.summary,a.output)
