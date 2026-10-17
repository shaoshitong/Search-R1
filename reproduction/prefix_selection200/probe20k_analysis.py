"""CPU-only aggregate and scientific curves; raw rows/weights stay remote."""
import argparse
import json
from pathlib import Path


def aggregate(base):
    base=Path(base);config=json.loads((base/'preparation/launch-config.json').read_text());arms=[]
    for arm in config['probe20k_study']['arms']:
        root=base/'arms'/arm['name'];epochs=[json.loads(p.read_text())for p in sorted(root.glob('epoch-*-metrics.json'))]
        updates=root/'global-loss-updates.jsonl';curve={}
        if updates.exists():
            for line in updates.open():
                row=json.loads(line);curve.setdefault(row['epoch'],[]).append(row)
        summaries=[]
        for epoch,rows in sorted(curve.items()):
            summaries.append(dict(epoch=epoch,actual_updates=len(rows),first=rows[0],last=rows[-1],
                mean={k:sum(r[k]for r in rows)/len(rows)for k in('trajectory_bce','group_bce','total_loss')},
                augmented_questions=sum(r['augmented_questions']for r in rows)))
        arms.append(dict(arm=arm,epochs=epochs,update_curve_summary=summaries,
                         complete=json.loads((root/'complete.json').read_text())if(root/'complete.json').exists()else None))
    return dict(actor_version=config['actor_version'],dataset_base=config['probe20k_study']['dataset_base'],
        questions=20480,trajectories=102400,test_questions=512,learning_rate=1e-5,ema_decay=.99,
        arms=arms,formal_actor_updated=False,raw_predictions_exported=False,
        same_data_source_and_head_initialization=True,post_update_real_unaugmented_evaluation=True)


def render(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    colors=['#1763a4','#d45b20','#218657'];titles=['20k + EMA','20k + EMA + group weight2','20k + EMA + hidden mixup']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    specs=[('balanced_accuracy','Mixed balanced accuracy',100),('trajectory_bce','Trajectory BCE',1),('group_bce','Mixed-group BCE',1)]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,split in enumerate(('train','test')):
        for ax,(key,title,scale)in zip(axes[row],specs):
            for i,arm in enumerate(summary['arms']):
                epochs=arm['epochs'];x=[r['epoch']for r in epochs]
                for mode,style,alpha in(('raw','--',.45),('ema','-',1.)):
                    ax.plot(x,[r[mode][split][key]*scale for r in epochs],style,color=colors[i],alpha=alpha,
                            marker='o'if mode=='ema'else None,markersize=3,label=titles[i]+' '+mode.upper())
            if key=='balanced_accuracy':ax.axhline(50,color='grey',linestyle=':',linewidth=1)
            ax.set(title=split.upper()+': '+title,xlabel='Epoch (0 = common initialization)',ylabel='Percent'if scale==100 else'Loss')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Frozen actor50 | identical 20480 G5 train / 512 G5 test | full question+20 readout\nSolid = EMA(0.99), dashed = raw; LR 1e-5; true post-update evaluation')
    fig.savefig(output/'matched20k-ema-curves.png',dpi=160);fig.savefig(output/'matched20k-ema-curves.pdf');plt.close(fig)
    for i,arm in enumerate(summary['arms']):
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');epochs=arm['epochs'];x=[r['epoch']for r in epochs]
        fields=[('accuracy','Mixed ordinary accuracy',100),('balanced_accuracy','Mixed balanced accuracy',100),
                ('trajectory_accuracy','Trajectory accuracy',100),('trajectory_bce','Trajectory BCE',1),
                ('group_bce','Mixed-group BCE',1),('reference_joint_loss_1plus1','Common reference loss1+1',1)]
        for ax,(key,title,scale)in zip(axes.flat,fields):
            for split,color in(('train','#1763a4'),('test','#d45b20')):
                for mode,style in(('raw','--'),('ema','-')):
                    ax.plot(x,[r[mode][split][key]*scale for r in epochs],style,color=color,label=split+' '+mode.upper())
            if key=='balanced_accuracy':ax.axhline(50,color='grey',linestyle=':')
            ax.set(title=title,xlabel='Epoch',ylabel='Percent'if scale==100 else'Loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(titles[i]+' | raw vs EMA on the same real trajectories')
        fig.savefig(output/(arm['arm']['name']+'-curves.png'),dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);summary=aggregate(a.base)
    (out/'aggregate-metrics.json').write_text(json.dumps(summary,indent=2)+'\n');render(summary,out)
    print(json.dumps(dict(arms=[dict(name=r['arm']['name'],epochs=len(r['epochs']),complete=bool(r['complete']))for r in summary['arms']],output=str(out))))
