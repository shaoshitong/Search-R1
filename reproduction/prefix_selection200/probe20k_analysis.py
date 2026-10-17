"""CPU-only aggregate and scientific curves; raw rows/weights stay remote."""
import argparse
import json
from pathlib import Path


def auc(scores,labels):
    ordered=sorted(zip(scores,labels));positives=sum(labels);negatives=len(labels)-positives
    if not positives or not negatives:return None
    total=0.;start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered)and ordered[end][0]==ordered[start][0]:end+=1
        total+=sum(y for _,y in ordered[start:end])*(start+end+1)/2
        start=end
    return (total-positives*(positives+1)/2)/(positives*negatives)


def score_summary(path):
    rows=[json.loads(line)for line in path.open()];prob=[p for r in rows for p in r['probabilities']]
    labels=[y for r in rows for y in r['strict_em']];q=[r['mixed_probability']for r in rows];m=[r['mixed_label']for r in rows]
    result=dict(trajectory_auc=auc(prob,labels),mixed_auc=auc(q,m),mean_p=sum(prob)/len(prob),
        trajectory_brier=sum((p-y)**2 for p,y in zip(prob,labels))/len(prob),
        mixed_brier=sum((p-y)**2 for p,y in zip(q,m))/len(q),
        saturation_below_001=sum(p<.01 for p in prob)/len(prob),saturation_above_099=sum(p>.99 for p in prob)/len(prob))
    for y in (0,1):
        values=[p for p,label in zip(prob,labels)if label==y]
        result['mean_p_given_em_'+str(y)]=sum(values)/len(values)if values else None
    result['by_correct_count']={}
    for k in range(6):
        selected=[r for r in rows if r['correct_count']==k]
        result['by_correct_count'][str(k)]=dict(count=len(selected),
            mean_p=sum(sum(r['probabilities'])for r in selected)/(5*len(selected))if selected else None,
            mean_q=sum(r['mixed_probability']for r in selected)/len(selected)if selected else None)
    return result


def aggregate(base):
    base=Path(base);config=json.loads((base/'preparation/launch-config.json').read_text());arms=[]
    spec=config['probe20k_study'];sources=[(arm,base/'arms'/arm['name'])for arm in spec['arms']]
    if spec.get('comparison_baseline_base'):
        old=Path(spec['comparison_baseline_base']);oldc=json.loads((old/'preparation/launch-config.json').read_text())
        sources.insert(0,(next(a for a in oldc['probe20k_study']['arms']if a['name']=='data_ema'),old/'arms/data_ema'))
    for arm,root in sources:
        epochs=[json.loads(p.read_text())for p in sorted(root.glob('epoch-*-metrics.json'))]
        epochs=[e for e in epochs if e['epoch']<=spec['epochs']]
        for epoch in epochs:
            for mode in ('raw','ema'):
                for split in ('train','test'):
                    epoch[mode][split]['score_distribution']=score_summary(root/('epoch-%02d-%s-%s-predictions.jsonl'%(epoch['epoch'],mode,split)))
        updates=root/'global-loss-updates.jsonl';curve={}
        if updates.exists():
            for line in updates.open():
                row=json.loads(line)
                if row['epoch']<=spec['epochs']:curve.setdefault(row['epoch'],[]).append(row)
        summaries=[]
        if arm.get('objective')=='group_vector_dist_softmix_kl':loss_keys=('dist_loss','soft_mixed_kl','total_loss')
        elif arm.get('objective')=='group_vector_dist_kl':loss_keys=('dist_loss','mean_correctness_kl','diversity_kl','total_loss')if arm['mean_weight']else('dist_loss','diversity_kl','total_loss')
        else:loss_keys=('trajectory_bce','group_bce','total_loss')
        for epoch,rows in sorted(curve.items()):
            summaries.append(dict(epoch=epoch,actual_updates=len(rows),first=rows[0],last=rows[-1],
                mean={k:sum(r[k]for r in rows)/len(rows)for k in loss_keys},
                augmented_questions=sum(r['augmented_questions']for r in rows)))
        arms.append(dict(arm=arm,epochs=epochs,update_curve_summary=summaries,
                         complete=json.loads((root/'complete.json').read_text())if(root/'complete.json').exists()else None))
    return dict(actor_version=config['actor_version'],dataset_base=config['probe20k_study']['dataset_base'],
        questions=spec['questions'],trajectories=spec['trajectories'],test_questions=spec.get('test_questions',512),learning_rate=spec['learning_rate'],ema_decay=spec['ema_decay'],
        arms=arms,formal_actor_updated=False,raw_predictions_exported=False,
        same_data_source_and_head_initialization=True,post_update_real_unaugmented_evaluation=True,question_split=spec.get('question_split'),test_protocol=spec.get('test_protocol','original_fixed512'))


def render(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    colors=['#1763a4','#d45b20','#218657','#8b4d9d']
    titles=[]
    for arm in summary['arms']:
        setting=arm['arm'];label=str(summary['questions'])+' + EMA'
        if setting.get('objective')=='group_vector_dist_softmix_kl':label+=' + DIST + soft mixed KL'+str(setting['soft_mixed_weight'])
        elif setting['group_weight']!=1:label+=' + group weight'+str(setting['group_weight'])
        if setting['mixup']:label+=' + hidden mixup'
        titles.append(label)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    specs=[('balanced_accuracy','Mixed event balanced accuracy(q)',100),('dist_loss','Groupwise DIST',1),('diversity_kl','Diversity KL',1)]if summary['arms'][0]['arm'].get('objective')=='group_vector_dist_kl'else[('balanced_accuracy','Mixed balanced accuracy',100),('trajectory_bce','Trajectory BCE',1),('group_bce','Mixed-group BCE',1)]
    if summary['arms'][0]['arm'].get('objective')=='group_vector_dist_softmix_kl':specs=[('balanced_accuracy','Mixed event balanced accuracy(h)',100),('dist_loss','Groupwise DIST',1),('soft_mixed_kl','Soft mixed-event KL',1)]
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
    fig.suptitle('Frozen actor50 | '+str(summary['questions'])+' G5 train / '+str(summary['test_questions'])+' G5 test | full question+20 readout\nSolid = EMA('+str(summary['ema_decay'])+'), dashed = raw; LR '+str(summary['learning_rate'])+'; true post-update evaluation')
    fig.savefig(output/'matched20k-ema-curves.png',dpi=160);fig.savefig(output/'matched20k-ema-curves.pdf');plt.close(fig)
    for i,arm in enumerate(summary['arms']):
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');epochs=arm['epochs'];x=[r['epoch']for r in epochs]
        fields=[('accuracy','Mixed ordinary accuracy',100),('balanced_accuracy','Mixed balanced accuracy',100),
                ('trajectory_accuracy','Trajectory accuracy',100),('trajectory_bce','Trajectory BCE',1),
                ('group_bce','Mixed-group BCE',1),('reference_joint_loss_1plus1','Common reference loss1+1',1)]
        if arm['arm'].get('objective')=='group_vector_dist_kl':
            fields=[('balanced_accuracy','Mixed event BA(q)',100),('diversity_mixed_balanced_accuracy','Diversity threshold BA(D)',100),
                ('vector_mae','5-vector MAE',1),('dist_loss','Groupwise DIST',1),('mean_correctness_kl','Mean-correctness KL',1),('diversity_kl','Diversity KL',1)]
        if arm['arm'].get('objective')=='group_vector_dist_softmix_kl':fields=[('accuracy','h mixed accuracy',100),('balanced_accuracy','h mixed balanced accuracy',100),('dist_loss','Groupwise DIST',1),('soft_mixed_kl','Soft mixed-event KL',1),('total_loss','DIST +50 soft mixed KL',1),('group_mean_mae','Mean correctness MAE (diagnostic)',1)]
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
