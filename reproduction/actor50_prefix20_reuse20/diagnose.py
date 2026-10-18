"""Paired checkpoint20 diagnostics on existing fixed512, using one idle aux GPU."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from reproduction.prefix_selection200.a800_runtime import UUIDLocks,gpu_inventory,compute_processes
from reproduction.prefix_selection200.prefix5_model import OnlineQuestionProbe,restore_native_buffers
from reproduction.prefix_selection200.native_features import NativeFeatures
from reproduction.prefix_selection200.probe import join_own_em,candidate_groups
from reproduction.prefix20_group_joint100.train_epoch import evaluate_predictions
from .model import Probe20
from .features import Features20
from .data import read,publish,sha


def auc(labels,values):
    labels=np.asarray(labels);values=np.asarray(values);order=np.argsort(values,kind='stable')
    ranks=np.empty(len(values),dtype=float);start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2;start=end
    n=int(labels.sum());m=len(labels)-n
    return float((ranks[labels==1].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def summarize(predictions,data):
    metric,groups=evaluate_predictions(predictions,data.index,512)
    z=np.array([x[1] for x in predictions]);y=np.array([x[2] for x in predictions]);p=1/(1+np.exp(-np.clip(z,-700,700)))
    metric['trajectory_accuracy']=float(((z>=0)==y).mean());metric['trajectory_AUROC']=auc(y,p)
    metric['trajectory_mean_p']=float(p.mean());metric['true_trajectory_correct_fraction']=float(y.mean())
    metric['group_AUROC']=auc([r['mixed_label'] for r in groups],[r['mixed_probability'] for r in groups])
    lookup={t:(v,int(label)) for t,v,label in predictions};ranges=[];wins=0.;pairs=0;hard_mixed=0
    for row in groups:
        values=np.array([lookup[t][0] for t in row['trajectory_ids']]);ps=1/(1+np.exp(-values));ranges.append(float(np.ptp(ps)))
        if row['mixed_label']:
            labels=[lookup[t][1] for t in row['trajectory_ids']]
            hard_mixed+=0<sum(v>=0 for v in values)<5
            for i in range(5):
                for j in range(5):
                    if labels[i]==1 and labels[j]==0:
                        wins+=(values[i]>values[j])+.5*(values[i]==values[j]);pairs+=1
    metric.update(all5_p_equal_at1e10=sum(x<1e-10 for x in ranges),mean_within_G5_probability_range=float(np.mean(ranges)),
        median_within_G5_probability_range=float(np.median(ranges)),within_trueMixed_correct_vs_wrong_pair_AUC=wins/pairs,
        trueMixed_hard_trajectory_predictions_areMixed=hard_mixed,trueMixed_pair_count=pairs)
    return metric


def run(base):
    base=Path(base);c=read(base/'preparation/config.json');out=base/'analysis/checkpoint20-diagnostic'
    out.mkdir(parents=True,exist_ok=False);publish(out/'status.json',dict(stage='CPU_LOADING_PINNED20',checked_unix=time.time()))
    current=read(base/'training/cycle-0010/complete.json');parent=Path(c['source_prefix5_base']);old=read(parent/'training/cycle-0010/complete.json')
    assert current['completed_epochs']==old['completed_epochs']==20 and current['ema_updates']==old['ema_updates']==276
    assert sha(current['checkpoint'])==current['sha256'] and sha(old['checkpoint'])==old['sha256']
    newstate=torch.load(current['checkpoint'],map_location='cpu',weights_only=False)
    oldstate=torch.load(old['checkpoint'],map_location='cpu',weights_only=False)
    head={}
    for kind in ('raw_model','ema_model'):
        weight=newstate[kind]['output.weight'].reshape(293,2048)
        head[kind]={name:dict(L2=float(torch.linalg.vector_norm(weight[a:b].double())),max_abs=float(weight[a:b].abs().max())) for name,a,b in [('question',0,273),('first5',273,278),('new15',278,293)]}
    moment=next(item['exp_avg'] for item in newstate['optimizer']['state'].values() if tuple(item['exp_avg'].shape)==(1,293*2048))
    head['new15_Adam_exp_avg_L2']=float(torch.linalg.vector_norm(moment[:,278*2048:].double()))
    inventory=gpu_inventory();used={x['uuid'] for x in compute_processes()}
    prior_config=read(parent/'preparation/launch-config.json')
    expected=dict(zip(prior_config['auxiliary_gpu_indices'],prior_config['auxiliary_gpu_uuids']))
    admitted=[i for i in (5,6) if inventory[i]['uuid']==expected[i] and inventory[i]['memory_MiB']<=64 and inventory[i]['uuid'] not in used]
    if not admitted:raise RuntimeError('No idle approved aux GPU; do not interrupt training')
    selected=admitted[0];lease=UUIDLocks([inventory[selected]['uuid']]);torch.cuda.set_device(selected);device=torch.device('cuda',selected)
    publish(out/'admission.json',dict(actual=inventory[selected],foreign_processes_stopped=False,training_GPUs_touched=False,checked_unix=time.time()))
    newdata=Features20(base/'data/test');bundle=read(parent/'fixed-test/complete.json');olddata=NativeFeatures(bundle['scan_path'])
    labels=[json.loads(line) for line in Path(bundle['own_em']).open()]
    olddata.index=join_own_em(olddata.index,labels,c['actor_version'],bundle['scan_sha256'],bundle['selection_sha256'])
    olddata.groups=candidate_groups(olddata.index,512)
    assert set(newdata.index)==set(olddata.index)
    results={};all_predictions={};deltas={}
    try:
        model=Probe20(c['official_model_dir']).to(device=device,dtype=torch.float32).eval()
        captured=[]
        def hook(_module,_args,value):captured.append((value[0] if isinstance(value,tuple) else value).detach())
        handle=model.block.register_forward_hook(hook)
        for kind in ('ema','raw'):
            model.load_state_dict(newstate[kind+'_model']);restore_native_buffers(model,newstate[kind+'_buffers'])
            predictions=[];ablated=[];diffs=[];weight=model.output.weight.detach().clone();weight[:,278*2048:]=0
            group_batches=[newdata.groups[rank::4][start:start+6] for rank in range(4) for start in range(0,128,6)]
            completed=0
            for part in group_batches:
                ids=[t for g in part for t in g['trajectory_ids']];batch=newdata.get_batch(ids,device)
                captured.clear()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(batch);hidden=captured.pop().masked_fill(~batch['attention_mask'].unsqueeze(-1),0).flatten(1)
                    without=torch.nn.functional.linear(hidden,weight,model.output.bias).squeeze(-1).float()
                values=logits.cpu().tolist();zeros=without.cpu().tolist()
                predictions.extend((t,float(z),int(newdata.index[t]['strict_em'])) for t,z in zip(ids,values))
                ablated.extend((t,float(z),int(newdata.index[t]['strict_em'])) for t,z in zip(ids,zeros))
                diffs.extend(float(a-b) for a,b in zip(values,zeros))
                completed+=len(part)
                publish(out/'status.json',dict(stage='PINNED20_DIAGNOSTIC_INFERENCE',weights=kind,done_questions=completed,checked_unix=time.time()))
            name='prefix20_'+kind;all_predictions[name]=predictions;all_predictions[name+'_new15_head_zero']=ablated
            deltas[name]=dict(mean_absolute_logit_contribution=float(np.mean(np.abs(diffs))),max_absolute_logit_contribution=float(np.max(np.abs(diffs))),rms_logit_contribution=float(np.sqrt(np.mean(np.square(diffs)))))
            results[name]=summarize(predictions,newdata);results[name+'_new15_head_zero']=summarize(ablated,newdata)
        handle.remove();del model;torch.cuda.empty_cache()
        model=OnlineQuestionProbe(c['official_model_dir']).to(device=device,dtype=torch.float32).eval()
        model.load_state_dict(oldstate['ema_model']);restore_native_buffers(model,oldstate['ema_buffers']);predictions=[]
        group_batches=[olddata.groups[rank::4][start:start+6] for rank in range(4) for start in range(0,128,6)]
        for part in group_batches:
            ids=[t for g in part for t in g['trajectory_ids']];batch=olddata.get_batch(ids,device)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):values=model(batch).cpu().tolist()
            predictions.extend((t,float(z),int(olddata.index[t]['strict_em'])) for t,z in zip(ids,values))
        results['prefix5_ema']=summarize(predictions,olddata);all_predictions['prefix5_ema']=predictions
        prediction_path=out/'predictions.jsonl.gz'
        with gzip.open(prediction_path,'wt') as f:
            for name,rows in all_predictions.items():
                for tid,z,label in rows:f.write(json.dumps(dict(model=name,trajectory_id=tid,logit=z,strict_em=label))+'\n')
        value=dict(role='paired20_existing_fixed512_diagnostic',checked_unix=time.time(),prefix20_checkpoint_sha256=current['sha256'],prefix5_checkpoint_sha256=old['sha256'],
            checkpoint_epochs=20,Adam_EMA=276,metrics=results,new15_head_weight_norms=head,new15_logit_contribution=deltas,
            paired_fixed512_labels_unchanged=True,model_parameters_changed=False,new_rollouts=0,
            auxiliary_gpu=inventory[selected],per_trajectory_predictions_remote=str(prediction_path),prediction_file_sha256=sha(prediction_path))
        publish(out/'complete.json',value);print(json.dumps(value,allow_nan=False))
    finally:
        newdata.close();olddata.close();lease.close();publish(out/'exit.json',dict(checked_unix=time.time(),terminal=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);run(p.parse_args().base)
