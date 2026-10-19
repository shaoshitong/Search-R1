"""Full-bank balanced native probe fitting; complete Adam/EMA/RNG recovery.

Each epoch uses all majority questions and whole-G5 minority oversampling from
the full169028-question bank. Diagnostic512 is evaluated without resampling.
No actor parameters or frozen-feature encoder are trained or initialized here.
"""
import argparse
import copy
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .download_model import atomic, fingerprints
from .packed_features import FeatureReader
from .model import Probe, native_buffers, restore_buffers
from reproduction.prefix20_group_joint100.loss import joint_loss

STOP=False


def stop(_signal,_frame):
    global STOP
    STOP=True


def model_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def actual_steps(optimizer):
    values={int(s['step'].item()) for s in optimizer.state.values()}
    return next(iter(values)) if len(values)==1 else (0 if not values else -1)


def optimizer_hash(optimizer):
    h=hashlib.sha256()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for key,value in sorted(optimizer.state[parameter].items()):
                h.update(key.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else repr(value).encode())
    return h.hexdigest()


def balanced_plan(groups, seed, epoch):
    buckets={'all_wrong':[],'mixed':[],'all_correct':[]}
    for i,g in enumerate(groups):
        k=sum(g['labels']);buckets['all_wrong' if k==0 else 'all_correct' if k==5 else 'mixed'].append(i)
    assert all(buckets.values()),'A genuine class is missing; cannot fabricate labels'
    rng=random.Random(seed+epoch*100003)
    n=max(map(len,buckets.values()));indices=[]
    for own in buckets.values():
        values=list(own);rng.shuffle(values);values.extend(rng.choice(own) for _ in range(n-len(own)));indices.extend(values)
    rng.shuffle(indices);batches=[indices[i:i+51] for i in range(0,len(indices),51)]
    if len(batches[-1])<4:
        deficit=4-len(batches[-1]);batches[-1][:0]=batches[-2][-deficit:];del batches[-2][-deficit:]
    assert sum(map(len,batches))==3*n and min(map(len,batches))>=4
    return batches,dict(original_counts={k:len(v) for k,v in buckets.items()},balanced_per_class=n,groups=3*n)


def evaluate(model, reader, groups, rank, device, output, epoch, split):
    model.eval();stats=torch.zeros(20,dtype=torch.float64,device=device)
    own=groups[rank::4]
    destination=output/('epoch-%04d-%s-rank%d.jsonl.gz'%(epoch,split,rank))
    temporary=destination.with_suffix('.tmp')
    with torch.no_grad(),gzip.open(temporary,'wt') as stream:
        for offset in range(0,len(own),2):
            selected=own[offset:offset+2];ids=[tid for g in selected for tid in g['trajectory_ids']]
            data=reader.get(ids,device);labels=torch.tensor([g['labels'] for g in selected],device=device,dtype=torch.float32)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(data).reshape(-1,5)
            loss=joint_loss(logits,labels,group_weight=1.);pred=loss['mixed_probability']>=.5;target=loss['mixed_label'].bool()
            predicted_count=(logits>=0).sum(1);true_count=labels.sum(1)
            classes=lambda k:torch.where(k==0,0,torch.where(k==5,2,1))
            confusion=torch.bincount(3*classes(true_count)+classes(predicted_count),minlength=9)
            values=[len(selected),float(loss['trajectory_bce'])*len(selected),float(loss['group_bce'])*len(selected),
                int((pred&target).sum()),int((pred&~target).sum()),int((~pred&target).sum()),int((~pred&~target).sum()),
                int(((logits>=0)==labels.bool()).sum()),labels.numel(),float(loss['mixed_probability'].sum()),int(target.sum())]+confusion.tolist()
            stats+=torch.tensor(values,dtype=torch.float64,device=device)
            probabilities=torch.sigmoid(logits).float().cpu().tolist()
            mixed_probabilities=loss['mixed_probability'].float().cpu().tolist()
            for g,p,q in zip(selected,probabilities,mixed_probabilities):
                stream.write(json.dumps(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],p_j=p,
                    predicted_mixed=q,strict_em=g['labels'],true_mixed=0<sum(g['labels'])<5,
                    epoch=epoch,split=split,weight_kind='EMA0.99'))+'\n')
    temporary.replace(destination)
    dist.all_reduce(stats);n,tb,gb,tp,fp,fn,tn,correct,trajectories,pmean,mixed=stats[:11].tolist()
    assert int(n)==len(groups)
    return dict(groups=int(n),trajectories=int(trajectories),trajectory_bce=tb/n,group_bce=gb/n,total_loss=(tb+gb)/n,
        mixed_accuracy=(tp+tn)/n,mixed_balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
        trajectory_accuracy=correct/trajectories,mean_predicted_mixed=pmean/n,true_mixed_fraction=mixed/n,
        confusion=dict(tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn)),weight_kind='EMA0.99',resampled=False,
        three_class_order=['all_wrong','mixed','all_correct'],three_class_confusion=stats[11:].reshape(3,3).tolist(),
        prediction_file_pattern=str(output/('epoch-%04d-%s-rank*.jsonl.gz'%(epoch,split))))


def save(root, model, ema, optimizer, epoch, cursor, steps, config_sha, bank_sha, metrics, rank):
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
    local=dict(rank=rank,model_sha256=model_hash(model),ema_sha256=model_hash(ema),
        optimizer_sha256=optimizer_hash(optimizer),steps=actual_steps(optimizer),rng=rng)
    all_states=[None]*4;dist.all_gather_object(all_states,local)
    assert len({s['model_sha256'] for s in all_states})==len({s['ema_sha256'] for s in all_states})==1
    assert [s['steps'] for s in all_states]==[steps]*4
    assert len({s['optimizer_sha256'] for s in all_states})==1
    if rank==0:
        name='epoch-%04d.pt'%epoch if cursor==0 else 'inflight.pt'
        path=root/name;temp=path.with_suffix('.tmp')
        state=dict(role='QWEN7B_NATIVE_LAYER26_BLOCK27_PREFIX5_FULL_BANK_WARMUP',epoch=epoch,cursor=cursor,
            completed_epoch=cursor==0,steps=steps,config_sha256=config_sha,bank_sha256=bank_sha,
            raw=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
            raw_buffers=native_buffers(model),ema_buffers=native_buffers(ema),
            rank_states=all_states,metrics=metrics,actor_updates=0,epoch_time=time.time())
        with temp.open('wb') as handle:
            torch.save(state,handle);handle.flush();os.fsync(handle.fileno())
        temp.replace(path)
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        assert loaded['steps']==steps and loaded['cursor']==cursor and len(loaded['rank_states'])==4
        for part in ('raw','ema'):
            assert all(bool(torch.isfinite(value).all()) for value in loaded[part].values())
        assert {int(s['step']) for s in loaded['optimizer']['state'].values()}=={steps}
        assert all(bool(torch.isfinite(s[key]).all()) for s in loaded['optimizer']['state'].values() for key in ('exp_avg','exp_avg_sq'))
        del loaded
        item=dict(path=str(path),**fingerprints(path),epoch=epoch,cursor=cursor,steps=steps,
            config_sha256=config_sha,bank_sha256=bank_sha,completed_epoch=cursor==0,actor_updates=0,
            independent_cpu_reload=True,all_rank_optimizer_moments_match=True)
        atomic(root/'latest.json',item)
        if cursor==0:atomic(root/('epoch-%04d.json'%epoch),dict(item,metrics=metrics))
    dist.barrier()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    base=Path(args.base);c=json.loads((base/'config.json').read_text())
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==4
    torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank);dist.init_process_group('nccl')
    for sig in (signal.SIGTERM,signal.SIGUSR1):signal.signal(sig,stop)
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']=str(c['question_max_tokens'])
    bank=json.loads((base/'bank/complete.json').read_text())
    assert bank['train_questions']==169028 and bank['fit_prefix_tokens']==5 and bank['feature_layer']==26
    config_sha=fingerprints(base/'config.json')['sha256'];bank_sha=fingerprints(base/'bank/complete.json')['sha256']
    groups=json.loads((base/'bank/groups.json').read_text());features=json.loads((base/'bank/features.json').read_text())
    assert fingerprints(base/'bank/groups.json')['sha256']==bank['groups']['sha256']
    assert fingerprints(base/'bank/features.json')['sha256']==bank['features']['sha256']
    reader=FeatureReader(features)
    model=Probe(c['model_dir'],c['question_max_tokens'],5).to(device)
    ddp=DDP(model,device_ids=[local_rank]);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
    root=base/'probe';root.mkdir(exist_ok=True);start_epoch,cursor,steps=1,0,0
    predictions=root/'predictions';predictions.mkdir(exist_ok=True)
    if (root/'latest.json').exists():
        receipt=json.loads((root/'latest.json').read_text())
        assert fingerprints(receipt['path'])['sha256']==receipt['sha256']
        saved=torch.load(receipt['path'],map_location='cpu',weights_only=False)
        assert saved['config_sha256']==config_sha and saved['bank_sha256']==bank_sha and saved['actor_updates']==0
        model.load_state_dict(saved['raw'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
        restore_buffers(model,saved['raw_buffers']);restore_buffers(ema,saved['ema_buffers']);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
        own=saved['rank_states'][rank]['rng'];random.setstate(own['python']);np.random.set_state(own['numpy']);torch.set_rng_state(own['torch']);torch.cuda.set_rng_state(own['cuda'])
        steps=saved['steps'];start_epoch=saved['epoch']+int(saved['completed_epoch']);cursor=saved['cursor']
        assert actual_steps(optimizer)==steps
    elif rank==0:
        atomic(root/'initialization.json',dict(native_block=27,feature_layer=26,fit_prefix=5,capture_prefix=20,
            actor_updates=0,rank_count=4,actual_adam=0,epoch=time.time()))
    if steps==0:
        before={split:evaluate(ema,reader,groups[split],rank,device,predictions,0,split) for split in ('train','diagnostic')}
        if rank==0:atomic(root/'epoch-0000-metrics.json',before)
    for epoch in range(start_epoch,c['fit_epochs']+1):
        batches,balance=balanced_plan(groups['train'],c['seed'],epoch)
        model.train()
        for index,indices in enumerate(batches):
            if epoch==start_epoch and index<cursor:continue
            own=indices[rank::4];optimizer.zero_grad(set_to_none=True);summaries=[]
            for offset in range(0,len(own),2):
                part=own[offset:offset+2];selected=[groups['train'][i] for i in part]
                data=reader.get([tid for g in selected for tid in g['trajectory_ids']],device)
                labels=torch.tensor([g['labels'] for g in selected],dtype=torch.float32,device=device)
                context=ddp.no_sync() if offset+2<len(own) else nullcontext()
                with context:
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=ddp(data).reshape(len(part),5)
                    loss=joint_loss(logits,labels,group_weight=1.)
                    (loss['loss']*(4*len(part)/len(indices))).backward()
                summaries.append((len(part),float(loss['trajectory_bce'].detach()),float(loss['group_bce'].detach())))
            finite=torch.tensor(int(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())),device=device)
            dist.all_reduce(finite,op=dist.ReduceOp.MIN);assert finite.item()==1,'Nonfinite probe gradient'
            optimizer.step();steps+=1;assert actual_steps(optimizer)==steps
            with torch.no_grad():
                for target,value in zip(ema.parameters(),model.parameters()):target.mul_(.99).add_(value,alpha=.01)
            stats=torch.tensor([sum(n*t for n,t,g in summaries),sum(n*g for n,t,g in summaries),len(own)],device=device,dtype=torch.float64)
            dist.all_reduce(stats)
            if rank==0:
                row=dict(epoch=epoch,batch=index+1,batches=len(batches),steps=steps,ema_steps=steps,
                    trajectory_bce=float(stats[0]/stats[2]),group_bce=float(stats[1]/stats[2]),
                    total_loss=float((stats[0]+stats[1])/stats[2]),actor_updates=0,epoch_time=time.time())
                with (root/'curve.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n');handle.flush()
                atomic(root/'progress.json',row)
            flag=torch.tensor(int(STOP),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            if (index+1)%100==0 or flag.item():
                save(root,model,ema,optimizer,epoch,index+1,steps,config_sha,bank_sha,None,rank)
            if flag.item():
                dist.destroy_process_group();return
        metrics={split:evaluate(ema,reader,groups[split],rank,device,predictions,epoch,split) for split in ('train','diagnostic')}
        metrics['balance']=balance
        save(root,model,ema,optimizer,epoch,0,steps,config_sha,bank_sha,metrics,rank)
    if rank==0:
        epochs=[json.loads((root/('epoch-%04d.json'%e)).read_text()) for e in range(1,21)]
        assert all(e['independent_cpu_reload'] for e in epochs)
        best=max(epochs,key=lambda e:(e['metrics']['diagnostic']['mixed_balanced_accuracy'],-e['metrics']['diagnostic']['total_loss'],-e['epoch']))
        atomic(root/'recommended-ema.json',dict(best,weight_key='ema',selection='maximize_diagnostic_mixed_BA_then_minimize_joint_loss_then_earlier_epoch'))
        atomic(root/'complete.json',dict(status='FULL_DATA_7B_PROBE_WARMUP20_COMPLETE',epochs=20,steps=steps,
            latest=json.loads((root/'latest.json').read_text()),recommended_epoch=best['epoch'],actor_updates=0,
            feature_layer=26,fit_prefix=5,capture_prefix=20,epoch=time.time()))
    dist.destroy_process_group()


if __name__=='__main__':main()
