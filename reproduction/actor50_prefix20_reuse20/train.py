"""Two new probe epochs with per-real-Adam EMA; actor weights never optimized."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from reproduction.prefix20_online.features import read, sha
from reproduction.prefix20_online.launch import publish
from reproduction.prefix20_online.train_epoch import append
from reproduction.prefix20_group_joint100.loss import joint_loss
from reproduction.prefix20_group_joint100.train_epoch import evaluate_predictions
from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps
from reproduction.prefix_selection200.prefix5_model import restore_native_buffers, nonpersistent_buffers
from .model import Probe20 as OnlineQuestionProbe, validate_model_state
from .features import Features20
from reproduction.prefix_selection200.native_features import NativeFeatures
from reproduction.prefix_selection200.probe import candidate_groups, join_own_em
from reproduction.actor50_probe100.plan import balanced_epoch


def dataset(bundle):
    data=Features20(bundle['path'])
    return data,data.groups


def evaluate(model, data, groups, rank, device):
    model.eval()
    own = groups[rank::4]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(own), 6):
            ids = [tid for g in own[start:start+6] for tid in g['trajectory_ids']]
            batch = data.get_batch(ids, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(batch)
            predictions.extend((tid,float(logit),int(data.index[tid]['strict_em']))
                               for tid,logit in zip(ids,z.float().cpu().tolist()))
    all_rows = [None]*4
    dist.all_gather_object(all_rows,predictions)
    if rank:
        return None
    rows = [x for part in all_rows for x in part]
    metrics,_ = evaluate_predictions(rows,data.index,512)
    metrics['trajectory_accuracy'] = sum(int(logit>=0)==label for _,logit,label in rows)/2560
    metrics['weight_kind'] = 'EMA0.99'
    metrics['resampled_evaluation'] = False
    return metrics


def save_checkpoint(base, cycle, value):
    directory = base/'probe';directory.mkdir(exist_ok=True)
    target = directory/('cycle-%04d.pt'%cycle)
    if target.exists():
        raise FileExistsError('Completed cycle must not be overwritten')
    temp = directory/('cycle-%04d.tmp'%cycle)
    with temp.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,target)
    receipt = dict(run_id=value['run_id'],cycle=cycle,completed_epochs=cycle*2,
                   actual_adam_steps_per_rank=[value['new_adam_updates']]*4,
                   ema_updates=value['ema_updates'],ema_decay=.99,
                   checkpoint=str(target),bytes=target.stat().st_size,sha256=sha(target),
                   source_probe50_sha256=value['source_probe50_sha256'],actor_updates=0,
                   metrics=value['metrics'])
    publish(directory/('cycle-%04d.json'%cycle),receipt)
    temp_link=directory/'latest-link.tmp'
    os.link(target,temp_link);os.replace(temp_link,directory/'latest.pt')
    publish(directory/'latest.json',receipt)
    return receipt


def run(config_path, cycle):
    c=read(config_path);base=Path(c['base']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    try:
        random.seed(c['seed']+rank);np.random.seed(c['seed']+rank)
        torch.manual_seed(c['seed']+rank);torch.cuda.manual_seed_all(c['seed']+rank)
        checkpoint = base/'preparation/initial-probe50-model.pt' if cycle==1 else base/'probe/latest.pt'
        prior=read(base/'probe/latest.json') if cycle>1 else None
        if prior and (prior['cycle']!=cycle-1 or sha(checkpoint)!=prior['sha256']):
            raise ValueError('Resume only the immediately preceding complete new warmup cycle')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['source_probe50_sha256']!=c['probe50_sha256']:
            raise ValueError('Source checkpoint50 differs')
        if cycle>1 and (saved['cycle']!=cycle-1 or saved['completed_epochs']!=2*(cycle-1)):
            raise ValueError('New warmup epoch boundary differs')
        model=OnlineQuestionProbe(c['official_model_dir']).to(device=device,dtype=torch.float32)
        model.load_state_dict(saved['raw_model'],strict=True)
        restore_native_buffers(model,saved['raw_buffers'])
        ema=copy.deepcopy(model).requires_grad_(False)
        ema.load_state_dict(saved['ema_model'],strict=True)
        restore_native_buffers(ema,saved['ema_buffers'])
        optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
        steps=saved['new_adam_updates'];ema_steps=saved['ema_updates']
        if cycle>1:
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
            rng=saved['rank_rng'][rank]
            random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state(rng['torch_cuda'],device)
        if actual_adam_steps(optimizer)!=steps or steps!=ema_steps:
            raise ValueError('Actual new Adam/EMA clock differs; no cycle reset')
        train_bundle=dict(path=str(base/'data'/('cycle-%04d'%cycle)),actor_version=c['actor_version'])
        test_bundle=dict(path=str(base/'data/test'),actor_version=c['actor_version'])
        train,groups=dataset(train_bundle);test,test_groups=dataset(test_bundle)
        if train_bundle['actor_version']!=c['actor_version'] or test_bundle['actor_version']!=c['actor_version']:
            raise ValueError('Actor must stay at actual actor0050')
        if ({str(x['qid']) for x in train.index.values()} & {str(x['qid']) for x in test.index.values()} or
            {x['question_hash'] for x in train.index.values()} & {x['question_hash'] for x in test.index.values()}):
            raise ValueError('Train/test identities overlap')
        before={name:evaluate(ema,data,gs,rank,device) for name,data,gs in
                [('train',train,groups),('test',test,test_groups)]}
        model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
        plans=[];began=time.monotonic();out=base/'training'/('cycle-%04d'%cycle)
        if rank==0:out.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        for epoch_in_cycle in (1,2):
            epoch=2*(cycle-1)+epoch_in_cycle
            batches,proof=balanced_epoch(groups,train.index,c['seed'],cycle,epoch)
            proof.update(epoch=epoch,optimizer_updates=len(batches))
            plans.append(proof);model.train()
            for batch_index,indices in enumerate(batches,1):
                local_indices=indices[rank::4]
                own=[groups[i] for i in local_indices]
                ids=[tid for g in own for tid in g['trajectory_ids']]
                inputs=train.get_batch(ids,device)
                labels=torch.tensor([[train.index[t]['strict_em'] for t in g['trajectory_ids']] for g in own],device=device,dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(inputs).reshape(len(own),5)
                loss=joint_loss(logits.float(),labels,group_weight=1.)
                backward=loss['loss']*(4*len(own)/len(indices))
                finite=torch.isfinite(backward).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError('Nonfinite actual probe loss')
                backward.backward()
                finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().int()
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite.item():raise FloatingPointError('Nonfinite actual probe gradient')
                optimizer.step();steps+=1
                if actual_adam_steps(optimizer)!=steps:raise RuntimeError('Real Adam return count differs')
                with torch.no_grad():
                    for target,source in zip(ema.parameters(),model.module.parameters()):target.mul_(.99).add_(source,alpha=.01)
                ema_steps+=1
                if ema_steps!=steps:raise RuntimeError('EMA must update exactly once per returned Adam')
                stats=torch.tensor([float(loss['trajectory_bce'].detach())*len(own),
                    float(loss['group_bce'].detach())*len(own),float(loss['loss'].detach())*len(own),len(own)],device=device,dtype=torch.float64)
                dist.all_reduce(stats)
                row=dict(cycle=cycle,epoch=epoch,batch=batch_index,new_adam_updates=steps,ema_updates=ema_steps,
                    learning_rate=5e-6,global_groups=len(indices),global_trajectories=5*len(indices),
                    trajectory_bce=float(stats[0]/stats[3]),group_bce=float(stats[1]/stats[3]),total_loss=float(stats[2]/stats[3]),
                    loss_clock='this_batch_before_actual_Adam',actor_updates=0,checked_unix=time.time())
                if rank==0:append(out/'updates.jsonl',row)
                publish(out/('rank-%d-progress.json'%rank),dict(row,rank=rank,actual_returned_adam=actual_adam_steps(optimizer)))
        after={name:evaluate(ema,data,gs,rank,device) for name,data,gs in
               [('train',train,groups),('test',test,test_groups)]}
        all_rng=[None]*4
        dist.all_gather_object(all_rng,dict(rank=rank,python=random.getstate(),numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device)))
        actual=[None]*4;dist.all_gather_object(actual,(steps,ema_steps))
        if len(set(actual))!=1:raise RuntimeError('Four native Adam/EMA replicas disagree')
        if rank==0:
            metrics=dict(cycle=cycle,completed_epochs=2*cycle,actor_version=c['actor_version'],actor_updates=0,
                new_adam_updates=steps,ema_updates=ema_steps,weight_kind='EMA0.99',before=before,after=after,
                epoch_sampling=plans,train_questions=512,test_questions=512,
                test_events_fixed_across_all_cycles=True,train_eval_on_original_unsampled512=True,
                training_seconds=time.monotonic()-began,checked_unix=time.time())
            raw={k:v.detach().cpu() for k,v in model.module.state_dict().items()}
            ema_state={k:v.detach().cpu() for k,v in ema.state_dict().items()}
            validate_model_state(raw);validate_model_state(ema_state)
            opt=optimizer.state_dict()
            for state in opt['state'].values():
                for k,v in list(state.items()):
                    if isinstance(v,torch.Tensor):state[k]=v.detach().cpu()
            state=dict(run_id=c['run_id'],role='frozen_actor50_prefix20_reuse20',cycle=cycle,completed_epochs=cycle*2,
                actor_updates=0,actor_version=c['actor_version'],source_probe50_sha256=c['probe50_sha256'],prefix_policy_tokens=20,source_prefix5_training_data_reused=True,
                raw_model=raw,raw_buffers=nonpersistent_buffers(model.module),ema_model=ema_state,
                ema_buffers=nonpersistent_buffers(ema),optimizer=opt,rank_rng=all_rng,
                new_adam_updates=steps,ema_updates=ema_steps,ema_decay=.99,metrics=metrics)
            receipt=save_checkpoint(base,cycle,state)
            publish(out/'complete.json',receipt);append(base/'metrics.jsonl',metrics)
        dist.barrier()
        train.close();test.close()
    finally:dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cycle',required=True,type=int)
    a=p.parse_args();run(a.config,a.cycle)


if __name__=='__main__':main()
