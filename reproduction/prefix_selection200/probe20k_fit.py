"""Three matched fixed-actor50 fits:20k+EMA, groupweight2, compatibleG5 mixup."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time

from reproduction.prefix20_online.launch import once, publish, read, sha


def mix_batch(batch,labels,pairs,seed,alpha):
    import torch
    from .actor_hidden import WIDTH,DIM
    result=dict(batch);features=batch['features'].reshape(len(labels),5,WIDTH,DIM)
    mixed=features.clone();target=labels.clone();event=((labels.sum(1)>0)&(labels.sum(1)<5)).float();soft_event=event.clone()
    rng=random.Random(seed);lambdas=[]
    for a,b in pairs:
        for key in('attention_mask','position_ids','question_mask','policy_mask'):
            v=batch[key].reshape(len(labels),5,-1);assert torch.equal(v[a],v[b]),'Mixup must preserve identical mask/position layout'
        lam=rng.betavariate(alpha,alpha);lambdas.append(lam)
        mixed[a]=(lam*features[a].float()+(1-lam)*features[b].float()).to(features.dtype)
        mixed[b]=((1-lam)*features[a].float()+lam*features[b].float()).to(features.dtype)
        target[a]=lam*labels[a]+(1-lam)*labels[b];target[b]=(1-lam)*labels[a]+lam*labels[b]
        soft_event[a]=lam*event[a]+(1-lam)*event[b];soft_event[b]=(1-lam)*event[a]+lam*event[b]
    result['features']=mixed.reshape_as(batch['features'])
    return result,target,soft_event,dict(augmented_questions=2*len(pairs),lambda_sum=sum(lambdas),pairs=len(pairs))


def train(base,arm_name):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from .probe import OnlineQuestionProbe,setup_ddp,candidate_groups,join_own_em,rank_question_slice,ddp_mean_scale,evaluate_predictions
    from .native_features import NativeFeatures
    from .probe20k_features import CombinedFeatures
    from .probe20k_plan import layout_key,plan_epoch
    from reproduction.prefix20_group_joint100.loss import joint_loss
    from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps
    base=Path(base);c=read(base/'preparation/launch-config.json');s=c['probe20k_study']
    arm=next(v for v in s['arms']if v['name']==arm_name);rank,local,device=setup_ddp()
    out=base/'arms'/arm_name;out.mkdir(parents=True,exist_ok=True)
    data=CombinedFeatures(Path(s['dataset_base'])/'dataset-complete.json')
    assert data.manifest['actor_version']==c['actor_version']
    diagnostic=data.manifest['diagnostic'];test=NativeFeatures(diagnostic['scan_path'])
    assert test.manifest['actor_version']==c['actor_version']
    own=Path(diagnostic['scan_path']).parent.parent/'full/own-em.jsonl'
    test.index=join_own_em(test.index,[json.loads(line)for line in own.open()],c['actor_version'],test.manifest_sha,'diagnostic')
    groups=candidate_groups(data.index,20480);testgroups=candidate_groups(test.index,512)
    assert not {g['qid']for g in groups}&{g['qid']for g in testgroups}
    assert not {g['question_hash']for g in groups}&{g['question_hash']for g in testgroups}
    keys=[layout_key(g,data.index)for g in groups]
    saved=torch.load(s['source_probe_checkpoint'],map_location='cpu',weights_only=False)
    assert saved['actual_adam_steps_per_rank']==[200]*8
    model=OnlineQuestionProbe(Path(c['official_model_dir']),273).to(device=device,dtype=torch.float32)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=0)
    optimizer.load_state_dict(saved['optimizer'])
    for g in optimizer.param_groups:g['lr']=s['learning_rate']
    for state in optimizer.state.values():
        for key,value in list(state.items()):
            if isinstance(value,torch.Tensor)and key!='step':state[key]=value.to(device)
    assert actual_adam_steps(optimizer)==200
    rng=saved['rank_rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state(rng['torch_cuda'],device)
    ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    model=DDP(model,device_ids=[local],broadcast_buffers=False)
    def evaluate(epoch):
        model.eval();ema.eval();result={name:{}for name in('raw','ema')};began=time.monotonic()
        with torch.no_grad():
            for split,features,question_groups in(('train',data,groups),('test',test,testgroups)):
                own_groups=question_groups[rank::8];predictions=[]
                for offset in range(0,len(own_groups),6):
                    ids=[tid for g in own_groups[offset:offset+6]for tid in g['trajectory_ids']]
                    batch=features.get_batch(ids,device)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):raw=model.module(batch);shadow=ema(batch)
                    predictions.extend((tid,float(z),float(e),int(features.index[tid]['strict_em']))for tid,z,e in zip(ids,raw.cpu().tolist(),shadow.cpu().tolist()))
                gathered=[None]*8;dist.all_gather_object(gathered,predictions)
                if rank==0:
                    flat=[r for rows in gathered for r in rows]
                    for name,column in(('raw',1),('ema',2)):
                        rows=[(r[0],r[column],r[3])for r in flat]
                        metrics,grouped=evaluate_predictions(rows,features.index,len(question_groups),arm['group_weight'])
                        metrics.update(trajectory_accuracy=sum(int(z>=0)==y for _,z,y in rows)/len(rows),
                            trajectory_correct_prevalence=sum(y for _,_,y in rows)/len(rows),
                            reference_joint_loss_1plus1=metrics['trajectory_bce']+metrics['group_bce'])
                        result[name][split]=metrics
                        with(out/('epoch-%02d-%s-%s-predictions.jsonl'%(epoch,name,split))).open('x')as f:
                            for row in grouped:f.write(json.dumps(row,sort_keys=True)+'\n')
        if rank==0:
            result.update(epoch=epoch,arm=arm,actor_version=c['actor_version'],actual_adam_per_rank=[actual_adam_steps(optimizer)]*8,
                new_ema_updates=epoch*400,ema_decay=s['ema_decay'],learning_rate=s['learning_rate'],
                dataset_manifest_sha256=data.manifest_sha,test_manifest_sha256=test.manifest_sha,
                evaluation_seconds=time.monotonic()-began,readout='question_plus20')
            once(out/('epoch-%02d-metrics.json'%epoch),result);publish(out/'latest-metrics.json',result)
        dist.barrier();return result
    try:
        if rank==0:once(out/'initialization.json',dict(source_probe=s['source_probe_checkpoint'],source_sha256=s['source_probe_sha256'],
            source_actual_adam=200,source_model_and_moments_restored=True,ema_initialized_from_same_raw_weights=True,
            actor_updated=False,encoder_forward_repeated=False,arm=arm,readout='question_plus20'))
        evaluate(0)
        for epoch in range(1,s['epochs']+1):
            plan=plan_epoch(keys,s['dataset_seed'],epoch)
            if rank==0:once(out/('epoch-%02d-order.json'%epoch),plan)
            model.train();began=time.monotonic();seen=[]
            for step,planned in enumerate(plan,1):
                chosen=[groups[i]for i in rank_question_slice(planned,rank)]
                ids=[t for g in chosen for t in g['trajectory_ids']];seen.extend(ids);batch=data.get_batch(ids,device)
                labels=torch.tensor([[data.index[t]['strict_em']for t in g['trajectory_ids']]for g in chosen],device=device,dtype=torch.float32)
                soft_event=None;augmentation=dict(augmented_questions=0,lambda_sum=0.,pairs=0)
                if arm['mixup']:
                    batch,labels,soft_event,augmentation=mix_batch(batch,labels,planned['mix_pairs'][rank],
                        s['dataset_seed']+epoch*1000003+step*8+rank,arm['mixup_alpha'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(batch).reshape(len(chosen),5)
                losses=joint_loss(logits.float(),labels,group_weight=arm['group_weight'],mixed_target=soft_event)
                backward=losses['loss']*ddp_mean_scale(len(chosen),planned['global_questions'])
                finite=torch.isfinite(backward).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite.item()):raise FloatingPointError('Actual nonfinite20k study loss')
                backward.backward();finite=torch.stack([torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None]).all().int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite.item()):raise FloatingPointError('Actual nonfinite20k study gradient')
                optimizer.step();actual=actual_adam_steps(optimizer);assert actual==200+(epoch-1)*400+step
                with torch.no_grad():
                    for p,q in zip(model.module.parameters(),ema.parameters()):q.mul_(s['ema_decay']).add_(p.detach(),alpha=1-s['ema_decay'])
                values=torch.tensor([float(losses[k].detach())*len(chosen)for k in('trajectory_bce','group_bce','total_loss')]+[
                    augmentation['augmented_questions'],augmentation['pairs'],augmentation['lambda_sum']],device=device,dtype=torch.float64)
                dist.all_reduce(values);stats=values.cpu().tolist()
                with(out/('rank-%d-updates.jsonl'%rank)).open('a')as f:f.write(json.dumps(dict(epoch=epoch,step=step,rank=rank,
                    actual_adam=actual,local_trajectories=len(ids),learning_rate=s['learning_rate'],ema_updates=actual-200))+'\n')
                if rank==0:
                    row=dict(epoch=epoch,step=step,actual_adam=actual,trajectory_bce=stats[0]/planned['global_questions'],
                        group_bce=stats[1]/planned['global_questions'],total_loss=stats[2]/planned['global_questions'],
                        augmented_questions=int(stats[3]),mixup_pairs=int(stats[4]),mixup_lambda_mean=stats[5]/stats[4]if stats[4]else None,
                        online_loss_on_augmented_inputs=arm['mixup'])
                    with(out/'global-loss-updates.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                    publish(out/'progress.json',dict(epoch=epoch,step=step,actual_adam=actual,ema_updates=actual-200))
            assert len(seen)==len(set(seen))==12800
            seconds=time.monotonic()-began;evaluate(epoch)
            rng=dict(rank=rank,python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))
            all_rng=[None]*8;dist.all_gather_object(all_rng,rng)
            if rank==0:
                saved=dict(run_id=c['run_id'],study_id=s['study_id'],arm=arm,epoch=epoch,actor_version=c['actor_version'],
                    model={k:v.detach().cpu()for k,v in model.module.state_dict().items()},
                    ema_model={k:v.detach().cpu()for k,v in ema.state_dict().items()},optimizer=copy.deepcopy(optimizer.state_dict()),
                    rank_rng=all_rng,actual_adam_per_rank=[actual]*8,ema_updates=actual-200,ema_decay=s['ema_decay'],
                    dataset_manifest_sha256=data.manifest_sha,test_manifest_sha256=test.manifest_sha,learning_rate=s['learning_rate'])
                for values in saved['optimizer']['state'].values():
                    for key,value in list(values.items()):
                        if isinstance(value,torch.Tensor):values[key]=value.detach().cpu()
                path=out/('head-epoch-%02d.pt'%epoch);temporary=path.with_suffix('.pt.partial');torch.save(saved,temporary)
                with temporary.open('rb')as f:os.fsync(f.fileno())
                os.replace(temporary,path);complete=dict(epoch=epoch,arm=arm_name,actual_adam_per_rank=[actual]*8,
                    checkpoint=str(path),checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha(path),
                    metrics_sha256=sha(out/('epoch-%02d-metrics.json'%epoch)),train_loop_seconds=seconds)
                once(out/('epoch-%02d-complete.json'%epoch),complete);publish(out/'latest.json',complete)
            dist.barrier()
        if rank==0:once(out/'complete.json',dict(status='ACTUAL_20K_PROBE_ARM_TEN_EPOCHS_COMPLETE',arm=arm_name,epochs=10,
            new_adam_per_rank=4000,actual_adam_per_rank=[4200]*8,ema_updates=4000,actor_updates=0,latest=read(out/'latest.json')))
    finally:
        data.close();test.close()
        if dist.is_initialized():dist.destroy_process_group()


def run(base):
    from .local_probe_fit50 import admit_local_allocation
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class,DeferredInterrupt,birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base);c=read(base/'preparation/launch-config.json');s=c['probe20k_study']
    assert read(Path(s['dataset_base'])/'dataset-complete.json')['trajectories']==102400
    admit_local_allocation(base,c);lease=UUIDLocks(c['physical_gpu_uuids']);owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();handlers={v:signal.getsignal(v)for v in(signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',NCCL_NVLS_ENABLE='0',HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1')
    if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
    for key in('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python')
    try:
        for arm in s['arms']:
            name=arm['name'];publish(base/'logs/pipeline-status.json',dict(stage=name,checked_unix=time.time(),actor_version=c['actor_version']))
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','reproduction.prefix_selection200.probe20k_fit','train','--base',str(base),'--arm',name]
            p=owned.start(name,command,env,str(base/'logs'/(name+'.log')),str(base/'repo'))
            while p.poll()is None:owned.observe(name);interrupt.check();time.sleep(1)
            owned.observe(name);owned.jobs[name]['exit_code']=p.returncode;owned.save();owned.stop(name)
            if p.returncode:raise RuntimeError('Actual20k probe arm failed: '+name)
            assert read(base/'arms'/name/'complete.json')['actual_adam_per_rank']==[4200]*8
        once(base/'logs/pipeline-complete.json',dict(status='THREE_MATCHED_20K_EMA_PROBE_ARMS_COMPLETE',arms=[v['name']for v in s['arms']],actor_updates=0,formal200_resumed=False))
    finally:
        try:
            owned.stop_all();owned.reap();births=[v for j in owned.jobs.values()for v in j['ledger'].values()];present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(owned_births_all_gone=not present,remaining_owned_births=present,checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)


def submit(base):
    base=Path(base);prep=base/'preparation'
    if(prep/'submit-intent.json').exists():return dict(status='EXISTING_INTENT_NO_REPLAY',receipt=read(prep/'submit-receipt.json')if(prep/'submit-receipt.json').exists()else None)
    c=read(prep/'launch-config.json');s=c['probe20k_study'];assert read(Path(s['dataset_base'])/'dataset-complete.json')['trajectories']==102400
    from .local_probe_fit50 import SHARED_AUTHORIZATION
    assert c['local_probe_fit50']['shared_slurm_physical8_authorization']==SHARED_AUTHORIZATION
    python=str(Path(c['environment_root'])/'bin/python');script=prep/'native-probe20k-fit.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport PYTHONPATH='+shlex.quote(str(base/'repo'))+'\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexec '+shlex.join([python,'-m','reproduction.prefix_selection200.probe20k_fit','run','--base',str(base)])+'\n');script.chmod(0o700)
    command=['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(s.get('slurm_cpus',48)),'--mem='+str(s.get('slurm_memory_gib',256))+'G',
        '--time=0','--signal=B:TERM@60','--export=NONE','--job-name=g1-probe20k-fit-'+s['study_id'][:12],
        '--comment=g1-probe20k-fit:'+s['study_id'],'--chdir='+str(base),'--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(script)]
    once(prep/'submit-intent.json',dict(command=command,config_sha256=sha(prep/'launch-config.json'),created_unix=time.time()))
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=30);job=r.stdout.strip().split(';')[0]
        value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,status='SUBMITTED'if r.returncode==0 and job.isdecimal()else'UNKNOWN_NO_REPLAY')
        if value['status']=='SUBMITTED':value['job_id']=job
    except subprocess.TimeoutExpired as error:value=dict(status='UNKNOWN_NO_REPLAY',error=repr(error))
    once(prep/'submit-receipt.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','train','submit'));p.add_argument('--base',required=True);p.add_argument('--arm');a=p.parse_args()
    print(json.dumps(train(a.base,a.arm)if a.phase=='train'else globals()[a.phase](a.base)))
