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


def update_ema(model,ema,decay):
    import torch
    with torch.no_grad():
        for p,q in zip(model.parameters(),ema.parameters()):q.mul_(decay).add_(p.detach(),alpha=1-decay)


def mix_batch(batch,labels,pairs,seed,alpha):
    import torch
    from .actor_hidden import WIDTH,DIM
    result=dict(batch);features=batch['features'].reshape(len(labels),5,WIDTH,DIM)
    mixed=features.clone();target=labels.clone();event=((labels.sum(1)>0)&(labels.sum(1)<5)).float();soft_event=event.clone()
    rng=random.Random(seed);lambdas=[]
    for a,b in pairs:
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
    vector_objective=arm.get('objective')=='group_vector_dist_kl'
    if vector_objective:
        from .probe_group_vector_dist import vector_dist_loss,dominant_hidden_mixup,evaluation_metrics
    out=base/'arms'/arm_name;out.mkdir(parents=True,exist_ok=True)
    pooled=None;split=s.get('question_split')
    if split:
        from .probe20k_features import split_question_features
        pooled=CombinedFeatures(Path(s['dataset_base'])/'dataset-complete.json')
        data,test=split_question_features(pooled,split['train_questions'],split['seed'])
        if rank==0:once(out/'question-split.json',dict(train=data.binding,test=test.binding,
            train_manifest_sha256=data.manifest_sha,test_manifest_sha256=test.manifest_sha,
            split_by_question=True,labels_used_for_partition=False,old_diagnostic_used=False))
    else:
        data=CombinedFeatures(Path(s['dataset_base'])/'dataset-complete.json',s.get('selected_data_blocks'))
        diagnostic=data.manifest['diagnostic'];test=NativeFeatures(diagnostic['scan_path'],validate_reads=False)
        own=Path(diagnostic['scan_path']).parent.parent/'full/own-em.jsonl'
        test.index=join_own_em(test.index,[json.loads(line)for line in own.open()],c['actor_version'],test.manifest_sha,'diagnostic')
    questions=len(data.index)//5;test_questions=len(test.index)//5;updates_per_epoch=questions*100//5120
    assert data.manifest['actor_version']==test.manifest['actor_version']==c['actor_version']
    groups=candidate_groups(data.index,questions);testgroups=candidate_groups(test.index,test_questions)
    assert questions==s['questions']and test_questions==s.get('test_questions',512)
    assert not {g['qid']for g in groups}&{g['qid']for g in testgroups}
    assert not {g['question_hash']for g in groups}&{g['question_hash']for g in testgroups}
    keys=[layout_key(g,data.index)for g in groups]
    assert Path(s['source_probe_checkpoint']).stat().st_size==s['source_probe_bytes']
    previous=read(out/'latest.json')if(out/'latest.json').exists()else None
    saved=torch.load(previous['checkpoint']if previous else s['source_probe_checkpoint'],map_location='cpu',weights_only=False)
    completed=previous['epoch']if previous else 0;expected=200+updates_per_epoch*completed
    if previous:
        assert saved['epoch']==completed and saved['actual_adam_per_rank']==[expected]*8 and saved['run_id']==c['run_id']
        assert saved['arm']==arm and saved['dataset_manifest_sha256']==data.manifest_sha and saved['test_manifest_sha256']==test.manifest_sha
        assert saved['ema_updates']==updates_per_epoch*completed and saved.get('selected_data_blocks',list(range(4)))==data.selected_blocks and saved['ema_decay']==s['ema_decay']
    else:assert saved['actual_adam_steps_per_rank']==[200]*8 and saved['run_id']==s['source_probe_run_id']
    class PrevalidatedStudyProbe(OnlineQuestionProbe):
        def forward(self,batch):
            return self.forward_validated(batch)
    model=PrevalidatedStudyProbe(Path(c['official_model_dir']),273).to(device=device,dtype=torch.float32)
    model.load_state_dict(saved['model'],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=s['learning_rate'],weight_decay=0)
    optimizer.load_state_dict(saved['optimizer'])
    for g in optimizer.param_groups:g['lr']=s['learning_rate']
    for state in optimizer.state.values():
        for key,value in list(state.items()):
            if isinstance(value,torch.Tensor)and key!='step':state[key]=value.to(device)
    assert actual_adam_steps(optimizer)==expected
    rng=saved['rank_rng'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state(rng['torch_cuda'],device)
    ema=copy.deepcopy(model).eval()
    if previous:ema.load_state_dict(saved['ema_model'],strict=True)
    for parameter in ema.parameters():parameter.requires_grad_(False)
    model=DDP(model,device_ids=[local],broadcast_buffers=False)
    def prepare_batch(features,ids):
        return features.get_batch(ids,device)
    def evaluate(epoch):
        model.eval();ema.eval();result={name:{}for name in('raw','ema')};began=time.monotonic();fallbacks=[]
        with torch.no_grad():
            for split,features,question_groups in(('train',data,groups),('test',test,testgroups)):
                own_groups=question_groups[rank::8];predictions=[];offset=0
                micro=s.get('train_eval_micro_questions',25)if split=='train'else s.get('test_eval_micro_questions',6)
                while offset<len(own_groups):
                    ids=[tid for g in own_groups[offset:offset+micro]for tid in g['trajectory_ids']]
                    batch=raw=shadow=None;failed=False
                    try:
                        batch=prepare_batch(features,ids)
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):raw=model.module(batch);shadow=ema(batch)
                    except torch.cuda.OutOfMemoryError:
                        if micro==1:raise
                        failed=True
                    if failed:
                        # Only a real typed inference OOM changes its microbatch.
                        del batch,raw,shadow
                        import gc
                        gc.collect();torch.cuda.empty_cache()
                        fallbacks.append(dict(split=split,previous_questions=micro,next_questions=max(1,micro//2),actual_typed_oom=True))
                        micro=max(1,micro//2);continue
                    predictions.extend((tid,float(z),float(e),int(features.index[tid]['strict_em']))for tid,z,e in zip(ids,raw.cpu().tolist(),shadow.cpu().tolist()))
                    offset+=len(ids)//5
                gathered=[None]*8;dist.all_gather_object(gathered,predictions)
                if rank==0:
                    flat=[r for rows in gathered for r in rows]
                    for name,column in(('raw',1),('ema',2)):
                        rows=[(r[0],r[column],r[3])for r in flat]
                        metrics,grouped=evaluate_predictions(rows,features.index,len(question_groups),arm['group_weight'])
                        tp=sum(z>=0 and y==1 for _,z,y in rows);fn=sum(z<0 and y==1 for _,z,y in rows)
                        tn=sum(z<0 and y==0 for _,z,y in rows);fp=sum(z>=0 and y==0 for _,z,y in rows)
                        metrics.update(trajectory_accuracy=sum(int(z>=0)==y for _,z,y in rows)/len(rows),
                            trajectory_correct_prevalence=sum(y for _,_,y in rows)/len(rows),
                            trajectory_tp=tp,trajectory_fn=fn,trajectory_tn=tn,trajectory_fp=fp,
                            trajectory_positive_recall=tp/(tp+fn)if tp+fn else None,
                            trajectory_negative_recall=tn/(tn+fp)if tn+fp else None,
                            trajectory_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2 if(tp+fn)and(tn+fp)else None,
                            reference_joint_loss_1plus1=metrics['trajectory_bce']+metrics['group_bce'])
                        if vector_objective:
                            by_id={tid:z for tid,z,_label in rows}
                            for grouped_row in grouped:grouped_row['trajectory_logits']=[by_id[t]for t in grouped_row['trajectory_ids']]
                            metrics.update(evaluation_metrics(grouped,arm['dist_weight'],arm['mean_weight'],arm['diversity_weight']))
                        result[name][split]=metrics
                        with(out/('epoch-%02d-%s-%s-predictions.jsonl'%(epoch,name,split))).open('x')as f:
                            for row in grouped:f.write(json.dumps(row,sort_keys=True)+'\n')
        all_fallbacks=[None]*8;dist.all_gather_object(all_fallbacks,fallbacks)
        if rank==0:
            result.update(epoch=epoch,arm=arm,actor_version=c['actor_version'],actual_adam_per_rank=[actual_adam_steps(optimizer)]*8,
                new_ema_updates=epoch*updates_per_epoch,ema_decay=s['ema_decay'],learning_rate=s['learning_rate'],
                dataset_manifest_sha256=data.manifest_sha,test_manifest_sha256=test.manifest_sha,test_questions=test_questions,test_trajectories=test_questions*5,
                evaluation_seconds=time.monotonic()-began,train_questions=questions,train_trajectories=questions*5,selected_data_blocks=data.selected_blocks,readout='question_plus20',inference_actual_oom_fallbacks=all_fallbacks,
                train_eval_initial_local_trajectories=s.get('train_eval_micro_questions',25)*5,
                test_eval_initial_local_trajectories=s.get('test_eval_micro_questions',6)*5)
            once(out/('epoch-%02d-metrics.json'%epoch),result);publish(out/'latest-metrics.json',result)
        dist.barrier();return result
    try:
        if rank==0 and not(out/'initialization.json').exists():once(out/'initialization.json',dict(source_probe=s['source_probe_checkpoint'],source_sha256=s['source_probe_sha256'],
            source_actual_adam=200,source_model_and_moments_restored=True,ema_initialized_from_same_raw_weights=True,
            actor_updated=False,encoder_forward_repeated=False,arm=arm,readout='question_plus20'))
        if previous and rank==0:once(out/('resume-after-epoch-%02d.json'%completed),dict(source_checkpoint=previous['checkpoint'],actual_adam_per_rank=[expected]*8,ema_updates=completed*updates_per_epoch,model_optimizer_ema_and_eight_rng_restored=True))
        if not previous:evaluate(0)
        for epoch in range(completed+1,s['epochs']+1):
            plan=plan_epoch(keys,s['dataset_seed'],epoch)
            if rank==0:once(out/('epoch-%02d-order.json'%epoch),plan)
            model.train();began=time.monotonic();seen=[]
            for step,planned in enumerate(plan,1):
                chosen=[groups[i]for i in rank_question_slice(planned,rank)]
                ids=[t for g in chosen for t in g['trajectory_ids']];seen.extend(ids);batch=prepare_batch(data,ids)
                labels=torch.tensor([[data.index[t]['strict_em']for t in g['trajectory_ids']]for g in chosen],device=device,dtype=torch.float32)
                soft_event=None;augmentation=dict(augmented_questions=0,lambda_sum=0.,pairs=0)
                if arm['mixup']:
                    if vector_objective:
                        batch,labels,augmentation=dominant_hidden_mixup(batch,labels,planned['mix_pairs'][rank],
                            s['dataset_seed']+epoch*1000003+step*8+rank,arm['mixup_alpha'])
                    else:
                        batch,labels,soft_event,augmentation=mix_batch(batch,labels,planned['mix_pairs'][rank],
                            s['dataset_seed']+epoch*1000003+step*8+rank,arm['mixup_alpha'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model(batch).reshape(len(chosen),5)
                if vector_objective:
                    losses=vector_dist_loss(logits.float(),labels,arm['dist_weight'],arm['mean_weight'],arm['diversity_weight'])
                    loss_keys=(('dist_loss','mean_correctness_kl','diversity_kl','total_loss')if arm['mean_weight']else('dist_loss','diversity_kl','total_loss'))
                else:
                    losses=joint_loss(logits.float(),labels,group_weight=arm['group_weight'],mixed_target=soft_event,validate=False)
                    loss_keys=('trajectory_bce','group_bce','total_loss')
                backward=losses['loss']*ddp_mean_scale(len(chosen),planned['global_questions'])
                backward.backward()
                optimizer.step();actual=int(next(iter(optimizer.state.values()))['step'].item())
                update_ema(model.module,ema,s['ema_decay'])
                values=torch.tensor([float(losses[k].detach())*len(chosen)for k in loss_keys]+[
                    augmentation['augmented_questions'],augmentation['pairs'],augmentation['lambda_sum']],device=device,dtype=torch.float64)
                dist.all_reduce(values);stats=values.cpu().tolist()
                with(out/('rank-%d-updates.jsonl'%rank)).open('a')as f:f.write(json.dumps(dict(epoch=epoch,step=step,rank=rank,
                    actual_adam=actual,local_trajectories=len(ids),learning_rate=s['learning_rate'],ema_updates=actual-200))+'\n')
                if rank==0:
                    nloss=len(loss_keys)
                    row=dict(epoch=epoch,step=step,actual_adam=actual,
                        **{key:stats[i]/planned['global_questions']for i,key in enumerate(loss_keys)},
                        augmented_questions=int(stats[nloss]),mixup_pairs=int(stats[nloss+1]),
                        mixup_lambda_mean=stats[nloss+2]/stats[nloss+1]if stats[nloss+1]else None,
                        online_loss_on_augmented_inputs=arm['mixup'])
                    if vector_objective:row.update(objective='group_vector_dist_kl',mixup_primary_coefficient_strictly_above_half=True,trajectory_BCE_in_loss=False,group_mean_KL_in_loss=bool(arm['mean_weight']))
                    with(out/'global-loss-updates.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
                    publish(out/'progress.json',dict(epoch=epoch,step=step,actual_adam=actual,ema_updates=actual-200))
            assert len(seen)==len(set(seen))==questions*5//8
            seconds=time.monotonic()-began;evaluate(epoch)
            rng=dict(rank=rank,python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))
            all_rng=[None]*8;dist.all_gather_object(all_rng,rng)
            if rank==0:
                saved=dict(run_id=c['run_id'],study_id=s['study_id'],arm=arm,epoch=epoch,actor_version=c['actor_version'],
                    model={k:v.detach().cpu()for k,v in model.module.state_dict().items()},
                    ema_model={k:v.detach().cpu()for k,v in ema.state_dict().items()},optimizer=copy.deepcopy(optimizer.state_dict()),
                    rank_rng=all_rng,actual_adam_per_rank=[actual]*8,ema_updates=actual-200,ema_decay=s['ema_decay'],
                    dataset_manifest_sha256=data.manifest_sha,test_manifest_sha256=test.manifest_sha,learning_rate=s['learning_rate'],
                    selected_data_blocks=data.selected_blocks,train_questions=questions,test_questions=test_questions,question_split=split,updates_per_epoch=updates_per_epoch)
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
        if rank==0:once(out/'complete.json',dict(status='ACTUAL_FIXED_DATA_PROBE_ARM_COMPLETE',arm=arm_name,epochs=s['epochs'],
            new_adam_per_rank=updates_per_epoch*s['epochs'],actual_adam_per_rank=[200+updates_per_epoch*s['epochs']]*8,ema_updates=updates_per_epoch*s['epochs'],actor_updates=0,latest=read(out/'latest.json')))
    finally:
        data.close();test.close()
        if pooled is not None:pooled.close()
        if dist.is_initialized():dist.destroy_process_group()


def run(base):
    from .local_probe_fit50 import admit_local_allocation
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class,DeferredInterrupt,birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base);c=read(base/'preparation/launch-config.json');s=c['probe20k_study']
    assert c['probe20k_role']=='fit'
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
            if(base/'arms'/name/'complete.json').exists():continue
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','reproduction.prefix_selection200.probe20k_fit','train','--base',str(base),'--arm',name]
            p=owned.start(name,command,env,str(base/'logs'/(name+'.log')),str(base/'repo'))
            while p.poll()is None:owned.observe(name);interrupt.check();time.sleep(1)
            owned.observe(name);owned.jobs[name]['exit_code']=p.returncode;owned.save();owned.stop(name)
            if p.returncode:raise RuntimeError('Actual20k probe arm failed: '+name)
            assert read(base/'arms'/name/'complete.json')['actual_adam_per_rank']==[200+s['planned_updates_per_epoch']*s['epochs']]*8
        once(base/'logs/pipeline-complete.json',dict(status='FIXED_DATA_EMA_PROBE_ARMS_COMPLETE',arms=[v['name']for v in s['arms']],epochs_per_arm=s['epochs'],actor_updates=0,formal200_resumed=False))
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
    c=read(prep/'launch-config.json');assert c['probe20k_role']=='fit'
    if c['probe20k_study'].get('resume_fit_base'):
        from .probe20k_resume import import_history
        import_history(base,c)
    s=c['probe20k_study'];assert read(Path(s['dataset_base'])/'dataset-complete.json')['trajectories']==102400
    from .local_probe_fit50 import SHARED_AUTHORIZATION
    assert c['local_probe_fit50']['shared_slurm_physical8_authorization']==SHARED_AUTHORIZATION
    python=str(Path(c['environment_root'])/'bin/python');script=prep/'native-probe20k-fit.sh'
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport PYTHONPATH='+shlex.quote(str(base/'repo'))+'\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexec '+shlex.join([python,'-m','reproduction.prefix_selection200.probe20k_fit','run','--base',str(base)])+'\n');script.chmod(0o700)
    command=['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(s.get('slurm_cpus',48)),'--mem='+str(s.get('slurm_memory_gib',256))+'G',
        '--time='+s.get('slurm_time_limit','0'),'--signal=B:TERM@60','--export=NONE','--job-name=g1-probe20k-fit-'+s['study_id'][:12],
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
