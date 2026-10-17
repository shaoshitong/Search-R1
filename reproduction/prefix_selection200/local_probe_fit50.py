"""Frozen actor50, one fixed G5 dataset, ten shuffled probe-only epochs.

The native sampling hidden is captured once. No actor/grounding/PPO update,
selection or formal200 continuation is reachable from this entry point.
"""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import time
import urllib.request

from reproduction.prefix20_online.launch import once, publish, read, sha, native_allocation


DATA_OUTER = 51  # actor0050 supplies every prefix and continuation.


def train(base):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from .probe import (OnlineQuestionProbe, setup_ddp, candidate_groups, join_own_em,
                        ddp_mean_scale, rank_question_slice, evaluate_predictions)
    from .native_features import NativeFeatures
    from .periodic import complete_pool_plan
    from reproduction.prefix20_group_joint100.loss import joint_loss
    from reproduction.prefix_only_correctness_ab.train_ddp import actual_adam_steps
    base = Path(base); c = read(base/'preparation/launch-config.json')
    spec = c['local_probe_fit50']; rank, local, device = setup_ddp()
    out = base/'train'; out.mkdir(exist_ok=True)
    root = base/'outers'/('outer-%06d'%DATA_OUTER)
    train_data = NativeFeatures(root/'prefix/scan')
    diagnostic = read(root/'diagnostic-complete.json')
    test_data = NativeFeatures(diagnostic['scan_path'])
    all_selection = root/'all-selection.json'
    actor_version = spec['actor_version']
    assert train_data.manifest['actor_version'] == test_data.manifest['actor_version'] == actor_version
    for data, own, selection in ((train_data, root/'full/own-em.jsonl', sha(all_selection)),
            (test_data, Path(diagnostic['scan_path']).parent.parent/'full/own-em.jsonl', 'diagnostic')):
        data.index = join_own_em(data.index, [json.loads(line) for line in own.open()],
                                 actor_version, data.manifest_sha, selection)
    assert not ({r['qid'] for r in train_data.index.values()} & {r['qid'] for r in test_data.index.values()})
    assert not ({r['question_hash'] for r in train_data.index.values()} & {r['question_hash'] for r in test_data.index.values()})
    groups = candidate_groups(train_data.index, 5120)
    test_groups = candidate_groups(test_data.index, 512)
    source = Path(spec['probe_checkpoint']); assert source.stat().st_size == spec['probe_bytes']
    saved = torch.load(source, map_location='cpu', weights_only=False)
    assert saved['run_id'] == c['run_id'] and saved['actual_adam_steps_per_rank'] == [200]*8
    model = OnlineQuestionProbe(Path(c['official_model_dir']), 273).to(device=device, dtype=torch.float32)
    model.load_state_dict(saved['model'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec['learning_rate'], weight_decay=0)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = spec['learning_rate']
    for values in optimizer.state.values():
        for key, value in list(values.items()):
            if isinstance(value, torch.Tensor) and key != 'step':
                values[key] = value.to(device)
    assert actual_adam_steps(optimizer) == 200
    rng = saved['rank_rng'][rank]
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state(rng['torch_cuda'], device)
    model = DDP(model, device_ids=[local], broadcast_buffers=False)
    def evaluate(epoch):
        model.eval(); result = {}; began = time.monotonic()
        with torch.no_grad():
            for split, data, question_groups in (('train',train_data,groups), ('test',test_data,test_groups)):
                own = question_groups[rank::8]; predictions = []
                for offset in range(0, len(own), 6):
                    ids = [tid for g in own[offset:offset+6] for tid in g['trajectory_ids']]
                    batch = data.get_batch(ids, device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model.module(batch)
                    predictions.extend((tid,float(z),int(data.index[tid]['strict_em']))
                                       for tid,z in zip(ids,logits.cpu().tolist()))
                all_predictions = [None]*8; dist.all_gather_object(all_predictions,predictions)
                if rank == 0:
                    flat = [row for values in all_predictions for row in values]
                    metrics, grouped = evaluate_predictions(flat,data.index,len(question_groups))
                    metrics['trajectory_accuracy'] = sum(int(z>=0)==y for _,z,y in flat)/len(flat)
                    metrics['trajectory_correct_prevalence'] = sum(y for _,_,y in flat)/len(flat)
                    result[split] = metrics
                    with (out/('epoch-%02d-%s-predictions.jsonl'%(epoch,split))).open('x') as stream:
                        for row in grouped: stream.write(json.dumps(row,sort_keys=True)+'\n')
        if rank == 0:
            result.update(epoch=epoch,actual_adam_per_rank=[actual_adam_steps(optimizer)]*8,
                learning_rate=spec['learning_rate'],actor_version=actor_version,
                fixed_train_scan_sha256=train_data.manifest_sha,fixed_test_scan_sha256=test_data.manifest_sha,
                evaluation_seconds=time.monotonic()-began)
            once(out/('epoch-%02d-metrics.json'%epoch),result)
            publish(out/'latest-metrics.json',result)
        dist.barrier()
        return result
    try:
        if rank == 0:
            once(out/'initialization.json',dict(actor_version=actor_version,
                source_probe_path=str(source),source_probe_sha256=spec['probe_sha256'],
                source_actual_adam_per_rank=[200]*8,learning_rate=spec['learning_rate'],
                optimizer_moments_imported=True,actor_updated=False,encoder_forward_repeated=False))
        evaluate(0)
        for epoch in range(1,11):
            # Reuse the tested whole-G5 planner, changing only its shuffle seed.
            plan = complete_pool_plan(c['seed']+epoch*100003,20)
            if rank == 0: once(out/('epoch-%02d-order.json'%epoch),plan)
            model.train(); began=time.monotonic(); seen=[]
            for step,planned in enumerate(plan,1):
                batch_groups=[groups[j] for j in rank_question_slice(planned,rank)]
                ids=[tid for g in batch_groups for tid in g['trajectory_ids']];seen.extend(ids)
                batch=train_data.get_batch(ids,device)
                labels=torch.tensor([[train_data.index[t]['strict_em'] for t in g['trajectory_ids']]
                                     for g in batch_groups],dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits=model(batch).reshape(len(batch_groups),5)
                losses=joint_loss(logits.float(),labels,group_weight=1.)
                backward=losses['loss']*ddp_mean_scale(len(batch_groups),planned['global_questions'])
                finite=torch.isfinite(backward).int();dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite.item()):raise FloatingPointError('Actual nonfinite local fitting loss')
                backward.backward()
                finite=torch.stack([torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None]).all().int()
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not int(finite.item()):raise FloatingPointError('Actual nonfinite local fitting gradient')
                optimizer.step()
                actual=actual_adam_steps(optimizer)
                assert actual==200+(epoch-1)*100+step
                stats=torch.tensor([float(losses[k].detach())*len(batch_groups)
                    for k in ('trajectory_bce','group_bce','total_loss')],device=device,dtype=torch.float64)
                dist.all_reduce(stats);stats=(stats/planned['global_questions']).tolist()
                row=dict(epoch=epoch,step=step,rank=rank,actual_adam=actual,
                    local_questions=len(batch_groups),local_trajectories=len(ids),learning_rate=spec['learning_rate'])
                with (out/('rank-%d-updates.jsonl'%rank)).open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                if rank==0:
                    with (out/'global-loss-updates.jsonl').open('a')as stream:
                        stream.write(json.dumps(dict(epoch=epoch,step=step,actual_adam=actual,
                            trajectory_bce=stats[0],group_bce=stats[1],total_loss=stats[2]))+'\n')
                    publish(out/'progress.json',dict(epoch=epoch,step=step,actual_adam=actual))
            assert len(seen)==len(set(seen))==3200
            loop_seconds=time.monotonic()-began
            metrics=evaluate(epoch)
            rank_rng=dict(rank=rank,python=random.getstate(),numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))
            all_rng=[None]*8;dist.all_gather_object(all_rng,rank_rng)
            if rank==0:
                checkpoint=dict(local_probe_fit50=True,epoch=epoch,source_probe_sha256=spec['probe_sha256'],
                    model={k:v.detach().cpu() for k,v in model.module.state_dict().items()},
                    optimizer=copy.deepcopy(optimizer.state_dict()),rank_rng=all_rng,actual_adam_per_rank=[actual]*8,
                    learning_rate=spec['learning_rate'],actor_version=actor_version,
                    train_scan_sha256=train_data.manifest_sha,test_scan_sha256=test_data.manifest_sha)
                for values in checkpoint['optimizer']['state'].values():
                    for key,value in list(values.items()):
                        if isinstance(value,torch.Tensor):values[key]=value.detach().cpu()
                path=out/('head-epoch-%02d.pt'%epoch);temp=path.with_suffix('.pt.partial')
                torch.save(checkpoint,temp)
                with temp.open('rb') as stream:os.fsync(stream.fileno())
                os.replace(temp,path)
                complete=dict(epoch=epoch,actual_adam_per_rank=[actual]*8,checkpoint=str(path),
                    checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha(path),
                    metrics_sha256=sha(out/('epoch-%02d-metrics.json'%epoch)),train_loop_seconds=loop_seconds)
                once(out/('epoch-%02d-complete.json'%epoch),complete);publish(out/'latest.json',complete)
            dist.barrier()
        if rank==0:once(out/'complete.json',dict(status='FROZEN_ACTOR50_FIXED_G5_TEN_EPOCHS_COMPLETE',
            epochs=10,new_adam_per_rank=1000,actual_adam_per_rank=[1200]*8,actor_updates=0,
            latest=read(out/'latest.json')))
    finally:
        train_data.close();test_data.close()
        if dist.is_initialized():dist.destroy_process_group()


def run(base):
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    from .rollout import root_dir
    base=Path(base);c=read(base/'preparation/launch-config.json');spec=c['local_probe_fit50']
    assert spec['epochs']==10 and spec['actor_outer']==50 and spec['learning_rate']==1e-5
    native_allocation(base,c);lease=UUIDLocks(c['physical_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();handlers={s:signal.getsignal(s)for s in (signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        SEARCH_G1_BASE=str(base),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',NCCL_NVLS_ENABLE='0',
        HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),TRITON_CACHE_DIR=str(base/'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost')
    if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
    for key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in ('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python');conf=str(base/'preparation/launch-config.json');services={}
    def start(name,command,child_env=None):
        p=owned.start(name,command,child_env or env,str(base/'logs'/(name+'.log')),str(base/'repo'));owned.observe(name);return p
    def phase(name,command):
        publish(base/'logs/pipeline-status.json',dict(stage=name,checked_unix=time.time(),actor_version=spec['actor_version']))
        p=start(name,command)
        while p.poll()is None:
            owned.observe(name);interrupt.check()
            if any(s.poll()is not None for s in services.values()):raise RuntimeError('Own retriever exited')
            time.sleep(1)
        owned.observe(name);owned.jobs[name]['exit_code']=p.returncode;owned.save();owned.stop(name)
        if p.returncode:raise RuntimeError('Local fitting phase failed: '+name)
    try:
        envroot=Path(c['environment_root']);libs=list((envroot/'lib/python3.10/site-packages/nvidia').glob('*/lib'))+[envroot/'lib/python3.10/site-packages/torch/lib']
        services['retriever']=start('retriever',[python,'-m','reproduction.prefix20_online.retriever',
            '--index',c['retrieval_index'],'--corpus',c['retrieval_corpus'],'--encoder',c['retrieval_encoder'],
            '--cache',str(base/'retrieval-cache'),'--run-id',c['run_id'],'--port',str(c['retrieval_port']),
            '--data-cache',c.get('retrieval_data_cache',str(base/'retrieval-cache')),'--batch-queries',str(c.get('retrieval_batch_queries',1024))],
            dict(env,PYTHONPATH=c['retriever_gpu_lib']+':'+str(base/'repo'),LD_LIBRARY_PATH=':'.join(map(str,libs))))
        while True:
            interrupt.check();owned.observe('retriever')
            if services['retriever'].poll()is not None:raise RuntimeError('Own retriever failed during startup')
            try:
                with urllib.request.urlopen(c['search_url'].replace('/retrieve','/health'),timeout=5)as response:health=json.load(response)
                if health.get('ok')and health.get('run_id')==c['run_id']and health.get('gpu_shards')==8:break
            except (OSError,ValueError):pass
            time.sleep(2)
        phase('fixed-actor50-prefix',[python,'-m','reproduction.prefix_selection200.rollout','--config',conf,'--outer',str(DATA_OUTER),'--phase','prefix'])
        root=root_dir(c,DATA_OUTER)
        from .probe import candidate_groups
        from .rollout import iter_rows
        prefix=read(root/'prefix/complete.json');index={r['trajectory_id']:r for r in iter_rows(Path(prefix['scan_path'])/'index.jsonl')}
        selection=root/'all-selection.json'
        once(selection,dict(run_id=c['run_id'],outer=DATA_OUTER,actor_version=spec['actor_version'],
            prefix_scan_sha256=prefix['scan_sha256'],selected=candidate_groups(index,5120),selected_questions=5120,
            outcome_labels_used=False,continuation_scope='all_candidates'))
        phase('fixed-actor50-full25600',[python,'-m','reproduction.prefix_selection200.rollout','--config',conf,'--outer',str(DATA_OUTER),'--phase','continue','--selection',str(selection)])
        phase('fixed-actor50-heldout2560',[python,'-m','reproduction.prefix_selection200.periodic_data','diagnostic','--config',conf,'--outer',str(DATA_OUTER)])
        owned.stop('retriever');services.clear()
        phase('probe-fixed-data-ten-epochs',[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',
            'reproduction.prefix_selection200.local_probe_fit50','train','--base',str(base)])
        once(base/'logs/pipeline-complete.json',dict(status='LOCAL_PROBE50_TEN_EPOCHS_COMPLETE',actor_updated=False,
            formal200_resumed=False,train_complete=read(base/'train/complete.json')))
    finally:
        try:
            owned.stop_all();owned.reap();births=[row for job in owned.jobs.values()for row in job['ledger'].values()];present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(owned_births_all_gone=not present,remaining_owned_births=present,checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)


def submit(base):
    base=Path(base);prep=base/'preparation'
    if (prep/'submit-intent.json').exists():
        return dict(status='EXISTING_INTENT_NO_REPLAY',receipt=read(prep/'submit-receipt.json')if(prep/'submit-receipt.json').exists()else None)
    c=read(prep/'launch-config.json');python=str(Path(c['environment_root'])/'bin/python');script=prep/'native-local-fit50.sh'
    once(prep/'local-only-scope.json',dict(actor_updates=0,probe_epochs=10,formal200_resumed=False))
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(str(base/'repo'))+'\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexec '+shlex.join([python,'-m','reproduction.prefix_selection200.local_probe_fit50','run','--base',str(base)])+'\n');script.chmod(0o700)
    cmd=['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task=96','--mem=512G','--gres=gpu:8','--time=0','--signal=B:TERM@60','--export=NONE',
        '--job-name=g1-fit50-'+c['local_probe_fit50']['experiment_id'][:12],
        '--comment=g1-local-fit50:'+c['local_probe_fit50']['experiment_id'],'--chdir='+str(base),
        '--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(script)]
    once(prep/'submit-intent.json',dict(command=cmd,config_sha256=sha(prep/'launch-config.json'),created_unix=time.time()))
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        job=r.stdout.strip().split(';')[0]
        if r.returncode==0 and job.isdecimal():value.update(status='SUBMITTED',job_id=job)
        else:value['status']='UNKNOWN_OR_REJECTED_NO_REPLAY'
    except subprocess.TimeoutExpired as error:value=dict(status='UNKNOWN_NO_REPLAY',error=repr(error))
    once(prep/'submit-receipt.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('run','train','submit'));parser.add_argument('--base',required=True);args=parser.parse_args()
    print(json.dumps(globals()[args.phase](args.base)),flush=True)
