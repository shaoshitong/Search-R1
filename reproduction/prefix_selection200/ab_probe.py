"""Current-outer-only, whole-G5 balanced online training and persistent EMA."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import ab_policy as policy
from . import probe as p
from .native_features import NativeFeatures
from .probe_plan import rank_question_slice, ddp_mean_scale


def update(args):
    rank, local, device = p.setup_ddp()
    features = test = None
    try:
        model, saved, receipt = p.load_online(args, device)
        if saved['binding'].get('selection_probe_experiment') != policy.BALANCED:
            raise ValueError('Frozen arm cannot enter probe optimizer')
        config = p.read(Path(args.state_dir).parent/'preparation/launch-config.json')
        policy.validate(config)
        m = p.read(Path(args.scored_features)/'manifest.json')
        features = NativeFeatures(m['native_hidden_scan'])
        if (m['run_id'] != saved['run_id'] or m['outer'] != args.outer
                or m['probe_before_checkpoint_sha256'] != receipt['sha256']
                or features.manifest['actor_version'] != m['actor_version']):
            raise ValueError('Current outer features/probe/actor provenance differs')
        own = Path(args.own_em)
        if p.sha(own) != args.own_em_sha256:
            raise ValueError('Own EM file differs from actual rollout receipt')
        records = [json.loads(line) for line in own.open()]
        # Ordinary rounds intentionally have labels only for the selected512.
        tids = {row['trajectory_id'] for row in records}
        if len(tids) != len(records) or not tids <= features.index.keys():
            raise ValueError('Own labels must be unique current sampled trajectories')
        index = p.join_own_em({tid:row for tid,row in features.index.items() if tid in tids},
            records, m['actor_version'], m['prefix_scan_sha256'], m['selection_sha256'])
        n = len(index)//5
        from .periodic import is_probe_update
        if n != (2560 if is_probe_update(args.outer) else 512):
            raise ValueError('Train exactly this round complete pool or selected512')
        groups = p.candidate_groups(index, n)
        features.index = index
        if args.test_cache:
            test = NativeFeatures(args.test_cache)
            if test.manifest_sha != args.test_manifest_sha or test.manifest['actor_version'] != m['actor_version']:
                raise ValueError('Diagnostic features must bind this same actor')
            path = Path(args.test_cache).parent.parent/'full/own-em.jsonl'
            test.index = p.join_own_em(test.index,[json.loads(line) for line in path.open()],
                m['actor_version'],test.manifest_sha,'diagnostic')
            if ({str(r['qid']) for r in index.values()} & {str(r['qid']) for r in test.index.values()}
                    or {r['question_hash'] for r in index.values()} & {r['question_hash'] for r in test.index.values()}):
                raise ValueError('Training and fixed diagnostics overlap')
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        ema.load_state_dict(saved['ema_model'], strict=True)
        p.restore_native_buffers(ema, saved['ema_nonpersistent_buffers'])

        def evaluate(head):
            head.eval(); result = {}
            splits = [('train', features, index, groups)]
            if test is not None:
                splits.append(('test', test, test.index, p.candidate_groups(test.index,512)))
            with torch.no_grad():
                for split, data, labels, rows in splits:
                    predictions = []
                    local_rows = rows[rank::p.WORLD]
                    for start in range(0,len(local_rows),6):
                        ids = [tid for g in local_rows[start:start+6] for tid in g['trajectory_ids']]
                        batch = data.get_batch(ids,device)
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                            logits = head(batch)
                        predictions.extend((tid,float(z),int(labels[tid]['strict_em'])) for tid,z in zip(ids,logits.cpu().tolist()))
                    all_predictions = [None]*p.WORLD
                    dist.all_gather_object(all_predictions,predictions)
                    if rank == 0:
                        merged = [r for part in all_predictions for r in part]
                        metrics, rows_out = p.evaluate_predictions(merged,labels,len(rows),group_weight=5.)
                        result[split] = metrics
                        with (out/(evaluation_clock+'-'+split+'-group-predictions.jsonl')).open('x') as handle:
                            for row in rows_out:
                                handle.write(p.canonical(row).decode()+'\n')
            return result

        out = Path(args.output)
        if rank == 0: out.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        evaluation_clock = 'raw-before'; before_raw = evaluate(model)
        evaluation_clock = 'ema-before'; before_ema = evaluate(ema)
        before = receipt['actual_adam_steps_per_rank'][0]
        optimizer = torch.optim.Adam(model.parameters(), lr=p.LR, weight_decay=0)
        if before:
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key,value in list(state.items()):
                    if isinstance(value,torch.Tensor) and key != 'step': state[key] = value.to(device)
        if saved.get('rank_rng') is not None:
            rng = saved['rank_rng'][rank]
            random.setstate(rng['python']); np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state(rng['torch_cuda'],device)
        else:
            random.seed(args.seed+rank); np.random.seed(args.seed+rank)
            torch.manual_seed(args.seed+rank); torch.cuda.manual_seed_all(args.seed+rank)
        if p.actual_adam_steps(optimizer) != before or any(x != p.LR for x in p.optimizer_learning_rates(optimizer)):
            raise ValueError('Actual probe Adam/LR differs from saved state')
        labels = [int(0<sum(index[t]['strict_em'] for t in g['trajectory_ids'])<5) for g in groups]
        plan, sampling = policy.balanced_plan(labels,args.seed,args.outer)
        all_plans = [None]*p.WORLD
        dist.all_gather_object(all_plans,p.digest(sampling))
        if len(set(all_plans)) != 1: raise ValueError('Ranks have different balancing selections')
        if rank == 0: p.once(out/'balanced-sampling.json',sampling)
        ddp = DDP(model,device_ids=[local],broadcast_buffers=False)
        ddp.train(); curve = []; began = time.monotonic()
        for step,batch_plan in enumerate(plan,1):
            rows = [groups[j] for j in rank_question_slice(batch_plan,rank)]
            ids = [tid for g in rows for tid in g['trajectory_ids']]
            batch = features.get_batch(ids,device)
            target = torch.tensor([[index[t]['strict_em'] for t in g['trajectory_ids']] for g in rows],device=device,dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                logits = ddp(batch).reshape(len(rows),5)
            loss = p.joint_loss(logits.float(),target,group_weight=5.)
            backward = loss['loss']*ddp_mean_scale(len(rows),batch_plan['global_questions'])
            finite = torch.isfinite(backward).int(); dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item(): raise FloatingPointError('Actual nonfinite probe loss')
            backward.backward()
            finite = torch.stack([torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None]).all().int()
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item(): raise FloatingPointError('Actual nonfinite probe gradients')
            optimizer.step()
            actual = p.actual_adam_steps(optimizer)
            if actual != before+step: raise RuntimeError('Probe actual Adam counter mismatch')
            policy.ema_step(ema,model,.99)
            stats = torch.tensor([float(loss[k].detach())*len(rows) for k in ('trajectory_bce','group_bce','loss')]+[len(rows)],device=device,dtype=torch.float64)
            dist.all_reduce(stats)
            row = dict(event='ACTUAL_PROBE_ADAM_WRITE',rank=rank,outer=args.outer,step=step,actual_adam_steps=actual,ema_updates=actual,
                learning_rate=p.LR,local_questions=len(rows),local_trajectories=len(ids),
                global_questions=int(stats[3]),trajectory_bce=float(stats[0]/stats[3]),
                group_bce=float(stats[1]/stats[3]),total_loss=float(stats[2]/stats[3]))
            p.append(out/('rank-%d-updates.jsonl'%rank),row)
            p.publish(out/('progress-rank-%d.json'%rank),dict(row,stage='ONLINE_AND_EMA_WRITE_COMPLETE'))
            curve.append(row)
            if rank == 0: p.append(out/'global-loss-updates.jsonl',row)
        loop_seconds = time.monotonic()-began
        evaluation_clock = 'raw-after'; after_raw = evaluate(model)
        evaluation_clock = 'ema-after'; after_ema = evaluate(ema)
        actual = p.actual_adam_steps(optimizer)
        rng = dict(rank=rank,actual_adam_steps=actual,python=random.getstate(),numpy=np.random.get_state(),
            torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state(device))
        all_rng = [None]*p.WORLD; dist.all_gather_object(all_rng,rng)
        if rank == 0:
            optimizer_state = optimizer.state_dict() if actual else None
            if optimizer_state:
                for state in optimizer_state['state'].values():
                    for key,value in list(state.items()):
                        if isinstance(value,torch.Tensor): state[key] = value.detach().cpu()
            checkpoint = dict(saved,completed_outer=args.outer,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                model_nonpersistent_buffers=p.nonpersistent_buffers(model),
                ema_model={k:v.detach().cpu() for k,v in ema.state_dict().items()},
                ema_nonpersistent_buffers=p.nonpersistent_buffers(ema),ema_updates=actual,
                optimizer=optimizer_state,rank_rng=all_rng,actual_adam_steps_per_rank=[actual]*p.WORLD,
                sampling=sampling,own_em_sha256=p.sha(own),actor_version=m['actor_version'],
                scored_features_manifest_sha256=features.manifest_sha)
            result = p.atomic_checkpoint(args.state_dir,checkpoint,args.outer)
            metrics = dict(run_id=saved['run_id'],outer=args.outer,status=policy.COMPLETE,
                actual_adam_steps_per_rank=[actual]*p.WORLD,new_adam_steps_per_rank=len(plan),ema_updates=actual,
                original_training_questions=n,group_loss_weight=5.,learning_rate=p.LR,ema_rate=.99,
                sampling={k:v for k,v in sampling.items() if k not in ('sampled_indices','group_multiplicities','rng_before','rng_after')},
                sampling_path=str(out/'balanced-sampling.json'),sampling_sha256=p.sha(out/'balanced-sampling.json'),
                before_update=dict(raw=before_raw,ema=before_ema),
                after_update=dict(raw=after_raw,ema=after_ema),train_loop_seconds=loop_seconds,
                global_step_loss_curve=curve,checkpoint=result,metrics_distribution='original_unresampled_current_outer_groups',
                own_em_sha256=p.sha(own),scored_features_manifest_sha256=features.manifest_sha)
            p.once(out/'metrics.json',metrics)
            p.once(out/'complete.json',dict(metrics,metrics_sha256=p.sha(out/'metrics.json'),observed_utc=p.utc()))
        dist.barrier()
    finally:
        if features is not None: features.close()
        if test is not None: test.close()
        if dist.is_initialized(): dist.destroy_process_group()
