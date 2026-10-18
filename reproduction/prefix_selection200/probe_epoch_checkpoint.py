"""Durable in-fit probe epoch boundaries without moving a live Adam to CPU."""
import hashlib
import json
import os
from pathlib import Path

import torch


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def save_epoch(directory,*,epoch,model,optimizer,rank_rng,base_saved,
               expected_before,steps,features_sha,labels_sha,plan_sha,totals,global_curve):
    from .prefix5_model import nonpersistent_buffers
    from .contract import PREFIX_TOKENS
    from .probe import actual_adam_steps
    if steps!=epoch*50 or actual_adam_steps(optimizer)!=expected_before+steps:
        raise ValueError('Only a real complete epoch with its actual Adam boundary can be saved')
    if len(rank_rng)!=4 or {r['rank'] for r in rank_rng}!=set(range(4)):
        raise ValueError('Every epoch must retain all four actual RNG states')
    directory=Path(directory);path=directory/('epoch-%03d.pt'%epoch)
    receipt_path=path.with_suffix('.json')
    if path.exists() or receipt_path.exists():raise FileExistsError('Keep existing epoch evidence; resume explicitly')
    saved=dict(role='PROBE_FIT_EPOCH_COMPLETE_NOT_JOINT',run_id=base_saved['run_id'],
        completed_epoch=epoch,steps_in_current_fit=steps,expected_before=expected_before,
        model=cpu_copy(model.state_dict()),model_nonpersistent_buffers=nonpersistent_buffers(model),
        optimizer=cpu_copy(optimizer.state_dict()),optimizer_parameter_names=[n for n,_ in model.named_parameters()],
        rank_rng=rank_rng,actual_adam_steps_per_rank=[expected_before+steps]*4,
        binding=base_saved['binding'],binding_sha256=base_saved['binding_sha256'],
        prefix_tokens=PREFIX_TOKENS,scored_features_manifest_sha256=features_sha,
        own_em_sha256=labels_sha,plan_sha256=plan_sha,rank_totals=totals,global_curve=global_curve)
    temporary=path.with_suffix('.tmp-%d'%os.getpid())
    with temporary.open('xb') as handle:torch.save(saved,handle);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for raw in iter(lambda:handle.read(8*1024**2),b''):digest.update(raw)
    proof=dict(role=saved['role'],path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest(),
               completed_epoch=epoch,steps_in_current_fit=steps,actual_adam_steps_per_rank=saved['actual_adam_steps_per_rank'],
               prefix_tokens=PREFIX_TOKENS,features_sha256=features_sha,labels_sha256=labels_sha,
               plan_sha256=plan_sha,actor_optimizer_updated=False)
    with receipt_path.open('x') as handle:json.dump(proof,handle,sort_keys=True);handle.flush();os.fsync(handle.fileno())
    return proof


def restore_epoch(directory,epoch,*,features_sha,labels_sha,plan_sha,base_saved,expected_before):
    """Explicit stopped-fit recovery; never silently discard later recorded writes."""
    from .contract import PREFIX_TOKENS
    from .probe import sha,actual_adam_steps
    directory=Path(directory);path=directory/('epoch-%03d.pt'%epoch)
    proof=json.loads(path.with_suffix('.json').read_text())
    if (proof['role']!='PROBE_FIT_EPOCH_COMPLETE_NOT_JOINT' or proof['completed_epoch']!=epoch
            or proof['prefix_tokens']!=PREFIX_TOKENS or proof['features_sha256']!=features_sha
            or proof['labels_sha256']!=labels_sha or proof['plan_sha256']!=plan_sha
            or path.stat().st_size!=proof['bytes'] or sha(path)!=proof['sha256']):
        raise ValueError('Epoch recovery needs its exact qualified feature/label/plan/payload SHA')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    steps=epoch*50
    if (saved['binding_sha256']!=base_saved['binding_sha256'] or saved['expected_before']!=expected_before
            or saved['steps_in_current_fit']!=steps or saved['actual_adam_steps_per_rank']!=[expected_before+steps]*4
            or len(saved['global_curve'])!=steps or len(saved['rank_totals'])!=4):
        raise ValueError('Epoch recovery model/Adam/RNG/fit counters differ')
    # Replaying uncheckpointed real writes requires a separate explicit recovery
    # decision and retained evidence, not an implicit command-line rollback.
    for name in ['global-loss-updates.jsonl']+['rank-%d-updates.jsonl'%rank for rank in range(4)]:
        rows=[json.loads(raw) for raw in (directory/name).read_text().splitlines()]
        if len(rows)!=steps or [row['step'] for row in rows]!=list(range(1,steps+1)):
            raise RuntimeError('Later real writes exist; qualify and archive them before an authorized epoch replay')
    return saved
