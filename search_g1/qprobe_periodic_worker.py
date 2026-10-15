"""Readback of a frozen *current* student, including nonempty Adam counters."""
import hashlib
from pathlib import Path

import torch


def capture(worker, directory):
    from search_g1.training_state import capture_worker
    from search_g1.qprobe_features import sha_json
    digest = hashlib.sha256()
    count = 0
    for name,parameter in worker.actor.actor_module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        raw = memoryview(value.view(torch.uint8).numpy()).cast('B')
        for offset in range(0,len(raw),8*1024**2):
            digest.update(raw[offset:offset+8*1024**2])
        count += value.numel()
    steps = sorted(set(int(s['step']) for s in worker.actor_optimizer.state.values() if 'step' in s))
    writes = getattr(worker.actor,'_optimizer_writes',0)
    if not count or steps != [writes]:
        raise ValueError('PERIODIC_CURRENT_STUDENT_ADAM_STEPS_DIVERGE')
    return dict(rank=worker.rank,parameter_sha256=digest.hexdigest(),local_parameter_count=count,
        actual_optimizer_writes=writes,completed_outer=getattr(worker.actor,'_complete_outers',0),
        scheduler_sha256=sha_json(worker.actor_lr_scheduler.state_dict()),
        adam_steps=steps,rng_receipt=capture_worker(worker,directory))


def restore_rng(worker, directory):
    import random
    import numpy as np
    from search_g1.training_state import cpu_tree,worker_rng
    from search_g1.full_state_restore import equal_tree
    saved = torch.load(Path(directory)/('rank-%d.pt'%worker.rank),map_location='cpu',weights_only=False)
    if (saved['rank'] != worker.rank or
            saved['actual_optimizer_writes'] != getattr(worker.actor,'_optimizer_writes',0) or
            saved['completed_outer'] != getattr(worker.actor,'_complete_outers',0)):
        raise ValueError('PERIODIC_SAMPLING_RNG_BELONGS_TO_DIFFERENT_STUDENT')
    engine = worker.rollout.inference_engine.llm_engine
    if engine.has_unfinished_requests():
        raise ValueError('PERIODIC_CANNOT_RESTORE_ACTIVE_SAMPLING_RNG')
    rng = saved['rng']
    generators = engine.model_executor.worker.model_runner.get_generators()
    generators.clear()
    for key,value in rng['request_generators'].items():
        generator = torch.Generator(device=value['device'])
        generator.set_state(value['state'])
        generators[key] = generator
    worker.rollout_sharding_manager.torch_random_states = cpu_tree(rng['actor_cuda_saved'])
    worker.rollout_sharding_manager.gen_random_states = cpu_tree(rng['rollout_cuda_saved'])
    torch.set_rng_state(rng['torch_cpu'])
    torch.cuda.set_rng_state(rng['torch_cuda'],device=torch.cuda.current_device())
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    equal_tree(worker_rng(worker),rng)
    return dict(rank=worker.rank,rng_equal=True,completed_outer=saved['completed_outer'])
