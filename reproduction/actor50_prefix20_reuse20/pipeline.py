"""Owned frozen actor replay, then ten matched two-epoch Prefix20 cycles."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from reproduction.prefix_selection200.a800_runtime import UUIDLocks,gpu_inventory,compute_processes
from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class,DeferredInterrupt,birth_states
from .data import read,sha,publish,prepare


def initial(base,c):
    import torch
    from .model import lift_state
    from reproduction.prefix_selection200.prefix5_model import validate_native_buffers
    parent=Path(c['source_prefix5_base'])
    proof=read(parent/'preparation/source50-qualified.json');pin=parent/'preparation/initial-probe50-model.pt'
    assert sha(pin)==proof['initial_model_sha256'] and proof['probe50_sha256']==c['probe50_sha256']
    assert proof['joint50_sha256']==c['joint50_sha256'] and proof['actor_version']==c['actor_version']
    old=torch.load(pin,map_location='cpu',weights_only=False)
    assert old['new_adam_updates']==old['ema_updates']==0 and not old['source_optimizer_imported']
    raw=lift_state(old['raw_model']);validate_native_buffers(old['raw_buffers'])
    value=dict(raw_model=raw,ema_model=raw,raw_buffers=old['raw_buffers'],ema_buffers=old['raw_buffers'],
        source_probe50_sha256=c['probe50_sha256'],new_adam_updates=0,ema_updates=0,actor_updates=0,
        source_optimizer_imported=False,prefix_policy_tokens=20)
    target=base/'preparation/initial-probe50-model.pt'
    with target.open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    result=dict(initial_state_sha256=sha(target),source50_qualification=proof,
        source_head_columns_preserved=278*2048,added_zero_head_columns=15*2048,initial_Adam=0,initial_EMA=0,
        source50_raw_model_only=True,actor_updates=0,checked_unix=time.time())
    publish(base/'preparation/initial-model-qualified.json',result)


def admit(base,c):
    mapping=gpu_inventory();indices=c['physical_gpu_indices'];uuids=c['physical_gpu_uuids']
    assert len(indices)==len(set(indices))==4 and [mapping[i]['uuid'] for i in indices]==uuids
    assert not {x['uuid'] for x in compute_processes()}&set(uuids)
    assert all(mapping[i]['memory_MiB']<=64 for i in indices)
    publish(base/'logs/native-allocation.json',dict(actual_inventory=[mapping[i] for i in indices],
        physical_gpu_uuids=uuids,no_auxiliary_GPUs_needed=True,foreign_processes_stopped=False,checked_unix=time.time()))


def run(base):
    base=Path(base);c=read(base/'preparation/config.json')
    assert c['target_probe_epochs']==20 and c['cycles']==10 and c['prefix_policy_tokens']==20 and not c['actor_training_enabled']
    assert c['actor_version']=='54ac33dcebe64e62a11fe4177f41f116-actor0050'
    success=False;owned=None;lease=None;interrupt=DeferredInterrupt()
    old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in old:signal.signal(s,interrupt.handler)
    def status(stage,**extra):publish(base/'logs/status.json',dict(stage=stage,actor_updates=0,checked_unix=time.time(),**extra))
    try:
        status('PREPARING_SAME_STORED_G5_DATA');prepare(base);initial(base,c)
        lease=UUIDLocks(c['physical_gpu_uuids']);admit(base,c)
        owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
        env=dict(os.environ,PYTHONPATH=str(base/'repo'),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            CUDA_VISIBLE_DEVICES=','.join(map(str,c['physical_gpu_indices'])),CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
            HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),TMPDIR=str(base/'tmp'),
            NCCL_NVLS_ENABLE='0',TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'))
        for key in ('PYTHONHOME','RAY_ADDRESS'):env.pop(key,None)
        for name in ('tmp','cache/hf','cache/triton','cache/inductor','training','probe'):(base/name).mkdir(parents=True,exist_ok=True)
        python=str(Path(c['environment_root'])/'bin/python')
        def phase(name,module,*args):
            status(name)
            command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m',module]+list(args)
            child=owned.start(name,command,env,str(base/'logs'/(name+'.log')),str(base/'repo'),pass_fds=tuple(lease.fds))
            while child.poll() is None:
                owned.observe(name);interrupt.check();time.sleep(2)
            owned.observe(name);owned.jobs[name]['exit_code']=child.returncode;owned.save();owned.stop(name)
            if child.returncode:raise RuntimeError('Own phase failed: '+name)
            interrupt.check()
        phase('REPLAY_STORED_PREFIX20_FEATURES','reproduction.actor50_prefix20_reuse20.extract','--base',str(base))
        for number in range(11):
            name='test' if number==0 else 'cycle-%04d'%number;target=base/'data'/name
            shards=[read(target/('rank-%d-complete.json'%rank)) for rank in range(4)]
            assert all(x['rank']==rank and x['preserved_question_and_first5_bitwise'] and x['new_sampled_tokens']==0 for rank,x in enumerate(shards))
            publish(target/'features-complete.json',dict(dataset=name,shards=shards,actor_version=c['actor_version'],prefix_tokens=20))
        for cycle in range(1,11):
            phase('cycle-%04d-probe-two-epochs'%cycle,'reproduction.actor50_prefix20_reuse20.train',
                  '--config',str(base/'preparation/config.json'),'--cycle',str(cycle))
            latest=read(base/'probe/latest.json')
            assert latest['cycle']==cycle and latest['completed_epochs']==2*cycle
            assert latest['actual_adam_steps_per_rank']==[latest['ema_updates']]*4
        latest=read(base/'probe/latest.json');baseline=read(Path(c['source_prefix5_base'])/'training/cycle-0010/complete.json')
        assert latest['completed_epochs']==20 and latest['ema_updates']==baseline['ema_updates']==276
        assert sha(latest['checkpoint'])==latest['sha256']
        publish(base/'complete.json',dict(role='frozen_actor50_prefix20_reuse_matched20_complete',run_id=c['run_id'],
            completed_epochs=20,complete_cycles=10,actual_Adam_per_rank=latest['actual_adam_steps_per_rank'],
            EMA_updates=latest['ema_updates'],latest_probe=latest,actor_updates=0,new_rollouts=0,
            prefix5_baseline=baseline,checked_unix=time.time()))
        success=True;status('PREFIX20_COMPLETE20',completed_epochs=20)
    except BaseException as error:
        publish(base/'logs/failure.json',dict(error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checked_unix=time.time()))
        raise
    finally:
        if owned:
            owned.stop_all();owned.reap();present=birth_states([v for job in owned.jobs.values() for v in job['ledger'].values()])
        else:present=[]
        publish(base/'logs/cleanup-status.json',dict(cleaned=not any(x['state']!='Z' for x in present),owned_births_all_gone=not present,remaining_owned_birth_count=len(present),checked_unix=time.time()))
        if lease:lease.close()
        for s,handler in old.items():signal.signal(s,handler)
        publish(base/'logs/exit.json',dict(success=success,checked_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);run(p.parse_args().base)
