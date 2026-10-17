"""Matched local50 ablation: retain question attention, read only policy outputs.

This entry never generates trajectories or starts an actor/retriever. It reuses
the completed fixed data and starts from the same original joint50 probe state.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from reproduction.prefix20_online.launch import once, publish, read, sha


def policy_source_state(saved, model):
    """Slice only removed Linear columns and their moments; keep Adam clock."""
    names = [name for name, _ in model.named_parameters()]
    ids = [pid for group in saved['optimizer']['param_groups'] for pid in group['params']]
    assert len(names) == len(ids) == len(saved['model']) == 14
    weight_id = dict(zip(names, ids))['output.weight']
    width = model.output.in_features
    old = saved['model']['output.weight']
    assert old.shape == (1,293*2048) and width == 20*2048
    result = dict(saved, model=dict(saved['model']), optimizer=dict(saved['optimizer']))
    result['model']['output.weight'] = old[:, -width:].contiguous()
    result['optimizer']['state'] = dict(saved['optimizer']['state'])
    moment = dict(saved['optimizer']['state'][weight_id])
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if key in moment:
            assert moment[key].shape == old.shape
            moment[key] = moment[key][:, -width:].contiguous()
    result['optimizer']['state'][weight_id] = moment
    return result


# Keep torch imports out of the deployment/controller process.
def probe_class():
    import torch
    from torch import nn
    from .probe import OnlineQuestionProbe, validate_online_layout, QMAX, WIDTH, DIM
    from reproduction.prefix20_question_replay.model import masked_flatten

    class PolicyOnly(OnlineQuestionProbe):
        def __init__(self, model_dir, question_max_tokens):
            super().__init__(model_dir, question_max_tokens)
            self.output = nn.Linear(20*DIM,1)

        def forward(self, batch):
            validate_online_layout(batch, self.max_positions)
            features, mask, positions = batch['features'].detach(), batch['attention_mask'], batch['position_ids']
            kwargs = dict(attention_mask=mask, position_ids=positions,
                position_embeddings=self.rotary(features,positions),use_cache=False,output_attentions=False)
            if self._cache_position:
                kwargs['cache_position'] = torch.arange(WIDTH,device=features.device)
            hidden = self.block(features,**kwargs)
            if isinstance(hidden,tuple):hidden=hidden[0]
            # Question participates in attention, but has no direct Linear path.
            return self.output(masked_flatten(hidden[:,QMAX:],batch['policy_mask'][:,QMAX:])).squeeze(-1).float()
    return PolicyOnly


def PolicyOnlyReadoutProbe(*args, **kwargs):
    return probe_class()(*args, **kwargs)


def run(base):
    from .local_probe_fit50 import admit_local_allocation
    from reproduction.prefix_only_correctness_ab.launch import torchrun_owned_class, DeferredInterrupt, birth_states
    from reproduction.h1008_qprobe.pipeline import UUIDLocks
    base=Path(base);c=read(base/'preparation/launch-config.json');spec=c['local_probe_fit50']
    assert spec['policy_only_readout'] is True and spec['epochs']==10 and spec['learning_rate']==1e-5
    source=Path(spec['fixed_data_source_base'])
    assert source!=base and read(source/'train/complete.json')['actual_adam_per_rank']==[1200]*8
    admit_local_allocation(base,c);lease=UUIDLocks(c['physical_gpu_uuids'])
    owned=torchrun_owned_class()(base/'logs/process-ledger.json',c['run_id'])
    interrupt=DeferredInterrupt();handlers={s:signal.getsignal(s)for s in(signal.SIGTERM,signal.SIGINT)}
    for sig in handlers:signal.signal(sig,interrupt.handler)
    env=dict(os.environ,PYTHONPATH=str(base/'repo'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
        OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        NCCL_NVLS_ENABLE='0',HF_HOME=str(base/'cache/hf'),XDG_CACHE_HOME=str(base/'cache'),
        TRITON_CACHE_DIR=str(base/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(base/'cache/inductor'),
        TMPDIR=str(base/'tmp'),PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost')
    if c.get('python_header_cpath'):env['CPATH']=c['python_header_cpath']
    for key in ('PYTHONHOME','RAY_ADDRESS','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):env.pop(key,None)
    for name in ('tmp','cache/hf','cache/triton','cache/inductor'):(base/name).mkdir(parents=True,exist_ok=True)
    python=str(Path(c['environment_root'])/'bin/python')
    try:
        publish(base/'logs/pipeline-status.json',dict(stage='probe-policy-only-readout-ten-epochs',checked_unix=time.time(),actor_version=spec['actor_version']))
        command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m',
            'reproduction.prefix_selection200.local_probe_fit50','train','--base',str(base)]
        p=owned.start('policy-only-readout-train',command,env,str(base/'logs/policy-only-readout-train.log'),str(base/'repo'))
        while p.poll()is None:
            owned.observe('policy-only-readout-train');interrupt.check();time.sleep(1)
        owned.observe('policy-only-readout-train');owned.jobs['policy-only-readout-train']['exit_code']=p.returncode;owned.save();owned.stop('policy-only-readout-train')
        if p.returncode:raise RuntimeError('Actual policy-only readout training failed')
        once(base/'logs/pipeline-complete.json',dict(status='LOCAL_PROBE50_POLICY_ONLY_READOUT_TEN_EPOCHS_COMPLETE',
            actor_updated=False,formal200_resumed=False,train_complete=read(base/'train/complete.json')))
    finally:
        try:
            owned.stop_all();owned.reap();births=[v for j in owned.jobs.values()for v in j['ledger'].values()];present=birth_states(births)
            once(base/'logs/cleanup-status.json',dict(owned_births_all_gone=not present,remaining_owned_births=present,checked_unix=time.time()))
        finally:
            lease.close()
            for sig,handler in handlers.items():signal.signal(sig,handler)


def submit(base):
    base=Path(base);prep=base/'preparation'
    if (prep/'submit-intent.json').exists():
        return dict(status='EXISTING_INTENT_NO_REPLAY',receipt=read(prep/'submit-receipt.json')if(prep/'submit-receipt.json').exists()else None)
    c=read(prep/'launch-config.json');spec=c['local_probe_fit50'];assert spec['policy_only_readout'] is True
    from .local_probe_fit50 import SHARED_AUTHORIZATION
    assert spec['shared_slurm_physical8_authorization']==SHARED_AUTHORIZATION
    script=prep/'native-local-fit50-policy-readout.sh';python=str(Path(c['environment_root'])/'bin/python')
    script.write_text('#!/bin/bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexport PYTHONPATH='+shlex.quote(str(base/'repo'))+'\nexec '+shlex.join([python,'-m','reproduction.prefix_selection200.local_probe_policy_readout50','run','--base',str(base)])+'\n');script.chmod(0o700)
    command=['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1',
        '--cpus-per-task=96','--mem=512G','--time=0','--signal=B:TERM@60','--export=NONE',
        '--job-name=g1-policyreadout-'+spec['experiment_id'][:12],'--comment=g1-local50-policy-only:'+spec['experiment_id'],
        '--chdir='+str(base),'--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(script)]
    once(prep/'submit-intent.json',dict(command=command,config_sha256=sha(prep/'launch-config.json'),created_unix=time.time()))
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=30);job=r.stdout.strip().split(';')[0]
        value=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,status='SUBMITTED'if r.returncode==0 and job.isdecimal()else'UNKNOWN_NO_REPLAY')
        if value['status']=='SUBMITTED':value['job_id']=job
    except subprocess.TimeoutExpired as error:value=dict(status='UNKNOWN_NO_REPLAY',error=repr(error))
    once(prep/'submit-receipt.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','submit'));p.add_argument('--base',required=True);a=p.parse_args();print(json.dumps(globals()[a.phase](a.base)))
