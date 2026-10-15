"""Qualify an already completed balanced20 result, on CPU only.

No generation, model inference, training or actor restart occurs. Preserve all
remote inputs and checkpoints; return only scalar/hash and owned-release proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):return json.loads(path.read_text())


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(8*1024**2),b''):value.update(raw)
    return value.hexdigest()


def audit(base):
    assert base.resolve()==base and re.fullmatch('/data/shaoshitong/Search-G1-qprobe-prefix-correctness-balanced20-[0-9a-f]{32}',str(base))
    intent=read(base/'intent.json');out=base/'warmup';ready=read(out/'ready.json');exit=read(base/'logs/exit.json')
    assert intent['balanced_sampling'] is True and exit['exit_code']==0 and exit['head_adam_updates']==2000
    assert ready['actual_head_adam_updates']==2000 and ready['completed_epochs']==20 and ready['epoch_records']==21
    assert ready['minibatch_records']==2000 and ready['final_epoch']==20 and ready['balanced_sampling'] is True
    assert ready['per_class_per_batch']==128 and ready['replace'] is True
    assert ready['actor_updated'] is False and ready['formal_training_started'] is False
    for name,value in intent['operator_sha256'].items():assert sha(base/name)==value
    epochs=[read(out/('epoch-%02d.json'%epoch)) for epoch in range(21)]
    for epoch,row in enumerate(epochs):
        assert row['epoch']==epoch and row['actual_head_adam_updates']==epoch*100
        assert row['train']['samples']==25600 and row['test']['samples']==2560
        assert row['train']['questions']==5120 and row['test']['questions']==512
        if epoch:assert row['sampling']['sampled_positive']==row['sampling']['sampled_negative']==12800
    minis=[json.loads(line) for line in (out/'minibatch-metrics.jsonl').read_text().splitlines() if line]
    assert len(minis)==2000
    for ordinal,row in enumerate(minis,1):
        assert row['head_update_ordinal']==row['head_adam_steps_after']==ordinal and row['head_adam_steps_before']==ordinal-1
        assert row['epoch']==(ordinal-1)//100+1 and row['minibatch_index']==(ordinal-1)%100
        assert row['sampled_positive']==row['sampled_negative']==128 and row['batch_size']==256
    final=Path(ready['final_checkpoint']['path'])
    assert final==out/'head-epoch-20.pt' and sha(final)==ready['final_checkpoint']['sha256']
    assert sha(out/'metrics.json')==ready['metrics_sha256']
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    assert not torch.cuda.is_initialized()
    state=torch.load(final,map_location='cpu',weights_only=False)
    assert state['epoch']==20 and state['actual_head_adam_updates']==2000
    assert state['actor_updated'] is False and state['formal_training_started'] is False
    assert state['config']==ready['binding']['config'] and state['config']['outputs']==1
    assert state['binding']['fit_source_sha256']==intent['operator_sha256']['qprobe_prefix_correctness_fit.py']
    assert state['binding']['cache_manifest_sha256']==intent['feature_manifest_sha256']
    assert state['model']['output.weight'].shape==(1,247*2048) and state['model']['output.bias'].shape==(1,)
    tensors=[]
    def walk(value):
        if isinstance(value,torch.Tensor):
            assert value.device.type=='cpu' and torch.isfinite(value).all();tensors.append(value)
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,(list,tuple)):
            for item in value:walk(item)
    walk(state)
    for item in state['optimizer']['state'].values():assert int(item['step'].item())==2000
    assert not torch.cuda.is_initialized()
    sid=read(base/'logs/launch.json')['process']['sid'];survivors=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            text=(path/'stat').read_text();values=text[text.rfind(')')+2:].split()
            if int(values[3])==sid and values[0] not in ('Z','X'):survivors.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    assert survivors==[], 'Owned fit session still releasing; inspect the same handle later'
    result=dict(role='balanced_prefix20_actual_final_cpu_qualification',checked_unix=time.time(),
        source_commit=intent['operator_commit'],operator_sha256=intent['operator_sha256'],
        head_adam_updates=2000,complete_epochs=20,epoch_evaluations=21,mini_batches=2000,
        all_batches_exact128_correct128_wrong=True,final_checkpoint_sha256=sha(final),
        final_checkpoint_bytes=final.stat().st_size,model_tensor_count=len(state['model']),
        all_checkpoint_tensors_finite=True,checked_tensor_count=len(tensors),
        optimizer_state_count=len(state['optimizer']['state']),all_saved_adam_steps=2000,
        fixed_test_questions=512,fixed_test_trajectories=2560,final_epoch=20,
        epoch0_metrics=epochs[0],final_metrics=epochs[-1],
        best_test_balanced_accuracy_epoch_diagnostic_only=max(epochs,key=lambda x:(x['test']['balanced_accuracy'],-x['epoch']))['epoch'],
        owned_live_session_survivors=survivors,actor_training_started=False,CUDA_initialized=False,
        feature_manifest_sha256=intent['feature_manifest_sha256'],metrics_sha256=sha(out/'metrics.json'))
    path=out/'final-cpu-qualified.json'
    with path.open('x') as stream:json.dump(result,stream,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.base),allow_nan=False))
