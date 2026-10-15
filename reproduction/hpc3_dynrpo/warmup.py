"""Fresh native block warmup, with exact private-RNG minibatch continuation."""
from dataclasses import asdict
from pathlib import Path
import os
import time

from .common import BASE, MODEL, own, read, sha, publish, progress, teacher_labels, normalize_preparation_gpu


def save_tensor(path, value):
    import torch
    path = Path(path)
    if not path.is_relative_to(BASE) or path.is_symlink():
        raise ValueError('Tensor output must remain in the own remote namespace')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def fit_continuation(engine, provider, lengths, counts, ids, history, callback=None):
    """The native fit objective/order, preserving accepted Adam and RNG on retry.

    Tests compare this full fit and an interrupted continuation against native
    QwenBlockWarmup.fit, including every parameter, Adam tensor, and private RNG.
    """
    import torch
    from search_g1.qprobe_token_model import token_binary_loss
    lengths, counts = torch.as_tensor(lengths,dtype=torch.long), torch.as_tensor(counts,dtype=torch.long)
    epochs, minis = history['epoch_history'], history['minibatch_history']
    batches = (len(ids)+engine.config.batch_size-1)//engine.config.batch_size
    if engine.update_count != len(minis) or engine._adam_steps() != engine.update_count:
        raise ValueError('Saved warmup Adam/history counters differ; cannot silently roll back')
    def emit(stage, row):
        if callback: callback(dict(stage=stage,**row))
    if not epochs:
        if engine.update_count: raise ValueError('Missing epoch0 warmup history')
        epochs.append(engine._summary(provider,lengths,counts,ids,0)); emit('epoch_training_summary',epochs[-1])
    reconstruction = torch.Generator(device='cpu').manual_seed(engine.config.seed+1)
    engine.model.train()
    for epoch in range(1,engine.config.epochs+1):
        prior = reconstruction.get_state().clone()
        order = torch.randperm(len(ids),generator=reconstruction)
        after = reconstruction.get_state().clone()
        lo,hi = (epoch-1)*batches,epoch*batches
        if engine.update_count >= hi:
            if len(epochs) == epoch:
                epochs.append(engine._summary(provider,lengths,counts,ids,epoch)); emit('epoch_training_summary',epochs[-1])
            continue
        if engine.update_count < lo:
            raise ValueError('Warmup epoch counter gap')
        offset = engine.update_count-lo
        actual = engine.rng.get_state()
        if offset == 0 and torch.equal(actual,prior):
            actual_order = torch.randperm(len(ids),generator=engine.rng)
            if not torch.equal(actual_order,order): raise ValueError('Warmup private permutation changed')
        elif not torch.equal(actual,after):
            raise ValueError('Saved private RNG cannot prove the current minibatch order')
        for index_number,index in enumerate(order.split(engine.config.batch_size)):
            if index_number < offset: continue
            engine.optimizer.zero_grad(set_to_none=True)
            values = dict(pre_step_loss=0.,binary_bce=0.,regression_mse_raw=0.,regression_mse_normalized=0.,regression_contribution=0.)
            correct = 0
            for micro in index.split(engine.config.micro_batch_size):
                features,sizes = engine._batch(provider,micro,lengths,ids)
                prediction = engine.model(features,sizes)
                loss,parts = token_binary_loss(prediction,counts[micro],engine.config)
                fraction = len(micro)/len(index)
                (loss*fraction).backward()
                for key,value in dict(pre_step_loss=loss,**parts).items(): values[key] += float(value.detach())*fraction
                truth = ((counts[micro]>0)&(counts[micro]<5)).to(engine.device)
                correct += int(((prediction['mixed_probability'].detach()>=.5)==truth).sum())
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in engine.model.parameters()):
                raise ValueError('Nonfinite native warmup gradient; retain the true current state')
            row = dict(epoch=epoch,minibatch_index=index_number,batch_size=len(index),micro_batch_size=engine.config.micro_batch_size,
                head_update_ordinal=engine.update_count+1,head_adam_steps_before=engine._adam_steps(),
                learning_rate=float(engine.optimizer.param_groups[0]['lr']),binary_accuracy=correct/len(index),accuracy=correct/len(index),**values)
            engine.optimizer.step(); engine.update_count += 1
            if any(not torch.isfinite(p).all() for p in engine.model.parameters()):
                raise ValueError('Nonfinite native warmup parameter; no automatic rollback')
            row['head_adam_steps_after'] = engine._adam_steps(); minis.append(row); emit('head_optimizer_update',row)
        epochs.append(engine._summary(provider,lengths,counts,ids,epoch)); emit('epoch_training_summary',epochs[-1])
    engine.fitted = True
    return dict(fit_epochs=engine.config.epochs,training_groups=len(ids),actual_head_adam_updates=engine._adam_steps(),
        expected_head_adam_updates=engine.config.epochs*batches,epoch_history=epochs,minibatch_history=minis,
        training_summary=epochs[-1],primary_report='binary_accuracy',dataset_scope='training_only_not_independent_holdout',
        initial_block_proof=engine.initial_block_proof,native_implementation=engine.native_implementation,
        actor_updated=False,reference_updated=False,formal_training_started=False)


def main():
    own(BASE); gpu = normalize_preparation_gpu()
    import torch
    from safetensors import safe_open
    from search_g1.qprobe_token_features import TokenFeatureCache
    from search_g1.qprobe_qwen_block_model import QwenBlockQProbeConfig,QwenBlockWarmup
    ready_path = BASE/'warmup/ready.json'
    if ready_path.is_file():
        accepted = read(ready_path)
        if sha(BASE/'warmup/fresh-head.pt') != accepted['head_sha256']:
            raise ValueError('Accepted warmup changed')
        progress('WARMUP_ALREADY_ACCEPTED',head_adam_steps=200); return
    feature = read(BASE/'preparation/warmup-features-ready.json')
    cache = TokenFeatureCache(BASE/'warmup-token-features',expected_spec_sha=feature['spec_sha'])
    if sha(cache.path/'manifest.json') != feature['manifest_sha256']:
        raise ValueError('Warmup input bank changed')
    labels,provenance = teacher_labels(); qids = [g['qid'] for g in labels]
    if any(cache.index[g['qid']]['question_hash'] != g['question_hash'] for g in labels):
        raise ValueError('Teacher labels and all-token features differ')
    index = read(MODEL/'model.safetensors.index.json')['weight_map']
    keys = sorted(k for k in index if k.startswith('model.layers.35.'))
    if len(keys) != 12: raise ValueError('Official layer35 must have exactly twelve tensors')
    state = {}
    for shard in sorted({index[k] for k in keys}):
        with safe_open(MODEL/shard,framework='pt',device='cpu') as f:
            for k in keys:
                if index[k] == shard: state[k] = f.get_tensor(k)
    engine = QwenBlockWarmup(QwenBlockQProbeConfig(),state,device='cuda:0')
    del state
    history = dict(epoch_history=[],minibatch_history=[])
    checkpoint = BASE/'warmup/latest-state.pt'
    if checkpoint.is_file():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['token_feature_spec_sha256'] != cache.spec_sha or saved['label_producer_provenance'] != provenance:
            raise ValueError('Warmup continuation namespace changed')
        h = saved['head']
        if h['config'] != asdict(engine.config) or h['native_implementation'] != engine.native_implementation:
            raise ValueError('Saved native model/config changed')
        engine.model.load_state_dict(h['model'],strict=True); engine.optimizer.load_state_dict(h['optimizer'])
        engine.rng.set_state(h['private_rng']); engine.update_count=h['update_count']; engine.fitted=h['fitted']
        engine._initial_block_proof=h['initial_block_proof']; engine.initial_readout_proof=h['initial_readout_proof']
        history=saved['history']
    def payload():
        return dict(token_feature_spec_sha256=cache.spec_sha,label_producer_provenance=provenance,
                    head=engine.state_dict(),history=history)
    def callback(event):
        stage=event['stage']; progress('NATIVE_QWEN_WARMUP',**{k:v for k,v in event.items() if k!='stage'},warmup_event=stage)
        if stage=='epoch_training_summary': save_tensor(checkpoint,payload())
    progress('FRESH_NATIVE_BLOCK_WARMUP_START',gpu_uuid=gpu,head_adam_steps=engine._adam_steps(),questions=5120,epochs=10)
    try:
        metrics=fit_continuation(engine,cache,[cache.index[q]['length'] for q in qids],
                                [g['correct_count'] for g in labels],qids,history,callback)
    except BaseException as error:
        save_tensor(checkpoint,payload())
        publish(BASE/'warmup/failure.json',dict(error=repr(error),actual_adam_steps=engine._adam_steps(),
                actual_update_count=engine.update_count,checkpoint_sha256=sha(checkpoint),checked_unix=time.time()))
        raise
    if engine.update_count != 200 or engine._adam_steps() != 200 or len(metrics['epoch_history'])!=11 or len(metrics['minibatch_history'])!=200:
        raise ValueError('Native warmup10 did not finish all actual Adam updates')
    save_tensor(BASE/'warmup/fresh-head.pt',payload())
    publish(BASE/'warmup/warmup-metrics.json',metrics,once=True)
    block_proof=dict(native_class='transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer',block_index=35,
        official_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',source_tensors_verified=True,
        official_block_copy_verified=True,head_initialized=True,fresh_linear=True,fresh_adam=True,
        transformers_version=engine.native_implementation['transformers_version'])
    ready=dict(stage='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE',source_commit=os.environ['SEARCH_G1_SOURCE_COMMIT'],
        head_sha256=sha(BASE/'warmup/fresh-head.pt'),metrics_sha256=sha(BASE/'warmup/warmup-metrics.json'),
        token_feature_spec_sha256=cache.spec_sha,token_feature_manifest_sha256=feature['manifest_sha256'],
        questions=5120,teacher_trajectories_reused=25600,trajectories_generated=0,head_optimizer_updates=200,head_adam_steps=200,
        epoch_records=11,minibatch_records=200,native_block_proof=block_proof,initial_block_proof=engine.initial_block_proof,
        initial_readout_proof=engine.initial_readout_proof,native_implementation=engine.native_implementation,
        label_producer_provenance=provenance,training_summary=metrics['training_summary'],formal_training_started=False,
        actor_updated=False,independent_validation=False,checked_unix=time.time())
    publish(ready_path,ready,once=True)
    progress('NATIVE_QWEN_WARMUP10_ACCEPTED',head_adam_steps=200,training_accuracy=metrics['training_summary']['binary_accuracy'])


if __name__=='__main__':
    main()
