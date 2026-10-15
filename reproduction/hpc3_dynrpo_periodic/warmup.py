"""Continue the accepted ten epochs to twenty; no student block copying."""
import copy
import os
from pathlib import Path
import signal
import time

from .common import BASE, own, read, sha, publish, progress, normalize_preparation_gpu

PREVIOUS = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-native-thread5-20261015')


def save(path, value):
    import torch
    if not Path(path).is_relative_to(BASE) or Path(path).is_symlink():
        raise ValueError('Periodic state must remain in its independent namespace')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    own(BASE)
    normalize_preparation_gpu()
    import torch
    from reproduction.hpc3_dynrpo.warmup import fit_continuation
    from search_g1.qprobe_periodic_model import restore_engine
    from search_g1.qprobe_token_features import TokenFeatureCache
    from reproduction.hpc3_dynrpo.common import teacher_labels
    ready = BASE/'warmup/ready.json'
    if ready.exists():
        accepted = read(ready)
        if sha(BASE/'warmup/continued-head.pt') != accepted['head_sha256']:
            raise ValueError('Accepted twenty-epoch head changed')
        progress('CONTINUOUS_WARMUP20_ALREADY_ACCEPTED', head_adam_steps=400)
        return
    old_ready = read(PREVIOUS/'warmup/ready.json')
    old_raw = PREVIOUS/'warmup/fresh-head.pt'
    if old_ready['head_adam_steps'] != 200 or sha(old_raw) != old_ready['head_sha256']:
        raise ValueError('Previous accepted real ten-epoch state changed')
    cache = TokenFeatureCache(PREVIOUS/'warmup-token-features',
                             expected_spec_sha=old_ready['token_feature_spec_sha256'])
    if sha(cache.path/'manifest.json') != old_ready['token_feature_manifest_sha256']:
        raise ValueError('Initial fixed feature bank changed')
    labels, provenance = teacher_labels()
    if provenance != old_ready['label_producer_provenance']:
        raise ValueError('Initial frozen student rollout labels changed')
    target = BASE/'warmup/latest-state.pt'
    source = target if target.exists() else old_raw
    saved = torch.load(source, map_location='cpu', weights_only=True)
    engine = restore_engine(saved['head'], device='cuda:0', epochs=20)
    history = copy.deepcopy(saved['history'])
    del saved
    if engine.update_count < 200:
        raise ValueError('Cannot discard the accepted ten warmup epochs')
    def payload():
        return dict(head=engine.state_dict(), history=history,
                    token_feature_spec_sha256=cache.spec_sha,
                    label_producer_provenance=provenance,
                    prior_head_sha256=old_ready['head_sha256'])
    def callback(event):
        progress('CONTINUOUS_NATIVE_WARMUP20', **event)
        if event['stage'] == 'epoch_training_summary':
            save(target, payload())
    def terminate(signum, frame):
        raise KeyboardInterrupt('Slurm signal %d; persist current probe state' % signum)
    signal.signal(signal.SIGTERM, terminate)
    progress('CONTINUOUS_WARMUP20_START', head_adam_steps=engine._adam_steps(),
             inherited_epochs=10, target_epochs=20, student_parameters_copied=False)
    try:
        metrics = fit_continuation(engine, cache,
            [cache.index[g['qid']]['length'] for g in labels],
            [g['correct_count'] for g in labels], [g['qid'] for g in labels], history, callback)
    except BaseException as error:
        save(target, payload())
        publish(BASE/'warmup/failure.json', dict(error=repr(error),
            head_adam_steps=engine._adam_steps(), state_sha256=sha(target), checked_unix=time.time()))
        raise
    if engine._adam_steps() != 400 or len(history['epoch_history']) != 21:
        raise ValueError('Twenty total warmup epochs did not really finish')
    output = BASE/'warmup/continued-head.pt'
    save(output, payload())
    # Read back the actual persisted optimizer, model and private RNG.
    restored = restore_engine(torch.load(output, map_location='cpu', weights_only=True)['head'])
    from search_g1.full_state_restore import equal_tree
    equal_tree(restored.model.state_dict(), {k:v.cpu() for k,v in engine.model.state_dict().items()})
    from search_g1.training_state import cpu_tree
    equal_tree(restored.optimizer.state_dict(), cpu_tree(engine.optimizer.state_dict()))
    if not torch.equal(restored.rng.get_state(), engine.rng.get_state()):
        raise ValueError('Saved probe private RNG differs')
    publish(BASE/'warmup/warmup-metrics.json', metrics, once=True)
    publish(ready, dict(stage='CONTINUOUS_QWEN_WARMUP20_FULL_STATE_READBACK_PASS',
        head_sha256=sha(output), head_adam_steps=400, fit_epochs=20,
        inherited_head_sha256=old_ready['head_sha256'], inherited_adam_steps=200,
        new_adam_updates=200, student_parameters_copied=False, optimizer_reset=False,
        token_feature_spec_sha256=cache.spec_sha,
        token_feature_manifest_sha256=old_ready['token_feature_manifest_sha256'],
        label_producer_provenance=provenance, training_summary=metrics['training_summary'],
        actual_saved_model_adam_rng_readback=True, checked_unix=time.time()), once=True)
    progress('CONTINUOUS_WARMUP20_ACCEPTED', head_adam_steps=400,
             training_accuracy=metrics['training_summary']['binary_accuracy'])


if __name__ == '__main__':
    main()
