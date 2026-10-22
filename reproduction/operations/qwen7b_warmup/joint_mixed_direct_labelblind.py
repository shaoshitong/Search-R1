"""Restore E1 raw+EMA; distinct-question E2 joint and E3 head-only at5e-7."""
import argparse
import fcntl
import gc
import json
import os
import signal
import time
import traceback
from pathlib import Path
import torch
from . import probe_ablation_train as u, mixed_group_checkpoint as original_ck
from . import mixed_group_unique_checkpoint as ck
from . import joint_mixed_direct_b128 as prior
from .labelblind_question_g5_sampler import LabelBlindQuestionG5Sampler
from .direct_update_labelblind import direct_update_labelblind
from .unique_selected_feature_reader import SelectedG5PrefetchReader
from .unique_fixed_train_evaluation import make_fixed_training_groups, evaluate_fixed_training

BASE = prior.source.B/'algorithm-improvement-20261021/mixed-direct-e1-uniform-g5-weighted-20261022-v1'
STOP = False


def configure(model, native, head, phase):
    frozen = phase == 'phase3_head_only'
    model.block.requires_grad_(not frozen)
    if frozen:
        for parameter in model.block.parameters():parameter.grad = None
    for optimizer in (native, head):
        for group in optimizer.param_groups:group['lr'] = 5e-7


def binding(base, config):
    return dict(config_sha256=u.sha(base/'config.json'), matched_binding_sha256=u.digest(config),
                initialization_source='official_initial', source_epoch1_checkpoint=config['source_e1']['source_checkpoint'])


def run(base):
    config = u.read(base/'config.json')
    spec = config['source_e1']
    old = u.read(u.checked(spec['source_config']))
    if (config['mode'] != 'e1_labelblind_g5_each_epoch' or config['epoch_end_cursors'] != [3898, 5219]
        or config['update_budget'] != 5219 or config['batch_questions'] != 128):
        raise ValueError('User all169028 unique-question phased recipe differs')
    for name, digest in config['code_hashes'].items():
        if u.sha(Path(__file__).resolve().parents[3]/name) != digest:
            raise ValueError('Unpublished runtime change ' + name)
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Owned one-GPU formal allocation required')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_:globals().__setitem__('STOP', True))
    u.atomic(base/'controller.json', dict(pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'],
        stage='RESTORING_COMPLETE_E1_RAW_AND_EMA', source_cursor=2577, time=time.time()))
    groups, blocked, hashes = prior.source.own_groups(old)
    lookup = {group['qid']:group for group in groups}
    bind = binding(base, config)
    model, ema, native, head, _ = prior.source.initialize(old, device)
    prior.source.warm_gradient_buckets(model)
    sampling = dict(seed=config.get('unique_sampling_seed', 2026102204), source_cursor=2577,
                    epoch_end_cursors=(3898, 5219), forbidden_qids=blocked, forbidden_hashes=hashes)
    pool_path = base/'fixed-training-pool.json'
    if (base/'latest.json').exists():
        saved = torch.load(u.checked(u.read(base/'latest.json')), map_location='cpu', mmap=True, weights_only=False)
        sampler = LabelBlindQuestionG5Sampler(groups, rng_state=saved['sampler_state']['rng_state'], **sampling)
        cursor, seen = ck.restore(saved, model, ema, native, head, device, sampler, bind)
        phase_state = dict(saved['phase_state'])
        provenance = saved['official_initialization']
        if not pool_path.exists():raise ValueError('Accepted fixed training evaluation pool missing on resume')
    else:
        saved = torch.load(u.checked(spec['source_checkpoint']), map_location='cpu', mmap=True, weights_only=False)
        old_sampler = prior.B128Sampler(groups, old, blocked, hashes)
        cursor, seen = original_ck.restore(saved, model, ema, native, head, device, old_sampler, prior.binding(old))
        if cursor != 2577:raise ValueError('Exact first-epoch raw/EMA/twoAdam source required')
        provenance = saved['official_initialization']
        sampler = LabelBlindQuestionG5Sampler(groups, rng_state=old_sampler.rng.getstate(), **sampling)
        phase_state = dict(training_phase='phase2_joint_low_lr', native_freeze_cursor=3898,
            native_frozen_boundary_sha256=None, source_epoch1_checkpoint=spec['source_checkpoint'])
        fixed_groups, fixed_proof = make_fixed_training_groups(groups, seen, seed=2026102201, count=2048)
        if pool_path.exists() and u.read(pool_path) != dict(groups=fixed_groups, proof=fixed_proof):
            raise ValueError('Fixed training evaluation pool changed')
        u.atomic(pool_path, dict(groups=fixed_groups, proof=fixed_proof))
        u.atomic(base/'source-restore-complete.json', dict(
            status='COMPLETE_E1_RAW_EMA_TWO_ADAM_RNG_RESTORED_NEW_UNIQUE_SAMPLER',
            source_checkpoint=spec['source_checkpoint'], source_cursor=cursor,
            source_accepted_hashes=saved['accepted_hashes'], seen_questions=len(seen),
            raw_replaced_by_EMA=False, raw_EMA_twoAdam_global_RNG_preserved=True,
            inherited_sampler_RNG=True, sampler_policy_intentionally_changed=True,
            previous_hard_pool_not_replayed=True, unique_sampling_seed=sampling['seed'],
            new_sampler_binding=sampler.binding, next_native_lr=5e-7, next_head_lr=5e-7,
            epoch_time=time.time()))
        del old_sampler
    fixed = u.read(pool_path)
    del saved
    gc.collect()
    configure(model, native, head, phase_state['training_phase'])
    inherited = u.read(u.checked(spec['reader_integrity']))
    seal = {key:inherited[key] for key in ('source_config', 'source_checkpoint', 'restoration_receipt')}
    reader = SelectedG5PrefetchReader.from_inherited_seal(u.read(u.checked(config['features'])), 26, seal)
    reader.proof.update(immediate_restored_checkpoint=spec['source_checkpoint'],
        fetch_policy='only_actual_selected_own_G5', max_prefetched_feature_bytes=8*128*5*293*3584*2)
    u.atomic(base/'reader-integrity.json', reader.proof)
    micro = u.read(base/'progress.json').get('micro_questions', 128) if (base/'progress.json').exists() else 128

    def epoch_due():return {2577:1, 3898:2, 5219:3}.get(cursor)

    def save():
        return ck.save(base, ck.snapshot(model, ema, native, head, device, sampler, cursor,
            bind, provenance, seen, 'evaluation_due' if epoch_due() else 'training',
            phase_state=phase_state), pin=epoch_due() is not None)

    def evaluate_all(receipt, epoch):
        reader.reset_prefetch()
        prior.evaluate(base, config, model, ema, native, head, device, sampler, provenance, receipt, epoch)
        reader.reset_prefetch()
        evaluate_fixed_training(base, config, model, ema, native, head, device, reader,
            fixed['groups'], fixed['proof'], receipt, epoch)
        reader.reset_prefetch()

    def evaluation_due(epoch):
        folder = base/'evaluations'/('epoch-%02d'%epoch)
        training = base/'fixed-training-evaluations'/('epoch-%02d'%epoch)/'complete.json'
        return not (folder/'complete.json').exists() or not training.exists()

    receipt = save()
    try:
        while cursor < 5219 and not STOP:
            epoch = epoch_due()
            if epoch and evaluation_due(epoch):evaluate_all(save(), epoch)
            if cursor >= 3898 and phase_state['training_phase'] != 'phase3_head_only':
                phase_state = dict(phase_state, training_phase='phase3_head_only',
                    native_frozen_boundary_sha256=ck.native_hash(model))
                configure(model, native, head, phase_state['training_phase'])
                u.atomic(base/'phase-transition.json', dict(cursor=cursor, phase_state=phase_state,
                    native_steps=original_ck.optimizer_steps(native.state_dict(), True),
                    head_steps=original_ck.optimizer_steps(head.state_dict(), True),
                    native_lr=5e-7, head_lr=5e-7, raw_native_frozen=True,
                    EMA_full_model_passive_tracking=True, time=time.time()))
            reader.prepare(sampler)
            u.atomic(base/'update-intent.json', dict(cursor=cursor+1, phase=phase_state['training_phase'], state='started', time=time.time()))
            metrics, batch = direct_update_labelblind(model, ema, native, head, reader, lookup,
                sampler, device, config['ema_decay'], micro_questions=micro, phase=phase_state['training_phase'])
            cursor += 1
            micro = metrics['micro_questions']
            seen.update(batch['source_qids'])
            sampler_state = sampler.state_dict()
            row = dict(metrics)
            row.update(cursor=cursor, budget=5219, source_cursor=2577, new_updates=cursor-2577,
                question_visits_since_E1=sum(r['questions'] for r in sampler_state['epoch_coverage_receipts']) + sampler_state['seen_current_epoch']['count'],
                epoch_unique_question_coverage=sampler_state['seen_current_epoch'],
                completed_epoch_coverage=sampler_state['epoch_coverage_receipts'],
                seen_questions=len(seen), time=time.time(), training_phase=phase_state['training_phase'],
                prefetch=dict(reader.prefetch_statistics))
            u.atomic(base/'update-intent.json', dict(cursor=cursor, phase=phase_state['training_phase'], state='accepted_update_complete'))
            u.atomic(base/'progress.json', row)
            with (base/'curve.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n');stream.flush()
            print(json.dumps(row), flush=True)
            if cursor in (2578, 3899) or cursor % 128 == 0 or epoch_due():save()
        receipt = save()
        epoch = epoch_due()
        if epoch and not STOP:evaluate_all(receipt, epoch)
        result = dict(status='INTERRUPTED_AFTER_COMPLETE_ACCEPTED_UPDATE' if STOP else 'E1_LABELBLIND_G5_CLASSWEIGHT_E2_JOINT_E3_HEADONLY_COMPLETE',
            cursor=cursor, native_steps=original_ck.optimizer_steps(native.state_dict(), True),
            head_steps=original_ck.optimizer_steps(head.state_dict(), True), phase_state=phase_state,
            checkpoint=receipt, global_batch_questions=128, final_batch_questions=68,
            source_cursor=2577, seen_questions=len(seen), epoch_coverage=sampler.state_dict()['epoch_coverage_receipts'],
            question_replay_within_epoch=False, final4096_scored=False, actor_updates=0)
        u.atomic(base/('interrupted.json' if STOP else 'complete.json'), result)
        return result
    finally:reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--action', choices=('train',), default='train')
    args = parser.parse_args()
    base = args.base.resolve()
    if base != BASE:parser.error('Independent unique-question E1 continuation namespace required')
    with (base/'training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:print(json.dumps(run(base)), flush=True)
        except BaseException as error:
            u.atomic(base/'failure.json', dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc(), time=time.time()))
            raise


if __name__ == '__main__':main()
