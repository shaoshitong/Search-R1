"""Native-only rolling state plus small, immediately flushed completion evidence."""
import json
import math
import os
from pathlib import Path
import shutil
import time

from . import EVALUATION_STEPS
from .contracts import read_receipt, sha


def replace_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending-' + str(os.getpid()))
    pending.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    os.replace(pending, path)


def save_rolling(trainer, step):
    from search_r1.baselines.training_state import atomic_save, digest, driver_snapshot
    from search_r1.baselines.tree_guard_continue import accounting
    cfg = trainer.config
    from search_r1.baselines.execution_identity import execution_id
    identity = dict(job_id=execution_id(), code_commit=os.environ['SEARCH_R1_BASELINE_COMMIT'],
        config_sha256=os.environ['SEARCH_R1_BASELINE_CONFIG_SHA256'],
        data_identity=os.environ['SEARCH_R1_BASELINE_DATA_IDENTITY'],
        model_identity=os.environ['SEARCH_R1_MODEL_IDENTITY'], completed_step=step,
        actual_update_limit=150, guard_continue_accounting=accounting(trainer))
    directory = Path(cfg.trainer.default_local_dir) / 'native-rolling' / f'global_step_{step}'
    print('ABLATION_ROLLING_STATE_BEGIN=' + json.dumps(dict(outer=step, time=time.time())), flush=True)
    started = time.monotonic()
    workers = trainer.actor_rollout_wg.save_training_state(str(directory), identity, verify=False)
    if len(workers) != 8 or {r['identity']['rank'] for r in workers} != set(range(8)):
        raise ValueError('Rolling snapshot has incomplete worker coverage')
    saved = driver_snapshot(trainer, identity)
    driver_path = directory / 'driver.pt'
    driver_sha = atomic_save(saved, driver_path)
    value = dict(status='COMPLETE_SAME_WORLD_STATE_SAVED', identity=identity,
        worker_states=workers, driver=dict(path=str(driver_path), sha256=driver_sha,
            logical_state_sha256=digest(saved), next_step=step+1, exact_reload_verified=False),
        hf_checkpoint=None, hf_identity=None, snapshot_kind='native_only_rolling_two',
        limits='Complete native parameters/Adam/scheduler/RNG/cursor; no redundant per-round HF checkpoint')
    path = directory / 'manifest.json'
    manifest_sha = atomic_save(value, path, json_format=True)
    # Only this task's own accepted native-only cache; sparse HF keypoints stay.
    for old in directory.parent.glob('global_step_*'):
        old_step = int(old.name.removeprefix('global_step_'))
        if old_step >= step-1 or not (old / 'manifest.json').is_file():
            continue
        record = json.loads((old / 'manifest.json').read_text())
        if (record.get('snapshot_kind') == 'native_only_rolling_two'
                and record['identity']['config_sha256'] == identity['config_sha256']):
            shutil.rmtree(old)
    elapsed = time.monotonic()-started
    print('ABLATION_ROLLING_STATE_END=' + json.dumps(dict(outer=step, seconds=elapsed, time=time.time())), flush=True)
    return dict(path=str(path), sha256=manifest_sha, native_only=True, save_seconds=elapsed)


def completed_round(trainer, metrics, step):
    if type(step) is not int or not 1 <= step <= 150:
        raise ValueError('Invalid actual completed round')
    receipt = read_receipt(os.environ['RECURSIVE_ABLATION_RECEIPT'])
    small = {key: value for key, value in metrics.items()
             if key.startswith(('actor/', 'training/', 'timing_s/', 'val/', 'reward/'))
             and isinstance(value, (int, float))}
    if any(not math.isfinite(value) for value in small.values()):
        raise ValueError('Completed-round scalar is nonfinite')
    if step in EVALUATION_STEPS:
        path = Path(trainer.config.trainer.default_local_dir) / 'training_state' / f'global_step_{step}' / 'post-evaluation' / 'manifest.json'
        if not path.is_file():
            raise ValueError('Keypoint must have its actual completed scheduled evaluation state')
        checkpoint = dict(path=str(path), sha256=sha(path), native_only=False)
    else:
        checkpoint = save_rolling(trainer, step)
    record = dict(time=time.time(), group=receipt['group'], run_id=receipt['run_id'],
        completed_outer=step, next_outer=step+1, config_sha256=receipt['config_sha256'],
        source_commit=receipt['source_commit'], checkpoint=checkpoint, metrics=small,
        optimizer_calls_per_rank=dict(trainer.completed_optimizer_calls_per_rank),
        consumed_sampling_rounds=trainer.actual_outer_updates,
        zero_write_sampling_rounds=getattr(trainer, 'zero_write_sampling_rounds', 0))
    output = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    with (output / 'completed-outers.jsonl').open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
        handle.flush()
    replace_json(output / 'progress.json', record)
    print('ABLATION_COMPLETE_OUTER=' + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record


def finished(trainer):
    receipt = read_receipt(os.environ['RECURSIVE_ABLATION_RECEIPT'])
    output = Path(os.environ['SEARCH_R1_BASELINE_OUTPUT'])
    progress = json.loads((output / 'progress.json').read_text())
    if progress['completed_outer'] != 150 or not hasattr(trainer, '_ablation_dev150'):
        raise ValueError('Missing actual completed150 and scheduled150dev')
    result = dict(status='COMPLETE_NATIVE_150_TRAINING_AND_FOUR_SCHEDULED_DEV',
        time=time.time(), run_id=receipt['run_id'], group=receipt['group'],
        completed_outer=150, evaluation_steps=list(EVALUATION_STEPS),
        actual_optimizer_calls_per_rank=dict(trainer.completed_optimizer_calls_per_rank),
        final_dev_metrics=trainer._ablation_dev150, checkpoint=progress['checkpoint'],
        config_sha256=receipt['config_sha256'], source_commit=receipt['source_commit'],
        actual_process_exit_verified=False)
    from search_r1.baselines.training_state import atomic_save
    atomic_save(result, output / 'training-complete.json', json_format=True)
    print('ABLATION_NATIVE_COMPLETE=' + json.dumps(result, sort_keys=True), flush=True)
    return result


def read(base):
    base = Path(base)
    start = json.loads((base / 'receipts' / 'current-entry.json').read_text())
    process = Path('/proc') / str(start['pid'])
    observed = None
    if process.exists():
        text = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        observed = dict(pid=start['pid'], uid=process.stat().st_uid, tick=int(text[19]), state=text[0])
        if observed['tick'] != start['tick'] or observed['uid'] != start['uid']:
            observed = dict(identity_mismatch=True, **observed)
    def optional(path):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
    output = Path(start['output'])
    return dict(time=time.time(), entry=start, process=observed,
        progress=optional(output / 'progress.json'),
        complete=optional(output / 'training-complete.json'),
        exit=optional(output / 'entry-exit.json'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    print(json.dumps(read(parser.parse_args().base), sort_keys=True), flush=True)
