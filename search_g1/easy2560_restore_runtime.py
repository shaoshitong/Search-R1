"""Actual native4 restore readbacks and honest saved-boundary provenance."""
import json
from pathlib import Path
import time


def verified_runtime_prefix(base, run):
    from search_g1.easy2560_restore import KIND, source
    from search_g1.full_state_restore import digest
    base = Path(base)
    binding = run['full_state_restore']
    assert binding['kind'] == KIND
    path, receipt, checkpoint, origin, parent = source(binding['source_receipt'])
    assert digest(path) == binding['source_receipt_sha256']
    workers = json.loads((base/'logs/full-state-restored-workers.json').read_text())['ranks']
    driver = json.loads((base/'logs/full-state-restored-driver.json').read_text())
    allocation = json.loads((base/'logs/native-allocation-admission.json').read_text())
    progress = receipt['attempt_progress']
    outer, writes = receipt['step'], progress['cumulative_writes']
    assert len(workers) == 4 and sorted(row['rank'] for row in workers) == list(range(4))
    assert {row['actual_cuda_uuid'] for row in workers} == set(allocation['actual_cuda_runtime_uuids'])
    assert len({(row['process_identity']['pid'], row['process_identity']['start_ticks']) for row in workers}) == 4
    assert all(row['slurm_job_id'] == allocation['slurm_job_id'] and
        row['source_run_id'] == run['run_id'] == receipt['run_id'] and
        row['source_checkpoint'] == str(checkpoint) and row['source_step'] == outer and
        row['actual_adam_scheduler_readback_equal'] is True and row['actual_rng_readback_equal'] is True and
        row['actual_optimizer_writes'] == writes and row['scheduler_last_epoch'] == outer and
        row['last_attempt_index'] == progress['attempt_index'] for row in workers)
    assert next(row for row in workers if row['rank'] == 0)['model_hf_exact_tensors_checked_on_rank0'] == receipt['hf_saved_tensor_count']
    assert driver['actual_driver_rng_readback_equal'] is True and driver['actual_candidate_stream_readback_equal'] is True
    assert driver['completed_outer'] == outer and driver['attempt_progress'] == progress
    assert driver['actual_candidate_source_cursor'] == receipt['driver_state']['source_consumed']
    assert len({row['policy_rejections'] for row in workers}) == 1
    return dict(completed_outer=outer, writes=writes, attempt_index=progress['attempt_index'],
        rejections=workers[0]['policy_rejections'], aborted=progress['aborted_attempts'],
        used_attempt_ids=receipt['ranks'][0]['used_attempt_ids'],
        source_base=str(origin), source_job=receipt['source_job'], source_commit=parent['source_commit'],
        actual_model_optimizer_scheduler_rng_data_equal=True, current_world_size=4,
        sampling_boundary='saved_checkpoint_before_development', historical_raw_prefix_reconstructed=False)


def record_development_history(base, run):
    """Link the real original QA5 without fabricating before0 or post-QA RNG."""
    from search_g1.easy2560_restore import source
    from search_g1.full_state_restore import digest, publish
    path, receipt, _, origin, parent = source(run['full_state_restore']['source_receipt'])
    history = receipt['historical_artifacts']['development']
    after = history.get('after5')
    measured = None
    if after is not None:
        actual = Path(after['path'])
        assert digest(actual) == after['full_read_sha256']
        measured = json.loads(actual.read_text())
        assert measured['run_id'] == run['run_id'] and measured['step'] == 5
    result = dict(role='easy2560_authenticated_historical_development', run_id=run['run_id'],
        source_base=str(origin), source_job=receipt['source_job'], source_commit=parent['source_commit'],
        source_receipt=str(path), source_receipt_sha256=digest(path), historical_after5=after,
        original_after5=measured, before0_available='before' in history,
        current_evaluation_performed=False, post_evaluation_rng_available=False,
        restored_rng_boundary='actual saved checkpoint before development callback', checked_unix=time.time())
    publish(Path(base)/'logs/restored-development-history.json', result)
    return result


def verify_reconstructed_stream(trainer, stream):
    from search_g1.full_state_restore import equal_tree, publish
    from search_g1.training_state import compact_stream_state, driver_rng
    import os
    equal_tree(compact_stream_state(stream.state()), trainer._restored_candidate_state)
    equal_tree(driver_rng(), trainer._restored_driver_rng)
    row = dict(actual_candidate_stream_readback_equal=True, actual_driver_rng_readback_equal=True,
        completed_outer=trainer.global_steps-1, current_world_size=4, question_probe=None,
        candidate_source_cursor=stream.state()['consumed_source_rows'], checked_unix=time.time())
    publish(Path(os.environ['SEARCH_G1_BASE'])/'logs/easy2560-restored-stream-before-next-outer.json', row)
    return row
