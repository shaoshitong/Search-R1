"""Native4 restoration of a committed actor and its actual callback phase.

The actor model is loaded by the existing FSDP initialisation from the bound HF
checkpoint. This adapter restores its actual local Adam/scheduler and sampling
RNG without an additional full-model gather/forward. JSON producer hashes are
reused with file size/mtime checks; tensor files are deserialized only when
needed by their owning rank or the driver.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

from .checkpoint import RuntimeCheckpoint
from .mixed_backstop import _atomic_json


KIND = "dynrpo7b_native4"
STATUSES = {"DYNRPO7B_ROLLING_READY", "DYNRPO7B_ACTOR_SAVED_CALLBACKS_PENDING"}


def _json(path):
    return json.loads(Path(path).read_text())


def is_receipt(path):
    return _json(path).get("kind") == KIND


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _check_files(actor, checkpoint):
    for name, size in actor["file_bytes"].items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint):
            raise ValueError("Restored state escapes its checkpoint")
        info = path.stat()
        if info.st_size != size:
            raise ValueError("A committed checkpoint file changed after producer hashing")
        if info.st_mtime_ns != actor["file_mtime_ns"][name]:
            # Shared filesystems can normalize timestamps after close. Only
            # exact producer SHA permits this slow-path; changed bytes fail.
            expected = actor.get("files_sha256", {}).get(name)
            if expected is None or digest(path) != expected:
                raise ValueError("A committed checkpoint file changed after producer hashing")


def source(receipt_path):
    path = Path(receipt_path).resolve()
    receipt = _json(path)
    if receipt.get("kind") != KIND or receipt.get("status") not in STATUSES:
        raise ValueError("Not an actual DynRPO7B native4 restore receipt")
    checkpoint = Path(receipt["checkpoint"]).resolve()
    origin = Path(receipt["source_base"]).resolve()
    parent = _json(origin / "logs/run-manifest.json")
    actor_origin = Path(receipt.get('actor_origin_base', origin)).resolve()
    if receipt["run_id"] != parent["run_id"] or not checkpoint.is_relative_to(actor_origin):
        raise ValueError("Restore receipt and original run differ")
    _check_files(receipt["actor_ref"], checkpoint)
    receipt = copy.deepcopy(receipt)
    receipt["step"] = receipt["completed_outer"]
    receipt["world_size"] = 4
    workers = receipt["actor_ref"]["worker_files"]
    if sorted(row["rank"] for row in workers) != [0, 1, 2, 3]:
        raise ValueError("Native4 resume requires all four actual local states")
    if len({row["actual_optimizer_writes"] for row in workers}) != 1:
        raise ValueError("Saved rank Adam counts differ")
    receipt["actual_optimizer_writes"] = workers[0]["actual_optimizer_writes"]
    # A pending callback can have committed probe epochs since actor-pending
    # was written. Its phase journal advances probe only, never actor weights.
    phase_path = Path(receipt.get("phase_path", ""))
    if receipt["status"] != "DYNRPO7B_ROLLING_READY" and phase_path.is_file():
        phase = _json(phase_path)
        if phase["completed_actor_outer"] == receipt["step"] and phase["run_id"] == receipt["run_id"]:
            receipt["phase"] = phase
    version = receipt['phase']['grounding_version']
    reward_path = None
    for reward_base in dict.fromkeys((origin, actor_origin)):
        paths = list((reward_base/'artifacts').glob('*/version-manifest.json'))
        for candidate in paths:
            value = _json(candidate)
            if value.get('version') == version:
                reward_path = candidate
                receipt['reward_probe_round'] = value['round']
                break
        if reward_path is not None:
            break
    if reward_path is None:
        raise ValueError('Saved grounding version has no actual archived manifest')
    else:
        receipt['reward_probe_manifest'] = dict(path=str(reward_path),sha256=digest(reward_path))
    return path, receipt, checkpoint, origin, parent


def bind(run, receipt_path):
    path, receipt, checkpoint, origin, parent = source(receipt_path)
    from .objective_revision import bind_transition
    bind_transition(run, parent, receipt)
    for key in ("input_sha256", "initial_model_sha256"):
        if key in run and key in parent and run[key] != parent[key]:
            raise ValueError("A restore cannot change the scientific " + key)
    if run.get("training_world_size", 4) != 4 or parent.get("training_world_size", 4) != 4:
        raise ValueError("Rank-local FSDP state resumes on the same four-rank layout")
    phase = receipt["phase"]
    run.update(run_id=receipt["run_id"], native_ray_execution_id=uuid.uuid4().hex,
               pretraining_baseline_policy="reuse_historical",
               initial_probe_version=phase["grounding_version"],
               full_state_restore=dict(kind=KIND, source_receipt=str(path),
                   source_receipt_sha256=digest(path), source_checkpoint=str(checkpoint),
                   source_base=str(origin), source_run_manifest_sha256=digest(origin / "logs/run-manifest.json"),
                   completed_outer=receipt["step"],
                   actual_optimizer_writes_per_rank=receipt["actual_optimizer_writes"],
                   probe_pending=phase["probe_pending"],
                   probe_completed_epoch=phase["probe_completed_epoch"],
                   reward_refresh_pending=phase["grounding_pending"],
                   reward_probe_version=phase["grounding_version"],
                   reward_probe_round=receipt.get('reward_probe_round'),
                   sampling_rng_stage="post_callbacks" if receipt["post_callback_rng_captured"] else "saved_actor_before_callbacks"),
               optimizer_semantics="actual native4 model/Adam/scheduler/RNG/data/probe restore; saved actor is not replayed")
    return run


def from_config(config):
    path, receipt, checkpoint, origin, parent = source(config.trainer.full_state_restore_receipt)
    if Path(config.actor_rollout_ref.model.actor_checkpoint_path).resolve() != checkpoint:
        raise ValueError("FSDP must initialise the actor from the actual saved HF model")
    if config.trainer.n_gpus_per_node * config.trainer.nnodes != 4:
        raise ValueError("The saved local Adam shards require native4")
    return receipt


def validate_rank_state(saved, rank, outer, writes):
    if (saved["rank"], saved["world_size"], saved["completed_outer"], saved["actual_optimizer_writes"]) != (
            rank, 4, outer, writes):
        raise ValueError("Rank/world/outer/Adam state does not match the saved actor")
    if saved["scheduler"]["last_epoch"] != outer:
        raise ValueError("Saved scheduler differs from the completed actor outer")
    values = saved["optimizer"]["state"].values()
    steps = {float(value["step"]) for value in values}
    if steps != {float(writes)}:
        raise ValueError("Actual rank Adam steps differ from the saved counter")
    return dict(rank=rank, completed_outer=outer, actual_optimizer_writes=writes,
                scheduler_last_epoch=outer)


def restore_worker(worker, receipt_path):
    import numpy as np
    import torch
    from search_g1.training_state import cpu_tree, worker_rng
    from search_g1.full_state_restore import equal_tree
    from verl.utils.weight_continuation import model_initialization_path
    path, receipt, checkpoint, origin, parent = source(receipt_path)
    rank, outer, writes = worker.rank, receipt["step"], receipt["actual_optimizer_writes"]
    if torch.distributed.get_world_size() != 4:
        raise ValueError("Saved FSDP shard layout requires four ranks")
    if Path(model_initialization_path(worker.config.model, actor=True)).resolve() != checkpoint:
        raise ValueError("The actor was not loaded from this exact saved HF checkpoint")
    proof = next(row for row in receipt["actor_ref"]["worker_files"] if row["rank"] == rank)
    saved = torch.load(proof["path"], map_location="cpu", weights_only=False)
    validate_rank_state(saved, rank, outer, writes)
    worker.actor_optimizer.load_state_dict(saved["optimizer"])
    worker.actor_lr_scheduler.load_state_dict(saved["scheduler"])
    # Shapes/step counters are sufficient here: optimizer.load_state_dict moves
    # the actual persisted tensors, avoiding a second huge CPU equality copy.
    for group in worker.actor_optimizer.param_groups:
        for parameter in group["params"]:
            state = worker.actor_optimizer.state[parameter]
            if not state or tuple(state["exp_avg"].shape) != tuple(parameter.shape):
                raise ValueError("Restored Adam shard does not match its current FSDP parameter")
            if float(state["step"]) != writes:
                raise ValueError("Restored actual Adam count changed")
    worker.actor._optimizer_writes = writes
    worker.actor._complete_outers = outer
    worker.actor._full_budget_outers = saved.get("full_budget_outers", 0)
    worker.actor._optimizer_calls = saved.get("optimizer_calls", writes)
    worker.actor._optimizer_rollbacks = saved.get("optimizer_rollbacks", 0)
    worker.actor._last_attempt_index = saved["last_attempt_index"]
    worker.actor._used_attempt_ids = set(saved["used_attempt_ids"])
    worker.actor._policy_rejections = saved["policy_rejections"]
    worker.actor._inline_initial_optimizer_published = True
    worker.actor._invalidate_monitor_forward_cache(model_changed=True)
    rng = saved["rng"]
    if receipt["post_callback_rng_captured"]:
        boundary = next(row for row in receipt["boundary_worker_files"] if row["rank"] == rank)
        rng = torch.load(boundary["path"], map_location="cpu", weights_only=False)["rng"]
    manager = worker.rollout_sharding_manager
    engine = worker.rollout.inference_engine.llm_engine
    if engine.has_unfinished_requests():
        raise ValueError("Sampling state can only restore into an idle engine")
    generators = engine.model_executor.worker.model_runner.get_generators()
    generators.clear()
    for key, value in rng["request_generators"].items():
        generator = torch.Generator(device=value["device"])
        generator.set_state(value["state"])
        generators[key] = generator
    manager.torch_random_states = cpu_tree(rng["actor_cuda_saved"])
    manager.gen_random_states = cpu_tree(rng["rollout_cuda_saved"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"], device=torch.cuda.current_device())
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    equal_tree(worker_rng(worker), rng)
    from reproduction.a800.owned_processes import identity
    from reproduction.common8.native_allocation import cuda_uuids
    row = dict(rank=rank, world_size=4, source_run_id=receipt["run_id"], source_step=outer,
               source_checkpoint=str(checkpoint), actual_optimizer_writes=writes,
               scheduler_last_epoch=outer, actual_adam_scheduler_readback_equal=True,
               actual_rng_readback_equal=True,
               actual_cuda_uuid=cuda_uuids(count=torch.cuda.device_count())[torch.cuda.current_device()],
               process_identity=identity(os.getpid()), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
               model_hf_exact_tensors_checked_on_rank0=0,
               model_loaded_from_bound_hf_path=True, full_model_gather_repeated=False,
               used_attempt_ids=list(saved["used_attempt_ids"]),
               last_attempt_index=saved["last_attempt_index"], policy_rejections=saved["policy_rejections"],
               checked_unix=time.time())
    base = Path(os.environ["SEARCH_G1_BASE"])
    _atomic_json(base / "logs" / ("full-state-restored-rank-%d.json" % rank), row)
    return row


def restore_driver(trainer, receipt_path):
    import numpy as np
    import torch
    from search_g1.training_state import compact_stream_state, driver_rng
    from search_g1.full_state_restore import equal_tree
    path, receipt, checkpoint, origin, parent = source(receipt_path)
    saved = torch.load(receipt["boundary_driver"]["path"], map_location="cpu", weights_only=False)
    outer, phase = receipt["step"], receipt["phase"]
    if saved["completed_outer"] != outer:
        raise ValueError("Driver and saved actor outers differ")
    for manager in (trainer.reward_fn, getattr(trainer, "val_reward_fn", None)):
        if manager is not None:
            manager.expected_probe_version = phase["grounding_version"]
    torch.set_rng_state(saved["rng"]["torch_cpu"])
    np.random.set_state(saved["rng"]["numpy"])
    random.setstate(saved["rng"]["python"])
    equal_tree(driver_rng(), saved["rng"])
    trainer._restored_candidate_state = copy.deepcopy(compact_stream_state(saved["candidate_stream"]))
    trainer._restored_driver_rng = copy.deepcopy(saved["rng"])
    trainer._restored_attempt_progress = copy.deepcopy(saved["attempt_progress"])
    # The original pending wrapper keeps all2560 same-G5 fitting inputs; a
    # partially fitted epoch reference is supplied separately in phase below.
    probe = receipt["question_probe"]
    trainer._restored_qprobe_state = dict(path=probe["path"], actor_snapshot_id=receipt["run_id"] + ":" + str(outer))
    trainer._dynrpo7b_restored_phase = copy.deepcopy(phase)
    trainer._dynrpo7b_boundary_pending = receipt["status"] != "DYNRPO7B_ROLLING_READY"
    trainer._restored_reward_refresh_pending = phase["grounding_pending"]
    current_base = Path(os.environ["SEARCH_G1_BASE"])
    journal = RuntimeCheckpoint(current_base / "preparation/dynrpo7b-checkpoints", receipt["run_id"])
    journal.state = copy.deepcopy(phase)
    journal._save()
    result = dict(kind=KIND, completed_outer=outer, source_checkpoint=str(checkpoint),
                  actual_driver_rng_readback_equal=True,
                  actual_candidate_source_cursor=saved["candidate_stream"]["consumed_source_rows"],
                  actual_probe_version=phase["grounding_version"],
                  probe_pending=phase["probe_pending"], probe_completed_epoch=phase["probe_completed_epoch"],
                  reward_refresh_pending=phase["grounding_pending"], actor_replayed=False)
    _atomic_json(current_base / "logs/full-state-restored-driver.json", result)
    return result


def verified_runtime_prefix(base, run):
    _, receipt, _, _, _ = source(run["full_state_restore"]["source_receipt"])
    driver = _json(Path(base) / "logs/full-state-restored-driver.json")
    workers = _json(Path(base) / "logs/full-state-restored-workers.json")["ranks"]
    if (driver["completed_outer"] != receipt["step"] or len(workers) != 4
            or not all(row["actual_adam_scheduler_readback_equal"] and row["actual_rng_readback_equal"] for row in workers)):
        raise ValueError("Actual native4 worker/driver restoration is incomplete")
    # Resume counters come from the saved driver, not inferred outer*20.
    import torch
    saved = torch.load(receipt["boundary_driver"]["path"], map_location="cpu", weights_only=False)
    progress = saved["attempt_progress"]
    return dict(completed_outer=receipt["step"], writes=progress["cumulative_writes"],
                attempt_index=progress["attempt_index"], rejections=workers[0]["policy_rejections"],
                aborted=progress["aborted_attempts"],
                used_attempt_ids=workers[0]["used_attempt_ids"])
