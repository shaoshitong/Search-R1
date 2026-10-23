"""Publish an actual full-state producer receipt and a small callback sidecar.

The actor writer already saves and hashes the model and rank Adam states.
Unlike the older audit publisher this path never reloads those tensors or
hashes the whole checkpoint again. It reads the committed producer JSON and
stats its files once. Only changed post-fit probe state and idle RNG/driver
state are saved after callbacks; the completed actor files stay immutable.
"""
import json
import os
from pathlib import Path
import re
import shutil

from .checkpoint import RuntimeCheckpoint, probe_update_due
from .mixed_backstop import _atomic_json


MILESTONES = {5, 50, 100, 150, 200}
MANAGER = "dynrpo7b_per_outer_full_state_v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _context(trainer, checkpoint, imported=False):
    base = Path(os.environ["SEARCH_G1_BASE"]).resolve()
    checkpoint = Path(checkpoint).resolve()
    actor_root = Path(trainer.config.trainer.default_local_dir).resolve() / "actor"
    run = _json(base / "logs/run-manifest.json")
    if imported:
        binding = run.get('full_state_restore', {})
        if binding.get('kind') != 'dynrpo7b_native4' or checkpoint != Path(binding['source_checkpoint']).resolve():
            raise ValueError('Imported boundary must use its actual bound source checkpoint')
    elif checkpoint.parent != actor_root or not checkpoint.is_relative_to(base):
        raise ValueError("Rolling retention only owns this run's actor checkpoint directory")
    step = int(trainer.global_steps)
    directory = base / "preparation/dynrpo7b-checkpoints" / ("global_step_%d" % step)
    directory.mkdir(parents=True, exist_ok=True)
    return base, checkpoint, actor_root, run, step, directory


def _producer_reference(checkpoint, run_id, step):
    checkpoint = Path(checkpoint).resolve()
    complete = _json(checkpoint / "complete.json")
    state = _json(checkpoint / "training_state/manifest.json")
    if (complete["run_id"] != run_id or complete["step"] != step
            or not complete.get("finite_all_saved_tensors")
            or state["completed_outer"] != step):
        raise ValueError("Actual actor checkpoint is not committed at this outer")
    workers = state["worker_files"]
    if sorted(row["rank"] for row in workers) != [0, 1, 2, 3]:
        raise ValueError("The full actor state requires its four real rank receipts")
    hashes = complete["files_sha256"]
    required = {"training_state/manifest.json", "training_state/driver.pt",
                "model.safetensors.index.json"}
    required.update("training_state/rank-%d.pt" % rank for rank in range(4))
    model_index = _json(checkpoint / "model.safetensors.index.json")
    required.update(model_index["weight_map"].values())
    if not required <= hashes.keys():
        raise ValueError("Producer receipt is missing actual model/rank/driver files")
    sizes, mtimes = {}, {}
    for name in hashes:
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError("A producer checkpoint file is absent or outside its checkpoint")
        info = path.stat()
        sizes[name], mtimes[name] = info.st_size, info.st_mtime_ns
    for row in [*workers, state["driver_file"]]:
        name = str(Path(row["path"]).resolve().relative_to(checkpoint))
        if hashes.get(name) != row["sha256"]:
            raise ValueError("Producer and native state-writer hashes disagree")
    return dict(path=str(checkpoint), complete_path=str(checkpoint / "complete.json"),
                files_sha256=hashes, file_bytes=sizes,
                file_mtime_ns=mtimes, worker_files=workers), state


def bind_actor_checkpoint(trainer, checkpoint, *, imported=False):
    """Call immediately after checkpoint_complete, before periodic probe fit."""
    base, checkpoint, actor_root, run, step, directory = _context(trainer, checkpoint, imported)
    receipt_path = directory / "actor-reference.json"
    if receipt_path.exists():
        actor = _json(receipt_path)
        state = _json(checkpoint / "training_state/manifest.json")
    else:
        actor, state = _producer_reference(checkpoint, run["run_id"], step)
        _atomic_json(receipt_path, actor)
    probe = state.get("question_probe")
    if probe is None:
        raise ValueError("Formal DynRPO checkpoints must bind the selection probe state")
    journal = RuntimeCheckpoint(base / "preparation/dynrpo7b-checkpoints", run["run_id"])
    rollout_ref = getattr(trainer.qprobe_runtime, "rollout_checkpoint_reference", None)
    if callable(rollout_ref):
        rollout_ref = rollout_ref()
    if rollout_ref is None:
        rollout_ref = probe
    journal.actor_complete(step, actor, state["driver_file"], probe,
                           prefix_tokens=80 if step <= 100 else 160,
                           rollout_ref=rollout_ref,
                           grounding_version=trainer.reward_fn.expected_probe_version)
    pending_path = directory / "actor-pending.json"
    if not pending_path.exists():
        pending = dict(schema_version=1, kind="dynrpo7b_native4", manager=MANAGER,
                       status="DYNRPO7B_ACTOR_SAVED_CALLBACKS_PENDING", run_id=run["run_id"],
                       source_base=str(base), source_commit=run["source_commit"],
                       actor_origin_base=(run['full_state_restore']['source_base'] if imported else str(base)),
                       imported_actor_checkpoint=imported,
                       completed_outer=step, checkpoint=str(checkpoint), actor_ref=actor,
                       boundary_driver=state["driver_file"], boundary_worker_files=[],
                       question_probe=probe, phase=journal.state,
                       phase_path=str(journal.path), post_callback_rng_captured=False)
        _atomic_json(pending_path, pending)
    return journal


def _attempt_state(trainer):
    progress = getattr(trainer, "inline_attempt_progress", None)
    if progress is None:
        return None
    return {key: getattr(progress, key) for key in (
        "completed_outer", "cumulative_writes", "attempt_index", "aborted_attempts", "full_budget_outer")}


def _referenced_steps(value, actor_root):
    """Keep an older managed snapshot only when a retained reference needs it."""
    if isinstance(value, dict):
        return set().union(*(_referenced_steps(item, actor_root) for item in value.values())) if value else set()
    if isinstance(value, (tuple, list)):
        return set().union(*(_referenced_steps(item, actor_root) for item in value)) if value else set()
    if isinstance(value, str) and value.startswith(str(actor_root) + "/"):
        match = re.match(r"global_step_(\d+)(?:/|$)", value[len(str(actor_root)) + 1:])
        return {int(match.group(1))} if match else set()
    return set()


def _retire_old(metadata, actor_root, run_id):
    managed = {}
    for path in metadata.glob("global_step_*/boundary.json"):
        value = _json(path)
        if value.get("manager") == MANAGER and value.get("run_id") == run_id:
            managed[value["completed_outer"]] = (path.parent, value)
    keep = (set(sorted(managed)[-2:]) | MILESTONES) & managed.keys()
    # A probe or grounding encoder can legitimately refer to an older model.
    # Follow retained receipts before deleting anything in this run namespace.
    previous = None
    while previous != keep:
        previous = set(keep)
        for step in list(keep):
            keep |= _referenced_steps(managed[step][1], actor_root) & managed.keys()
    retired = []
    for step in sorted(managed.keys() - keep):
        directory, value = managed[step]
        checkpoint = actor_root / ("global_step_%d" % step)
        imported = value.get('imported_actor_checkpoint', False)
        if (not imported and value["checkpoint"] != str(checkpoint)) or checkpoint.is_symlink() or directory.is_symlink():
            raise ValueError("Refusing to retire a checkpoint outside this manager's owned namespace")
        if not imported and checkpoint.exists():
            shutil.rmtree(checkpoint)
        shutil.rmtree(directory)
        retired.append(step)
    return retired


def publish_boundary(trainer, checkpoint, *, imported=False):
    """Call once all scheduled probe/reward callbacks finished successfully.

    This is metadata/CPU I/O plus small idle-worker RNG RPCs. It does not run
    GPU forward passes, reload full Adam, or rewrite the actor checkpoint.
    """
    base, checkpoint, actor_root, run, step, directory = _context(trainer, checkpoint, imported)
    metadata = directory.parent
    published = directory / "boundary.json"
    if published.exists():
        receipt = _json(published)
        _atomic_json(metadata / "latest.json", receipt)
        _retire_old(metadata, actor_root, run["run_id"])
        return receipt
    journal = bind_actor_checkpoint(trainer, checkpoint, imported=imported)
    from search_g1.training_state import atomic_save, driver_rng, compact_stream_state
    boundary = directory / "boundary"
    boundary.mkdir(exist_ok=True)
    stream = compact_stream_state(trainer.mixed_stream.state())
    probe = journal.state["probe_ref"]
    if probe_update_due(step):
        # A post-fit state supplied by runtime is preferred. Otherwise save
        # only the changed probe here, leaving the actor checkpoint untouched.
        # Runtime writes the tiny metadata wrapper pointing at its existing E3
        # tensor file; its saver can avoid serializing the same tensors again.
        probe = trainer.qprobe_runtime.save_checkpoint(
            boundary, step, stream, trainer.reward_fn.expected_probe_version,
            actor_snapshot_id=run["run_id"] + ":" + str(step))
        if journal.state["probe_pending"]:
            # Per-epoch commits come from the actual fit callback, never from
            # the publisher inferring completion from a running process.
            if journal.state["probe_completed_epoch"] != 3:
                actual_epochs = probe.get("fit_completed_epochs") if isinstance(probe, dict) else None
                if actual_epochs != 3:
                    raise ValueError("Probe publication requires its actual three completed epoch receipts")
                # A resumed callback may supply one durable E3 receipt after
                # its own earlier epoch snapshots. Do not invent E1/E2 refs.
                journal.state["probe_completed_epoch"] = actual_epochs
                journal.state["probe_ref"] = probe
                journal._save()
            # Publish the runtime metadata wrapper, which references the same
            # committed E3 state and also retains driver/candidate bindings.
            journal.state["probe_ref"] = probe
            journal._save()
            journal.probe_published(probe)
    if step in (50, 100, 150):
        live = _json(base / "artifacts/live/manifest.json")
        if live["version"] != trainer.reward_fn.expected_probe_version or live["round"] != step:
            raise ValueError("Reward refresh must actually adopt this boundary before publication")
        journal.grounding_published(live["version"], live)
    workers_path = boundary / "workers.json"
    if workers_path.exists():
        workers = _json(workers_path)
    else:
        workers = trainer.actor_rollout_wg.capture_mixed_state(str(boundary), checkpoint=False)
        _atomic_json(workers_path, workers)
    driver_path = boundary / "driver.pt"
    if driver_path.exists():
        driver = _json(boundary / "driver-reference.json")
    else:
        driver = atomic_save(driver_path, dict(
            completed_outer=step, candidate_stream=stream, rng=driver_rng(),
            probe_version=trainer.reward_fn.expected_probe_version,
            question_probe=probe, attempt_progress=_attempt_state(trainer),
            dynrpo7b_phase=journal.state, callbacks_complete=True))
        _atomic_json(boundary / "driver-reference.json", driver)
    receipt = dict(schema_version=1, kind="dynrpo7b_native4", manager=MANAGER, status="DYNRPO7B_ROLLING_READY",
                   run_id=run["run_id"], source_commit=run["source_commit"],
                   source_base=str(base), completed_outer=step, checkpoint=str(checkpoint),
                   actor_origin_base=(run['full_state_restore']['source_base'] if imported else str(base)),
                   imported_actor_checkpoint=imported,
                   actor_ref=journal.state["actor_ref"],
                   boundary_driver=driver, boundary_worker_files=workers,
                   question_probe=probe, phase=journal.state,
                   phase_path=str(journal.path), post_callback_rng_captured=True,
                   producer_hashes_reused=True, full_model_rehashed=False)
    _atomic_json(published, receipt)
    _atomic_json(metadata / "latest.json", receipt)
    receipt["retired_outers"] = _retire_old(metadata, actor_root, run["run_id"])
    return receipt


def publish_imported_boundary(trainer, checkpoint):
    """Commit finished pending callbacks without copying/re-saving the actor.

    Its original model/rank files remain owned by the source experiment; the
    new run only owns the post-callback driver/probe/RNG sidecar and never
    retires the original checkpoint.
    """
    return publish_boundary(trainer, checkpoint, imported=True)
