"""Config-driven fresh question+policy5 joint probe200 warmup with EMA0.99.

Run on the admitted remote GPUs with torchrun. This entrypoint never launches
generation/retrieval/actor jobs, performs GPU admission, or kills any process.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import signal
import stat
import subprocess
import sys
import time

import torch
import torch.distributed as dist

from .checkpoint import (capture_rng, make_checkpoint, restore_checkpoint,
                         restore_rng, save_checkpoint)
from .ema import ParameterEMA
from .model import OFFICIAL_REVISION, Question5CorrectnessProbe
from .schedule import cosine_learning_rate
from .training import (TrainCounters, actual_adam_step, build_epoch_plan,
                       evaluate_groups, train_update, validate_groups)

COMMON_MANIFEST = "0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24"
DEFAULT_PATTERN = [51, 51, 51, 51, 52]
_stop_requested = False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _hex(value, length, name):
    if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid lowercase hexadecimal " + name)
    return value


def validate_config(value):
    """Pure CPU validation; remote assets are inspected only by the real runner."""
    config = dict(value)
    if config.get("schema_version") != 1:
        raise ValueError("Require config schema_version1")
    for key in ("run_id", "execution_id"):
        _hex(config[key], 32, key)
    _hex(config["source_commit"], 40, "source_commit")
    _hex(config["feature_manifest_sha256"], 64, "feature_manifest_sha256")
    for key in ("feature_manifest", "model_dir", "output_dir"):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError("Require an explicit absolute remote " + key)
        if config[key].startswith("/Users/"):
            raise ValueError("Large experiment assets cannot use Mac storage: " + key)
    deployment = config.get("deployment_handle_path")
    if deployment is not None and (not isinstance(deployment, str)
            or not Path(deployment).is_absolute() or deployment.startswith("/Users/")):
        raise ValueError("Require an explicit absolute remote deployment_handle_path")
    if config.get("official_revision", OFFICIAL_REVISION) != OFFICIAL_REVISION:
        raise ValueError("Official model revision differs")
    if config.get("common_manifest_sha256", COMMON_MANIFEST) != COMMON_MANIFEST:
        raise ValueError("Common canonical question manifest differs")
    config.setdefault("official_revision", OFFICIAL_REVISION)
    config.setdefault("common_manifest_sha256", COMMON_MANIFEST)
    config.setdefault("prefix_tokens", 5)
    config.setdefault("epochs", 200)
    config.setdefault("world_size", 4)
    config.setdefault("seed", 1729)
    config.setdefault("global_group_pattern", DEFAULT_PATTERN)
    config.setdefault("micro_groups", 6)
    config.setdefault("lr_start", 3e-5)
    config.setdefault("lr_end", 3e-6)
    config.setdefault("ema_decay", .99)
    if "save_every" in config:
        if "save_every_epochs" in config and config["save_every_epochs"] != config["save_every"]:
            raise ValueError("Conflicting save_every/save_every_epochs values")
        config["save_every_epochs"] = config.pop("save_every")
    config.setdefault("save_every_epochs", 10)
    config.setdefault("ledger_flush_updates", 10)
    config.setdefault("progress_every_updates", 25)
    config.setdefault("attention_implementation", "sdpa")
    config.setdefault("resume", None)
    if (config["prefix_tokens"] != 5 or config["epochs"] != 200 or config["world_size"] != 4
            or config["global_group_pattern"] != DEFAULT_PATTERN or config["micro_groups"] != 6
            or config["lr_start"] != 3e-5 or config["lr_end"] != 3e-6
            or config["ema_decay"] != .99 or config["save_every_epochs"] != 10):
        raise ValueError("Require the bound question5/joint1:1/200epoch/world4/micro6/EMA.99/cosine recipe")
    for key in ("seed", "ledger_flush_updates", "progress_every_updates"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer " + key)
    if config["attention_implementation"] not in ("sdpa", "eager", "flash_attention_2"):
        raise ValueError("Use a native supported Qwen attention backend")
    for key in ("expected_train_groups", "expected_test_groups"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Require actual validated full-split count: " + key)
    uuids = config.get("physical_gpu_uuids")
    pattern = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    if (not isinstance(uuids, list) or len(uuids) != 4
            or any(not isinstance(value, str) or re.fullmatch(pattern, value) is None for value in uuids)
            or len(set(value.lower() for value in uuids)) != 4):
        raise ValueError("Require four unique admitted physical GPU UUIDs in visible-rank order")
    if config["resume"] is not None:
        resume = config["resume"]
        if not isinstance(resume, dict) or not Path(resume["checkpoint"]).is_absolute():
            raise ValueError("Resume requires one explicit same-run checkpoint")
        _hex(resume["sha256"], 64, "resume checkpoint SHA")
    return config


def verify_runtime_source(config, repo):
    """Verify a real Git checkout or an explicitly bound staged source archive."""
    repo = Path(repo)
    if config.get("deployment_handle_path") is None:
        actual_source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        tracked_changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                                  cwd=repo, text=True).strip()
        if actual_source != config["source_commit"] or tracked_changes:
            raise ValueError("Runtime source is not the exact clean published commit")
        return dict(source_kind="genuine_git_checkout", source_commit=actual_source, tracked_clean=True)

    base = repo.parent
    if (repo.resolve() != repo or base.resolve() != base or repo.name != "repo"
            or (repo / ".git").exists()):
        raise ValueError("Archive mode requires its canonical isolated repo without Git metadata")
    limit = 64 * 1024**2

    def owned_file(path):
        path = Path(path)
        info = path.lstat()
        if (path.resolve() != path or not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid() or info.st_size > limit):
            raise ValueError("Unsafe, foreign or oversized source file: " + str(path))
        return info

    def metadata(path):
        owned_file(path)
        with Path(path).open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Oversized source metadata")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Source metadata must be an object")
        return value, hashlib.sha256(raw).hexdigest()

    handle_path = Path(config["deployment_handle_path"])
    if handle_path != base / "preparation/deployment.json":
        raise ValueError("Deployment handle must belong to this source archive base")
    handle, handle_sha = metadata(handle_path)
    if handle.get("base") != str(base) or any(handle.get(key) != config[key]
            for key in ("run_id", "execution_id", "source_commit")):
        raise ValueError("Archive deployment run/execution/source identity differs")
    _hex(handle.get("source_tree"), 40, "archive source tree")
    _hex(handle.get("source_archive_sha256"), 64, "source archive SHA")
    _hex(handle.get("source_plan_sha256"), 64, "source plan SHA")
    files = handle.get("source_files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("Require a bounded full declared source inventory")
    receipt, receipt_sha = metadata(base / "preparation/source-receipt.json")
    expected = dict(stage="QUESTION5_EXACT_SOURCE_ARCHIVE_STAGED", base=str(base),
                    source_commit=config["source_commit"], source_tree=handle["source_tree"],
                    source_archive_sha256=handle["source_archive_sha256"], source_files_verified=len(files),
                    teacher_or_actor_started=False, training_started=False)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Archive staging source receipt differs from the exact deployment")
    source_plan = base / "preparation/source-plan.json"
    owned_file(source_plan)
    if sha(source_plan) != handle["source_plan_sha256"]:
        raise ValueError("Archive source plan differs from the authenticated deployment")
    actual, total_bytes = {}, 0
    for name, expected_sha in files.items():
        if not isinstance(name, str):
            raise ValueError("Source inventory paths must be strings")
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or str(relative) != name
                or any(part in (".", "..", ".git") for part in relative.parts)):
            raise ValueError("Unsafe declared source-relative path")
        _hex(expected_sha, 64, "declared source file SHA")
        path = repo / relative
        before = owned_file(path)
        actual[name] = sha(path)
        after = owned_file(path)
        if (actual[name] != expected_sha or
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError("Archive runtime file differs or changed during verification: " + name)
        total_bytes += after.st_size
    inventory = set()
    for path in repo.rglob("*"):
        info = path.lstat()
        if path.resolve() != path or info.st_uid != os.getuid():
            raise ValueError("Unsafe or foreign archive inventory entry")
        if stat.S_ISREG(info.st_mode):
            inventory.add(path.relative_to(repo).as_posix())
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("Archive inventory contains a nonregular entry")
    if inventory != set(files) or "reproduction/question5_ema_warmup/train.py" not in actual:
        raise ValueError("Actual archive inventory differs from the full declared source map")
    return dict(source_kind="authenticated_staged_archive", source_commit=config["source_commit"],
                source_tree=handle["source_tree"], source_archive_sha256=handle["source_archive_sha256"],
                deployment_handle_path=str(handle_path), deployment_handle_sha256=handle_sha,
                source_receipt_sha256=receipt_sha, source_plan_sha256=handle["source_plan_sha256"],
                source_files_verified=len(actual), source_bytes_verified=total_bytes,
                actual_source_file_digests_sha256=digest(actual), exact_inventory_verified=True,
                train_source_sha256=actual["reproduction/question5_ema_warmup/train.py"])


def _bound_cuda_uuid(config, local):
    """Verify the local CUDA device against the parent's physical UUID binding.

No GPU is selected or leased here. On old Torch without a UUID property, the
parent must expose UUIDs directly in CUDA_VISIBLE_DEVICES; an index guess is
deliberately rejected. A bounded nvidia-smi read validates that exact UUID.
"""
    expected = config["physical_gpu_uuids"][local]
    value = getattr(torch.cuda.get_device_properties(local), "uuid", None)
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            value = None
    actual = str(value) if value is not None else ""
    method = "torch_cuda_device_properties_uuid"
    if not actual.startswith("GPU-"):
        visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
        if len(visible) != 4 or visible[local].lower() != expected.lower():
            raise ValueError("Torch lacks device UUID: require explicit admitted UUIDs in CUDA_VISIBLE_DEVICES")
        actual = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
                                          "--id=" + expected], text=True, timeout=10).strip()
        method = "explicit_CUDA_UUID_mapping_and_nvidia_smi_exact_UUID_readback"
    if actual.lower() != expected.lower():
        raise ValueError("Actual local CUDA device UUID differs from the parent's admitted binding")
    return dict(local_rank=local, expected_uuid=expected, actual_uuid=actual, method=method)


def identity():
    path = Path("/proc/self/stat")
    birth = int(path.read_text().rsplit(") ", 1)[1].split()[19]) if path.exists() else None
    return dict(pid=os.getpid(), parent_pid=os.getppid(), birth=birth, sid=os.getsid(0),
                uid=os.getuid(), hostname=os.uname().nodename,
                observed_unix=time.time(), observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + ".partial-%d" % os.getpid())
        with temporary.open("xb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rank0_action(action, rank):
    result = [None]
    if rank == 0:
        try:
            result[0] = dict(ok=True, value=action())
        except BaseException as error:
            result[0] = dict(ok=False, error_type=type(error).__name__, error=str(error))
    dist.broadcast_object_list(result, src=0)
    if not result[0]["ok"]:
        raise RuntimeError("Rank0 operation rejected: " + result[0]["error_type"] + ": " + result[0]["error"])
    return result[0]["value"]


def _agree(value, world, name):
    values = [None] * world
    dist.all_gather_object(values, value)
    if values != [value] * world:
        raise RuntimeError("Ranks disagree on " + name)
    return values


def _signal_stop(_signum, _frame):
    global _stop_requested
    # Finish the in-flight complete minibatch before serializing raw/EMA/Adam.
    _stop_requested = True


def _save_boundary(config, model, optimizer, ema, counters, binding, epoch, next_step,
                   plan_sha, rank, world, device, completed_epoch, epoch_metrics=None):
    rng = capture_rng(rank=rank, device=device,
                      sampler_state=dict(epoch=epoch, next_update_in_epoch=next_step, epoch_plan_sha256=plan_sha))
    rank_rng = [None] * world
    dist.all_gather_object(rank_rng, rng)
    _agree((actual_adam_step(optimizer), ema.updates, counters.state_dict()), world, "checkpoint counters")

    def write():
        output = Path(config["output_dir"])
        saved = make_checkpoint(model, optimizer, ema, counters, binding=binding, epoch=epoch,
                                next_update_in_epoch=next_step, epoch_plan_sha256=plan_sha,
                                rank_rng=rank_rng, completed_epoch=completed_epoch)
        if completed_epoch:
            if epoch_metrics is None:
                raise ValueError("A committed epoch requires its same-EMA train/test metrics")
            saved["epoch_metrics"] = epoch_metrics
            saved["epoch_metrics_sha256"] = digest(epoch_metrics)
        receipt = save_checkpoint(output / "latest.pt", saved)
        receipt.update(run_id=config["run_id"], execution_id=config["execution_id"],
                       source_commit=config["source_commit"], binding_sha256=digest(binding),
                       counters=counters.state_dict(), next_update_in_epoch=next_step,
                       epoch_plan_sha256=plan_sha, observed_unix=time.time())
        if completed_epoch:
            receipt["epoch_metrics_sha256"] = digest(epoch_metrics)
        if completed_epoch and (epoch % config["save_every_epochs"] == 0 or epoch == config["epochs"]):
            immutable = output / ("head-epoch-%04d.pt" % epoch)
            os.link(output / "latest.pt", immutable)
            receipt["immutable_path"] = str(immutable)
        atomic_json(output / "latest.json", receipt)
        if completed_epoch:
            atomic_json(output / "epochs" / ("epoch-%04d" % epoch) / "checkpoint.json", receipt, exclusive=True)
        return receipt

    return _rank0_action(write, rank)


def _publish_epoch_commit(output, metrics, receipt):
    """Idempotent closure after a complete state file was fsync'ed and SHA read.

An interrupted diagnostic before checkpoint writes only inside its execution
namespace. A valid completed checkpoint carries its own exact metrics so a
same-run continuation can close a missing marker without replaying any epoch.
"""
    directory = Path(output) / "epochs" / ("epoch-%04d" % metrics["epoch"])
    values = {"metrics.json": metrics,
              "complete.json": dict(status="FULL_MERGED_EPOCH_COMPLETE", metrics=metrics, checkpoint=receipt)}
    for name, value in values.items():
        path = directory / name
        if path.exists():
            prior = json.loads(path.read_text())
            if name == "metrics.json":
                if prior != metrics:
                    raise ValueError("Existing epoch metrics belong to a different update/state")
            elif (prior.get("metrics") != metrics
                  or prior.get("checkpoint", {}).get("sha256") != receipt["sha256"]):
                raise ValueError("Existing complete marker belongs to a different update/state")
        else:
            atomic_json(path, value, exclusive=True)
    return dict(committed=True, epoch=metrics["epoch"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-config", action="store_true", help="CPU-only schema check, no asset or GPU access")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = validate_config(json.loads(config_path.read_text()))
    if args.validate_config:
        print(json.dumps(dict(status="CONFIG_SCHEMA_VALID", config_sha256=digest(config),
                              epochs=config["epochs"], actor_updates=0), sort_keys=True))
        return 0
    if sys.platform == "darwin":
        raise RuntimeError("The real warmup and all large artifacts must run on admitted remote GPUs")
    rank, local, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != config["world_size"] or torch.cuda.device_count() != world:
        raise ValueError("Require exactly the four admitted visible CUDA GPUs/ranks")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    output = Path(config["output_dir"])
    execution = output / "executions" / config["execution_id"]
    execution.mkdir(parents=True, exist_ok=True)
    owner = dict(identity(), rank=rank, local_rank=local, world_size=world,
                 run_id=config["run_id"], execution_id=config["execution_id"],
                 source_commit=config["source_commit"], config_sha256=digest(config))
    atomic_json(execution / ("identity-rank-%d.json" % rank), owner, exclusive=True)
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    lock = None
    initialized = False
    counters = None
    ledger = None
    terminal = "FAILED"
    epoch = 0
    next_step = 0
    started = time.monotonic()
    try:
        cuda_binding = _bound_cuda_uuid(config, local)
        owner["physical_gpu_uuid"] = cuda_binding["actual_uuid"]
        atomic_json(execution / ("cuda-binding-rank-%d.json" % rank),
                    dict(identity=owner, binding=cuda_binding, parent_physical_gpu_uuids=config["physical_gpu_uuids"]),
                    exclusive=True)
        repo = Path(__file__).resolve().parents[2]
        source_binding = verify_runtime_source(config, repo)
        source_record = dict(stage="EXACT_RUNTIME_SOURCE_VERIFIED_BEFORE_NCCL", identity=owner,
                             binding=source_binding, observed_unix=time.time())
        if len(canonical(source_record)) > 16 * 1024:
            raise ValueError("Source binding receipt exceeds its scalar/hash bound")
        atomic_json(execution / ("source-binding-rank-%d.json" % rank), source_record, exclusive=True)
        if sha(config["feature_manifest"]) != config["feature_manifest_sha256"]:
            raise ValueError("Pinned feature manifest SHA differs")
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
        initialized = True
        actual_uuids = [None] * world
        dist.all_gather_object(actual_uuids, cuda_binding["actual_uuid"].lower())
        if actual_uuids != [value.lower() for value in config["physical_gpu_uuids"]]:
            raise ValueError("Four rank-to-physical-GPU UUID mapping differs from parent admission")

        def take_lock():
            nonlocal lock
            output.mkdir(parents=True, exist_ok=True)
            lock = (output / "run.lock").open("a+b")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return dict(locked=True, owner=owner)

        _rank0_action(take_lock, rank)
        from .cache import PackedFeatureReader
        train = PackedFeatureReader(config["feature_manifest"], split="train", device=device,
                                    expected_manifest_sha=config["feature_manifest_sha256"])
        test = PackedFeatureReader(config["feature_manifest"], split="test", device=device,
                                   expected_manifest_sha=config["feature_manifest_sha256"])
        train_counts, test_counts = validate_groups(train.groups), validate_groups(test.groups)
        if (train_counts["groups"] != config["expected_train_groups"]
                or test_counts["groups"] != config["expected_test_groups"]
                or train.question_max_tokens != test.question_max_tokens):
            raise ValueError("Actual full merged train/test counts or shared Qmax differ")
        if set(tid for group in train.groups for tid in group["trajectory_ids"]).intersection(
                tid for group in test.groups for tid in group["trajectory_ids"]):
            raise ValueError("Train/test trajectory events overlap")
        random.seed(config["seed"] + rank)
        import numpy as np
        np.random.seed(config["seed"] + rank)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        raw = Question5CorrectnessProbe(config["model_dir"], train.question_max_tokens,
                                       attention_implementation=config["attention_implementation"]).to(device)
        model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_start"], weight_decay=0)
        ema = ParameterEMA(model, config["ema_decay"])
        counters = TrainCounters()
        initial_plan = build_epoch_plan(len(train.groups), epoch=1, seed=config["seed"], world=world,
                                        global_group_pattern=config["global_group_pattern"])
        updates_per_epoch = len(initial_plan)
        total_updates = updates_per_epoch * config["epochs"]
        binding = dict(schema_version=1, role="fresh_question5_joint200_ema099", run_id=config["run_id"],
                       source_commit=config["source_commit"], official_revision=config["official_revision"],
                       common_manifest_sha256=config["common_manifest_sha256"],
                       feature_manifest_sha256=config["feature_manifest_sha256"],
                       feature_spec_sha256=digest(getattr(train, "spec", {})),
                       test_feature_spec_sha256=digest(getattr(test, "spec", {})),
                       question_max_tokens=train.question_max_tokens, prefix_tokens=5, head_copy=raw.copy_metadata,
                       train_counts=train_counts, test_counts=test_counts, epochs=config["epochs"],
                       world_size=world, micro_groups=config["micro_groups"], seed=config["seed"],
                       global_group_pattern=config["global_group_pattern"], updates_per_epoch=updates_per_epoch,
                       total_actual_adam_updates=total_updates, trajectory_bce_weight=1., mixed_bce_weight=1.,
                       ema_decay=.99, lr_start=config["lr_start"], lr_end=config["lr_end"],
                       learning_rate_clock="successful_actual_Adam", old_probe_imported=False,
                       old_optimizer_imported=False, actor_updates=0)
        _agree(digest(binding), world, "run/science/feature/batch binding")
        _agree((actual_adam_step(optimizer), ema.updates), world, "fresh Adam/EMA boundary")

        def claim_run():
            claim_path = output / "run-claim.json"
            if config["resume"] is None:
                if any((output / name).exists() for name in ("latest.pt", "latest.json", "run-claim.json")):
                    raise FileExistsError("Fresh initialization cannot overwrite a previous run; explicit same-run resume required")
                atomic_json(claim_path, dict(binding=binding, binding_sha256=digest(binding),
                                            first_execution_id=config["execution_id"], owner=owner), exclusive=True)
            else:
                claim = json.loads(claim_path.read_text())
                if claim.get("binding") != binding or claim.get("binding_sha256") != digest(binding):
                    raise ValueError("Resume claim is not this exact new experiment")
            atomic_json(execution / "binding.json", dict(binding=binding, binding_sha256=digest(binding),
                                                        config=config, actual_config_file_sha256=sha(config_path)), exclusive=True)
            return dict(claimed=True)

        _rank0_action(claim_run, rank)
        first_epoch, resume_step = 1, 0
        resume_rng = None
        if config["resume"] is not None:
            resume = config["resume"]
            _rank0_action(lambda: True if sha(resume["checkpoint"]) == resume["sha256"] else
                          (_ for _ in ()).throw(ValueError("Same-run checkpoint SHA differs")), rank)
            saved = torch.load(resume["checkpoint"], map_location="cpu", weights_only=False)
            counters = restore_checkpoint(saved, model, optimizer, ema, expected_binding=binding, device=device)
            if len(saved["rank_rng"]) != world:
                raise ValueError("Saved RNG rank count differs")
            if saved["completed_epoch"]:
                if counters.complete_epochs != saved["epoch"] or saved["next_update_in_epoch"] != 0:
                    raise ValueError("Saved complete-epoch marker/counters differ")
                first_epoch, resume_step = saved["epoch"] + 1, 0
            else:
                first_epoch, resume_step = saved["epoch"], saved["next_update_in_epoch"]
                if counters.complete_epochs != first_epoch - 1 or not 0 <= resume_step <= updates_per_epoch:
                    raise ValueError("Saved partial epoch cursor differs")
            expected_steps = counters.complete_epochs * updates_per_epoch + resume_step
            if counters.optimizer_steps != expected_steps:
                raise ValueError("Saved Adam count differs from full-data epoch/cursor")
            if counters.optimizer_steps:
                previous_lr = cosine_learning_rate(counters.optimizer_steps - 1, total_updates,
                                                    start=config["lr_start"], end=config["lr_end"])
                if any(float(group["lr"]) != previous_lr for group in optimizer.param_groups):
                    raise ValueError("Restored optimizer LR differs from the successful-Adam cosine clock")
            expected_plan = build_epoch_plan(len(train.groups), epoch=saved["epoch"], seed=config["seed"],
                                              world=world, global_group_pattern=config["global_group_pattern"])
            if digest([step.state_dict() for step in expected_plan]) != saved["epoch_plan_sha256"]:
                raise ValueError("Saved exact-once sampler plan SHA differs")
            expected_groups = counters.complete_epochs * len(train.groups)
            if not saved["completed_epoch"]:
                expected_groups += sum(step.global_groups for step in expected_plan[:resume_step])
            if (counters.global_groups_seen != expected_groups or counters.optimizer_steps > total_updates
                    or not 1 <= first_epoch <= config["epochs"] + 1):
                raise ValueError("Saved full-dataset exposure/budget boundary differs")
            if saved["completed_epoch"]:
                stored_metrics = saved.get("epoch_metrics")
                if stored_metrics is None or digest(stored_metrics) != saved.get("epoch_metrics_sha256"):
                    raise ValueError("Complete state lacks its exact same-EMA train/test metrics")
                repair_receipt = dict(path=resume["checkpoint"], sha256=resume["sha256"],
                    bytes=Path(resume["checkpoint"]).stat().st_size,
                    actual_adam_steps_per_rank=saved["actual_adam_steps_per_rank"],
                    ema_updates=saved["ema"]["updates"], counters=saved["counters"],
                    epoch=saved["epoch"], completed_epoch=True, actor_updates=0,
                    binding_sha256=digest(binding), epoch_metrics_sha256=digest(stored_metrics),
                    marker_recovered_from_full_checkpoint=True)
                _rank0_action(lambda: _publish_epoch_commit(output, stored_metrics, repair_receipt), rank)
            resume_rng = saved["rank_rng"][rank]
            del saved
        elif optimizer.state or counters.optimizer_steps or ema.updates:
            raise RuntimeError("Fresh probe must have empty Adam/zero EMA updates")
        # Constructor, DDP, official copy and restore have finished before RNG restore.
        if resume_rng is not None:
            restore_rng(resume_rng, rank=rank, device=device)
        atomic_json(execution / ("initialization-rank-%d.json" % rank),
                    dict(owner, stage="READY_FOR_COMPLETE_DATASET_EPOCHS", counters=counters.state_dict(),
                         first_epoch=first_epoch, next_update_in_epoch=resume_step,
                         fresh=config["resume"] is None, total_updates=total_updates), exclusive=True)
        ledger = (execution / ("updates-rank-%d.jsonl" % rank)).open("x", encoding="utf-8", buffering=64 * 1024)
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(first_epoch, config["epochs"] + 1):
            plan = initial_plan if epoch == 1 else build_epoch_plan(len(train.groups), epoch=epoch,
                seed=config["seed"], world=world, global_group_pattern=config["global_group_pattern"])
            plan_sha = digest([step.state_dict() for step in plan])
            _agree(plan_sha, world, "exact-once complete-data epoch plan")
            start_step = resume_step if epoch == first_epoch else 0
            prior_groups = sum(step.global_groups for step in plan[:start_step])
            if counters.global_groups_seen != (epoch - 1) * len(train.groups) + prior_groups:
                raise RuntimeError("Retained group exposure count differs from exact epoch cursor")
            epoch_started = time.monotonic()
            online = [0., 0., 0]
            for index in range(start_step, len(plan)):
                next_step = index
                lr = cosine_learning_rate(counters.optimizer_steps, total_updates,
                                          start=config["lr_start"], end=config["lr_end"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                record = train_update(model, optimizer, ema, counters, train.groups, plan[index], rank=rank,
                                      reader=train, micro_groups=config["micro_groups"], device=device)
                next_step = index + 1
                record.update(epoch=epoch, epoch_update=next_step, updates_per_epoch=updates_per_epoch,
                              plan_sha256=plan_sha, run_id=config["run_id"], execution_id=config["execution_id"],
                              rank=rank, source_commit=config["source_commit"], observed_unix=time.time())
                ledger.write(canonical(record).decode() + "\n")
                if next_step % config["ledger_flush_updates"] == 0:
                    ledger.flush()
                online[0] += record["online_raw_trajectory_bce"] * record["global_groups"]
                online[1] += record["online_raw_group_bce"] * record["global_groups"]
                online[2] += record["global_groups"]
                if next_step % config["progress_every_updates"] == 0 or next_step == len(plan):
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    atomic_json(execution / ("progress-rank-%d.json" % rank),
                        dict(stage="RETURNED_ADAM_AND_EMA", epoch=epoch, epoch_update=next_step,
                             counters=counters.state_dict(), latest_online_raw=record,
                             identity=owner, observed_unix=time.time()))
                stop = torch.tensor(int(_stop_requested), device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if bool(stop.item()):
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    receipt = _save_boundary(config, model, optimizer, ema, counters, binding, epoch,
                                              next_step, plan_sha, rank, world, device, False)
                    terminal = "STOPPED_AT_COMPLETE_ADAM_BOUNDARY"
                    atomic_json(execution / ("stopped-rank-%d.json" % rank),
                                dict(identity=owner, counters=counters.state_dict(), checkpoint=receipt), exclusive=True)
                    return 0
            if (counters.optimizer_steps != epoch * updates_per_epoch
                    or counters.global_groups_seen != epoch * len(train.groups)):
                raise RuntimeError("Full merged epoch actual update/exposure count differs")
            counters.complete_epochs = epoch
            _agree(counters.state_dict(), world, "full merged epoch counters")
            # Same EMA version for train and fixed test; neither split trains here.
            atomic_json(execution / ("progress-rank-%d.json" % rank),
                        dict(stage="SAME_EMA_TRAIN_TEST_DIAGNOSTICS", epoch=epoch,
                             counters=counters.state_dict(), identity=owner, observed_unix=time.time()))
            train_metrics = evaluate_groups(ema.model, train.groups, rank=rank, world=world, reader=train,
                                            micro_groups=config["micro_groups"], device=device)
            test_metrics = evaluate_groups(ema.model, test.groups, rank=rank, world=world, reader=test,
                                           micro_groups=config["micro_groups"], device=device)
            peaks = [None] * world
            dist.all_gather_object(peaks, dict(rank=rank, allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                                              reserved_bytes=int(torch.cuda.max_memory_reserved(device))))
            metrics = dict(epoch=epoch, run_id=config["run_id"], execution_id=config["execution_id"],
                           source_commit=config["source_commit"], observed_unix=time.time(),
                           train_ema=train_metrics, test_ema=test_metrics, ema_updates=ema.updates,
                           train_mixed_group_balanced_accuracy=train_metrics["mixed_group"]["balanced_accuracy"],
                           test_mixed_group_balanced_accuracy=test_metrics["mixed_group"]["balanced_accuracy"],
                           same_train_test_weight_kind="ema0.99", counters=counters.state_dict(),
                           raw_online_window=dict(groups=online[2], resumed_epoch_suffix=start_step > 0,
                              trajectory_bce=online[0] / online[2] if online[2] else None,
                              group_bce=online[1] / online[2] if online[2] else None),
                           epoch_seconds=time.monotonic() - epoch_started,
                           execution_seconds=time.monotonic() - started, actor_updates=0,
                           execution_cuda_peaks_per_rank=peaks)
            ledger.flush()
            os.fsync(ledger.fileno())
            _rank0_action(lambda: atomic_json(execution / "epochs" / ("epoch-%04d" % epoch) / "metrics.json",
                                               metrics, exclusive=True), rank)
            receipt = _save_boundary(config, model, optimizer, ema, counters, binding, epoch,
                                      0, plan_sha, rank, world, device, True, epoch_metrics=metrics)
            _rank0_action(lambda: _publish_epoch_commit(output, metrics, receipt), rank)
            if rank == 0:
                print(json.dumps(dict(stage="FULL_MERGED_EPOCH_COMPLETE", **metrics), allow_nan=False), flush=True)
            resume_step = 0
            dist.barrier()
            stop = torch.tensor(int(_stop_requested), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop.item()) and epoch < config["epochs"]:
                terminal = "STOPPED_AT_COMPLETE_EPOCH_BOUNDARY"
                return 0
        if counters.complete_epochs != 200 or counters.optimizer_steps != total_updates or ema.updates != total_updates:
            raise RuntimeError("Final real200epoch/Adam/EMA budget differs")
        terminal = "WARMUP_200_COMPLETE"
        def finish():
            value = dict(status=terminal, run_id=config["run_id"], execution_id=config["execution_id"],
                         binding_sha256=digest(binding), counters=counters.state_dict(),
                         total_actual_adam_updates=total_updates, source_commit=config["source_commit"],
                         actor_updates=0, observed_unix=time.time())
            path = output / "complete.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if previous.get("binding_sha256") != digest(binding) or previous.get("counters") != counters.state_dict():
                    raise ValueError("Existing final completion belongs to another state")
                return previous
            atomic_json(path, value, exclusive=True)
            return value
        _rank0_action(finish, rank)
        return 0
    except BaseException as error:
        try:
            atomic_json(execution / ("error-rank-%d.json" % rank),
                        dict(identity=owner, error_type=type(error).__name__, error=str(error),
                             epoch=epoch, next_update_in_epoch=next_step,
                             counters=counters.state_dict() if counters is not None else None,
                             partial_writes_not_rolled_back=True, observed_unix=time.time()), exclusive=True)
        except BaseException:
            pass
        raise
    finally:
        if ledger is not None:
            ledger.flush()
            os.fsync(ledger.fileno())
            ledger.close()
        try:
            atomic_json(execution / ("exit-rank-%d.json" % rank),
                        dict(identity=owner, terminal=terminal, epoch=epoch,
                             counters=counters.state_dict() if counters is not None else None,
                             elapsed_seconds=time.monotonic() - started, observed_unix=time.time()), exclusive=True)
        except BaseException:
            pass
        if initialized:
            dist.destroy_process_group()
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
