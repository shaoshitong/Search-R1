#!/usr/bin/env python3
"""Append one immutable completed outer to the local nested-tree v3 cost book.

No SSH, subprocess, model/tensor imports, training controls, or STATUS/REPORT writes.
Existing history through20 is preserved byte-for-byte. Missing current metrics
remain pending/null, including final outer200 without a complete metric line.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

BOOK = "PER_OUTER_METRICS_COMPACT.json"
BOOK_MD = "PER_OUTER_METRICS_COMPACT.md"
PROTECTED_THROUGH = 20
BACKFILL_INDEX = "CANDIDATE_COST_BACKFILL_THROUGH19.json"
BACKFILL_SHA256 = "d219a4f1af593d6dc304154e115aab218168113974109ce032d0b8402c568699"
MAX_BYTES = 100000
PREFIXES = {
    "variance_metrics": "tree_variance/",
    "stage_timing_seconds": "timing_s/",
    "rollout_iteration_metric_ledger": "rollout/",
    "tree_sampling_metrics": "tree_sampling/",
}
OPT_KEYS = [
    "actor/optimizer_steps", "actor/optimizer_opportunities_processed",
    "actor/optimizer_opportunities_skipped", "actor/skipped_empty_minibatches",
    "actor/scheduler_step_before_outer", "actor/scheduler_step_after_outer",
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(path.read_bytes())


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decode(value, fieldsets):
    if isinstance(value, dict) and "field_set" in value:
        return dict(zip(fieldsets[value["field_set"]], value["values"]))
    return value if isinstance(value, dict) else {}


def atomic_copy(base, mirrors, relative, raw, immutable=False):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Output must be a generated relative audit filename")
    for root in [base] + mirrors:
        dst = root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if immutable and dst.exists() and dst.read_bytes() != raw:
            raise RuntimeError("Existing immutable local audit bytes differ: " + str(dst))
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, dst)
        if sha(dst.read_bytes()) != sha(raw):
            raise RuntimeError("Local audit publication byte readback differs")


def local_input(base, name, expected=None):
    # Only a direct, explicitly named small JSON in this task's local audit dir.
    name = Path(name).name
    if not name.endswith(".json") or name in ("STATUS.json", "REPORT.json") or name.startswith("CURRENT"):
        raise ValueError("Mutable live state is not an immutable source")
    path = base / name
    if path.stat().st_size > 4_000_000:
        raise ValueError("Input is not small scalar metadata: " + name)
    raw = path.read_bytes()
    if expected and sha(raw) != expected:
        raise RuntimeError("Immutable input SHA differs: " + name)
    return name, json.loads(raw), {"bytes": len(raw), "sha256": sha(raw)}


def actual_inputs(base, receipt_path):
    name, full, binding = local_input(base, receipt_path)
    step = full.get("completed_outer")
    if not isinstance(step, int) or not name.startswith("FULL_OUTER" + str(step) + "_"):
        raise ValueError("Expected an explicit FULL_OUTERN immutable receipt")
    target = full.get("target_outer", 200)
    if not 1 <= step <= target:
        raise ValueError("Completed outer outside actual target")
    if step < target and (full.get("actual_next_outer_entered") or 0) < step + 1:
        raise ValueError("Full receipt does not yet confirm next outer entry")
    if not full.get("source_poll_sha256") or not full.get("native_metadata_readback_sha256"):
        raise ValueError("Immutable poll/native input SHA is required")
    pn, poll, pb = local_input(base, full["source_poll"], full["source_poll_sha256"])
    nn, native, nb = local_input(base, full["native_metadata_readback"], full["native_metadata_readback_sha256"])
    source = full["source_commit"]
    run = full["run"]
    if poll.get("source_commit") != source or native.get("source_commit") != source or poll.get("run") != run:
        raise ValueError("Source/run identity differs across immutable evidence")
    if full.get("unit") != poll.get("unit") or native.get("outer") != step:
        raise ValueError("Unit/outer differs across immutable evidence")
    nr = native.get("receipt", {})
    workers = nr.get("workers", [])
    if native.get("manifest_status") != "COMPLETE_NATIVE_ROLLING_STATE_SAVED" or nr.get("status") != "COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED":
        raise ValueError("Native metadata is not COMPLETE")
    if native.get("archive_matches_original_manifest") is not True or len(workers) != 8 or sorted(w.get("rank") for w in workers) != list(range(8)):
        raise ValueError("Missing complete eight-rank native metadata")
    if any(w.get("full_worker_state_finite") is not True for w in workers):
        raise ValueError("Original writer full-state finite proof incomplete")
    if nr.get("driver", {}).get("next_step") != step + 1:
        raise ValueError("Complete native driver next_step differs")
    manifest = native.get("manifest_sha256")
    if manifest != native.get("recorded_original_manifest_sha256") or manifest != nr.get("source_manifest_sha256") or manifest != full.get("native_manifest_sha256"):
        raise ValueError("Native manifest identity differs")
    progress = poll.get("progress", {})
    if progress.get("errors") or progress.get("actual_completed_outer_from_attempt_results", 0) < step:
        raise ValueError("Actual completed progress is absent or has errors")
    found = [m for m in poll.get("iteration_metrics", []) if m.get("step") == step]
    fm = full.get("iteration_metrics")
    # A present but inconsistent metric is an error, never relabelled pending.
    if isinstance(fm, dict) and fm and fm.get("step") != step:
        raise ValueError("Full receipt metric belongs to a different outer")
    metric = found[-1] if found else fm if isinstance(fm, dict) and fm.get("step") == step else {}
    if found and isinstance(fm, dict) and fm and metric != fm:
        raise ValueError("Available current metric differs from immutable receipt")
    controller = next((c for c in poll.get("rollout_cost", []) if c.get("candidate_outer") == step and c.get("status") == "TRAINING_ROLLOUT_COST_COMPLETE"), {})
    ranks = [r for r in poll.get("last_rank_summaries", []) if r.get("outer_step") == step]
    if not ranks:
        ranks = [r for r in full.get("last_rank_summaries", []) if r.get("outer_step") == step]
    if ranks and (len(ranks) != 8 or sorted(r.get("rank") for r in ranks) != list(range(8))):
        raise ValueError("Available direct rank evidence is inconsistent")
    bindings = {name: binding, pn: pb, nn: nb}
    retrospective = None
    if full.get("evidence_mode") == "RETROSPECTIVE_SOURCE_READBACK":
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_retrospective_full", Path(__file__).with_name("backfill_nested_historical_full.py"))
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        retrospective = bridge.cost_evidence(base, full, poll, native)
        metric, controller, ranks = [retrospective[k] for k in ("metric", "controller", "ranks")]
        bindings.update({k: {"bytes": v["bytes"], "sha256": v["sha256"]} for k, v in retrospective["bindings"].items()})
    resource_ref = None
    resource_binding = full.get("input_bindings", {}).get("resource")
    if isinstance(resource_binding, dict):
        rn, _, rb = local_input(base, resource_binding["path"], resource_binding["sha256"])
        if rb["bytes"] != resource_binding["bytes"]:
            raise ValueError("Resource metadata byte length differs")
        bindings[rn] = rb
        resource_ref = {"source_file": rn, "source_sha256": rb["sha256"], "json_selector": "original immutable resource snapshot; not an added cost phase"}
    return {"step": step, "target": target, "source": source, "full_name": name,
            "full": full, "poll_name": pn, "poll": poll, "native_name": nn,
            "native": native, "metric": metric, "controller": controller,
            "ranks": ranks, "bindings": bindings, "resource_ref": resource_ref,
            "retrospective": retrospective}


def supplemental_controller_input(base, reference, evidence):
    name, poll, binding = local_input(base, reference["file"], reference.get("sha256"))
    if reference.get("bytes") is not None and binding["bytes"] != reference["bytes"]:
        raise ValueError("Supplemental controller poll bytes differ")
    full, step = evidence["full"], evidence["step"]
    if poll.get("source_commit") != evidence["source"] or poll.get("run") != full["run"] or poll.get("unit") != full["unit"] or poll.get("experiment_origin") != "fresh":
        raise ValueError("Supplemental controller poll source/run/unit/origin differs")
    if not numeric(poll.get("epoch")) or not numeric(evidence["poll"].get("epoch")) or poll["epoch"] > evidence["poll"]["epoch"]:
        raise ValueError("Supplemental controller poll must precede its immutable FULL observation")
    found = [c for c in poll.get("rollout_cost", []) if c.get("candidate_outer") == step]
    if len(found) != 1:
        raise ValueError("Supplemental controller poll must contain exactly one matching outer")
    controller = found[0]
    if controller.get("status") != "TRAINING_ROLLOUT_COST_COMPLETE" or controller.get("attempt_id") != step or controller.get("completed_outer_before") != step - 1 or controller.get("code_commit") != evidence["source"] or controller.get("config_sha256") != full.get("config_sha256"):
        raise ValueError("Supplemental controller identity/completion differs")
    return name, controller, binding


def new_row(index, evidence):
    e = evidence
    step, source, full, p, n = e["step"], e["source"], e["full"], e["poll"], e["native"]
    nr, m, c = n["receipt"], e["metric"], copy.deepcopy(e["controller"])
    workers = nr["workers"]
    row_id = "child_native_continuation:" + source + ":" + str(step)
    identity = copy.deepcopy(n.get("identity", {}))
    common_keys = ["allocation_kind", "physical_gpus", "slurm_reserved_gpus", "resource_protocol",
                   "job_id", "run_id", "code_commit", "config_sha256", "data_identity",
                   "model_identity", "observer_source_sha256", "scope"]
    common = {k: c.pop(k) for k in common_keys if k in c}
    common_id = sha(encoded(common))[:16]
    index["controller_common_fields"].setdefault(common_id, common)
    c["common_fields_ref"] = "controller_common_fields." + common_id
    c.setdefault("costs", {})
    c.setdefault("status", "PENDING_CURRENT_OUTER_CONTROLLER_COST")
    old_identity = index["source_identity_by_commit"].get(source)
    if old_identity and old_identity["identity"].get("config_sha256") != identity.get("config_sha256"):
        raise ValueError("Actual source config identity changed without a new source")
    index["source_identity_by_commit"].setdefault(source, {
        "identity": {**identity, "individual_split_file_sha256": None,
                     "split_boundary": "Recorded data identity only; split-file SHA absent unless explicitly stored."},
        "run_directory": full["run"], "unit": full["unit"]})
    def ref(name, selector):
        return {"source_file": name, "source_sha256": e["bindings"][name]["sha256"], "json_selector": selector}
    actual_counts = p.get("progress", {}).get("completed_actual_optimizer_calls_per_rank")
    retrospective = e.get("retrospective")
    metric_name = retrospective["gap_name"] if retrospective else e["poll_name"]
    if retrospective:
        actual_counts = retrospective["completed"]
    row = {
        "row_id": row_id, "scope": "child_native_continuation", "source_commit": source,
        "outer": step, "effective_history": True, "experiment_origin": p.get("experiment_origin"),
        "identity_ref": "source_identity_by_commit." + source,
        "rollout_controller_cost": c, "metric_evidence": metric_name + ":iteration_metrics[step=" + str(step) + "]",
        "full_outer_evidence": e["full_name"], "full_outer_status": full.get("status"),
        "full_metric_status": ("ACTUAL_HISTORICAL_OUTER_METRIC_RETROSPECTIVE_SOURCE_READBACK" if retrospective else "ACTUAL_CURRENT_OUTER_METRIC") if m and numeric(m.get("timing_s/step")) else "PENDING_CURRENT_OUTER_FULL_METRIC",
        "original_other_iteration_scalars": {k: v for k, v in m.items() if not any(k.startswith(pre) for pre in PREFIXES.values()) and k not in OPT_KEYS},
        "candidate_cost_detail": {
            "candidate_counts": {k: m.get(v) for k, v in [("all", "rollout/candidate_attempts"), ("accepted", "rollout/candidate_accepted"), ("discarded", "rollout/candidate_discarded"), ("fallback", "rollout/candidate_fallbacks")]},
            "candidate_outcome_token_retrieval_scoring_prefix_sums": None, "A1_B1_B2": None,
            "candidate_round_elapsed_seconds": None, "padding_cost_by_candidate_outcome": None,
            "training_leaf_count": None, "root_leaf_quotas": None,
            "fallback_overlap_crosschecked_from_rows": None},
        "optimizer": {
            "iteration_metric_fields": {k: m.get(k) for k in OPT_KEYS},
            "per_outer_direct_rank_summaries": [{k: v for k, v in r.items() if k != "local_shard_sample_delta"} for r in e["ranks"]],
            "cumulative_calls_per_rank": actual_counts,
            "count_boundary": "Actual source/native cumulative state includes inherited ancestors. New successful credit uses only this completed outer direct records; failed2/6 and bootstrap costs remain separate.",
            "direct_rank_detail_ref": ref(metric_name, "direct_rank_summaries_by_outer." + str(step) if retrospective else "last_rank_summaries[outer_step=" + str(step) + "]")},
        "native_manifest_metadata": {
            "evidence_file": e["native_name"], "source_manifest_path": nr.get("source_manifest_path"),
            "source_manifest_sha256": nr.get("source_manifest_sha256"), "source_manifest_bytes": nr.get("source_manifest_bytes"),
            "manifest_status": n.get("manifest_status"), "history_receipt_sha256": n.get("receipt_sha256"),
            "history_manifest_sha256": nr.get("history_manifest_sha256"),
            "archive_matches_original_manifest": n.get("archive_matches_original_manifest"),
            "full_worker_state_finite_by_rank": {str(w["rank"]): w.get("full_worker_state_finite") for w in workers},
            "scheduler_last_epoch_by_rank": {str(w["rank"]): w.get("scheduler_last_epoch") for w in workers},
            "actual_optimizer_calls_per_rank_from_native_writer": {str(w["rank"]): w.get("actual_total_optimizer_calls") for w in workers},
            "observed_adam_steps_by_rank": {str(w["rank"]): w.get("observed_adam_steps") for w in workers},
            "actual_optimizer_calls_per_rank_from_progress": actual_counts,
            "driver_next_step": nr["driver"].get("next_step"),
            "recorded_worker_driver_identity_detail_ref": ref(e["native_name"], "receipt.workers + receipt.driver"),
            "tensor_contents_read": n.get("tensor_contents_read"),
            "independent_cold_reload_performed": n.get("independent_cold_reload_performed"),
            "current_PT_availability": None,
            "scope": "Original writer finite/count/scheduler and archived JSON metadata; no tensor read/hash/reload and no current PT availability claim."},
        "completion_evidence": {
            "actual_completed_outer_from_attempt_results": p["progress"].get("actual_completed_outer_from_attempt_results"),
            "progress_errors": p["progress"].get("errors"),
            "actual_next_outer_entered": full.get("actual_next_outer_entered"),
            "source_poll": e["poll_name"], "source_poll_sha256": e["bindings"][e["poll_name"]]["sha256"],
            "resource_snapshot_ref": e["resource_ref"]},
        "statistics_scope": {
            "training_question_count": m.get("tree_variance/question_count"),
            "tree_root_count": m.get("tree_variance/tree_count"),
            "controller_ledger_question_rows": c.get("question_rows"),
            "controller_rows_meaning": "1024 original root rows for512 training questions before five-leaf expansion; not1024 questions."},
        "missing_fields": ["Missing current-outer fields remain null; a prior outer metric is never used.",
                           "Candidate outcome padding and split-file SHA are not guessed."],
        "development_evaluation": copy.deepcopy(full.get("actual_dev512")),
        "development_evaluation_receipt_value": copy.deepcopy(full.get("actual_dev512")),
        "development_evaluation_status": "ACTUAL_CURRENT_OUTER_LEDGER" if full.get("actual_dev512") is not None else "NO_CURRENT_OUTER_DEV_IN_IMMUTABLE_RECEIPT",
        "full_outer_receipt_accounting": {k: full[k] for k in [
            "local_control_script_sha256", "per_rank_actual_total_adam", "partial_optimizer_calls_per_rank",
            "actual_outer_successful_writes_per_rank", "actual_outer_skipped_opportunities_per_rank",
            "actual_outer_processed_opportunities_per_rank", "full_metric_processed_opportunities",
            "full_metric_skipped_opportunities", "no_all20_writes_or_zero_skip_assumption", "all_skipped_attempts",
            "normal_dev512_wall_seconds", "training_archive_wall_seconds_excluding_current_dev"] if k in full},
    }
    if retrospective:
        row["retrospective_source_readback"] = {"evidence_mode": full["evidence_mode"],
            "observed_epoch": full["observed_epoch"], "scope": full["retrospective_evidence_scope"],
            "source_readback": ref(metric_name, "metric/rank/controller/dev original file bindings"),
            "live_poll_completed_outer_at_collection": p["progress"]["actual_completed_outer_from_attempt_results"],
            "historical_counter_source": full["historical_counter_source"],
            "actual_next_outer_entry_scope": full["actual_next_outer_entry_scope"]}
    for key, prefix in PREFIXES.items():
        row[key] = {k: v for k, v in m.items() if k.startswith(prefix)}
    dev = row["development_evaluation"]
    if dev is not None and (not isinstance(dev, dict) or dev.get("outer") != step or dev.get("rows") != 512 or dev.get("status") != "VALIDATION_COST_COMPLETE" or dev.get("metrics", {}).get("val/qa_count/all") != 512):
        raise ValueError("Present current dev ledger is incomplete/inconsistent")
    return row


def materialized_rows(base, index):
    rows = []
    for info in index["scalar_blocks"]:
        raw = (base / info["file"]).read_bytes()
        if sha(raw) != info["sha256"]:
            raise RuntimeError("Existing scalar block SHA differs")
        block = json.loads(raw)
        sets = block["scalar_fieldsets"]
        for item in block["rows"]:
            row = copy.deepcopy(item)
            for key in ("variance_metrics", "rollout_iteration_metric_ledger", "tree_sampling_metrics"):
                row[key] = decode(row.get(key), sets)
            row["rollout_controller_cost"] = copy.deepcopy(row.get("rollout_controller_cost", {}))
            row["rollout_controller_cost"]["costs"] = decode(row["rollout_controller_cost"].get("costs"), sets)
            records = row.get("optimizer", {}).get("per_outer_direct_rank_summaries")
            if isinstance(records, dict) and "field_set" in records:
                row["optimizer"]["per_outer_direct_rank_summaries"] = [dict(zip(sets[records["field_set"]], v)) for v in records["values"]]
            row["_block_file"] = info["file"]
            row["_block_row_index"] = block["rows"].index(item)
            rows.append(row)
    return sorted(rows, key=lambda r: r["outer"])


def n(value):
    if value is None or value == "unknown":
        return "null"
    return format(value, ".10g") if isinstance(value, (int, float)) else str(value)


def group(values, keys):
    return "/".join(n(values.get(k)) for k in keys)


def append_md(old, row, entry, previous_outer):
    lines = old.splitlines()
    m, v, c = row["rollout_iteration_metric_ledger"], row["variance_metrics"], row["rollout_controller_cost"]
    opts = row["optimizer"]["iteration_metric_fields"]
    calls = row["native_manifest_metadata"]["actual_optimizer_calls_per_rank_from_native_writer"]
    label = "子" + row["source_commit"][:3] + " / " + str(row["outer"])
    cvalue = next(iter(calls.values())) if calls and len(set(calls.values())) == 1 else None
    new_lines = [
        "|" + "|".join([label] + [n(v.get(k)) for k in ["tree_variance/within_root_raw_reward_population_variance", "tree_variance/intra_adv_population_variance", "tree_variance/group_actual_adv_population_variance", "tree_variance/mixed_reward_question_fraction"]] + [
            group(v, ["tree_variance/exact_reward_one_question_count", "tree_variance/exact_reward_zero_question_count", "tree_variance/mixed_question_count"]),
            group(v, ["tree_variance/exact_reward_one_tree_count", "tree_variance/exact_reward_zero_tree_count", "tree_variance/mixed_tree_count"])]) + "|",
        "|" + "|".join([label, group(opts, OPT_KEYS[1:2] + OPT_KEYS[:1] + OPT_KEYS[2:3]), n(cvalue), n(c.get("wall_seconds"))] + [n(m.get("rollout/" + k)) for k in ["generated_tokens", "prefill_tokens", "retrieval_calls"]] + [
            group(m, ["rollout/scoring_calls", "rollout/scoring_wall_seconds"]),
            group(row["candidate_cost_detail"]["candidate_counts"], ["all", "accepted", "discarded", "fallback"])]) + "|",
        "|" + "|".join([label] + [n(row["stage_timing_seconds"].get("timing_s/" + k)) for k in ["gen", "ref", "adv", "update_actor", "testing", "stage_testing", "save_checkpoint", "step"]]) + "|",
    ]
    anchors = [i for i, line in enumerate(lines) if line.startswith("|") and (" / " + str(previous_outer) + "|") in line]
    if len(anchors) != 3:
        raise ValueError("Expected existing three per-outer readable tables")
    for position, line in reversed(list(zip(anchors, new_lines))):
        lines.insert(position + 1, line)
    for i, line in enumerate(lines):
        if line.startswith("本次新增 "):
            lines[i] = "本次新增 [" + str(row["outer"]) + "]；仅本地固定凭据记账；不修改STATUS/REPORT/运行源码。"
        if line.startswith("固定源证据只使用"):
            latest = row.get("last_confirmed_dev512_ref") or {}
            last_dev = latest.get("row_id", "unknown").rsplit(":", 1)[-1]
            lines[i] = "固定源证据只使用immutable FULL_OUTER/source poll/native/resource；当前轮" + str(row["outer"]) + "，最新实际dev轮" + last_dev + "。旧dev不作为本轮新评分或本轮testing耗时。"
        if entry["file"] in line:
            lines[i] = ""
    lines += ["", "- " + entry["file"] + "；实际轮次 " + ",".join(map(str, entry["actual_outers"])) + "；" + str(entry["bytes"]) + " bytes；SHA " + entry["sha256"] + "。",
              "第" + str(row["outer"]) + "轮metric状态：" + row["full_metric_status"] + "；未知字段保留null，不借上一轮cost/dev/opportunity。"]
    dev = row["development_evaluation"]
    if dev:
        lines.append("第" + str(row["outer"]) + "轮实际dev512 F1/EM：" + group(dev["metrics"], ["val/f1/all", "val/em/all"]) + "；ledger SHA " + str(dev.get("ledger_sha256")) + "。")
    else:
        lines.append("本轮actual_dev512=null；latest dev由既有实际dev行引用保留，不作为本轮评测或本轮testing耗时。")
    return ("\n".join(lines) + "\n").encode()


def append_one(base, receipt, max_bytes):
    mainraw = (base / BOOK).read_bytes()
    index = json.loads(mainraw)
    mirrors = [Path(p) for p in index["mirrors"]]
    e = actual_inputs(base, receipt)
    step = e["step"]
    existing = [r for r in index["effective_rows"] if r["outer"] == step]
    if existing:
        current = materialized_rows(base, index)
        row = next(r for r in current if r["outer"] == step)
        block = read_json(base / row["_block_file"])
        old = block["source_inputs"].get(e["full_name"], {})
        if old.get("sha256") != e["bindings"][e["full_name"]]["sha256"]:
            raise ValueError("Already recorded outer has different immutable receipt; no history overwritten")
        return index, False
    previous = index["effective_history"]["latest_effective_outer"]
    if step != previous + 1:
        raise ValueError("Append requires exactly next effective outer; no missing/duplicate credit")
    previous_rows = materialized_rows(base, index)
    row = new_row(index, e)
    # Preserve original1..19 blocks, including the already-published first tool block.
    active = next((v for v in reversed(index["scalar_blocks"]) if v.get("created_by") == "append_nested_local_costs_v1" and v["first_outer"] > PROTECTED_THROUGH and not v.get("sealed")), None)
    if active:
        block = read_json(base / active["file"])
    else:
        block = None
    def fresh():
        first = step
        value = {
            "summary_schema": "tree_nested_per_outer_scalar_block_v3",
            "created_by": "append_nested_local_costs_v1", "first_outer": first,
            "last_outer_capacity": min(first + 9, e["target"]), "actual_outers": [],
            "sealed": False, "input_directory": str(base), "rows": [],
            "scalar_fieldsets": copy.deepcopy(index["scalar_fieldsets"]),
            "scalar_fieldsets_semantics": index["scalar_fieldsets_semantics"],
            "source_identity_by_commit": copy.deepcopy(index["source_identity_by_commit"]),
            "controller_common_fields": copy.deepcopy(index["controller_common_fields"]),
            "native5_restoration_evidence": copy.deepcopy(index.get("native5_restoration_evidence")),
            "statistical_and_cost_boundaries": copy.deepcopy(index["statistical_and_cost_boundaries"]),
            "source_inputs": {}, "missing_field_representation": None,
            "detail_compaction": "Original named scalar mappings; existing vectors still resolve with original field sets. No raw rollout/tensor."}
        filename = "per_outer_metrics_blocks/PER_OUTER_METRICS_OUTER" + format(first, "04d") + "_" + format(value["last_outer_capacity"], "04d") + ".json"
        info = {"file": filename, "first_outer": first, "last_outer_capacity": value["last_outer_capacity"],
                "actual_outers": [], "sealed": False, "created_by": "append_nested_local_costs_v1",
                "scope": "Complete original scalar fields; byte budget rolls into a new block without dropping fields."}
        return value, info
    if not block:
        block, active = fresh()
        index["scalar_blocks"].append(active)
    candidate = copy.deepcopy(block)
    candidate["rows"].append(row)
    candidate["source_inputs"].update(e["bindings"])
    candidate["actual_outers"] = [r["outer"] for r in candidate["rows"]]
    candidate["source_identity_by_commit"] = copy.deepcopy(index["source_identity_by_commit"])
    candidate["controller_common_fields"] = copy.deepcopy(index["controller_common_fields"])
    if len(encoded(candidate)) > max_bytes - 2048:
        if not block["rows"]:
            raise ValueError("Single scalar row exceeds byte budget; no fields discarded")
        active["sealed"] = True
        active["append_disabled_due_to_byte_budget"] = True
        candidate, active = fresh()
        index["scalar_blocks"].append(active)
        candidate["rows"] = [row]
        candidate["source_inputs"].update(e["bindings"])
        candidate["actual_outers"] = [step]
    candidate["sealed"] = step == candidate["last_outer_capacity"] or step == e["target"]
    raw = encoded(candidate)
    if len(raw) > max_bytes:
        raise ValueError("Single scalar block exceeds byte budget; no fields discarded")
    active.update(bytes=len(raw), sha256=sha(raw), actual_outers=candidate["actual_outers"], sealed=candidate["sealed"])
    row_index = len(candidate["rows"]) - 1
    known_dev = next((r for r in reversed(previous_rows) if isinstance(r.get("development_evaluation"), dict)), None)
    if row["development_evaluation"]:
        row["last_confirmed_dev512_ref"] = {"row_id": row["row_id"], "source_file": active["file"], "json_path": "rows[" + str(row_index) + "].development_evaluation"}
    elif known_dev:
        row["last_confirmed_dev512_ref"] = {"row_id": known_dev["row_id"], "source_file": known_dev["_block_file"], "json_path": "rows[" + str(known_dev["_block_row_index"]) + "].development_evaluation", "scope": "Actual prior dev, not current outer dev or timing."}
    else:
        row["last_confirmed_dev512_ref"] = None
    # Candidate contains the same row object only when freshly constructed; bind explicitly.
    candidate["rows"][row_index] = row
    raw = encoded(candidate)
    if len(raw) > max_bytes:
        raise ValueError("Complete row references exceed block byte budget; no publication")
    active.update(bytes=len(raw), sha256=sha(raw))
    index["effective_rows"].append({"row_id": row["row_id"], "scope": row["scope"], "source_commit": row["source_commit"],
                                    "outer": step, "effective_history": True, "block_file": active["file"], "block_row_index": row_index})
    history = index["effective_history"]
    history["latest_effective_outer"] = step
    history["latest_cumulative_calls_per_rank"] = row["optimizer"]["cumulative_calls_per_rank"]
    segments = history["effective_source_segments"]
    matching = next((s for s in segments if s["code_commit"] == row["source_commit"]), None)
    if matching:
        matching["last_outer"] = step
    else:
        identity = e["native"]["identity"]
        segments.append({"source_run": e["full"]["run"], "run_id": identity["run_id"], "code_commit": row["source_commit"],
                         "config_sha256": identity.get("config_sha256"), "first_outer": step, "last_outer": step})
    if row["source_commit"] == history.get("native5_child_source_commit"):
        history["native5_child_outer_steps"] = [r["outer"] for r in index["effective_rows"] if r["source_commit"] == row["source_commit"]]
    index.setdefault("local_append_tool", {"schema": "append_nested_local_costs_v1", "preserve_history_through_outer": PROTECTED_THROUGH,
        "initial_index_sha256": sha(mainraw), "initial_block_bytes_sha256": {v["file"]: v["sha256"] for v in index["scalar_blocks"] if max(v.get("actual_outers") or [0]) <= PROTECTED_THROUGH}})
    history_binding = index["local_append_tool"]
    history_binding["preserve_history_through_outer"] = max(history_binding.get("preserve_history_through_outer", 0), PROTECTED_THROUGH)
    history_binding.setdefault("preserved_extension_block_sha256", {}).update({v["file"]: v["sha256"] for v in index["scalar_blocks"] if max(v.get("actual_outers") or [0]) <= PROTECTED_THROUGH})
    index["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    index["preceding_compact_summary_sha256"] = sha(mainraw)
    md = append_md((base / BOOK_MD).read_text(), row, active, previous)
    atomic_copy(base, mirrors, active["file"], raw)
    atomic_copy(base, mirrors, BOOK_MD, md)
    atomic_copy(base, mirrors, BOOK, encoded(index))
    return index, True


def bounded_cumulative(result, through, max_bytes):
    """Keep every field, moving oversized evidence maps/lists into SHA-bound parts."""
    result = copy.deepcopy(result)
    artifacts = {}
    filename = "CUMULATIVE_COST_THROUGH" + str(through) + ".json"
    for key in sorted(result, key=lambda k: len(encoded(result[k])), reverse=True):
        if len(encoded(result)) <= max_bytes:
            break
        value = result[key]
        if not isinstance(value, (dict, list)):
            continue
        items = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        chunks = []
        current = {} if isinstance(value, dict) else []
        for itemkey, item in items:
            candidate = copy.deepcopy(current)
            if isinstance(value, dict):
                candidate[itemkey] = item
            else:
                candidate.append(item)
            payload = {"schema": "cumulative_scalar_field_part_v1", "original_field": key, "value": candidate}
            if len(encoded(payload)) > max_bytes and current:
                chunks.append(current)
                current = {itemkey: item} if isinstance(value, dict) else [item]
            else:
                current = candidate
        if current:
            chunks.append(current)
        refs = []
        for part, chunk in enumerate(chunks):
            name = "cumulative_cost_blocks/THROUGH" + format(through, "04d") + "_" + key + "_PART" + format(part, "03d") + ".json"
            raw = encoded({"schema": "cumulative_scalar_field_part_v1", "original_field": key, "value": chunk})
            if len(raw) > max_bytes:
                raise ValueError("One scalar evidence item exceeds byte budget; nothing discarded")
            artifacts[name] = raw
            refs.append({"file": name, "bytes": len(raw), "sha256": sha(raw), "json_selector": "value"})
        result[key] = {"storage": "complete_original_field_parts", "container_type": "mapping" if isinstance(value, dict) else "list",
                       "original_entries": len(value), "parts": refs}
    if len(encoded(result)) > max_bytes:
        raise ValueError("Cumulative index exceeds budget after retaining all fields in parts")
    artifacts[filename] = encoded(result)
    return artifacts


def join_candidate_backfill(args, index, rows):
    """SHA-join immutable candidate scalars into cumulative copies only."""
    base = args.base
    by_outer = {r["outer"]: r for r in rows}
    selected = args.candidate_backfill_index
    path = selected if selected.is_absolute() else base / selected
    if path.resolve().parent != base:
        raise ValueError("Backfill index must be a named file in the local audit dir")
    sources = []
    if path.exists():
        raw = path.read_bytes()
        if len(raw) > MAX_BYTES or sha(raw) != args.candidate_backfill_sha256:
            raise ValueError("Fixed candidate backfill index SHA/size differs")
        sources.append(("legacy3_19", path.name, raw, json.loads(raw)))
    registry_path = base / args.candidate_extension_inputs.name
    registry_binding = None
    if args.candidate_extension_inputs.is_absolute() and args.candidate_extension_inputs.resolve().parent != base:
        raise ValueError("Candidate extension input list must stay in local audit dir")
    if registry_path.exists():
        registry_raw = registry_path.read_bytes()
        if len(registry_raw) > MAX_BYTES:
            raise ValueError("Candidate input list exceeds scalar size budget")
        registry = json.loads(registry_raw)
        if registry.get("summary_schema") != "tree_candidate_cost_extension_inputs_v1":
            raise ValueError("Unexpected candidate extension input list schema")
        registry_binding = {"file": registry_path.name, "bytes": len(registry_raw), "sha256": sha(registry_raw),
                            "scope": "Derived input list snapshot; each original index/part is independently SHA bound."}
        registered = set()
        for ref in registry["indexes"]:
            outer = ref["outer"]
            if not isinstance(outer, int) or not 20 <= outer <= 200 or outer in registered:
                raise ValueError("Extension input list requires unique20..200 outers")
            registered.add(outer)
            if outer not in by_outer:
                continue
            name = ref["file"]
            if Path(name).name != name or name != "CANDIDATE_COST_EXTENSION_OUTER" + format(outer, "04d") + ".json":
                raise ValueError("Unexpected immutable candidate extension index filename")
            raw = (base / name).read_bytes()
            if len(raw) != ref["bytes"] or len(raw) > MAX_BYTES or sha(raw) != ref["sha256"]:
                raise ValueError("Candidate extension index immutable SHA/bytes differs")
            data = json.loads(raw)
            if data.get("requested_completed_outers") != [outer] or [v.get("outer") for v in data.get("parts", [])] != [outer]:
                raise ValueError("Candidate extension index does not bind its single completed outer")
            sources.append(("extension20_200", name, raw, data))
    applied, seen, index_bindings = [], {}, []
    for kind, name, raw, data in sources:
        if data.get("summary_schema") != "tree_candidate_cost_backfill_index_v1":
            raise ValueError("Unexpected candidate backfill schema")
        index_bindings.append({"kind": kind, "file": name, "bytes": len(raw), "sha256": sha(raw)})
        for info in data["parts"]:
            outer = info["outer"]
            allowed = range(3, 20) if kind == "legacy3_19" else range(20, 201)
            if outer not in allowed:
                raise ValueError("Candidate supplement outer outside its source scope; original1/2 stay separate")
            if outer in seen:
                if seen[outer] != info["sha256"]:
                    raise ValueError("Conflicting candidate summaries for one outer")
                continue
            seen[outer] = info["sha256"]
            if outer not in by_outer:
                continue
            partpath = (base / info["file"]).resolve()
            if not partpath.is_relative_to(base) or partpath.parent != base / "candidate_cost_backfill":
                raise ValueError("Candidate part is outside its local scalar directory")
            part_raw = partpath.read_bytes()
            if len(part_raw) != info["bytes"] or len(part_raw) > MAX_BYTES or sha(part_raw) != info["sha256"]:
                raise ValueError("Candidate part immutable SHA/bytes differs")
            part = json.loads(part_raw)
            if part.get("status") != "ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED":
                continue
            row = by_outer[outer]
            config = index["source_identity_by_commit"][row["source_commit"]]["identity"]["config_sha256"]
            if info["source_commit"] != row["source_commit"] or part.get("source_commit") != row["source_commit"] or info["config_sha256"] != config or part.get("config_sha256") != config or part.get("requested_outer") != outer:
                raise ValueError("Candidate outer/source/config join differs")
            if len(part.get("rollouts", [])) != 1:
                raise ValueError("Expected exactly one captured completed candidate rollout")
            value = part["rollouts"][0]
            if value.get("global_step") != outer or value.get("runtime_source_commit") != row["source_commit"] or value.get("config_sha256") != config:
                raise ValueError("Original candidate rollout identity differs")
            source = part["source_files"]
            audit, controller = source["audit"], value["rollout_controller_cost"]
            if not audit.get("stable_read") or audit["sha256"] != value["audit_file_sha256"] or audit["bytes"] != value["audit_file_bytes"] or source["resolved_config"]["sha256"] != config or not source["resolved_config"].get("stable_read") or not source["controller"].get("stable_read"):
                raise ValueError("Original candidate/config/controller file identity differs")
            if controller.get("status") != "TRAINING_ROLLOUT_COST_COMPLETE" or controller.get("code_commit") != row["source_commit"] or controller.get("config_sha256") != config or controller.get("candidate_outer") != outer:
                raise ValueError("Original candidate controller identity differs")
            if kind == "extension20_200":
                if row["source_commit"] != "778b2b43dad90fc0c1c5d95d77c2326286011b73" or config != "355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3":
                    raise ValueError("Candidate extension must use fixed778/config355")
                pin = part["source_pin"]
                if pin.get("experiment_origin") != "fresh" or pin.get("full_outer_receipt") != row.get("full_outer_evidence"):
                    raise ValueError("Candidate extension origin/credited receipt differs")
                evidence = actual_inputs(base, base / pin["full_outer_receipt"])
                if pin.get("immutable_local_inputs") != evidence["bindings"] or pin.get("native_manifest_sha256") != evidence["full"].get("native_manifest_sha256") or pin.get("source_poll") != evidence["poll_name"] or pin.get("source_poll_sha256") != evidence["bindings"][evidence["poll_name"]]["sha256"]:
                    raise ValueError("Candidate extension immutable receipt/poll/native SHA join differs")
                if pin.get("full_outer_receipt_sha256") != evidence["bindings"][evidence["full_name"]]["sha256"] or pin.get("native_metadata_readback_sha256") != evidence["bindings"][evidence["native_name"]]["sha256"]:
                    raise ValueError("Candidate extension FULL/native SHA differs")
                supplement = pin.get("supplemental_controller_poll")
                if supplement is not None:
                    sn, captured, sb = supplemental_controller_input(base, supplement, evidence)
                    if any(captured.get(key) != val for key, val in controller.items()):
                        raise ValueError("Supplemental controller differs from actual remote stable ledger")
                    original = evidence["controller"]
                    if original and any(original.get(key) != val for key, val in controller.items()):
                        raise ValueError("Supplement conflicts with original FULL controller evidence")
                    if not original:
                        row["rollout_controller_cost"] = copy.deepcopy(captured)
                        row["statistics_scope"]["controller_ledger_question_rows"] = captured.get("question_rows")
                    row["supplemental_controller_cost_evidence"] = {"source_file": sn, "source_sha256": sb["sha256"], "bytes": sb["bytes"], "scope": "Earlier COMPLETE sampling evidence only; completion/update credit remains bound to original FULL/native. Applied to this cumulative memory copy only; old scalar blocks and snapshots remain unchanged."}
            for field in ("candidate_outcomes", "candidate_round_controller_ledger"):
                row["candidate_cost_detail"][field] = {"source_file": info["file"], "source_sha256": info["sha256"], "json_selector": "rollouts[0]." + field}
            applied.append({"outer": outer, "source_commit": row["source_commit"], "config_sha256": config, "file": info["file"], "sha256": info["sha256"], "audit_file_sha256": audit["sha256"], "audit_file_bytes": audit["bytes"], **({"supplemental_controller_poll": pin["supplemental_controller_poll"]} if kind == "extension20_200" and pin.get("supplemental_controller_poll") is not None else {})})
    return {"status": "APPLIED_FIXED_LOCAL_BACKFILL" if sources else "NO_LOCAL_BACKFILL_INDEX", "index_file": path.name,
            "index_sha256": args.candidate_backfill_sha256 if path.exists() else None, "bound_indexes": index_bindings,
            "extension_input_list_snapshot": registry_binding, "applied_outers": [v["outer"] for v in applied], "inputs": applied,
            "original1_and2_refs_unchanged": True, "old_blocks_or_cumulative_files_modified": False,
            "scope": "Each original1/2 and each joined3..200 outer once. Missing next-outer candidate evidence stays null; no guessed padding or independent outcome wall."}


def cumulative(args):
    import json,hashlib,math,os,copy
    from pathlib import Path
    from datetime import datetime,timezone
    base=args.base;through=args.outer;mainbytes=(base/'PER_OUTER_METRICS_COMPACT.json').read_bytes();index=json.loads(mainbytes)
    def sha(b):return hashlib.sha256(b).hexdigest()
    def numeric(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    def decode(v,sets):return dict(zip(sets[v['field_set']],v['values'])) if isinstance(v,dict) and 'field_set' in v else v if isinstance(v,dict) else {}
    inputs={};snapshots=[];rows=[];immutable_inputs={}
    for info in index['scalar_blocks']:
     raw=(base/info['file']).read_bytes()
     if sha(raw)!=info['sha256']:raise RuntimeError('Scalar block snapshot differs from current index')
     b=json.loads(raw);sets=b['scalar_fieldsets'];immutable_inputs.update(b.get('source_inputs',{}))
     snapshots.append({'file':info['file'],'sha256_at_collection':sha(raw),'bytes':len(raw),'sealed':info['sealed'],'original_evidence':False,'scope':'Derived scalar snapshot;active block is appendable. Original fixed receipt/poll/metadata hashes are preserved separately.'})
     for r in b['rows']:
      if r['effective_history'] and 1<=r['outer']<=through:
       v=copy.deepcopy(r)
       for k in ['rollout_iteration_metric_ledger','variance_metrics','tree_sampling_metrics']:v[k]=decode(v.get(k),sets)
       v['rollout_controller_cost']=copy.deepcopy(v.get('rollout_controller_cost',{}))
       v['rollout_controller_cost']['costs']=decode(v['rollout_controller_cost'].get('costs'),sets)
       a=v.get('optimizer',{}).get('per_outer_direct_rank_summaries')
       if isinstance(a,dict) and 'field_set' in a:v['optimizer']['per_outer_direct_rank_summaries']=[dict(zip(sets[a['field_set']],x)) for x in a['values']]
       v['_block_file']=info['file'];rows.append(v)
    rows.sort(key=lambda r:r['outer'])
    if [r['outer'] for r in rows]!=list(range(1,through+1)):raise RuntimeError('Effective1..N rows are not uniquely present')
    candidate_backfill_join=join_candidate_backfill(args,index,rows)
    def resolve(ref):
     if not isinstance(ref,dict) or 'source_file' not in ref:return ref if isinstance(ref,dict) else {}
     name=ref['source_file'];raw=(base/name).read_bytes()
     if sha(raw)!=ref['source_sha256']:raise RuntimeError('Pinned referenced cost bytes changed')
     inputs[name]={'bytes':len(raw),'sha256':sha(raw)}
     d=json.loads(raw)
     if ref['json_selector']=='rollouts[0].candidate_round_controller_ledger':return d['rollouts'][0]['candidate_round_controller_ledger']
     if ref['json_selector']=='rollouts[0].candidate_outcomes':return d['rollouts'][0]['candidate_outcomes']
     raise RuntimeError('Unexpected scalar reference selector')
    def aggregate(values,scope):
     known=[(r['outer'],value) for r,value in zip(rows,values) if numeric(value)]
     missing=[r['outer'] for r,value in zip(rows,values) if not numeric(value)]
     total=math.fsum(value for _,value in known)
     if total.is_integer():total=int(total)
     return {'total':total if not missing else None,'known_sum':total,'known_outers':[s for s,_ in known],'missing_outers':missing,'complete':not missing,'scope':scope}
    def cost_value(r,key):
     v=r['rollout_controller_cost']['costs'].get(key)
     return v if numeric(v) else r['rollout_iteration_metric_ledger'].get('rollout/'+key)
    sampling={}
    for key in ['generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_emitted_tokens','raw_retained_tokens','raw_discarded_tail_tokens','raw_inherited_leaf_tokens','semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds']:
     sampling[key]=aggregate([cost_value(r,key) for r in rows],'Original Controller.generate native scalar event ledger;generated/emitted includes DP padding,prefill is submitted events,not uniqueKV/FLOPs.')
    for key in ['scoring_calls','scoring_wall_seconds']:
     sampling[key]=aggregate([r['rollout_iteration_metric_ledger'].get('rollout/'+key) for r in rows],'Whole training rollout scoring metric;never inferred from trajectories or candidate count.')
    candidate={}
    for key in ['all','accepted','discarded','fallback']:
     candidate[key]=aggregate([r['candidate_cost_detail'].get('candidate_counts',{}).get(key) for r in rows],'Original candidate count/flag;fallback overlaps accepted and is not a third additive bucket.')
    branch_ledgers={r['outer']:resolve(r['candidate_cost_detail'].get('candidate_round_controller_ledger')) for r in rows}
    branch={}
    for key in ['generated_tokens','prefill_tokens','retrieval_calls','raw_emitted_tokens','scoring_calls','scoring_wall_seconds']:
     vals=[]
     for r in rows:
      value=r['rollout_iteration_metric_ledger'].get('rollout/branch_'+key)
      if not numeric(value):value=branch_ledgers[r['outer']].get(key)
      vals.append(value)
     branch[key]=aggregate(vals,'Additional branch/candidate controller ledger;original parent1/ba9_2 candidate round ledger only where bound. Narrower than whole rollout;no unknown outcome padding allocated.')
    timings={key:aggregate([r['stage_timing_seconds'].get('timing_s/'+key) for r in rows],'Original stage label;whole step already contains its phases,so stage sums are not added again to step.')
     for key in ['step','gen','ref','adv','update_actor','stage_testing','testing','save_checkpoint']}
    rolloutwall=aggregate([r['rollout_controller_cost'].get('wall_seconds') for r in rows],'Controller rollout wall;included inside whole outer. Each actual outer ledger counted once.')
    drivercpu=aggregate([r['rollout_controller_cost'].get('trainer_process_cpu_seconds') for r in rows],'Trainer process CPU during Controller.generate only;excludes worker/retriever/service CPU and is not whole job CPU.')
    rankwrites={str(rank):0 for rank in range(8)};rankskips={str(rank):0 for rank in range(8)};direct_outers=[]
    for r in rows:
     a=r['optimizer'].get('per_outer_direct_rank_summaries')
     if isinstance(a,list) and len(a)==8 and all(numeric(item.get('successful_optimizer_calls')) and numeric(item.get('skipped_empty_minibatches')) for item in a):
      direct_outers.append(r['outer'])
      for item in a:
       rankwrites[str(item['rank'])]+=item['successful_optimizer_calls'];rankskips[str(item['rank'])]+=item['skipped_empty_minibatches']
    actual_counts=rows[-1]['native_manifest_metadata']['actual_optimizer_calls_per_rank_from_native_writer']
    optimizer={'successful_calls_per_rank_through'+str(through):actual_counts,'successful_rank_call_events_through'+str(through):sum(actual_counts.values()) if all(numeric(v) for v in actual_counts.values()) else None,
     'source':f'Actual native{through} writer cumulative counters already include inherited parent1 and ba9 2..5;do not sum repeated cumulative snapshots or add inherited100 again.',
     'direct_observed_calls_per_rank':rankwrites,'direct_observed_rank_call_events':sum(rankwrites.values()),'direct_record_outers':direct_outers,'missing_direct_record_outers':[r['outer'] for r in rows if r['outer'] not in direct_outers],
     'skipped_empty_minibatches_total':None if len(direct_outers)!=through else sum(rankskips.values()),'known_skips_per_rank':rankskips,
     'optimizer_opportunities_per_rank':aggregate([r['optimizer'].get('iteration_metric_fields',{}).get('actor/optimizer_opportunities_processed') if isinstance(r['optimizer'].get('iteration_metric_fields'),dict) else None for r in rows],'Sum of original reported actor opportunity metric across recorded effective outers. Parent1 opportunity20 is from complete metric line;source rank files prove writes/skips but lack separate opportunity fields.')}
    optimizer['latest_outer_original_receipt_opportunity_fields']=copy.deepcopy(rows[-1].get('full_outer_receipt_accounting',{}))
    optimizer['opportunity_evidence_scope']='Original per-outer actor metric20/skip0 is kept apart from individual rank opportunity records. Latest helper preserves actual per-rank processed/skipped opportunity fields including null;successful writes and empty-minibatch skips remain separate directly captured fields. No independent rank opportunity0 inferred.'
    dev=[]
    for r in rows:
     e=r.get('development_evaluation')
     if isinstance(e,dict) and e.get('rows')==512 and e.get('status')=='VALIDATION_COST_COMPLETE':
      metrics=e.get('metrics',{})
      dev.append({'outer':r['outer'],'source_commit':r['source_commit'],'ledger':e['path'],'ledger_sha256':e.get('ledger_sha256'),
       'rows':e['rows'],'phase':e['phase'],'testing_phase_wall_seconds':r['stage_timing_seconds'].get('timing_s/testing',r.get('full_outer_receipt_accounting',{}).get('normal_dev512_wall_seconds')),
       'generated_token_event_total':None,'prefill_token_event_total':None,'retrieval_call_event_total':None,'scoring_call_event_total':None,
       'recorded_means':{k:v for k,v in metrics.items() if 'generated_tokens' in k or 'search_calls' in k},
       'F1':metrics.get('val/f1/all'),'EM':metrics.get('val/em/all'),
       'scope':'Only original completed dev ledger and testing phase wall. Recorded means are not expanded into native cost totals. This phase is already included in outer step wall.'})
    devwall=math.fsum(d['testing_phase_wall_seconds'] for d in dev) if all(numeric(d['testing_phase_wall_seconds']) for d in dev) else None
    historical=[]
    for r in index['historical_failed_rows']:
     c=r['rollout_controller_cost'];cv=decode(c.get('costs'),index['scalar_fieldsets'])
     historical.append({'row_id':r['row_id'],'source_commit':r['source_commit'],'outer':r['outer'],'effective_outer_credit':0,'effective_successful_write_credit':0,
     'rollout_scalar_costs':cv,'controller_rollout_wall_seconds':c.get('wall_seconds'),'trainer_process_cpu_seconds':c.get('trainer_process_cpu_seconds'),
     'whole_failed_outer_wall_seconds':None,'scoring_calls':None,'scoring_wall_seconds':None,'candidate_counts':None,
     'actual_failed_outer_optimizer_writes':None,'failure_component':r.get('failure_provenance',{}).get('failure_component',r.get('failure_provenance',{}).get('failed_terminal',{}).get('failure_component')),
     'historical_cumulative_calls_before_failure':r.get('optimizer',{}).get('historical_cumulative_calls_per_rank_before_failure'),'cost_evidence':r.get('cost_evidence'),
     'scope':'Separate historical failed cost,excluded from the effective chain and successful counts. Unknown partial physical writes are not asserted zero.'})
    boot=[{**copy.deepcopy(r),'wall_seconds':None} for r in index['historical_bootstrap_failures']]
    outcomes={}
    for outcome in ['all','accepted','discarded','fallback']:
     data_by_outer={}
     for r in rows:
      d=resolve(r['candidate_cost_detail'].get('candidate_outcomes'))
      if isinstance(d.get(outcome),dict):data_by_outer[r['outer']]=d[outcome].get('field_sums',{})
     fields=sorted({field for d in data_by_outer.values() for field in d})
     outcomes[outcome]={'known_outers':sorted(data_by_outer),'missing_outers':[r['outer'] for r in rows if r['outer'] not in data_by_outer],
     'field_sums':{field:{'total':math.fsum(d[field] for d in data_by_outer.values()) if len(data_by_outer)==through and all(numeric(d.get(field)) for d in data_by_outer.values()) else None,'known_sum':math.fsum(d[field] for d in data_by_outer.values() if numeric(d.get(field))),'known_outers':[o for o,d in data_by_outer.items() if numeric(d.get(field))],'missing_outers':[r['outer'] for r in rows if not numeric(data_by_outer.get(r['outer'],{}).get(field))]} for field in fields},
     'scope':'Original exact candidate outcome scalar sums only. Amortized parallel wall shares and inherited-prefix event counts retain original scopes;fallback overlaps accepted.'}
    outerwall=timings['step']['total'];boundseconds=8*outerwall if numeric(outerwall) else None
    result={'summary_schema':'tree_nested_cumulative_cost_v1','created_utc':datetime.now(timezone.utc).isoformat(),'through_effective_outer':through,'target_outer':200,
     'effective_source_segments':[{**v,'last_outer':min(v['last_outer'],through)} for v in index['effective_history']['effective_source_segments'] if v['first_outer']<=through],
     'origin':'fresh','rows_counted_once':[{'outer':r['outer'],'source_commit':r['source_commit'],'scope':r['scope'],'row_id':r['row_id']} for r in rows],
     'derived_input_snapshots':{'index':{'file':'PER_OUTER_METRICS_COMPACT.json','sha256_at_collection':sha(mainbytes),'bytes':len(mainbytes),'original_evidence':False},'scalar_blocks':snapshots},
     'immutable_metadata_input_hashes':{k:v for k,v in immutable_inputs.items() if k!='CURRENT_NATIVE_RECOVERY_FOLLOWUP.json'},'resolved_fixed_cost_inputs':inputs,
     'effective_training_sampling':sampling,'effective_extra_candidates':candidate,'effective_branch_controller_costs':branch,'candidate_outcome_costs_partial':outcomes,
     'complete_outer_wall_seconds':outerwall,'complete_outer_wall_hours':outerwall/3600 if numeric(outerwall) else None,'stage_timing_seconds':timings,'controller_rollout_wall_seconds':rolloutwall,
     'controller_trainer_process_cpu_seconds':drivercpu,'actual_total_service_cpu_seconds':None,'successful_optimizer_accounting':optimizer,
     'completed_dev_evaluations':dev,'completed_dev_testing_phase_wall_seconds':devwall,'dev_phase_already_in_outer_wall':True,
     'physical8gpu_shared_capacity_upper_bound':{'physical_gpu_count':8,'effective_outer_wall_seconds':outerwall,'capacity_gpu_seconds_upper_bound':boundseconds,'capacity_gpu_hours_upper_bound':boundseconds/3600 if numeric(boundseconds) else None,
     'actual_gpu_active_seconds':None,'actual_gpu_active_hours':None,'exclusive_billable_gpu_hours':None,'gpu_energy_kwh':None,
     'scope':'Physical8GPU shared wall capacity upper bound only. Includes CPU waits/dev/save phases;not measured utilization and not exclusive GPU occupancy,billing,or incremental consumption.'},
     'historical_failed_costs':historical,'historical_failed_rollout_known_subtotal':{'generated_tokens':sum(v['rollout_scalar_costs']['generated_tokens'] for v in historical),'prefill_tokens':sum(v['rollout_scalar_costs']['prefill_tokens'] for v in historical),'retrieval_calls':sum(v['rollout_scalar_costs']['retrieval_calls'] for v in historical),'controller_rollout_wall_seconds':math.fsum(v['controller_rollout_wall_seconds'] for v in historical)},
     'historical_bootstrap_costs':boot,'historical_whole_service_segment_context':copy.deepcopy(index['historical_segment_costs']),
     'no_combined_effective_plus_failed_or_segment_grand_total':True,
     'boundaries':['Cutoff is the actual effective chain through the requested outer;all later outers excluded.','Actual latest native cumulative counts already include ancestors;failed mother2,failed ba9_6 and EB6 bootstrap receive no effective credit.','Whole ba9 terminal service CPU includes accepted2..5 plus failed6/startup; its run/supervisor wall is a separately recorded scope;never add it again to its subphase costs.','Generated/emitted includes DP padding;discarded candidate counts differ from raw discarded tail tokens. Unknown padding shares are null.','Question ledger1024 means two original roots for512 training questions,not1024 questions.','Stage/dev wall is contained in complete outer wall;do not add phases again. No round elapsed is multiplied by question count.','Total is null if any required outer lacks its field;known_sum and coverage remain explicit. Scoring totals are not guessed from trajectory count or evaluation means.','Only local scalar JSON was read;no tensor/rollout transfer,no remote query,no review/test,no train/status/source mutation.'],
     'mirrors':index['mirrors']}
    def clean(v):
     if v=='unknown':return None
     if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
     if isinstance(v,list):return [clean(x) for x in v]
     return v
    result=clean(result)
    result['candidate_cost_backfill_join']=candidate_backfill_join
    result['pending_full_metric_outers']=[r['outer'] for r in rows if r.get('full_metric_status','').startswith('PENDING')]
    result['completed_dev_testing_known_sum_seconds']=math.fsum(d['testing_phase_wall_seconds'] for d in dev if numeric(d['testing_phase_wall_seconds']))
    result['completed_dev_testing_missing_wall_outers']=[d['outer'] for d in dev if not numeric(d['testing_phase_wall_seconds'])]
    artifacts=bounded_cumulative(result,through,args.max_block_bytes)
    def n(v):return 'null' if v is None else format(v,',.6f').rstrip('0').rstrip('.') if isinstance(v,float) else format(v,',') if isinstance(v,int) else str(v)
    lines=['# 截至有效第'+str(through)+'轮累计成本','',
     '只计实际有效ancestry与原生完成轮次，每轮一次；未来轮次不计。完整目标仍200轮，失败与启动成本另列。','',
     '|项目|完整累计|已知部分累计|缺失轮次|','|---|---:|---:|---|']
    for label,record in [('训练emitted token事件（含padding）',sampling['generated_tokens']),('训练prefill事件',sampling['prefill_tokens']),('训练检索调用',sampling['retrieval_calls']),('训练评分调用',sampling['scoring_calls']),('训练评分wall秒',sampling['scoring_wall_seconds']),('额外候选总数',candidate['all']),('accepted候选',candidate['accepted']),('discarded候选',candidate['discarded']),('fallback重叠标志',candidate['fallback']),('额外branch emitted事件',branch['generated_tokens']),('额外branch prefill事件',branch['prefill_tokens']),('额外branch检索',branch['retrieval_calls'])]:
     lines.append('|'+ '|'.join([label,n(record['total']),n(record['known_sum']),','.join(map(str,record['missing_outers'])) or '无'])+'|')
    lines+=['',f'完整outer墙钟累计 **{n(outerwall)}秒（{n(outerwall/3600 if numeric(outerwall) else None)}小时）**；controller rollout wall {n(rolloutwall["total"])}秒。两者是包含关系，不相加。',
     f'实际native{through}累计各rank调用为 {actual_counts}；原生累计已含祖先，不重复加恢复计数。直接记录/机会/empty-skip按原字段及覆盖范围保存在JSON，未知为null。母1后补证据及原修正记录保留。','',
     f'训练评分1–{through}实际累计 {n(sampling["scoring_calls"]["total"])} 次，评分wall累计 {n(sampling["scoring_wall_seconds"]["total"])}秒。母1由此前新读取的原stage.log完整metric行432补齐3553次/30.04257808253169秒，不由trajectory推算。Candidate fallback与accepted重叠，不加到候选总数；discarded候选不是原始discarded-tail token。','',
     '每rank独立processed/skipped opportunity保留实际值，缺失为null；完整metric、成功写入与empty-skip字段分开，不由成功次数推算机会。','',
     '## 已实际完成的dev512','',
     '|dev轮|rows|testing wall秒|F1|EM|native token/prefill/检索/评分总量|','|---|---:|---:|---:|---:|---|']
    for d in dev:lines.append('|'+ '|'.join([str(d['outer']),str(d['rows']),n(d['testing_phase_wall_seconds']),n(d['F1']),n(d['EM']),'null（只保留原均值，不展开）'])+'|')
    lines +=['',f'已完成dev testing阶段共 {n(devwall)}秒，已包含在outer墙钟中。保存阶段不是dev成本；缺失的阶段或原生token成本不填0。原ledger及recorded means保存在JSON。','',
     '## 共享GPU上限与实际成本','',
     f'按物理8GPU容量计算，共享墙钟容量上限为 **{n(boundseconds)} GPU秒（{n(boundseconds/3600 if numeric(boundseconds) else None)} GPU小时）**。这包括CPU等待、dev及保存时间，只是上限；实际GPU活跃时间、独占占用/计费及能耗均为null。不能把8×outer墙钟当作独占GPU账单。','',
     '## 历史失败成本（不计有效更新）','',
     '|记录|emitted含padding|prefill|检索|rollout wall秒|有效更新信用|','|---|---:|---:|---:|---:|---:|']
    for h in historical:
     v=h['rollout_scalar_costs'];lines.append('|'+ '|'.join([h['source_commit'][:8]+'失败'+str(h['outer']),n(v['generated_tokens']),n(v['prefill_tokens']),n(v['retrieval_calls']),n(h['controller_rollout_wall_seconds']),'0'])+'|')
    lines+=['',f'两次历史失败rollout已知emitted共 {n(result["historical_failed_rollout_known_subtotal"]["generated_tokens"])}，rollout wall共 {n(result["historical_failed_rollout_known_subtotal"]["controller_rollout_wall_seconds"])}秒；完整失败outer wall、部分实际写入、候选/评分缺失为null。',
     'EB6是独立启动失败：CPUUsageNSec482013000（0.482013 CPU秒），wall null，未创建训练run/加载native/开始outer；有效信用0。',
     'ba9实际整服务CPUUsageNSec29067275364000、已记录run/supervisor wall5552.515588521957秒，范围涵盖有效2–5、失败6及相关启动恢复阶段，单列为上下文，不能再加到同段细分或称失败6专属成本。','',
     '原始rollout/tensor仍留远端。主v3索引及active块哈希仅标识本次派生读取快照；原始固定证据由immutable receipt/poll/native metadata的SHA绑定，未固定动态CURRENT/STATUS。','',
     '生成时间：'+result['created_utc']+'。本次仅本地汇总和复制小摘要，无新增训练门槛、远端查询、review/测试或训练/STATUS/REPORT/源码修改。']
    md=('\n'.join(lines)+'\n').encode()
    artifacts['CUMULATIVE_COST_THROUGH'+str(through)+'.md']=md
    for name,raw in artifacts.items():
     for root in [base]+[Path(v) for v in index['mirrors']]:
      dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.tmp');tmp.write_bytes(raw);os.replace(tmp,dst)
      if dst.read_bytes()!=raw:raise RuntimeError('Small summary mirror readback differs')
    return {'through':through,'artifacts':{name:{'bytes':len(raw),'sha256':sha(raw)} for name,raw in artifacts.items()},'pending_full_metric_outers':result['pending_full_metric_outers'],'complete_outer_wall_seconds':outerwall}
    


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Existing local task audit directory")
    parser.add_argument("--receipt", required=True, type=Path, help="Actual immutable FULL_OUTERN_PROGRESS_RECEIPT.json")
    parser.add_argument("--max-block-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--candidate-backfill-index", type=Path, default=Path(BACKFILL_INDEX))
    parser.add_argument("--candidate-backfill-sha256", default=BACKFILL_SHA256)
    parser.add_argument("--candidate-extension-inputs", type=Path, default=Path("CANDIDATE_COST_EXTENSION_INPUTS.json"))
    args = parser.parse_args()
    args.base = args.base.resolve(strict=True)
    if args.receipt.is_absolute() and args.receipt.resolve().parent != args.base:
        raise ValueError("Receipt must be within the explicitly selected local audit directory")
    receipt_path = args.base / args.receipt.name
    if not receipt_path.exists():
        print(json.dumps({"status": "PENDING_IMMUTABLE_FULL_RECEIPT", "receipt": str(receipt_path)}))
        return
    if not 10000 <= args.max_block_bytes <= MAX_BYTES:
        raise ValueError("Block byte budget must be10000..100000")
    args.outer = read_json(receipt_path)["completed_outer"]
    before = read_json(args.base / BOOK)
    protected = {info["file"]: (args.base / info["file"]).read_bytes()
                 for info in before["scalar_blocks"] if max(info.get("actual_outers") or [0]) <= PROTECTED_THROUGH}
    index, appended = append_one(args.base, receipt_path, args.max_block_bytes)
    cumulative_path = args.base / ("CUMULATIVE_COST_THROUGH" + str(args.outer) + ".json")
    cost = cumulative(args) if not cumulative_path.exists() else {"status": "EXISTING_CUMULATIVE_PRESERVED"}
    for name, raw in protected.items():
        if (args.base / name).read_bytes() != raw:
            raise RuntimeError("Protected1..20 history bytes changed")
    print(json.dumps({"status": "APPENDED" if appended else "ALREADY_RECORDED",
                      "outer": args.outer, "source_receipt": args.receipt.name,
                      "index_sha256": sha((args.base / BOOK).read_bytes()),
                      "protected_history_bytes_unchanged": True, "cumulative": cost,
                      "remote_operations": False, "training_status_or_source_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

