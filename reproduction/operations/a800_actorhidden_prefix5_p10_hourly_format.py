"""Format an existing P10 observer receipt against an actually delivered anchor.

Pure local scalar/hash processing. No SSH, network, model reads, evaluation or
anchor mutation. Output is JSON; --markdown PATH additionally saves a summary.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


UNKNOWN = "unknown_not_explicitly_recorded"
HISTORY_FIELDS = (
    "outer", "actor_adam_per_rank", "probe_completed_outer", "probe_adam_per_rank",
    "reward_mean", "reward_min", "reward_max", "selected_mixed_fraction",
    "selected_mixed_count", "selected_strict_correct", "actual_em_selection",
    "candidate_mixed_count", "fallback_count", "all_correct_groups",
    "all_wrong_groups", "mixed_groups", "joint_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, low=0, high=None):
    require(type(value) is int and value >= low, name + " must be an integer in range")
    require(high is None or value <= high, name + " exceeds range")
    return value


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    return value


def score(value, name):
    value = number(value, name)
    require(0 <= value <= 1, name + " is outside [0,1]")
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), name + " is not SHA256")
    return value


def clock(outer):
    integer(outer, "outer", 0, 200)
    last = min(outer, 199) // 10 * 10
    return last, last // 10 * 50


def four_counts(value, name, expected=None, high=None):
    require(isinstance(value, list) and len(value) == 4, name + " requires four ranks")
    for count in value:
        integer(count, name, high=high)
    require(len(set(value)) == 1, name + " differs across ranks")
    require(expected is None or value == [expected] * 4, name + " disagrees with boundary")
    return value


def safe_identity(value):
    if value is None:
        return UNKNOWN
    result = {name: integer(value[name], "identity." + name) for name in
              ("pid", "start_ticks", "sid", "uid")}
    require(type(value.get("alive")) is bool, "identity.alive must be recorded")
    result["alive"] = value["alive"]
    return result


def actor_boundary(value, run_id, outer, bound_projection=False):
    require(value.get("run_id", run_id if bound_projection else None) == run_id, "Actor run differs")
    require(value["actor_version"] == "%s-actor%04d" % (run_id, outer), "Actor version differs")
    require(value["completed_outer"] == outer, "Actor boundary differs")
    require(value["scheduler_completed_outer"] == outer, "Actor scheduler differs")
    return four_counts(value["actual_adam_steps_per_rank"], "actor Adam", 20 * outer)


def probe_boundary(value, run_id, outer, bound_projection=False):
    last, updates = clock(outer)
    require(value.get("run_id", run_id if bound_projection else None) == run_id, "Probe run differs")
    require(value["completed_outer"] == last, "P10 probe completed boundary differs")
    return four_counts(value["actual_adam_steps_per_rank"], "probe Adam", updates)


def safe_qa(value, complete_outer, anchor_outer, anchor_qa):
    outer = integer(value["outer"], "QA outer", 1, 200)
    require(outer in (5, 50, 100, 150, 200), "Unexpected QA milestone")
    if value["status"] != "EXISTING_NATIVE_QA_COMPLETE":
        return {"outer": outer, "status": "not_generated_or_unknown_not_zero"}
    require(outer <= complete_outer and value["qa_count"] == 512, "QA boundary/count differs")
    sources = value["per_source"]
    require(set(sources) == {"nq", "hotpotqa"}, "QA source names differ")
    previous = anchor_qa.get(outer)
    if previous is not None:
        changed = previous != value["metadata_sha256"]
    elif outer > anchor_outer:
        changed = True
    else:
        changed = UNKNOWN
    result = {"outer": outer, "status": value["status"], "qa_count": 512,
              "em": score(value["em"], "QA EM"), "f1": score(value["f1"], "QA F1"),
              "new_milestone_since_anchor": changed, "per_source": {}}
    for name, row in sources.items():
        require(row["qa_count"] == 256, "QA source count differs")
        result["per_source"][name] = {"qa_count": 256, "em": score(row["em"], "source EM"),
                                              "f1": score(row["f1"], "source F1")}
    for key in ("metadata_sha256", "question_source_sha256"):
        result[key] = digest(value[key], "QA " + key)
    require(len(value["output_shards"]) == len(value["rank_receipt_sha256"]) == 4,
            "QA needs four shard and rank seals")
    result["output_shards"] = [
        {"bytes": integer(row["bytes"], "QA shard bytes"),
         "sha256": digest(row["sha256"], "QA shard SHA")} for row in value["output_shards"]]
    result["rank_receipt_sha256"] = [digest(item, "QA rank SHA") for item in value["rank_receipt_sha256"]]
    return result


def format_report(record, anchor, receipt_sha, anchor_sha):
    require(record.get("exit_code") == 0 and not record.get("timed_out"),
            "Observer failed or timed out; preserve unknown")
    require(record.get("remote_mutations") is False, "Receipt is not read-only")
    actual = json.loads(record["stdout"])
    require(actual.get("operator_contract") == "a800_actorhidden_prefix5_p10_bounded_scalar_observer_v1",
            "Receipt is not the current P10 scalar observer")
    require(actual.get("metadata_only") is True and actual.get("models_or_tensors_read") is False,
            "Receipt is not metadata-only")
    require(actual.get("no_new_evaluation_or_model_readback") is True,
            "Receipt lacks no-new-evaluation scope")
    for key in ("run_id", "execution_id", "source_commit", "base"):
        require(isinstance(anchor.get(key), str) and anchor[key] == actual.get(key),
                "Delivered anchor " + key + " differs from receipt")
        if key != "base":
            require(record.get(key) == actual[key], "Receipt envelope " + key + " differs")
    require(anchor.get("delivered") is True, "Anchor was not actually delivered")
    observed = number(actual["observed_unix"], "observed_unix")
    delivered = number(anchor["delivered_unix"], "delivered_unix")
    require(delivered <= observed, "Anchor delivery is after observation")
    start = integer(anchor["complete_outers"], "anchor complete_outers", 0, 200)
    require(anchor.get("completed_outer") == start, "Anchor boundary aliases differ")
    end = integer(actual["complete_outer"], "complete_outer", 0, 200)
    require(start <= end, "Anchor boundary is newer than observation")
    if "actor_adam" in anchor:
        require(anchor["actor_adam"] == 20 * start, "Anchor actor clock differs")
    if "online_probe_adam" in anchor:
        require(anchor["online_probe_adam"] == clock(start)[1], "Anchor probe clock differs")

    columns = actual["history_columns"]
    require(len(columns) == len(set(columns)) and set(HISTORY_FIELDS).issubset(columns),
            "Scalar history columns differ")
    raw_history = actual["complete_outer_scalar_history"]
    require(len(raw_history) == end, "Complete history must contain exactly 1..n")
    history = []
    for outer, row in enumerate(raw_history, 1):
        require(isinstance(row, list) and len(row) == len(columns), "History row shape differs")
        item = dict(zip(columns, row))
        require(item["outer"] == outer and type(item["outer"]) is int, "History has gaps or duplicates")
        last, probe_steps = clock(outer)
        for key in ("actor_adam_per_rank", "probe_completed_outer", "probe_adam_per_rank"):
            integer(item[key], key)
        require(item["actor_adam_per_rank"] == 20 * outer, "History actor clock differs")
        require(item["probe_completed_outer"] == last and item["probe_adam_per_rank"] == probe_steps,
                "History P10 probe clock differs")
        mean = number(item["reward_mean"], "reward mean")
        for key in ("reward_min", "reward_max"):
            if item[key] is not None:
                number(item[key], key)
        require(item["reward_min"] is None or item["reward_min"] <= mean, "Reward below minimum")
        require(item["reward_max"] is None or item["reward_max"] >= mean, "Reward above maximum")
        mixed = integer(item["selected_mixed_count"], "mixed count", 0, 512)
        require(abs(score(item["selected_mixed_fraction"], "mixed fraction") - mixed / 512) <= 1e-12,
                "Mixed count and fraction differ")
        groups = [integer(item[key], key, 0, 512) for key in
                  ("all_correct_groups", "all_wrong_groups", "mixed_groups")]
        require(sum(groups) == 512 and groups[2] == mixed, "Group counts differ from 512/mixed selection")
        integer(item["selected_strict_correct"], "selected strict correct", 0, 2560)
        require(type(item["actual_em_selection"]) is bool, "EM-selection flag is not boolean")
        due = outer % 10 == 0 and outer < 200
        require(item["actual_em_selection"] is due, "EM selection is outside its P10 update boundary")
        if due:
            candidate = integer(item["candidate_mixed_count"], "candidate mixed", 0, 2560)
            fallback = integer(item["fallback_count"], "fallback", 0, 512)
            require(mixed == min(candidate, 512) and fallback == 512 - mixed, "EM-priority fallback differs")
        digest(item["joint_sha256"], "joint SHA")
        history.append({key: item[key] for key in HISTORY_FIELDS})
    if start and "reward_mean" in anchor:
        require(abs(number(anchor["reward_mean"], "anchor reward") - history[start - 1]["reward_mean"]) <= 1e-12,
                "Anchor reward does not match its own outer")
    contract = actual["probe_clock_contract"]
    require(contract["period"] == 10 and contract["updates_per_period"] == 50
            and contract["skip_update_at_outer200"] is True
            and contract["last_update_outer"] == clock(end)[0], "P10 contract differs")
    four_counts(contract["actual_adam_steps_per_rank"], "contract probe count", clock(end)[1])
    probe_history = actual["probe_update_history"]
    due_outers = list(range(10, min(end, 199) + 1, 10))
    require(len(probe_history) == len(due_outers), "Probe update history has gaps")
    for outer, item in zip(due_outers, probe_history):
        require(item["outer"] == outer and item["new_adam_steps_per_rank"] == 50,
                "Probe update event clock differs")
        four_counts(item["actual_adam_steps_per_rank"], "probe event count", clock(outer)[1])
        if "actual_learning_rate" in item:
            require(number(item["actual_learning_rate"], "probe LR") == 5e-6, "Probe LR differs")

    joint = actual.get("latest_joint")
    if end:
        require(joint is not None and joint["outer"] == end, "Latest joint boundary differs")
        # The current observer deliberately projects run_id out of the nested
        # joint clock, after checking its producer identity remotely. Bind this
        # typed scalar projection to the validated envelope and actor version.
        actor_boundary(joint["actor"], actual["run_id"], end, bound_projection=True)
        probe_boundary(joint["probe"], actual["run_id"], end, bound_projection=True)
    latest_actor = actual.get("actor_latest")
    latest_actor_boundary = end
    latest_actor_steps = [0] * 4
    if latest_actor is not None:
        latest_actor_boundary = integer(latest_actor["completed_outer"], "latest actor boundary", end, min(end + 1, 200))
        latest_actor_steps = actor_boundary(latest_actor, actual["run_id"], latest_actor_boundary)
    latest_probe = actual.get("probe_latest")
    latest_probe_steps = [0] * 4
    if latest_probe is not None:
        p_outer = integer(latest_probe["completed_outer"], "latest probe boundary", 0, 190)
        require(p_outer in {clock(end)[0], clock(latest_actor_boundary)[0]}, "Uncommitted probe clock differs")
        latest_probe_steps = probe_boundary(latest_probe, actual["run_id"], p_outer)
    attempts = []
    for item in actual.get("actor_attempts", []):
        counts = four_counts(item["confirmed_returned_per_rank"], "current returned Adam", high=20)
        require(item["four_ranks_agree"] is True, "Current actor four-rank agreement missing")
        ranks = item["ranks"]
        require(len(ranks) == 4 and {rank["rank"] for rank in ranks} == set(range(4)), "Current actor ranks differ")
        for rank in ranks:
            require(rank["confirmed_returned_adam"] == counts[rank["rank"]], "Current actor rank count differs")
            digest(rank["ledger_sha256"], "current actor ledger SHA")
        attempts.append({"confirmed_returned_Adam_per_rank": counts,
                         "returned_Adam_counted_once": counts[0],
                         "ledger_sha256_by_rank": [rank["ledger_sha256"] for rank in sorted(ranks, key=lambda r: r["rank"])],
                         "failure_recorded_by_rank": [rank.get("failure") is not None for rank in sorted(ranks, key=lambda r: r["rank"])],
                         "inflight": UNKNOWN, "processed": UNKNOWN, "skip": UNKNOWN})

    window = history[start:end]
    means = [row["reward_mean"] for row in window]
    mixed_count = sum(row["selected_mixed_count"] for row in window)
    denominator = len(window) * 512
    latest_reward = history[-1]["reward_mean"] if history else UNKNOWN
    baseline_reward = history[start - 1]["reward_mean"] if start else None
    anchor_qa = {}
    for item in anchor.get("qa_milestones", []):
        if item.get("metadata_sha256"):
            anchor_qa[integer(item["outer"], "anchor QA outer", 1, 200)] = digest(
                item["metadata_sha256"], "anchor QA SHA")
    qa = [safe_qa(item, end, start, anchor_qa) for item in actual.get("qa_milestones", [])]
    require(len({item["outer"] for item in qa}) == len(qa), "Duplicate QA milestones")
    new_qa = [item["outer"] for item in qa if item.get("new_milestone_since_anchor") is True]
    unknown_qa = [item["outer"] for item in qa if item.get("new_milestone_since_anchor") == UNKNOWN]
    result = {"role": "p10_existing_receipt_hourly_scalar_window", "asof_unix": observed,
              **{key: actual[key] for key in ("run_id", "execution_id", "source_commit", "base")},
              "receipt_file_sha256": digest(receipt_sha, "receipt file SHA"),
              "anchor_file_sha256": digest(anchor_sha, "anchor file SHA"),
              "anchor_delivered_unix": delivered, "formal_hourly_due_unix": delivered + 3600,
              "formal_hourly_due_at_observation": observed >= delivered + 3600,
              "anchor_mutated": False, "new_evaluation_or_remote_call": False,
              "complete_outers": end, "target_outers": 200,
              "actor_returned_Adam_counted_once": 20 * end,
              "actor_processed": UNKNOWN, "actor_skip": UNKNOWN,
              "online_probe_Adam_counted_once": clock(end)[1],
              "window": {"anchor_complete_outer_excluded": start,
                         "first_outer": start + 1 if window else None,
                         "last_outer": end if window else None, "new_complete_outers": len(window),
                         "contractual_optimizer_opportunities": 20 * len(window),
                         "actual_processed": UNKNOWN, "actual_skip": UNKNOWN,
                         "returned_actor_Adam_counted_once": 20 * len(window),
                         "returned_probe_Adam_counted_once": clock(end)[1] - clock(start)[1],
                         "reward_mean": sum(means) / len(means) if means else UNKNOWN,
                         "reward_outer_mean_range": [min(means), max(means)] if means else UNKNOWN,
                         "selected_mixed_count": mixed_count, "selected_questions": denominator,
                         "selected_mixed_fraction": mixed_count / denominator if denominator else UNKNOWN,
                         "all_correct_zero_anchor_groups": sum(row["all_correct_groups"] for row in window),
                         "per_outer": window},
              "latest_reward_mean": latest_reward,
              "latest_reward_change_from_anchor_outer": latest_reward - baseline_reward if baseline_reward is not None else UNKNOWN,
              "latest_actor_complete_boundary": latest_actor_boundary,
              "latest_actor_returned_Adam_per_rank": latest_actor_steps,
              "latest_probe_returned_Adam_per_rank": latest_probe_steps,
              "current_outer": integer(actual.get("current_outer", end), "current outer", end, min(end + 1, 200)),
              "current_actor_attempts": attempts, "current_actor_inflight": UNKNOWN,
              "pipeline_identity": safe_identity(actual.get("pipeline_process")),
              "qa_milestones": qa,
              "new_QA_since_anchor": new_qa if new_qa else "none_new_or_not_generated",
              "QA_newness_unknown_since_anchor": unknown_qa,
              "GPU_current": UNKNOWN, "GPU_window_peak": UNKNOWN, "GPU_execution_peak": UNKNOWN,
              "GPU_sampling_coverage": "not_recorded_in_this_existing_receipt"}
    return result


def markdown(result):
    window = result["window"]
    lines = ["# P10 hourly scalar window", "",
             "Existing receipt only; this formatter does not deliver a report or advance its anchor.", "",
             "Complete: %d/200; returned actor Adam: %d; online probe Adam: %d."
             % (result["complete_outers"], result["actor_returned_Adam_counted_once"], result["online_probe_Adam_counted_once"]),
             "Window: (%d, %d], %d complete outers; returned actor Adam %d; probe Adam %d."
             % (window["anchor_complete_outer_excluded"], result["complete_outers"], window["new_complete_outers"],
                window["returned_actor_Adam_counted_once"], window["returned_probe_Adam_counted_once"]),
             "Processed/skip and inflight are unknown unless explicitly recorded; contractual opportunities are not a processed counter.", "",
             "Latest reward mean: %s; window mean: %s; outer-mean range: %s."
             % (result["latest_reward_mean"], window["reward_mean"], window["reward_outer_mean_range"]),
             "Selected mixed: %d/%d; fraction: %s."
             % (window["selected_mixed_count"], window["selected_questions"], window["selected_mixed_fraction"]), "",
             "| Outer | Actor Adam | Reward mean | Selected mixed / 512 | Probe Adam |",
             "|---:|---:|---:|---:|---:|"]
    for row in window["per_outer"]:
        lines.append("| %d | %d | %.9f | %d | %d |" %
                     (row["outer"], row["actor_adam_per_rank"], row["reward_mean"],
                      row["selected_mixed_count"], row["probe_adam_per_rank"]))
    lines += ["", "Existing QA (new milestone since anchor: %s):" % result["new_QA_since_anchor"], ""]
    for qa in result["qa_milestones"]:
        if "em" in qa:
            lines.append("- Outer %d: EM %.8f; F1 %.8f; metadata SHA256 `%s`."
                         % (qa["outer"], qa["em"], qa["f1"], qa["metadata_sha256"]))
        else:
            lines.append("- Outer %d: not generated/unknown, not zero." % qa["outer"])
    lines += ["", "GPU current/window/execution peaks: unknown; receipt has no recorded sample coverage.", "",
              "Receipt SHA256 `%s`; delivered-anchor SHA256 `%s`."
              % (result["receipt_file_sha256"], result["anchor_file_sha256"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="Also save Markdown here; cannot overwrite either input")
    args = parser.parse_args()
    receipt_bytes = args.receipt.read_bytes()
    anchor_bytes = args.anchor.read_bytes()
    result = format_report(json.loads(receipt_bytes), json.loads(anchor_bytes),
                           hashlib.sha256(receipt_bytes).hexdigest(), hashlib.sha256(anchor_bytes).hexdigest())
    if args.markdown:
        require(args.markdown.resolve() not in {args.receipt.resolve(), args.anchor.resolve()},
                "Markdown output must not overwrite receipt or anchor")
        args.markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
