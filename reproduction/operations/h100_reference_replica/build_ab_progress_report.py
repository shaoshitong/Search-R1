"""Build a descriptive A/B report from already-read small committed records.

Local JSON only: no remote access, tensors, trainer mutation, or deployment.
Missing metrics remain missing; pending iterations are never inferred complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def display(value, percent=False):
    if value is None:
        return "pending"
    return f"{100 * value:.4f}%" if percent else f"{value:.6f}"


def build(source, output):
    source, output = Path(source), Path(output)
    snapshot = json.loads(source.read_text())
    result = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "actual_complete_by_arm": snapshot["actual_complete_by_arm"],
        "requested_through": snapshot["requested_through"],
        "pending": snapshot.get("pending"),
        "scope": "Descriptive committed records; not a causal claim or final acceptance.",
        "arms": {},
    }
    for arm, records in snapshot["data"].items():
        rows, losses, qa = [], [], {}
        for outer, record in sorted(records.items(), key=lambda item: int(item[0])):
            joint = (record.get("joint_metrics") or {}).get("value")
            if joint is None:
                continue
            n = int(outer)
            probe = joint.get("probe") or {}
            wall = (record.get("iteration_wall") or {}).get("value") or {}
            # This is the actual reference schedule, with ordinary outer200.
            periodic = n < 200 and n % 10 == 0
            rows.append({
                "outer": n,
                "periodic_actual_em_priority": periodic,
                "selection": joint.get("selection"),
                "actor": joint.get("actor"),
                "reward": joint.get("reward"),
                "whole_wall_seconds": wall.get("total_wall_seconds_including_qa_grounding_storage"),
                "training_wall_seconds": wall.get("training_wall_seconds"),
                "phase_timings": joint.get("phase_timings"),
                "probe_new_steps_per_rank": probe.get("new_adam_steps_per_rank"),
                "probe_actual_steps_per_rank": probe.get("actual_adam_steps_per_rank"),
                "ema_updates": probe.get("ema_updates"),
                "probe_learning_rate": probe.get("learning_rate"),
                "sampling": probe.get("sampling"),
                "before_update": probe.get("before_update"),
                "after_update": probe.get("after_update"),
            })
            curve = (record.get("loss_curve") or {}).get("value")
            if curve is not None:
                losses.extend(curve)
            if record.get("qa") is not None:
                qa[str(n)] = record["qa"]
        ordinary = [r for r in rows if r["outer"] >= 6 and not r["periodic_actual_em_priority"]]
        result["arms"][arm] = {
            "rows": rows,
            "ordinary_from6_count": len(ordinary),
            "ordinary_from6_selected_mixed_mean": mean([(r.get("selection") or {}).get("mixed_fraction") for r in ordinary]),
            "ordinary_from6_whole_wall_mean_seconds": mean([r["whole_wall_seconds"] for r in ordinary]),
            "global_probe_loss_rows": losses,
            "qa_records": qa,
        }
    lines = [
        "# Search-G1 A/B committed-progress comparison", "",
        "This report uses already-read small records only. It neither changes training nor proves final acceptance.",
        "Periodic actual-EM-priority selected mixing is reported separately from ordinary pre-label selection.",
        "Probe classification metrics are not QA EM/F1. Different actor snapshots produce different diagnostic labels;", 
        "across-outer metrics are not a fixed-label learning curve. Missing records remain pending.", "",
        "| Arm | Actual complete | Included records | Ordinary from6 | Mean selected mixed | Mean whole wall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        lines.append(f"|{arm}|{result['actual_complete_by_arm'].get(arm)}|{len(data['rows'])}|"
                     f"{data['ordinary_from6_count']}|{display(data['ordinary_from6_selected_mixed_mean'], True)}|"
                     f"{display(data['ordinary_from6_whole_wall_mean_seconds'])}s|")
    lines.extend(["", "## Scheduled QA source aggregates", "",
                  "These are source-reported QA aggregates, not a substitute for final question-identity and worker-output acceptance.", "",
                  "| Arm | Outer | Question count | EM | F1 | Actual worker exits |",
                  "|---|---:|---:|---:|---:|---|"])
    for arm, data in result["arms"].items():
        for outer in [5, 50, 100, 150, 200]:
            qa = (data["qa_records"].get(str(outer)) or {}).get("value") or {}
            exits = qa.get("actual_worker_exit_codes")
            lines.append(f"|{arm}|{outer}|{qa.get('qa_count', 'pending')}|"
                         f"{display(qa.get('em'), True)}|{display(qa.get('f1'), True)}|"
                         f"{json.dumps(exits, sort_keys=True) if exits is not None else 'pending'}|")
    for arm, data in result["arms"].items():
        lines.extend(["", f"## {arm}: per-outer selected outcomes", "",
                      "Uncontinued candidate groups have no labels; selected metrics do not establish candidate-pool accuracy.", "",
                      "| Outer | Periodic actual-EM priority | Selected mixed | All wrong | All correct | Predicted mean mixing | Whole wall seconds |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in data["rows"]:
            s = row.get("selection") or {}
            values = [str(row["outer"]), str(row["periodic_actual_em_priority"])]
            values.extend(display(s.get(k), True) for k in ["mixed_fraction", "all_wrong_fraction", "all_correct_fraction", "mean_selection_q"])
            values.append(display(row["whole_wall_seconds"]))
            lines.append("|" + "|".join(values) + "|")
        lines.extend(["", f"## {arm}: periodic same-data probe comparisons", "",
                      "Train is the original unresampled distribution. Each pre/post pair uses the same trajectories and labels.", "",
                      "| Outer | Head / distribution | BA before→after | Ordinary accuracy before→after | Trajectory BCE before→after | Mixed BCE before→after |",
                      "|---|---|---:|---:|---:|---:|"])
        for row in data["rows"]:
            if not row["periodic_actual_em_priority"]:
                continue
            before, after = row["before_update"] or {}, row["after_update"] or {}
            for head in sorted(set(before) | set(after)):
                for dist in ["train", "test"]:
                    a, b = before.get(head, {}).get(dist, {}), after.get(head, {}).get(dist, {})
                    vals = [str(row["outer"]), f"{head} / {dist}"]
                    for key, percent in [("balanced_accuracy", True), ("accuracy", True), ("trajectory_bce", False), ("group_bce", False)]:
                        vals.append(display(a.get(key), percent) + "→" + display(b.get(key), percent))
                    lines.append("|" + "|".join(vals) + "|")
        lines.extend(["", f"Actual stored global probe loss rows: {len(data['global_probe_loss_rows'])}.",
                      "QA source records, full paired metrics, recall/confusion matrices, balancing statistics and clocks are retained in the JSON companion."])
    lines.extend(["", "Required final scope remains both200/no201, all scheduled QA and final51713, saved/recoverable state and owned resource closure.",
                  "This partial report does not certify completion or recommend changing the experiment.", ""])
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison-progress.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "comparison-progress.md").write_text("\n".join(lines))
    return {"output": str(output), "actual_complete_by_arm": result["actual_complete_by_arm"],
            "included": {arm: len(data["rows"]) for arm, data in result["arms"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output)))


if __name__ == "__main__":
    main()
