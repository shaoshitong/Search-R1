"""Render only existing scalar receipts; no network, training, or evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    envelope = json.loads(args.receipt.read_text())
    assert envelope["exit_code"] == 0
    source = json.loads(envelope["stdout"])
    rows = [dict(zip(source["history_columns"], r))
            for r in source["complete_outer_scalar_history"] if r[0] <= 50]
    assert [r["outer"] for r in rows] == list(range(1, 51))
    for row in rows:
        assert row["selected_mixed_count"] == row["mixed_groups"]
        assert row["all_correct_groups"] + row["all_wrong_groups"] + row["mixed_groups"] == 512
        assert row["selected_mixed_fraction"] == row["mixed_groups"] / 512
        row["selection_mode"] = "true_EM_priority" if row["actual_em_selection"] else "predicted_Top512"
        row["selected_questions"] = 512
    blocks = []
    for low in (1, 11, 21, 31, 41):
        subset = [r for r in rows if low <= r["outer"] <= low + 8]
        assert len(subset) == 9 and not any(r["actual_em_selection"] for r in subset)
        total = sum(r["mixed_groups"] for r in subset)
        blocks.append({"first_outer": low, "last_outer": low + 8,
                       "outers": 9, "mixed_count": total, "selected_questions": 9 * 512,
                       "mean_mixed_groups_per_outer": total / 9,
                       "mixed_fraction": total / (9 * 512),
                       "min_mixed_groups": min(r["mixed_groups"] for r in subset),
                       "max_mixed_groups": max(r["mixed_groups"] for r in subset),
                       "all_correct_fraction": sum(r["all_correct_groups"] for r in subset) / (9 * 512),
                       "all_wrong_fraction": sum(r["all_wrong_groups"] for r in subset) / (9 * 512)})
    oracle = [{"outer": r["outer"], "candidate_mixed_count": r["candidate_mixed_count"],
               "candidate_questions": 2560, "candidate_mixed_fraction": r["candidate_mixed_count"] / 2560,
               "selected_mixed_count": r["mixed_groups"], "selected_questions": 512,
               "fallback_count": r["fallback_count"]} for r in rows if r["actual_em_selection"]]
    assert [r["outer"] for r in oracle] == [10, 20, 30, 40, 50]
    qa = [r for r in source["qa_milestones"] if r["outer"] in (5, 50)]
    assert all(r["status"] == "EXISTING_NATIVE_QA_COMPLETE" and r["qa_count"] == 512 for r in qa)
    assert qa[0]["question_source_sha256"] == qa[1]["question_source_sha256"]
    report = {"role": "existing_scalar_history_not_new_evaluation", "asof_unix": source["observed_unix"],
              "run_id": source["run_id"], "execution_id": source["execution_id"],
              "runtime_source_commit": source["source_commit"],
              "source_receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
              "complete_outers_at_capture": source["complete_outer"],
              "selected_denominator": "512 questions, five trajectories per same-event group",
              "mixed_definition": "0 < sum(strict_EM[0:5]) < 5; K=1 included",
              "ordinary_blocks": blocks, "oracle_update_rounds": oracle,
              "qa_milestones": qa, "probe_update50": source["probe_update_history"][-1],
              "per_outer": rows, "new_rollout_or_evaluation": False,
              "caveat": "Selected mixture is not classifier accuracy. Candidate-pool mixture is observed only on update rounds; no ordinary-round candidate EM is imputed."}
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["outer", "selection_mode", "selected_questions", "mixed_groups", "selected_mixed_fraction",
              "all_correct_groups", "all_wrong_groups", "candidate_mixed_count", "fallback_count", "joint_sha256"]
    with stem.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    cache = tempfile.mkdtemp(prefix="search-g1-scalar-plot-")
    os.environ["MPLCONFIGDIR"] = cache
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True,
                                 gridspec_kw={"height_ratios": [2.15, 1]})
        ax = axes[0]
        for idx, block in enumerate(blocks):
            subset = [r for r in rows if block["first_outer"] <= r["outer"] <= block["last_outer"]]
            ax.plot([r["outer"] for r in subset], [100 * r["selected_mixed_fraction"] for r in subset],
                    color="#2563a6", marker="o", markersize=4, linewidth=1.7,
                    label="Ordinary round: predicted Top512" if idx == 0 else None)
            ax.hlines(100 * block["mixed_fraction"], block["first_outer"], block["last_outer"],
                      color="#142d4e", linestyle="--", linewidth=1, alpha=.65)
            ax.text((block["first_outer"] + block["last_outer"]) / 2,
                    100 * block["mixed_fraction"] + 4, f"mean {100 * block['mixed_fraction']:.2f}%",
                    ha="center", color="#142d4e", fontsize=9)
        ax.scatter([r["outer"] for r in oracle], [100] * len(oracle), color="#d07a24",
                   marker="D", s=45, label="Update round: true-EM priority", zorder=4)
        for r in oracle:
            ax.axvline(r["outer"], color="#d07a24", alpha=.15, linewidth=1)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Selected mixed groups / 512 (%)")
        ax.set_title("A800 P10: true mixed-group fraction, complete outer iterations 1-50", loc="left", weight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="lower left", frameon=False)
        count_ax = ax.secondary_yaxis("right", functions=(lambda y: y * 5.12, lambda y: y / 5.12))
        count_ax.set_ylabel("Mixed groups (count)")
        count_ax.set_yticks([0, 128, 256, 384, 512])
        bottom = axes[1]
        bottom.plot([r["outer"] for r in oracle], [100 * r["candidate_mixed_fraction"] for r in oracle],
                    marker="o", color="#6c488b", linewidth=1.6)
        for r in oracle:
            bottom.annotate(f"{r['candidate_mixed_count']}/2560\n{100 * r['candidate_mixed_fraction']:.2f}%",
                            (r["outer"], 100 * r["candidate_mixed_fraction"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9)
        bottom.set_ylim(20, 42)
        bottom.set_xlim(.5, 51)
        bottom.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        bottom.set_xlabel("Complete outer iteration")
        bottom.set_ylabel("Candidate mixed / 2560 (%)")
        bottom.grid(axis="y", alpha=.2)
        bottom.set_title("Full candidate EM is available on update rounds only", loc="left", fontsize=10)
        fig.text(.08, .015, "Mixed means 1-4 correct trajectories out of G5. Oracle-selected 100% is not probe accuracy.\n"
                 "QA512 at outer50: EM 157/512 = 30.66%, F1 = 0.3741. Existing scalar receipts; no new evaluation.", fontsize=9)
        fig.tight_layout(rect=(0, .065, 1, 1))
        fig.savefig(stem.with_suffix(".png"), dpi=165, facecolor="white")
        fig.savefig(stem.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
        plt.close(fig)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    lines = ["# A800 P10: QA50 and mixed groups across outer1-50 (2026-10-17)", "",
             "This report projects existing native scalar receipts; it performs no new training or evaluation.", "",
             "QA50 uses the same fixed512 question identities as QA5. EM157/512=30.6640625%, F1=.37414124503968255. "
             "QA5 EM118/512=23.046875%, F1=.2888516865079365; improvement39 correct /7.6171875 percentage points.", "",
             "Mixed means 0<K<5 among the five strict-EM labels, including K=1. The main denominator is selected512 questions, not2560 trajectories.", "",
             "| Ordinary outers | Mean mixed count /512 | Mean mixed % | Count range |",
             "|---|---:|---:|---:|"]
    for b in blocks:
        lines.append(f"| {b['first_outer']}-{b['last_outer']} | {b['mean_mixed_groups_per_outer']:.2f} | {100*b['mixed_fraction']:.4f}% | {b['min_mixed_groups']}-{b['max_mixed_groups']} |")
    lines += ["", "Update rounds10/20/30/40/50 all selected512 mixed groups with zero fallback, using realEM priority. This is not classifier accuracy.", "",
              "| Update outer | Full candidate mixed /2560 | Candidate mixed % | Selected mixed /512 |",
              "|---|---:|---:|---:|"]
    for o in oracle:
        lines.append(f"| {o['outer']} | {o['candidate_mixed_count']} | {100*o['candidate_mixed_fraction']:.4f}% | {o['selected_mixed_count']} |")
    lines += ["", "Probe50 mixed-group classification on its existing512-question diagnostic: ordinary accuracy44.140625%, balanced accuracy50.84892995%, all-nonmixed baseline378/512=73.828125%. These are not QA answer EM. The diagnostic uses newly sampled G5 events, so different update rounds do not share identical trajectory labels.", "",
              "The ordinary selected fraction and full update-round candidate fraction both decline. This observation alone does not identify overfitting or a code defect.", "",
              "![Full mixed-group history](" + stem.name + ".png)", "", "## Every complete outer", "",
              "| Outer | Selection | Mixed /512 | Mixed % | All correct | All wrong |",
              "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['outer']} | {r['selection_mode']} | {r['mixed_groups']} | {100*r['selected_mixed_fraction']:.4f}% | {r['all_correct_groups']} | {r['all_wrong_groups']} |")
    lines += ["", f"Runtime source `{source['source_commit']}`; run `{source['run_id']}` / execution `{source['execution_id']}`. Complete53 at capture; same pipeline continued54. Full scalar JSON and CSV preserve each joint SHA. No raw QA, token, rollout, weight, or credential is included.", ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"qa50_em": qa[1]["em"], "complete": source["complete_outer"], "rows": len(rows),
                      "files": [str(stem.with_suffix(ext)) for ext in (".json", ".csv", ".png", ".svg", ".md")]}))


if __name__ == "__main__":
    main()
