"""Build a seven-dataset EM table from already collected small audit summaries.

No remote jobs, scoring runs, or raw predictions are touched. The seven-dataset
macro mean and the pooled 51,713-question micro mean are kept separate.
"""
import copy
import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
OLD = D.parent / "training-and-fixed512-em-20261014-2030/final-seven-datasets.json"
DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
LABELS = ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2Wiki", "MuSiQue", "Bamboogle"]
COUNTS = dict(zip(DATASETS, [3610, 11313, 14267, 7405, 12576, 2417, 125]))


def read(path):
    return json.loads(Path(path).read_text())


def source(path, pointer=None):
    p = Path(path)
    out = {"local_audit": str(p), "local_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    if pointer:
        out["json_pointer"] = pointer
    return out


history = read(OLD)
latest = read(D / "all-seven-results.json")
ours = copy.deepcopy(history["ours"])
by_name = {r["experiment"]: r for r in ours}
for row in ours:
    row["sources"] = [source(OLD)]
    row["scope"] = "historical_full_test_summary_observed_2026-10-14"
    row["count"] = 51713

# The current collection independently re-reads these final result identities.
current_bindings = {
    "递归Tree A": "Recursive A",
    "Tree-GRPO": "Tree-GRPO original guarded200",
    "TOP-D α=0.1": "TOP-D alpha0.1",
    "Tree-BRANPO 3360": "Tree BRANPO fresh200 3360",
    "GiGPO": "GiGPO safety fresh200 3347",
    "G1→R1": "Soft Reward -> Hard Reward",
    "Search-G1": "Soft Reward (Search-G1)",
}
latest_by_name = {r["name"]: r for r in latest["runs"]}
for label, actual in current_bindings.items():
    row = by_name[label]
    final = latest_by_name[actual]["final51713"]
    assert final["complete"] and final["n"] == 51713
    assert abs(row["micro_em"] - final["em"]) < 1e-12
    row["final_identity_current"] = final
    row["sources"].append(source(D / "all-seven-results.json", "/runs/name=" + actual + "/final51713"))
    row["scope"] = "full_test_identity_rechecked_2026-10-15_dataset_breakdown_previously_verified"

# Fresh per-dataset full-test results, not the fixed 512-question validation.
mix = read(D / "g1-mixed-final-scores-second.json")
mix_row = {
    "experiment": "Easy Selection+Soft Reward", "checkpoint_iter": 200,
    "dataset_em": {k: mix["source_metrics"][k]["em"] for k in DATASETS},
    "micro_em": mix["em"], "f1": mix["f1"], "count": 51713,
    "source_counts": {k: mix["source_metrics"][k]["count"] for k in DATASETS},
    "evidence": {"path": mix["result_file"], "sha256": latest_by_name["Easy Selection+Soft Reward"]["final51713"]["sha256"]},
    "sources": [source(D / "g1-mixed-final-scores-second.json", "/source_metrics")],
    "kind": "ours", "scope": "full_test_dataset_breakdown_rechecked_2026-10-15",
    "final_identity_current": latest_by_name["Easy Selection+Soft Reward"]["final51713"],
    "note": "203 complete scoring batches and full 51,713 rows verified; scoring complete, formal job cleanup acceptance pending.",
}
ours.append(mix_row)

topd = read(D / "topd-latest.json")
t1 = topd["runs"][1]["complete"]
metrics = t1["data"]["metrics"]
t1_row = {
    "experiment": "TOP-D α=1", "checkpoint_iter": 200,
    "dataset_em": {k: metrics["val/em/" + k] for k in DATASETS},
    "micro_em": metrics["val/em/all"], "f1": metrics["val/f1/all"], "count": 51713,
    "source_counts": {k: metrics["val/qa_count/" + k] for k in DATASETS},
    "evidence": latest_by_name["TOP-D alpha1"]["final51713"]["complete_source"],
    "sources": [source(D / "topd-latest.json", "/runs/1/complete/data/metrics")],
    "kind": "ours", "scope": "full_test_dataset_breakdown_rechecked_2026-10-15",
    "final_identity_current": latest_by_name["TOP-D alpha1"]["final51713"],
}
ours.append(t1_row)

renames = {"G1→R1": "Soft Reward → Hard Reward", "Search-G1": "Soft Reward（旧 Search-G1）"}
checks = []
for row in ours:
    row["experiment"] = renames.get(row["experiment"], row["experiment"])
    assert list(row["dataset_em"]) == DATASETS
    assert row["source_counts"] == COUNTS
    assert sum(row["source_counts"].values()) == 51713
    values = row["dataset_em"]
    row["macro_em"] = sum(values.values()) / 7
    integer_correct = {k: round(values[k] * COUNTS[k]) for k in DATASETS}
    assert all(abs(values[k] * COUNTS[k] - integer_correct[k]) < 1e-8 for k in DATASETS)
    pooled = sum(integer_correct.values()) / 51713
    assert abs(pooled - row["micro_em"]) < 1e-12
    row["dataset_correct"] = integer_correct
    row["correct"] = sum(integer_correct.values())
    checks.append({"experiment": row["experiment"], "dataset_counts_sum": 51713,
                   "pooled_em_matches": True, "integer_correct_counts_verified": True,
                   "macro_recomputed_from_unrounded_dataset_em": True})

papers = copy.deepcopy(history["paper_references"])
for row in papers:
    row["paper_reported_macro_em"] = row["macro_em"]
    vals = list(row["dataset_em"].values())
    row["macro_recomputed_from_displayed_dataset_values"] = sum(vals) / 7 if all(v is not None for v in vals) else None
    row["micro_em"] = None
    row["scope"] = "user_supplied_paper_reference_not_reaudited_in_this_collection"
    row["sources"] = [source(OLD), {"source": "latest user-provided table", "verification": "not independently reread from paper this turn"}]
    row["note"] = "Use paper-reported average; rounded per-dataset entries may give a different recomputed last digit. Different paper settings are not a matched-budget comparison."

ours.sort(key=lambda r: r["macro_em"], reverse=True)
rows = sorted(ours + papers, key=lambda r: r["macro_em"] if r["macro_em"] is not None else -1, reverse=True)
out = {
    "created_cst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    "observed_window_cst": latest["observed_report_at_cst"],
    "datasets": DATASETS, "dataset_labels": LABELS, "source_counts": COUNTS,
    "full_test_questions": 51713,
    "macro_definition": "unweighted arithmetic mean of the seven dataset EM values; each dataset has equal weight",
    "micro_definition": "total correct / 51,713; dataset-size-weighted pooled EM, not the seven-dataset average",
    "paper_reference_note": "Paper rows supplied by the user and previous comparison; not independently reverified from papers in this refresh.",
    "ours": ours, "paper_references": papers, "rows": rows, "checks": checks,
    "pending": [
        {"experiment": "CoSearch 双模型 iter200", "reason": "Full-test generation is incomplete; no final EM/F1 ranked."},
        {"experiment": "递归Tree B / Tree 嵌套分叉 / 新 Q-probe", "reason": "Current training has not produced a complete final 51,713-question score."},
        {"experiment": "Search-E1 Archive四卡", "reason": "No verified complete 51,713-question final score found in this audit; paper row is separate."},
    ],
}
target = D / "final-seven-table-data.json"
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")

def pct(v):
    return "—" if v is None else f"{100 * v:.2f}%"

lines = ["完整 51,713 题终评；七集平均为等权平均（macro），不是全题加权准确率（micro）。", "",
         "| 实验／checkpoint | " + " | ".join(LABELS) + " | 七集平均 |",
         "|---|" + "---:|" * 8]
for row in rows:
    name = row["experiment"] + (f"，iter{row['checkpoint_iter']}" if row["checkpoint_iter"] is not None else "")
    lines.append("| " + name + " | " + " | ".join(pct(row["dataset_em"][k]) for k in DATASETS) + " | **" + (pct(row["macro_em"]) if row["macro_em"] is not None else "未报告") + "** |")
lines.extend(["", "论文行沿用用户给出的原论文成绩；本次没有重新核对原论文。论文平均数保留其报告值，舍入后的各列重算可能有末位差异。",
              "", "BAPO 为 iter150，其余本地终评行为 iter200。CoSearch 尚未完成完整终评，未填入部分题目的成绩。",
              "", "Easy Selection+Soft Reward 的评分和 51,713 行对账已完成；正式作业收尾验收仍待确认。",
              "", "BAPO 与 OPSD 沿用 10 月 14 日直接读取的已核验评分摘要；其余行包含 10 月 15 日重读的终评身份或逐集结果。"])
(D / "final-seven-table.md").write_text("\n".join(lines) + "\n")

# Read back the generated artifact and independently recalculate all own rows.
again = read(target)
assert len(again["ours"]) == 11 and len(again["paper_references"]) == 3
for row in again["ours"]:
    assert abs(sum(row["dataset_em"].values()) / 7 - row["macro_em"]) < 1e-15
    assert sum(row["dataset_correct"].values()) == row["correct"]
    assert abs(row["correct"] / 51713 - row["micro_em"]) < 1e-12
print(json.dumps({"rows": len(rows), "ours": len(ours), "paper_references": len(papers),
                  "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                  "readback_validation": "passed"}, ensure_ascii=False))
