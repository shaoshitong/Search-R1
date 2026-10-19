"""Validate small independently accepted method bundles before final reporting.

This is a CPU-only report assembler. It never generates answers, changes a
classification, estimates missing costs, or reads large remote model assets.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


METHODS = (
    "Search-G1", "TOP-D alpha=0.1", "OPD alpha=1", "OPSD3304",
    "Soft-to-Hard", "Easy Selection",
)
NQ_ORDER = "23c0a8307f38a7a8d1cbd9e71fb115b7bd091421a3d3b10952fb71c8210b551c"


def equal_number(actual, expected):
    if expected is None:
        assert actual is None, (actual, expected)
    else:
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
        assert math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-12)


def verify_metric(metric, normal):
    counts = metric["counts"]
    direct, true, amb, hall = (counts.get(k, 0) for k in ("Direct", "True", "Amb", "Hall"))
    assert all(type(v) is int and v >= 0 for v in (direct, true, amb, hall))
    classified = true + amb + hall
    coverage = metric["coverage"]
    assert coverage["classified"] == coverage["inference_completed"] == classified
    assert coverage["pending_or_error"] == 0
    assert classified == coverage["perturbable"] <= coverage["correct_searched"]
    assert coverage["correct_searched"] + direct == normal["correct"]
    assert sum(coverage["invalid_reasons"].values()) == coverage["correct_searched"] - classified
    assert metric["TC_denominator"] == classified + direct
    assert metric["DInv_denominator"] == classified
    denominator = classified + direct
    equal_number(metric["TC"], (direct + true + .6 * amb + .3 * hall) / denominator if denominator else None)
    equal_number(metric["D_Inv"], hall / classified if classified else None)
    proof = metric["proof"]
    assert proof["independent_verified"] is True
    for key in ("acceptance_sha256", "normal_sha256", "manifest_sha256"):
        assert len(proof[key]) == 64 and int(proof[key], 16) >= 0
    assert proof["acceptance_path"].startswith("/")
    assert proof["normal_sha256"] == normal["source_sha256"]


def verify_tokens(tokens, count):
    for name, stat in tokens.items():
        if not isinstance(stat, dict) or "total" not in stat:
            continue
        total = stat["total"]
        if total is None:
            assert stat.get("null_reason"), name
            continue
        assert type(total) is int and total >= 0, name
        assert stat["measured_questions"] == count, name
        equal_number(stat["mean"], total / count if count else None)


def assemble(paths):
    rows, sources = {}, []
    for path in paths:
        raw = Path(path).read_bytes()
        bundle = json.loads(raw)
        for row in bundle["methods"]:
            method = row["method"]
            assert method in METHODS and method not in rows, method
            assert row["checkpoint_iteration"] == 200
            assert row["canonical_question_gold_order_sha256"] == NQ_ORDER
            normal = row["normal"]
            assert normal["questions"] == 3610
            assert 0 <= normal["correct"] <= normal["questions"]
            equal_number(normal["EM"], normal["correct"] / 3610)
            verify_tokens(normal["tokens"], 3610)
            for name in ("requested_changed_rule_on_v4", "corrected_strict_v5"):
                metric = row[name]
                verify_metric(metric, normal)
                verify_tokens(metric["CF_tokens"], metric["coverage"]["classified"])
            assert row["requested_changed_rule_on_v4"]["protocol"] == "official-changed-rule-replay-on-v4-responses"
            assert row["corrected_strict_v5"]["protocol"] == "strict-v5-conservative-typed"
            assert row["requested_changed_rule_on_v4"]["counts"].get("Amb", 0) == 0
            # The two different intervention populations must remain separate.
            assert row["requested_changed_rule_on_v4"]["proof"]["manifest_sha256"] != row["corrected_strict_v5"]["proof"]["manifest_sha256"]
            rows[method] = row
        sources.append({"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    assert set(rows) == set(METHODS), {"missing_methods": sorted(set(METHODS) - set(rows))}
    return {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "ALL_SIX_METHODS_COMPLETE_INDEPENDENTLY_VERIFIED",
        "normal_dataset": "Fixed NQ 3610 occurrences from common final51713 parquet",
        "canonical_question_gold_order_sha256": NQ_ORDER,
        "checkpoint_iteration": 200,
        "measured_runs_per_method": 1,
        "seed_sample_SD": None,
        "scope": {
            "requested_changed_rule_on_v4": "Same frozen v4 CF responses, changed-answer rule only; diagnostic, not full official protocol reproduction",
            "corrected_strict_v5": "New CF generations on conservative finite typed subset; conditional coverage differs by method; not a population-wide grounding ranking",
            "normal_tokens": "Accepted unique formal trajectories. Raw generated output includes tails discarded by the historical action parser. Retained trajectory counts initial prompt, retained output and seen environment once; cumulative input includes repeated history.",
            "all_job_cost": "Pilots, discarded retries and retriever startup are separate operational diagnostics; formal token totals do not claim complete hardware billing cost.",
        },
        "sources": sources,
        "methods": [rows[m] for m in METHODS],
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = assemble(args.input)
    raw = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
    # A final report is created only after all six bundles pass validation.
    with Path(args.output).open("xb") as f:
        f.write(raw)
    print(json.dumps({"path": args.output, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "methods": len(report["methods"])}))


if __name__ == "__main__":
    main()
