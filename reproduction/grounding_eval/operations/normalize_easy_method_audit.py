"""Adapt the independently accepted Easy delivery to the six-method report."""
import argparse
import hashlib
import json
from pathlib import Path


def normalize(path):
    raw = path.read_bytes()
    x = json.loads(raw)
    assert x["state"] == "EASY_NORMAL_AND_DUAL_CF_INDEPENDENTLY_VERIFIED"
    assert x["two_CF_protocols_kept_separate"] is True
    assert x["all_raw_outputs_remain_remote"] is True
    assert all(v == 0 for v in x["exit_codes"].values())
    assert x["assets"]["all_selected_file_source_and_target_sha_verified"] is True
    assert x["assets"]["verified_files"] == 13
    assert x["assets"]["model_bytes_through_Mac"] == 0
    n = x["normal"]
    normal = dict(questions=n["questions"], correct=n["correct"], EM=n["EM"], source_sha256=n["source_sha256"], tokens=n["token_usage"], actual_batch_sizes=n["actual_batch_sizes"], per_batch_execution_metrics=n["per_batch_execution_metrics"], independent_verification_path=n["independent_verification_path"], independent_verification_sha256=n["independent_verification_sha256"])
    metrics = {}
    for src, name, protocol in (
        ("v4_diagnostic", "requested_changed_rule_on_v4", "official-changed-rule-replay-on-v4-responses"),
        ("v5_corrected_strict", "corrected_strict_v5", "strict-v5-conservative-typed"),
    ):
        report = x[src]
        assert report["source_normal_sha256"] == normal["source_sha256"]
        assert report["normal_content_order_sha256"] == x["normal_content_order_sha256"]
        assert report["cf_attempt_tokens"]["errors"] == report["cf_attempt_tokens"]["failed_or_interrupted_attempts"] == 0
        if src == "v4_diagnostic":
            values = dict(report["changed_rule_only"])
        else:
            values = {k: report[k] for k in ("counts", "TC", "D_Inv", "TC_denominator")}
            values["DInv_denominator"] = report["D_Inv_denominator"]
        proof = dict(independent_verified=True, acceptance_path=report["acceptance_path"], acceptance_sha256=report["acceptance_sha256"], normal_sha256=report["source_normal_sha256"], manifest_sha256=report["manifest_file_sha256"])
        metrics[name] = dict(values, protocol=protocol, coverage=report["coverage"], CF_tokens=report["cf_tokens"], actual_batch_sizes=report["actual_inference_batch_sizes"], proof=proof)
    row = dict(method="Easy Selection", checkpoint_iteration=200, canonical_question_gold_order_sha256=x["normal_content_order_sha256"], normal=normal, assets=x["assets"], exit_codes=x["exit_codes"], **metrics, evidence_sources=[dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())])
    # Keep pilot costs and the repaired pre-generation failure visible without
    # counting them as accepted formal trajectories or masking active failures.
    row["pilot_execution"] = x["pilot_execution"]
    row["execution_incidents"] = x["execution_incidents"]
    prior = row["execution_incidents"]["prior_zero_generation_admission_failure"]
    assert prior["raw_rows"] == 0 and prior["exit_code"] == 1
    assert row["pilot_execution"]["separate_from_formal_3610_costs"] is True
    return dict(state="EASY_ACCEPTED_SIXTH_METHOD", methods=[row])


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = normalize(Path(args.input))
    with Path(args.output).open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(dict(path=args.output, methods=1)))
