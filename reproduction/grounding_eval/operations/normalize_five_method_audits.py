"""Normalize five completed read-back audits; Easy remains a separate sixth input."""
import argparse
import hashlib
import json
from pathlib import Path


def normalize(base):
    def load(name):
        return json.loads((base / name).read_text())

    def evidence(name):
        p = base / name
        return dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    old = {}
    rules = load("official-code-audit/all-complete-same-response-rule-replay.json")["methods"]
    for method, filename, audit in (
        ("Search-G1", "a800-search-g1-full-v4-final-file-audit.json", "g1-shared-independent-verification.json"),
        ("Soft-to-Hard", "a800-soft-to-hard-full-v4-final-file-audit.json", "soft-hard-shared-independent-verification.json"),
    ):
        d = load(filename)
        d = d.get("audit", d)
        manifest = next(f["sha256"] for f in d["files"] if f["path"].endswith("/manifest.json"))
        source = next(f["sha256"] for f in d["files"] if f["path"].endswith("/normal.jsonl"))
        old[method] = (d["summary"], rules[method]["changed_rule_only"], audit, manifest, source)
    filename = "hpc3-topd-opd-independent-acceptance.json"
    for accepted in load(filename).values():
        d = accepted["report"]
        method = d["method"]
        rule = load("hpc3-topd-opd-official-changed-rule.json")[method]["report"]["methods"][method]["changed_rule_only"]
        old[method] = (d["cf_summary"], rule, filename, d["cf_manifest_sha256"], d["normal_sha256"])
    filename = "opsd-cf-v4-independent-acceptance-final-readback.json"
    d = load(filename)["acceptance"]["report"]
    rule = load("opsd-cf-v4-official-changed-rule.json")["report"]["methods"]["OPSD3304"]["changed_rule_only"]
    old["OPSD3304"] = (d["cf_summary"], rule, filename, d["cf_manifest_sha256"], d["normal_sha256"])

    new = {}
    filename = "a800-three-strict-v5-token-coverage-summary-20261019.json"
    for x in load(filename)["methods"]:
        p = x["artifact_proofs"]
        coverage = dict(x["coverage"])
        coverage["invalid_reasons"] = coverage["exclusion_reasons"]
        # This denominator counts actual completed unique CF token records.
        coverage["inference_completed"] = x["CF_token_denominator"]
        new[x["method"]] = dict(
            counts=x["classification"]["counts"], coverage=coverage,
            TC=x["strict_v5"]["TC"], D_Inv=x["strict_v5"]["D_Inv"],
            TC_denominator=x["classification"]["TC_denominator"], DInv_denominator=x["classification"]["DInv_denominator"],
            CF_tokens=x["CF_tokens"], order=x["canonical_question_gold_order_sha256"], normal_tokens=x["normal_tokens"],
            normal_SHA=p["normal_sha256"], origin=filename,
            proof=dict(independent_verified=True, acceptance_path=p["independent_acceptance_path"], acceptance_sha256=p["independent_acceptance_sha256"], normal_sha256=p["normal_sha256"], manifest_sha256=p["manifest_file_sha256"]),
            extra_telemetry=x.get("normal_additional_telemetry"),
        )
    filename = "hpc3-topd-opd-strict-v5-final-small-summary.json"
    bundle = load(filename)
    assert bundle["complete"] is True
    for x in bundle["summary"]["models"]:
        p = x["acceptance_path"]
        new[x["method"]] = dict(
            counts=x["counts"], coverage=x["coverage"], TC=x["TC"], D_Inv=x["D_Inv"],
            TC_denominator=x["TC_denominator"], DInv_denominator=x["DInv_denominator"], CF_tokens=x["token_usage"]["counterfactual"],
            order=x["canonical_question_gold_order_sha256"], normal_tokens=x["token_usage"]["normal"], normal_SHA=x["normal_sha256"], origin=filename,
            proof=dict(independent_verified=True, acceptance_path=p, acceptance_sha256=bundle["acceptance_files"][p]["sha256"], normal_sha256=x["normal_sha256"], manifest_sha256=x["manifest_sha256"]),
        )
    rows = []
    for method, (summary, rule, src, manifest, source) in old.items():
        x = new[method]
        assert source == x["normal_SHA"]
        normal = dict(questions=3610, correct=summary["counts"]["normal_correct"], EM=summary["counts"]["normal_correct"] / 3610, source_sha256=source, tokens=x.pop("normal_tokens"))
        if x.get("extra_telemetry"):
            normal["additional_telemetry"] = x["extra_telemetry"]
        proof = dict(independent_verified=True, acceptance_path=evidence(src)["path"], acceptance_sha256=evidence(src)["sha256"], normal_sha256=source, manifest_sha256=manifest)
        changed = dict(rule, protocol="official-changed-rule-replay-on-v4-responses", coverage=summary["coverage"], CF_tokens=summary["token_usage"]["counterfactual"], proof=proof)
        strict = dict(protocol="strict-v5-conservative-typed", **{k: v for k, v in x.items() if k not in ("normal_SHA", "order", "origin", "extra_telemetry")})
        rows.append(dict(method=method, checkpoint_iteration=200, canonical_question_gold_order_sha256=x["order"], normal=normal, requested_changed_rule_on_v4=changed, corrected_strict_v5=strict, evidence_sources=[evidence(src), evidence(x["origin"])]))
    return dict(state="FIVE_ACCEPTED_METHODS_AWAITING_SIXTH", methods=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = normalize(Path(args.root))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(path=args.output, methods=len(result["methods"]))))
