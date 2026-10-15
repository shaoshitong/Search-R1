#!/usr/bin/env python3
"""Local, immutable actual scalar-record inventory through effective outer171.

This is evidence extraction, not a test, code review, training gate or remote
operation. Existing data and code are read only. Only the three named new small
inventory products are written to BASE and its two existing result mirrors.
"""
import datetime
import hashlib
import json
import math
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/tree-nested-branch-handoff-20261013"
NAME = "METRIC_RECORD_COVERAGE_THROUGH171_LOCAL"
PINS = {
    "LEARNING_CURVE_COMPLETE_THROUGH170.json": (992265, "5c542620c45852c1570f5dcfca79e8b4ae6e4352335a8452d31abed0ecb791c9"),
    "LEARNING_CURVE_INITIAL1_10_SUPPLEMENT.json": (26045, "94f8506dd8c258ef53ea298f117223f284ce1692863721814a895f3a2268fa65"),
    "PER_OUTER_METRICS_COMPACT.json": (101394, "c5f201b38087718b17f7a8434e84b0814a0e1aaef94ec364b2c1efbeea6247f0"),
    "FULL_OUTER171_PROGRESS_RECEIPT.json": (29637, "52ee76bd3cdb1b3f5926e9f361167586ea2d097a611dd5b4eb09b47783c40e14"),
    "NATIVE5_CONTINUE_POLL_2121.json": (637184, "2e1f367cee626de17e4893fc1a17e2e023db47f7b1f5187da88033041d35dc08"),
}
COMMITS = ["0befbaf99fc44942105831b3fbec94e4bf1b8ccb", "ba9f6627ec9a4e9749830d7b6acca53f6feec6b4", "778b2b43dad90fc0c1c5d95d77c2326286011b73"]
FIELDS = {
    "training_reward_mean": "critic/rewards/mean",
    "training_score_mean": "critic/score/mean",
    "raw_within_tree_reward_population_variance": "tree_variance/within_root_raw_reward_population_variance",
    "pure_intra_adv_population_variance": "tree_variance/intra_adv_population_variance",
    "total_actual_adv_population_variance": "tree_variance/group_actual_adv_population_variance",
    "five_leaf_same_reward_question_fraction": "tree_variance/same_reward_question_fraction",
    "five_leaf_mixed_reward_question_fraction": "tree_variance/mixed_reward_question_fraction",
    "all_reward_one_question_count": "tree_variance/exact_reward_one_question_count",
    "all_reward_zero_question_count": "tree_variance/exact_reward_zero_question_count",
    "root0_same_reward_fraction": "tree_variance/root0_same_reward_fraction",
    "root0_mixed_reward_fraction": "tree_variance/root0_mixed_reward_fraction",
    "root1_same_reward_fraction": "tree_variance/root1_same_reward_fraction",
    "root1_mixed_reward_fraction": "tree_variance/root1_mixed_reward_fraction",
    "all_roots_same_reward_tree_fraction": "tree_variance/same_reward_tree_fraction",
    "all_roots_mixed_reward_tree_fraction": "tree_variance/mixed_reward_tree_fraction",
}
SERIES_EXTRA = ["question_count", "tree_count", "mixed_question_count", "exact_reward_one_tree_count", "exact_reward_zero_tree_count", "mixed_tree_count", "root0_exact_reward_one_count", "root0_exact_reward_zero_count", "root0_mixed_count", "root1_exact_reward_one_count", "root1_exact_reward_zero_count", "root1_mixed_count"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


cache, sources = {}, {}


def local_path(value):
    p = pathlib.Path(value)
    if p.is_absolute():
        p = p.resolve()
    elif str(p).startswith("outputs/"):
        p = (ROOT / p).resolve()
    else:
        p = (BASE / p).resolve()
    if not p.is_relative_to(BASE):
        raise ValueError(f"Not a local BASE source: {p}")
    return p


def read_source(value, expected_sha=None, expected_bytes=None):
    p = local_path(value)
    name = str(p.relative_to(BASE))
    if name not in cache:
        raw = p.read_bytes()
        source_id = f"s{len(sources):03d}"
        sources[source_id] = {"path": name, "bytes": len(raw), "sha256": sha(raw)}
        cache[name] = (source_id, json.loads(raw))
    source_id, data = cache[name]
    bound = sources[source_id]
    if expected_sha is not None and bound["sha256"] != expected_sha:
        raise ValueError(f"Source SHA mismatch: {name}")
    if expected_bytes is not None and bound["bytes"] != expected_bytes:
        raise ValueError(f"Source byte mismatch: {name}")
    return source_id, data


def exact_metric(data, outer):
    container = data.get("iteration_metrics", [])
    rows = [container] if isinstance(container, dict) else container
    matches = [m for m in rows if m.get("step") == outer]
    if not matches:
        raise ValueError(f"No exact outer{outer} metric record")
    if any(m != matches[0] for m in matches):
        raise ValueError(f"Conflicting outer{outer} metric records")
    return matches[0]


def present(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for name, (size, digest) in PINS.items():
        read_source(name, digest, size)
    curve = cache["LEARNING_CURVE_COMPLETE_THROUGH170.json"][1]
    supplement = cache["LEARNING_CURVE_INITIAL1_10_SUPPLEMENT.json"][1]
    compact = cache["PER_OUTER_METRICS_COMPACT.json"][1]
    full171 = cache["FULL_OUTER171_PROGRESS_RECEIPT.json"][1]
    points = curve["training_points"]
    if [r["outer"] for r in points] != list(range(1, 171)):
        raise ValueError("Curve outer lineage is not exactly1..170")
    if any(points[i] != {**r, "input_summary": "LEARNING_CURVE_INITIAL1_10_SUPPLEMENT.json", "input_summary_sha256": PINS["LEARNING_CURVE_INITIAL1_10_SUPPLEMENT.json"][1]} for i, r in enumerate(supplement["training_points"])):
        raise ValueError("Initial1..10 supplement differs from curve")
    index = {r["outer"]: r for r in compact["effective_rows"]}
    blocks = {r["file"]: r for r in compact["scalar_blocks"]}
    if sorted(index) != list(range(1, 172)) or len(compact["effective_rows"]) != 171:
        raise ValueError("Compact effective lineage is not exactly1..171")
    records, bindings, issues, relation_failures = {}, [], [], []
    for outer in range(1, 172):
        expected_commit = COMMITS[0 if outer == 1 else 1 if outer <= 5 else 2]
        point = points[outer - 1] if outer <= 170 else {"outer": 171, "source_commit": full171["source_commit"], "source_metric_file": full171["source_poll"], "source_metric_file_sha256": full171["source_poll_sha256"], "source_receipt": "FULL_OUTER171_PROGRESS_RECEIPT.json", "source_receipt_sha256": PINS["FULL_OUTER171_PROGRESS_RECEIPT.json"][1]}
        entry = index[outer]
        if point["source_commit"] != expected_commit or entry["source_commit"] != expected_commit or not entry["effective_history"]:
            raise ValueError(f"Invalid effective source lineage at outer{outer}")
        block_pin = blocks[entry["block_file"]]
        block_id, block = read_source(entry["block_file"], block_pin["sha256"], block_pin["bytes"])
        scalar = block["rows"][entry["block_row_index"]]
        if scalar["row_id"] != entry["row_id"] or scalar["outer"] != outer or scalar["source_commit"] != expected_commit:
            raise ValueError(f"Scalar block row identity mismatch at outer{outer}")
        source_name = point.get("source_metric_file") or point.get("source_poll")
        source_sha = point.get("source_metric_file_sha256") or point.get("source_poll_sha256")
        metric_id, source = read_source(source_name, source_sha)
        original_line_sha = point.get("original_metric_line_sha256")
        if outer == 1:
            rows = source["stage_log"]["outer1_complete_metric_lines"]
            chosen = [r for r in rows if r["metrics"].get("step") == 1]
            if len(chosen) != 1 or chosen[0]["line_sha256"] != original_line_sha:
                raise ValueError("Parent1 original metric line binding mismatch")
            metric = chosen[0]["metrics"]
            selector = "stage_log.outer1_complete_metric_lines[metrics.step=1].metrics"
        else:
            metric = exact_metric(source, outer)
            selector = "iteration_metrics" if isinstance(source["iteration_metrics"], dict) else f"iteration_metrics[step={outer}]"
        if source.get("source_commit", expected_commit) != expected_commit:
            raise ValueError(f"Original metric source commit mismatch at outer{outer}")
        if original_line_sha and outer != 1:
            lines = [r for r in source.get("metric_line_sources", []) if r["outer"] == outer]
            if len(lines) != 1 or lines[0]["line_sha256"] != original_line_sha:
                raise ValueError(f"Original historical metric line binding mismatch outer{outer}")
        full_id = None
        if outer >= 6:
            full_name = point.get("source_receipt") or (point["source_metric_file"] if pathlib.Path(point["source_metric_file"]).name.startswith("FULL_") else f"FULL_OUTER{outer}_PROGRESS_RECEIPT.json")
            full_sha = point.get("source_receipt_sha256") or point.get("source_metric_file_sha256")
            full_id, full = read_source(full_name, full_sha)
            if full["source_commit"] != expected_commit or full["completed_outer"] != outer:
                raise ValueError(f"FULL identity mismatch outer{outer}")
            fm = exact_metric(full, outer)
            if any(fm.get(k) != metric.get(k) for k in FIELDS.values()):
                issues.append({"outer": outer, "kind": "FULL_original_required_metric_mismatch"})
            # Early curve6..10 selects FULL; also bind its actual original poll.
            if source_name == full_name and full.get("source_poll"):
                metric_id, source = read_source(full["source_poll"], full["source_poll_sha256"])
                metric = exact_metric(source, outer)
                selector = f"iteration_metrics[step={outer}]"
                if any(fm.get(k) != metric.get(k) for k in FIELDS.values()):
                    issues.append({"outer": outer, "kind": "FULL_original_poll_required_metric_mismatch"})
        vm = scalar["variance_metrics"]
        decoded = dict(zip(block["scalar_fieldsets"][vm["field_set"]], vm["values"])) if isinstance(vm, dict) and "field_set" in vm else vm if isinstance(vm, dict) else {}
        if any(decoded.get(k) != metric.get(k) for k in FIELDS.values() if k.startswith("tree_variance/")):
            issues.append({"outer": outer, "kind": "compact_block_original_required_metric_mismatch"})
        for key, original in FIELDS.items():
            if outer <= 170:
                curve_key = {"five_leaf_same_reward_question_fraction": "same_reward_question_fraction", "five_leaf_mixed_reward_question_fraction": "mixed_reward_question_fraction", "all_roots_same_reward_tree_fraction": "same_reward_tree_fraction", "all_roots_mixed_reward_tree_fraction": "mixed_reward_tree_fraction"}.get(key, key)
                if curve_key in point and point[curve_key] != metric.get(original):
                    issues.append({"outer": outer, "kind": "curve_original_required_metric_mismatch", "field": key})
        def relation(label, a, b):
            if not (present(a) and present(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)):
                relation_failures.append({"outer": outer, "relation": label, "left": a, "right": b})
        def v(k):
            return metric.get("tree_variance/" + k)
        q, t = v("question_count"), v("tree_count")
        relation("question_count512", q, 512)
        relation("tree_count1024", t, 1024)
        relation("same_question+mixed_question=1", v("same_reward_question_fraction") + v("mixed_reward_question_fraction"), 1)
        relation("exact1+exact0+mixed_question=question_count", v("exact_reward_one_question_count") + v("exact_reward_zero_question_count") + v("mixed_question_count"), q)
        relation("mixed_question_fraction=mixed_question_count/question_count", v("mixed_reward_question_fraction"), v("mixed_question_count") / q)
        relation("same_tree+mixed_tree=1", v("same_reward_tree_fraction") + v("mixed_reward_tree_fraction"), 1)
        relation("exact1+exact0+mixed_tree=tree_count", v("exact_reward_one_tree_count") + v("exact_reward_zero_tree_count") + v("mixed_tree_count"), t)
        relation("mixed_tree_fraction=mixed_tree_count/tree_count", v("mixed_reward_tree_fraction"), v("mixed_tree_count") / t)
        relation("pure_intra_alias", v("intra_adv_population_variance"), v("within_root_normalized_component_population_variance"))
        for slot in [0, 1]:
            prefix = f"root{slot}_"
            relation(prefix + "same+mixed=1", v(prefix + "same_reward_fraction") + v(prefix + "mixed_reward_fraction"), 1)
            relation(prefix + "exact1+exact0+mixed=question_count", v(prefix + "exact_reward_one_count") + v(prefix + "exact_reward_zero_count") + v(prefix + "mixed_count"), q)
            relation(prefix + "mixed_fraction=mixed_count/question_count", v(prefix + "mixed_reward_fraction"), v(prefix + "mixed_count") / q)
            relation(prefix + "same_fraction=(exact1+exact0)/question_count", v(prefix + "same_reward_fraction"), (v(prefix + "exact_reward_one_count") + v(prefix + "exact_reward_zero_count")) / q)
        relation("pooled_tree_mixed_count=root0+root1_mixed_count", v("mixed_tree_count"), v("root0_mixed_count") + v("root1_mixed_count"))
        for k, value in metric.items():
            if k.startswith("tree_variance/") and k.endswith("_count") and (not present(value) or value < 0 or value != int(value)):
                issues.append({"outer": outer, "kind": "noninteger_original_count", "field": k, "value": value})
        records[outer] = metric
        bindings.append({"outer": outer, "lineage": expected_commit[:4], "metric_source": metric_id, "selector": selector, "FULL_receipt": full_id, "scalar_block": block_id, "scalar_row_index": entry["block_row_index"], "original_metric_line_sha256": original_line_sha})
    source_repo = ROOT / "work/tree-speed-repair-20261008/nested-native5-controls-source"
    definition_sources = []
    for commit in COMMITS:
        blob = subprocess.run(["git", "-C", str(source_repo), "show", commit + ":search_r1/baselines/advantages.py"], check=True, capture_output=True).stdout
        definition_sources.append({"local_repository": str(source_repo), "source_commit": commit, "repository_path": "search_r1/baselines/advantages.py", "bytes": len(blob), "sha256": sha(blob), "definition_lines": [65, 145], "scope": "Local field-definition read only; no code review or execution."})
    coverage = {}
    for key, original in FIELDS.items():
        found = [n for n in records if present(records[n].get(original))]
        coverage[key] = {"original_field": original, "actual_covered_outers": found, "unknown_outers": [n for n in records if n not in found], "values_in_outer1_to171_order": [records[n].get(original) if present(records[n].get(original)) else None for n in records], "value_origin": "direct original scalar, not imputed or reconstructed"}
    mirrors = [pathlib.Path(p) for p in compact["mirrors"]]
    if len(mirrors) != 2 or any(not p.is_dir() for p in mirrors):
        raise ValueError("Expected exactly two existing result mirrors")
    output = {
        "schema": "tree_nested_metric_record_coverage_through171_local_v1", "created_at_utc": now,
        "scope": "Purely local actual scalar-record coverage inventory. No training/implementation/theory review, tests, SSH, remote operations, tensors, raw rollouts or full evaluation outputs. Coverage is through effective outer171 only and is not final200 acceptance or continuous observation coverage.",
        "source_path_base": str(BASE), "source_catalog": sources,
        "effective_lineage": [{"source_commit": COMMITS[0], "outers": [1]}, {"source_commit": COMMITS[1], "outers": [2, 3, 4, 5]}, {"source_commit": COMMITS[2], "outers": list(range(6, 172))}],
        "failed_or_duplicate_attempts_counted": False, "latest_values_used_for_earlier_outers": False,
        "coverage": coverage, "per_outer_source_bindings": bindings,
        "original_denominator_and_count_series": {k: [records[n].get("tree_variance/" + k) for n in records] for k in SERIES_EXTRA},
        "definitions": {
            "source_bindings": definition_sources,
            "training_reward_score": "Direct critic/rewards/mean and critic/score/mean; same scalar values in the actual records, neither converted into question all-correct/all-wrong counts.",
            "raw_within_tree_reward_population_variance": "Population variance(correction=0) of raw leaf rewards within each root tree, averaged equally over all1024 roots; not the five-leaf question-group reward variance.",
            "pure_intra_adv_population_variance": "Population variance(correction=0) within each root of the normalized intra component; averaged equally over1024 roots. intra_adv_population_variance and within_root_normalized_component_population_variance are equal aliases in the actual records. Neither within_root_actual_adv nor sample/leaf-mass variance is substituted.",
            "total_actual_adv_population_variance": "group_actual_adv_population_variance: population variance of actual scalar(inter+intra) advantage among the five leaves of each question; averaged equally over512 questions. This is not a variance pooled across the entire batch or token weighted.",
            "five_leaf_question_group": "same/mixed fractions directly check raw max==min / max!=min over all5 leaves per question; denominator512. exact_reward_one/zero_question_count are sums of per-question all(r==1)/all(r==0), not averaged indicators.",
            "root_slots": "root0 and root1 same/mixed fractions directly check the leaves belonging to each root slot; denominator512 for each slot. Each question has two roots, with2 and3 leaves. Pooled tree fractions have denominator1024. Question-group equality is not used to infer either root's equality.",
            "allowed_exact_conversions": ["mixed_question_count/question_count equals mixed_reward_question_fraction", "(exact_reward_one_question_count+exact_reward_zero_question_count)/question_count equals same_reward_question_fraction for these binary reward records", "rootN_mixed_count/question_count equals rootN_mixed_reward_fraction", "(rootN_exact_reward_one_count+rootN_exact_reward_zero_count)/question_count equals rootN_same_reward_fraction", "mixed_tree_count/tree_count equals mixed_reward_tree_fraction"],
            "conversion_policy": "All required values are direct original fields; these exact relations only verify aliases/denominators. No converted values replace missing original fields, and no counts are reconstructed from a mean. null means unknown.",
        },
        "actual_local_relations": {"checked_outers": list(range(1, 172)), "relation_failures": relation_failures, "FULL_original_and_compact_or_curve_mismatches": issues, "count_fields_are_nonnegative_integers": not any(x["kind"] == "noninteger_original_count" for x in issues)},
        "historical_gap_scope": "107..118 use each curve row's SHA-bound HISTORICAL_GAP107_119_SCALAR_READBACK source and original line SHA. Source poll1265 contains only117..119; it is never represented as a contemporaneous poll for107..116.119 retains original live poll binding.",
        "required_field_gaps": {k: v["unknown_outers"] for k, v in coverage.items() if v["unknown_outers"]},
        "not_final_acceptance": True, "remote_operations": False, "training_modified": False, "existing_products_modified": False,
        "generator": {"path": str(pathlib.Path(__file__).resolve()), "bytes": pathlib.Path(__file__).stat().st_size, "sha256": sha(pathlib.Path(__file__).read_bytes())},
        "publication": {"copies": [str(BASE)] + [str(p) for p in mirrors], "byte_readback_rounds": 2, "readback_receipt": NAME + "_READBACKS.json"},
    }
    missing = output["required_field_gaps"]
    md = ("# outer1..171 本地实际指标记录覆盖\n\n"
          "有效 lineage：0bef 的1，ba9 的2..5，778 的6..171；失败和重复尝试不计。\n\n"
          + ("所有要求的 reward/score、三类方差、五叶题组同奖/混奖比例、全1/全0题计数和 root0/root1 同奖/混奖比例均有逐轮实际原始标量（171/171）。\n\n" if not missing else "存在原始字段缺口，见 JSON 的 required_field_gaps。\n\n")
          + f"原字段与 FULL、紧凑块、曲线的值差异：{len(issues)}；已记录分母/计数关系不符：{len(relation_failures)}。\n\n"
          "JSON 保存逐轮原值、可读本地来源路径及 SHA、各轮选择器和 null 缺口。题分母为512，每个 root 槽512，合并树分母1024；全1/全0是原始求和计数。纯树内方差不替换为实际总优势、样本或叶质量加权方差。\n\n"
          "107..118 使用历史原日志补录和逐条行 SHA；未将 live119 值当作早期轮值。没有由题组同奖推断各树同奖，没有用均值反推计数，也没有补造缺失值。\n\n"
          "这仅为本地记录覆盖清点，不是代码 review、测试、训练门禁、最终200验收或连续监控覆盖声明。训练和已有产物未修改。BASE 与既有两个 mirror 的新清单进行两轮字节读回，见同名 READBACKS.json。\n").encode()
    products = {NAME + ".json": encode(output), NAME + ".md": md}
    copies = [BASE] + mirrors
    for folder in copies:
        for name in products:
            if (folder / name).exists() or (folder / (NAME + "_READBACKS.json")).exists():
                raise FileExistsError("Immutable inventory product already exists")
    for folder in copies:
        for name, raw in products.items():
            with (folder / name).open("xb") as handle:
                handle.write(raw)
    readbacks = []
    for round_no in [1, 2]:
        round_rows = []
        for folder in copies:
            for name, expected in products.items():
                actual = (folder / name).read_bytes()
                if actual != expected:
                    raise ValueError(f"Published bytes differ: {folder / name}")
                round_rows.append({"path": str(folder / name), "bytes": len(actual), "sha256": sha(actual), "byte_equal": True})
        readbacks.append({"round": round_no, "read_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "files": round_rows})
    # A separate receipt avoids a self-referential hash in the frozen inventory.
    receipt = encode({"schema": "tree_nested_metric_record_coverage_through171_two_round_readbacks_v1", "inventory_generator_sha256": output["generator"]["sha256"], "rounds": readbacks, "all_three_copies_byte_equal_in_both_rounds": True, "source_reads": len(sources), "remote_operations": False, "training_modified": False, "existing_products_modified": False})
    for folder in copies:
        with (folder / (NAME + "_READBACKS.json")).open("xb") as handle:
            handle.write(receipt)
    for round_no in [1, 2]:
        for folder in copies:
            if (folder / (NAME + "_READBACKS.json")).read_bytes() != receipt:
                raise ValueError("Readback receipt copy differs")
    # Recheck every frozen local input after actual extraction/publication.
    for ref in sources.values():
        raw = (BASE / ref["path"]).read_bytes()
        if len(raw) != ref["bytes"] or sha(raw) != ref["sha256"]:
            raise ValueError(f"Local source changed during inventory: {ref['path']}")
    print(json.dumps({"required_field_gaps": missing, "original_binding_mismatches": issues, "denominator_relation_failures": relation_failures, "source_files": len(sources), "generator": output["generator"], "products": [{"path": str(BASE / n), "bytes": len(v), "sha256": sha(v)} for n, v in {**products, NAME + "_READBACKS.json": receipt}.items()], "all_three_copies_two_round_byte_readback": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
