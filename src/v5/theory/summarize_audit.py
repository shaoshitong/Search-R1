"""Summarize the frozen-grid audit; never adds or reruns model conditions."""
import argparse
import json
from pathlib import Path


def states(result):
    profiles = result["pure_equilibria"]
    if not profiles:
        return None
    extra = [sum(a["k"] - 1 for a in e["actions"]) for e in profiles]
    return {"total_extra_accounts_range": [min(extra), max(extra)],
            "k_ranges": [[min(e["actions"][i]["k"] for e in profiles), max(e["actions"][i]["k"] for e in profiles)] for i in (0, 1)],
            "strict_profiles_listed": sum(e["strict"] for e in profiles)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.results.read_text())
    results = audit["conditions"]
    exact = []
    for row in results:
        b = row["continuous_benchmark"]
        if not any(b["physical_or_integer_violations"].values()):
            matched = any(all(abs(a["n"] - b["n"][i]) < 1e-9 and abs(a["k"] - b["k"]) < 1e-9
                              for i, a in enumerate(e["actions"])) for e in row["pure_equilibria"])
            exact.append({"case_id": row["condition"]["case_id"], "continuous_candidate_is_finite_NE": matched})
    weight_groups, cost_groups = {}, {}
    for row in results:
        p = row["condition"]
        if p["family"] == "symmetric":
            weight_groups.setdefault((p["alpha"], p["tau"], p["c"][0]), []).append(row)
        else:
            cost_groups.setdefault((p["alpha"], p["ratio"], p["tau"], tuple(p["theta"]), p["c"][1]), []).append(row)
    weights = []
    for group in weight_groups.values():
        ordered = sorted(group, key=lambda x: -x["condition"]["ratio"])
        for old, new in zip(ordered, ordered[1:]):
            a, b = states(old), states(new)
            weights.append({"case_ids": [old["condition"]["case_id"], new["condition"]["case_id"]],
                            "ratios": [old["condition"]["ratio"], new["condition"]["ratio"]],
                            "states": [a, b],
                            "possible_more_splitting_at_lower_new_weight": a is not None and b is not None and b["total_extra_accounts_range"][1] > a["total_extra_accounts_range"][0],
                            "all_equilibria_more_splitting_at_lower_new_weight": a is not None and b is not None and b["total_extra_accounts_range"][0] > a["total_extra_accounts_range"][1]})
    costs = []
    for group in cost_groups.values():
        ordered = sorted(group, key=lambda x: -x["condition"]["c"][0])
        for old, new in zip(ordered, ordered[1:]):
            a, b = states(old), states(new)
            delta = new["continuous_benchmark"]["k"] - old["continuous_benchmark"]["k"]
            reverse_players = []
            if a is not None and b is not None:
                for i in (0, 1):
                    if delta > 1e-9 and b["k_ranges"][i][1] < a["k_ranges"][i][0]:
                        reverse_players.append(i + 1)
                    if delta < -1e-9 and b["k_ranges"][i][0] > a["k_ranges"][i][1]:
                        reverse_players.append(i + 1)
            costs.append({"case_ids": [old["condition"]["case_id"], new["condition"]["case_id"]],
                          "c1": [old["condition"]["c"][0], new["condition"]["c"][0]],
                          "continuous_p1": [old["continuous_benchmark"]["shares"][0], new["continuous_benchmark"]["shares"][0]],
                          "continuous_M": [old["continuous_benchmark"]["M"], new["continuous_benchmark"]["M"]],
                          "continuous_k": [old["continuous_benchmark"]["k"], new["continuous_benchmark"]["k"]],
                          "finite_states": [a, b], "players_with_strict_direction_reversal": reverse_players})
    output = {"protocol_sha256": audit["protocol_sha256"],
              "exactly_implementable_continuous_candidates": exact,
              "weight_comparisons": weights, "unilateral_cost_comparisons": costs,
              "summary": {"implementable_candidates": len(exact), "implementable_candidates_missing_from_finite_NE": sum(not e["continuous_candidate_is_finite_NE"] for e in exact),
                          "weight_comparisons": len(weights), "weight_strong_reversals": sum(x["all_equilibria_more_splitting_at_lower_new_weight"] for x in weights),
                          "weight_possible_reversals": sum(x["possible_more_splitting_at_lower_new_weight"] for x in weights),
                          "cost_comparisons": len(costs), "cost_comparisons_without_both_pure_NE": sum(None in x["finite_states"] for x in costs),
                          "cost_comparisons_with_strict_direction_reversals": sum(bool(x["players_with_strict_direction_reversal"]) for x in costs)}}
    path = args.output or args.results.with_name("weighted_integer_comparisons.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))
    for row in costs:
        if row["players_with_strict_direction_reversal"]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
