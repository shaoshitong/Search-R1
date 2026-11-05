"""Exact V4 weighted-account finite games under a frozen V5 protocol.

No platform behavior, AI adoption or unknown-owner linkage is inferred here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allocation(n, k, alpha, ratio):
    if k == 1:
        return n ** alpha, {"old": n, "new_accounts": 0, "new_base": 0, "new_extra": 0}
    best = None
    for old in range(1, n - k + 2):
        base, extra = divmod(n - old, k - 1)
        score = old ** alpha + ratio * ((k - 1 - extra) * base ** alpha + extra * (base + 1) ** alpha)
        if best is None or score > best[0] + 1e-12:
            best = score, {"old": old, "new_accounts": k - 1, "new_base": base, "new_extra": extra}
    return best


def compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for first in range(1, total - parts + 2):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest


def verify_allocation(protocol):
    maximum_error, checked = 0.0, 0
    grid = protocol["symmetric_grid"]
    for alpha, ratio in itertools.product(grid["alpha"], grid["new_to_old_weight_ratio"]):
        for n in range(1, 8):
            for k in range(1, n + 1):
                exact = max(x[0] ** alpha + ratio * sum(t ** alpha for t in x[1:])
                            for x in compositions(n, k))
                maximum_error = max(maximum_error, abs(exact - allocation(n, k, alpha, ratio)[0]))
                checked += 1
    return {"action_scores_checked": checked, "maximum_exhaustive_composition_error": maximum_error,
            "n_upper_bound_for_independent_compositions": 7}


def conditions(protocol):
    symmetric, heterogeneous = protocol["symmetric_grid"], protocol["heterogeneous_grid"]
    result = []
    for alpha, ratio, tau, c in itertools.product(symmetric["alpha"], symmetric["new_to_old_weight_ratio"],
                                                 symmetric["tau"], symmetric["common_c"]):
        result.append({"family": "symmetric", "R": protocol["R"], "alpha": alpha, "ratio": ratio,
                       "tau": tau, "c": [c, c], "theta": symmetric["theta"]})
    for alpha, ratio, tau, theta, c1 in itertools.product(heterogeneous["alpha"], heterogeneous["new_to_old_weight_ratio"],
            heterogeneous["tau"], heterogeneous["theta_pairs"], heterogeneous["c1"]):
        result.append({"family": "heterogeneous", "R": protocol["R"], "alpha": alpha, "ratio": ratio,
                       "tau": tau, "c": [c1, heterogeneous["c2"]], "theta": theta})
    return [dict(case_id=i, **x) for i, x in enumerate(result)]


def continuous(case):
    R, alpha, tau = case["R"], case["alpha"], case["tau"]
    b = [t * c ** (-alpha) for t, c in zip(case["theta"], case["c"])]
    share = [x / sum(b) for x in b]
    M = R * share[0] * share[1]
    eta = case["ratio"] ** (1 / (1 - alpha)) if alpha < 1 else None
    k = 1 + max(0.0, (1 - alpha) * M / tau - 1 / eta) if alpha < 1 else 1.0
    n = [alpha * M / c for c in case["c"]]
    old = [x / (1 + eta * (k - 1)) if eta is not None else x for x in n]
    new = [eta * x if eta is not None and k > 1 else None for x in old]
    flags = {
        "total_content_below_account_count": any(x + 1e-9 < k for x in n),
        "old_account_content_below_one": any(x < 1 - 1e-9 for x in old),
        "new_account_content_below_one": any(x is not None and x < 1 - 1e-9 for x in new),
        "noninteger_total_content_or_accounts": any(abs(x - round(x)) > 1e-9 for x in n + [k]),
        "noninteger_account_allocation": any(abs(x - round(x)) > 1e-9 for x in old + [x for x in new if x is not None]),
    }
    return {"n": n, "k": k, "shares": share, "M": M, "eta": eta, "old_content": old,
            "new_content_per_account": new,
            "single_account_threshold": (1 - alpha) * M * eta if eta is not None else 0.0,
            "split": k > 1 + 1e-9, "physical_or_integer_violations": flags}


def action_table(case, player, cache):
    maximum_n = math.floor(case["R"] / case["c"][player] + 1)
    key = case["alpha"], case["ratio"], maximum_n
    if key not in cache:
        actions = [(n, k) for n in range(1, maximum_n + 1) for k in range(1, n + 1)]
        allocated = [allocation(n, k, case["alpha"], case["ratio"]) for n, k in actions]
        cache[key] = actions, allocated
    actions, allocated = cache[key]
    scores = case["theta"][player] * np.array([x[0] for x in allocated])
    costs = np.array([case["c"][player] * n + case["tau"] * (k - 1) for n, k in actions])
    return maximum_n, actions, allocated, scores, costs


def solve(case, cache, tolerance):
    tables = [action_table(case, player, cache) for player in (0, 1)]
    _, a1, alloc1, s1, costs1 = tables[0]
    _, a2, alloc2, s2, costs2 = tables[1]
    share = s1[:, None] / (s1[:, None] + s2[None, :])
    u1 = case["R"] * share - costs1[:, None]
    u2 = case["R"] * (1 - share) - costs2[None, :]
    best1, best2 = u1.max(axis=0), u2.max(axis=1)
    gain1, gain2 = best1[None, :] - u1, best2[:, None] - u2
    regret = np.maximum(gain1, gain2)
    matches = np.argwhere(regret <= tolerance)
    symmetric = case["c"][0] == case["c"][1] and case["theta"][0] == case["theta"][1]
    benchmark = continuous(case)

    def describe(i, j, strictness=True):
        actions = [a1[i], a2[j]]
        result = {"actions": [{"n": n, "k": k} for n, k in actions],
                  "allocation": [alloc1[i][1], alloc2[j][1]],
                  "payoffs": [float(u1[i, j]), float(u2[i, j])],
                  "shares": [float(share[i, j]), float(1 - share[i, j])],
                  "maximum_unilateral_gain": float(regret[i, j]),
                  "difference_from_continuous": [{"n": n - benchmark["n"][p], "k": k - benchmark["k"]}
                                                  for p, (n, k) in enumerate(actions)]}
        if strictness:
            alternatives1, alternatives2 = u1[:, j].copy(), u2[i, :].copy()
            alternatives1[i], alternatives2[j] = -np.inf, -np.inf
            x, y = int(alternatives1.argmax()), int(alternatives2.argmax())
            margins = [float(u1[i, j] - alternatives1[x]), float(u2[i, j] - alternatives2[y])]
            result.update({"strict_margins": margins, "strict": min(margins) > tolerance,
                           "best_distinct_actions": [{"n": a1[x][0], "k": a1[x][1]}, {"n": a2[y][0], "k": a2[y][1]}]})
        return result

    equilibria = [describe(int(i), int(j)) for i, j in matches if not symmetric or i <= j]
    minimum = np.unravel_index(regret.argmin(), regret.shape)
    result = {"condition": case, "maximum_n": [t[0] for t in tables], "actions_per_player": [len(a1), len(a2)],
              "profiles_enumerated": len(a1) * len(a2), "continuous_benchmark": benchmark,
              "pure_count_including_player_exchange": len(matches), "profiles_listed_up_to_exchange": symmetric,
              "pure_equilibria": equilibria, "minimum_maximum_unilateral_gain": float(regret[minimum])}
    if not equilibria:
        result["least_regret_profile"] = describe(int(minimum[0]), int(minimum[1]), False)
    return result


def verify_equilibria(results, cache):
    checks, max_gain, max_error = 0, 0.0, 0.0
    for result in results:
        case = result["condition"]
        tables = [action_table(case, p, cache) for p in (0, 1)]
        for profile in result["pure_equilibria"]:
            for p in (0, 1):
                own, other = profile["actions"][p], profile["actions"][1 - p]
                other_score = case["theta"][1 - p] * allocation(other["n"], other["k"], case["alpha"], case["ratio"])[0]
                own_score = case["theta"][p] * allocation(own["n"], own["k"], case["alpha"], case["ratio"])[0]
                current = case["R"] * own_score / (own_score + other_score) - case["c"][p] * own["n"] - case["tau"] * (own["k"] - 1)
                max_error = max(max_error, abs(current - profile["payoffs"][p]))
                for (n, k), (value, _) in zip(tables[p][1], tables[p][2]):
                    score = case["theta"][p] * value
                    payoff = case["R"] * score / (score + other_score) - case["c"][p] * n - case["tau"] * (k - 1)
                    max_gain = max(max_gain, payoff - current)
                    checks += 1
    return {"unilateral_actions_recomputed": checks, "maximum_positive_unilateral_gain": max_gain,
            "maximum_payoff_recomputation_error": max_error}


def summary(results):
    exist = [r for r in results if r["pure_equilibria"]]
    over, under, mixed_status = [], [], []
    for result in exist:
        states = {any(a["k"] > 1 for a in eq["actions"]) for eq in result["pure_equilibria"]}
        cid = result["condition"]["case_id"]
        if len(states) > 1:
            mixed_status.append(cid)
        if result["continuous_benchmark"]["split"] and False in states:
            over.append(cid)
        if not result["continuous_benchmark"]["split"] and True in states:
            under.append(cid)
    return {"conditions": len(results), "with_pure_equilibria": len(exist),
            "no_pure_case_ids": [r["condition"]["case_id"] for r in results if not r["pure_equilibria"]],
            "continuous_split_but_finite_has_no_split_case_ids": over,
            "continuous_no_split_but_finite_has_split_case_ids": under,
            "finite_equilibria_have_both_split_statuses_case_ids": mixed_status,
            "continuous_has_total_content_below_accounts": sum(r["continuous_benchmark"]["physical_or_integer_violations"]["total_content_below_account_count"] for r in results),
            "continuous_has_new_content_below_one": sum(r["continuous_benchmark"]["physical_or_integer_violations"]["new_account_content_below_one"] for r in results),
            "profiles_enumerated": sum(r["profiles_enumerated"] for r in results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cases, cache = conditions(protocol), {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "progress.jsonl"
    results = []
    with progress.open("w") as stream:
        for case in cases:
            result = solve(case, cache, protocol["enumeration"]["pure_equilibrium_tolerance"])
            results.append(result)
            item = {"completed": len(results), "planned": len(cases), "case_id": case["case_id"],
                    "pure_count": result["pure_count_including_player_exchange"],
                    "minimum_regret": result["minimum_maximum_unilateral_gain"]}
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            if len(results) % 16 == 0 or len(results) == len(cases):
                print(json.dumps(item), flush=True)
    audit = {"protocol_sha256": digest(args.protocol), "source_sha256": digest(Path(__file__)),
             "completed_utc": datetime.now(timezone.utc).isoformat(), "summary": summary(results), "conditions": results}
    target = args.output_dir / "weighted_integer_results.json"
    target.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"stage": "enumeration_complete", **audit["summary"]}), flush=True)
    verification = {"protocol_sha256": digest(args.protocol), "source_sha256": digest(Path(__file__)),
                    "results_sha256": digest(target), "allocation": verify_allocation(protocol),
                    "equilibria": verify_equilibria(results, cache), "condition_count": len(results),
                    "expected_condition_count": protocol["expected_conditions"]}
    (args.output_dir / "weighted_integer_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"stage": "verified_complete", **verification}), flush=True)


if __name__ == "__main__":
    main()
