"""Exact finite two-creator games; integer works and integer accounts.

This is a computational mechanism check, not human behavioral evidence.
Each creator chooses 1 <= k <= n. All works have equal fixed quality.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


RULES = (
    ("account_linear", 1.0, False),
    ("account_concave", 0.5, False),
    ("owner_aggregated_concave", 0.5, True),
)


def exact_score(n, k, alpha, aggregate_owner):
    if aggregate_owner:
        return n ** alpha
    q, r = divmod(n, k)
    return (k - r) * q ** alpha + r * (q + 1) ** alpha


def continuous_benchmark(R, c, tau, alpha, aggregate_owner):
    A = R / 4
    if aggregate_owner:
        k = 1.0
    else:
        k = max(1.0, min(A * (1 - alpha) / tau, A / (c + tau)))
    n = max(k, A * alpha / c)
    return {
        "n_per_creator": n,
        "k_per_creator": k,
        "payoff_per_creator": R / 2 - c * n - tau * (k - 1),
    }


def solve_condition(rule, alpha, aggregate_owner, R, c, tau):
    # Baseline (1,1) guarantees payoff > -c against every opponent.
    # Any n > R/c+1 has payoff <= R-cn < -c and is dominated.
    max_n = math.floor(R / c + 1)
    actions = [(n, k) for n in range(1, max_n + 1) for k in range(1, n + 1)]
    scores = np.array([exact_score(n, k, alpha, aggregate_owner) for n, k in actions])
    costs = np.array([c * n + tau * (k - 1) for n, k in actions])
    payoff = R * scores[:, None] / (scores[:, None] + scores[None, :]) - costs[:, None]
    best_value = payoff.max(axis=0)
    deviation_gain = best_value[None, :] - payoff
    maximum_profile_gain = np.maximum(deviation_gain, deviation_gain.T)
    tolerance = 1e-10
    pure = np.argwhere(maximum_profile_gain <= tolerance)
    benchmark = continuous_benchmark(R, c, tau, alpha, aggregate_owner)

    def describe(i, j):
        profiles = [actions[i], actions[j]]
        return {
            "actions": [{"n": n, "k": k} for n, k in profiles],
            "payoffs": [float(payoff[i, j]), float(payoff[j, i])],
            "maximum_unilateral_deviation_gain": float(maximum_profile_gain[i, j]),
            "difference_from_continuous_benchmark": [
                {"n": n - benchmark["n_per_creator"], "k": k - benchmark["k_per_creator"]}
                for n, k in profiles
            ],
        }

    equilibria = [describe(int(i), int(j)) for i, j in pure if i <= j]
    minimum_index = np.unravel_index(np.argmin(maximum_profile_gain), maximum_profile_gain.shape)
    result = {
        "rule": rule,
        "alpha": alpha,
        "parameters": {"R": R, "c": c, "tau": tau},
        "maximum_n_from_dominance": max_n,
        "action_count_per_player": len(actions),
        "pure_equilibrium_count_including_player_exchange": len(pure),
        "pure_equilibria_up_to_player_exchange": equilibria,
        "continuous_benchmark": benchmark,
        "minimum_maximum_unilateral_deviation_gain": float(maximum_profile_gain[minimum_index]),
    }
    if not equilibria:
        result["least_regret_profile"] = describe(*[int(x) for x in minimum_index])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("finite_market_results.json"))
    args = parser.parse_args()
    results = []
    for rule, alpha, aggregate_owner in RULES:
        for c in (0.25, 0.5, 1.0, 2.0):
            result = solve_condition(rule, alpha, aggregate_owner, 12.0, c, 1.0)
            results.append(result)
            print(json.dumps({
                "rule": rule, "c": c, "actions": result["action_count_per_player"],
                "pure_equilibria": [x["actions"] for x in result["pure_equilibria_up_to_player_exchange"]],
                "minimum_regret": result["minimum_maximum_unilateral_deviation_gain"],
            }), flush=True)
    output = {
        "purpose": "Exact finite-action mechanism calculation, not a human experiment or an estimate of Civitai behavior.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "equilibrium_tolerance": 1e-10,
        "reporting": "Asymmetric profiles are included once; exchanging the two players gives the companion equilibrium.",
        "conditions": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "conditions": len(results)}), flush=True)


if __name__ == "__main__":
    main()
