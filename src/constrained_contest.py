"""Two creators: continuous production n >= integer accounts k >= 1.

Enumerate bounded integer-account candidates, solve each quantity subgame,
then check simultaneous joint deviations in both quantity and account count.
Only the Python standard library is needed.
"""

import argparse
import json
import math
from pathlib import Path


def score(n, k, alpha):
    return k ** (1 - alpha) * n ** alpha


def payoff(n, k, opponent_score, R, c, tau, alpha):
    s = score(n, k, alpha)
    return R * s / (s + opponent_score) - c * n - tau * (k - 1)


def production_derivative(n, k, opponent_score, R, c, alpha):
    s = score(n, k, alpha)
    return R * alpha * s * opponent_score / (n * (s + opponent_score) ** 2) - c


def best_n(k, opponent_score, R, c, alpha):
    if production_derivative(k, k, opponent_score, R, c, alpha) <= 0:
        return float(k)
    lo, hi = float(k), max(float(k), R * alpha / (4 * c))
    for _ in range(70):
        mid = (lo + hi) / 2
        if production_derivative(mid, k, opponent_score, R, c, alpha) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def continuous_equilibrium(R, c, tau, alpha):
    A = R / 4
    k = max(1.0, min(A * (1 - alpha) / tau, A / (c + tau)))
    n = max(k, A * alpha / c)
    if k <= 1 + 1e-10:
        regime = "single_account_quantity_interior" if n > 1 + 1e-10 else "one_work_one_account"
    else:
        regime = "one_work_per_account_boundary" if abs(n - k) < 1e-10 else "two_inputs_interior"
    return {"n": n, "k": k, "regime": regime}


def fixed_accounts_quantities(k1, k2, R, c, alpha):
    ratio = (k1 / k2) ** (1 - alpha)
    equal_n = R * alpha * ratio / (c * (1 + ratio) ** 2)
    if equal_n >= max(k1, k2):
        return equal_n, equal_n
    if k1 >= k2:
        n1 = float(k1)
        return n1, best_n(k2, score(n1, k1, alpha), R, c, alpha)
    n2 = float(k2)
    return best_n(k1, score(n2, k2, alpha), R, c, alpha), n2


def best_joint_response(opponent_score, R, c, tau, alpha, max_k):
    values = []
    for k in range(1, max_k + 1):
        n = best_n(k, opponent_score, R, c, alpha)
        values.append((payoff(n, k, opponent_score, R, c, tau, alpha), n, k))
    return max(values)


def integer_no_split_threshold(R, c, alpha):
    # For a fixed opponent, V(k)=max_{n>=k}[gross reward-cn] is concave
    # in continuous k. Its secant slopes from k=1 therefore decrease;
    # among integer deviations, k=2 sets the exact no-split threshold.
    n1 = max(1.0, R * alpha / (4 * c))
    other = score(n1, 1, alpha)
    n2 = best_n(2, other, R, c, alpha)
    incumbent_value = R / 2 - c * n1
    two_account_value = payoff(n2, 2, other, R, c, 0.0, alpha)
    return {
        "tau_threshold": max(0.0, two_account_value - incumbent_value),
        "single_account_quantity": n1,
        "optimal_two_account_deviation_quantity": n2,
    }


def solve_case(R, c, tau, alpha):
    # Any k > 1 + R/(c+tau) has payoff below -c, while n=k=1
    # guarantees payoff greater than -c. Such k cannot be a best response.
    max_k = math.floor(1 + R / (c + tau))
    pure = []
    least_regret = None
    max_quantity_subgame_error = 0.0
    for k1 in range(1, max_k + 1):
        for k2 in range(k1, max_k + 1):
            n1, n2 = fixed_accounts_quantities(k1, k2, R, c, alpha)
            s1, s2 = score(n1, k1, alpha), score(n2, k2, alpha)
            max_quantity_subgame_error = max(
                max_quantity_subgame_error,
                abs(best_n(k1, s2, R, c, alpha) - n1),
                abs(best_n(k2, s1, R, c, alpha) - n2),
            )
            u1 = payoff(n1, k1, s2, R, c, tau, alpha)
            u2 = payoff(n2, k2, s1, R, c, tau, alpha)
            br1 = best_joint_response(s2, R, c, tau, alpha, max_k)
            br2 = best_joint_response(s1, R, c, tau, alpha, max_k)
            regret = max(br1[0] - u1, br2[0] - u2, 0.0)
            record = {
                "n": [n1, n2], "k": [k1, k2], "payoff": [u1, u2],
                "maximum_joint_deviation_gain": regret,
            }
            if least_regret is None or regret < least_regret["maximum_joint_deviation_gain"]:
                least_regret = record
            if regret < 1e-9:
                pure.append(record)
    return {
        "parameters": {"R": R, "c": c, "tau": tau, "alpha": alpha},
        "continuous_constrained_equilibrium": continuous_equilibrium(R, c, tau, alpha),
        "integer_no_split_threshold": integer_no_split_threshold(R, c, alpha),
        "integer_account_upper_bound": max_k,
        "pure_equilibria_up_to_player_exchange": pure,
        "least_regret_candidate": least_regret,
        "quantity_subgame_best_response_error": max_quantity_subgame_error,
    }


def finite_difference_check():
    pars = {"R": 4.0, "c": 0.3, "tau": 0.2, "alpha": 0.5}
    n, k, opponent = 2.6, 2, 1.7
    eps = 1e-5
    numeric = (
        payoff(n + eps, k, opponent, **pars)
        - payoff(n - eps, k, opponent, **pars)
    ) / (2 * eps)
    analytic = production_derivative(n, k, opponent, pars["R"], pars["c"], pars["alpha"])
    return {"numeric": numeric, "analytic": analytic, "absolute_error": abs(numeric - analytic)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    costs = [1.0, 0.75, 0.6, 0.45, 0.3, 0.2, 0.18, 0.1, 0.03]
    if args.dense:
        costs = [i / 100 for i in range(3, 101)]
    cases = [solve_case(4.0, c, 0.2, 0.5) for c in costs]
    cases += [solve_case(4.0, 0.3, 0.4, 0.7), solve_case(4.0, 0.05, 0.6, 0.5)]
    result = {"finite_difference_check": finite_difference_check(), "cases": cases}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    displayed_cases = cases if not args.dense else [
        case for case in cases if not case["pure_equilibria_up_to_player_exchange"]
    ]
    print(json.dumps({
        "output": str(args.output.resolve()),
        "case_count": len(cases),
        "finite_difference_error": result["finite_difference_check"]["absolute_error"],
        "cases": [{
            "parameters": case["parameters"],
            "continuous": case["continuous_constrained_equilibrium"],
            "integer_no_split_tau": case["integer_no_split_threshold"]["tau_threshold"],
            "integer_pure": [{"n": x["n"], "k": x["k"]} for x in case["pure_equilibria_up_to_player_exchange"]],
            "minimum_candidate_regret": case["least_regret_candidate"]["maximum_joint_deviation_gain"],
        } for case in displayed_cases],
    }, indent=2))


if __name__ == "__main__":
    main()
