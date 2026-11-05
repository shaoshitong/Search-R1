"""Independent payoff/deviation audit of every selected two-period policy."""
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar


def choices(number, cap):
    for total_accounts in range(number, 3):
        opening = total_accounts - number
        for production in itertools.product(range(cap + 1), repeat=total_accounts):
            if sum(production) <= cap and (opening == 0 or production[-1] >= 1):
                yield dict(opened=opening, production=production)


def independent_stage(state, pair, p):
    accounts = [list(s) + [(0, 0)] * a["opened"] for s, a in zip(state, pair)]
    raw = [[(1 + h)**p["beta"] * (b + x)**p["alpha"]
            for (b, h), x in zip(s, a["production"])] for s, a in zip(accounts, pair)]
    denominator = sum(sum(v) for v in raw)
    exposure = [[v / denominator for v in rr] for rr in raw]
    utility = [p["prize"] * sum(e) - p["costs"][i] * sum(a["production"])
               - p["opening"] * a["opened"] - p["maintenance"] * (len(s) - 1)
               for i, (s, a, e) in enumerate(zip(accounts, pair, exposure))]
    following = [[(p["rho"]*b + x, p["delta"]*h + p["capital_gain"]*e)
                  for (b, h), x, e in zip(s, a["production"], ee)]
                 for s, a, ee in zip(accounts, pair, exposure)]
    return np.array(utility), following


def expected(state, policies, p):
    answer = np.zeros(2)
    for a, b in itertools.product(*[pp["support"] for pp in policies]):
        u, _ = independent_stage(state, (a, b), p)
        answer += a["probability"] * b["probability"] * u
    return answer


def terminal_regret(state, policies, p):
    value = expected(state, policies, p)
    gains = [0.]
    for player in range(2):
        for deviated in choices(len(state[player]), p["capacity"]):
            gain = 0.
            for other in policies[1-player]["support"]:
                pair = [deviated, other] if player == 0 else [other, deviated]
                u, _ = independent_stage(state, pair, p)
                gain += other["probability"] * u[player]
            gains.append(gain - value[player])
    return max(gains), value


root = Path(sys.argv[1])
summary = json.loads((root / "dynamic_game_summary.json").read_text())
details = json.loads((root / "payoff_matrices_and_subgames.json").read_text())
errors, regrets = [], []
for row, detail in zip(summary["results"], details):
    p, state = row["parameters"], row["initial_state"]
    act = [list(choices(len(s), p["capacity"])) for s in state]
    for sub in detail["continuation"]:
        i, j = sub["first_indices"]
        pay, following = independent_stage(state, (act[0][i], act[1][j]), p)
        errors.append(max(abs(v-w) for creator, recorded in zip(following, sub["state"])
                          for account, stored in zip(creator, recorded)
                          for v,w in zip(account, stored)))
        gain, next_value = terminal_regret(following, sub["second_actions"], p)
        regrets.append(gain)
        errors.append(float(np.max(np.abs(next_value-sub["value"]))))
        for who, name in enumerate(["payoff_A", "payoff_B"]):
            value = pay[who] + p["discount"] * next_value[who]
            errors.append(abs(value - detail[name][i][j]))
    weights = []
    for aa, policy in zip(act, row["first_actions"]):
        weights.append(np.array([sum(s["probability"] for s in policy["support"]
                                     if s["opened"] == a["opened"] and
                                     tuple(s["production"]) == tuple(a["production"])) for a in aa]))
    A, B = np.array(detail["payoff_A"]), np.array(detail["payoff_B"])
    x, y = weights
    regrets.extend([float(max(A@y)-x@A@y), float(max(x@B)-x@B@y)])
    errors.append(float(np.max(np.abs(np.array([x@A@y, x@B@y])-row["value"]))))

# An independently optimised analytic two-period slice supplied in the derivation.
R, old, decay, discount = 12., 2.5, .25, .25
def value_first(x, rival):
    share = (old+x)/(2*old+x+rival)
    next_share = (1+share)/3
    next_stock = decay*old+x
    next_production = R*next_share*(1-next_share)-next_stock
    return R*share-x+discount*(R*next_share-next_production)
xstar = 11/6
best = minimize_scalar(lambda x: -value_first(x, xstar), bounds=(0,2), method="bounded")
closed = dict(first_production=xstar, numeric_best_response=best.x,
              next_stock=59/24, next_capital=.5, second_production=13/24,
              value=value_first(xstar,xstar), stated_value=177/32,
              deviation_gain=max(0., -best.fun-value_first(xstar,xstar)))
result = dict(utc=datetime.now(timezone.utc).isoformat(), conditions=len(details),
              terminal_subgames=sum(len(d["continuation"]) for d in details),
              maximum_payoff_or_state_reconstruction_error=max(errors),
              maximum_independent_deviation_gain=max(0.,max(regrets)),
              threshold_conditions=summary["threshold_audit"]["conditions"],
              maximum_two_account_threshold_gap=summary["threshold_audit"]["max_two_account_threshold_gap"],
              analytic_two_period_example=closed)
result["pass"] = (max(errors)<1e-8 and max(regrets)<1e-8
                  and result["maximum_two_account_threshold_gap"]<1e-7
                  and abs(closed["value"]-closed["stated_value"])<1e-10
                  and closed["deviation_gain"]<1e-8)
(root/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
assert result["pass"], result
