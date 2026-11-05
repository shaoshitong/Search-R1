"""Two-period public-state creator game; exact finite action sets, verified NE.

The capacity and account limits are economic constraints, not truncation claims.
Raw payoff matrices and per-subgame results stay on the remote machine.
"""
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog, minimize_scalar


@dataclass(frozen=True)
class Parameters:
    alpha: float = .5
    beta: float = .5
    rho: float = .5
    delta: float = .8
    capital_gain: float = 1.
    prize: float = 4.
    discount: float = .9
    opening: float = .25
    maintenance: float = .1
    costs: tuple = (1., 1.)
    capacity: int = 2


def actions(accounts, capacity):
    """One permanent original account; at most one extra account may be opened."""
    result = []
    for opened in range(2 - accounts + 1):
        count = accounts + opened
        for x in itertools.product(range(capacity + 1), repeat=count):
            if sum(x) <= capacity and (not opened or x[-1] >= 1):
                result.append((opened, x))
    return result


def stage(state, action_pair, par):
    scores, after = [], []
    for account_state, (opened, production) in zip(state, action_pair):
        padded = tuple(account_state) + ((0., 0.),) * opened
        scores.append(np.array([(1 + h) ** par.beta * (b + x) ** par.alpha
                                for (b, h), x in zip(padded, production)]))
        after.append(padded)
    denominator = sum(float(s.sum()) for s in scores)
    shares = [s / denominator for s in scores]
    payoff, next_state = [], []
    for i, (p, accounts, (opened, x)) in enumerate(zip(shares, after, action_pair)):
        payoff.append(par.prize * p.sum() - par.costs[i] * sum(x)
                      - par.opening * opened - par.maintenance * (len(accounts) - 1))
        next_state.append(tuple((par.rho * b + produced,
                                 par.delta * h + par.capital_gain * float(share))
                                for (b, h), produced, share in zip(accounts, x, p)))
    return np.array(payoff), tuple(next_state)


def regret(A, B, x, y):
    return max(float(np.max(A @ y) - x @ A @ y),
               float(np.max(x @ B) - x @ B @ y), 0.)


def equilibrium(A, B, tolerance=1e-8):
    """Pure equilibrium first; otherwise feasible-support LPs, verified regrets.

    An equilibrium with supports I,J is a pair of independent LP feasibility
    problems: supported actions tie, all other actions weakly underperform.
    Enumerating all nonempty supports is complete for these <=6x6 games.
    """
    pure = np.argwhere((A >= A.max(axis=0)[None, :] - tolerance)
                       & (B >= B.max(axis=1)[:, None] - tolerance))
    if len(pure):
        # Predeclared selection: maximum payoff sum, then lexicographic indices.
        i, j = sorted(pure.tolist(), key=lambda ij: (-(A[tuple(ij)] + B[tuple(ij)]), *ij))[0]
        x, y = np.eye(len(A))[i], np.eye(A.shape[1])[j]
        return dict(x=x, y=y, value=np.array([A[i, j], B[i, j]]),
                    kind="pure", pure_count=len(pure), regret=regret(A, B, x, y))
    nr, nc = A.shape
    supports_r = [s for k in range(1, nr + 1) for s in itertools.combinations(range(nr), k)]
    supports_c = [s for k in range(1, nc + 1) for s in itertools.combinations(range(nc), k)]
    for I, J in sorted(itertools.product(supports_r, supports_c),
                       key=lambda pair: (len(pair[0]) + len(pair[1]), pair)):
        # [y_J, u], A y <= u, equality for every supported row.
        eq_y = np.vstack([np.r_[np.ones(len(J)), 0.],
                          np.c_[A[np.ix_(I, J)], -np.ones(len(I))]])
        sol_y = linprog(np.zeros(len(J) + 1),
                        A_ub=np.c_[A[:, J], -np.ones(nr)], b_ub=np.zeros(nr),
                        A_eq=eq_y, b_eq=np.r_[1., np.zeros(len(I))],
                        bounds=[(0., None)] * len(J) + [(None, None)], method="highs")
        if not sol_y.success:
            continue
        eq_x = np.vstack([np.r_[np.ones(len(I)), 0.],
                          np.c_[B[np.ix_(I, J)].T, -np.ones(len(J))]])
        sol_x = linprog(np.zeros(len(I) + 1),
                        A_ub=np.c_[B[I, :].T, -np.ones(nc)], b_ub=np.zeros(nc),
                        A_eq=eq_x, b_eq=np.r_[1., np.zeros(len(J))],
                        bounds=[(0., None)] * len(I) + [(None, None)], method="highs")
        if not sol_x.success:
            continue
        x, y = np.zeros(nr), np.zeros(nc)
        x[list(I)], y[list(J)] = sol_x.x[:-1], sol_y.x[:-1]
        error = regret(A, B, x, y)
        if error <= tolerance:
            return dict(x=x, y=y, value=np.array([x @ A @ y, x @ B @ y]),
                        kind="mixed", pure_count=0, regret=error)
    raise RuntimeError("No verified equilibrium found in exhaustive support search")


def matrices(state, par):
    act = [actions(len(s), par.capacity) for s in state]
    A, B = np.zeros((len(act[0]), len(act[1]))), np.zeros((len(act[0]), len(act[1])))
    for i, a in enumerate(act[0]):
        for j, b in enumerate(act[1]):
            pay, _ = stage(state, (a, b), par)
            A[i, j], B[i, j] = pay
    return act, A, B


def expected_action(act, weights):
    return dict(production=float(sum(p * sum(a[1]) for p, a in zip(weights, act))),
                opened=float(sum(p * a[0] for p, a in zip(weights, act))),
                at_capacity_probability=float(sum(p for p, a in zip(weights, act)
                                                  if sum(a[1]) == max(sum(b[1]) for b in act))),
                support=[dict(probability=float(p), opened=a[0], production=list(a[1]))
                         for p, a in zip(weights, act) if p > 1e-9])


def solve_two_period(par, state):
    act, myopic_A, myopic_B = matrices(state, par)
    A, B = myopic_A.copy(), myopic_B.copy()
    continuations = []
    for i, a in enumerate(act[0]):
        for j, b in enumerate(act[1]):
            _, following = stage(state, (a, b), par)
            next_act, next_A, next_B = matrices(following, par)
            solved = equilibrium(next_A, next_B)
            A[i, j] += par.discount * solved["value"][0]
            B[i, j] += par.discount * solved["value"][1]
            continuations.append(dict(first_indices=[i, j], state=following,
                                      kind=solved["kind"], pure_count=solved["pure_count"],
                                      regret=solved["regret"], value=solved["value"].tolist(),
                                      second_actions=[expected_action(aa, ww) for aa, ww in
                                                      zip(next_act, [solved["x"], solved["y"]])]))
    first = equilibrium(A, B)
    myopic = equilibrium(myopic_A, myopic_B)
    result = dict(parameters=par.__dict__, initial_state=state,
                  first_kind=first["kind"], first_pure_count=first["pure_count"],
                  first_regret=first["regret"], value=first["value"].tolist(),
                  first_actions=[expected_action(aa, ww) for aa, ww in
                                 zip(act, [first["x"], first["y"]])],
                  myopic_actions=[expected_action(aa, ww) for aa, ww in
                                  zip(act, [myopic["x"], myopic["y"]])],
                  terminal_subgames=len(continuations),
                  terminal_mixed_subgames=sum(s["kind"] == "mixed" for s in continuations),
                  maximum_regret=max([first["regret"]] + [s["regret"] for s in continuations]))
    second = [dict(production=0., opened=0.) for _ in range(2)]
    for continuation in continuations:
        i, j = continuation["first_indices"]
        weight = first["x"][i] * first["y"][j]
        for player in range(2):
            for key in second[player]:
                second[player][key] += float(weight * continuation["second_actions"][player][key])
    result["second_actions"] = second
    return result, dict(payoff_A=A.tolist(), payoff_B=B.tolist(), continuation=continuations)


def optimal_score(n, m, stock, weight, alpha, new_weight=1.):
    if not m:
        return weight * (stock + n) ** alpha
    eta = (new_weight / weight) ** (1 / (1 - alpha))
    old_effective = np.clip((stock + n) / (1 + m * eta), stock, stock + n - m)
    new_per_account = (stock + n - old_effective) / m
    return weight * old_effective ** alpha + new_weight * m * new_per_account ** alpha


def gross_value(m, stock, weight, alpha, cost, rival, prize):
    # n=m is feasible and gives utility >-c*m; n>m+R/c is dominated.
    lower, upper = float(m), float(m) + prize / cost
    def objective(n):
        score = optimal_score(n, m, stock, weight, alpha)
        return prize * score / (score + rival) - cost * n
    candidate = minimize_scalar(lambda n: -objective(n), bounds=(lower, upper), method="bounded")
    options = [(objective(n), n) for n in [lower, upper, candidate.x]]
    return max(options)


def threshold_audit():
    records = []
    for stock, capital, alpha, cost, rival in itertools.product(
            [.25, 2., 8.], [0., 2.], [.5, .8], [.5, 1.], [1., 4.]):
        weight = (1 + capital) ** .5
        base, n0 = gross_value(0, stock, weight, alpha, cost, rival, 4.)
        differences = []
        for m in range(1, int(4 / cost) + 2):
            value, n = gross_value(m, stock, weight, alpha, cost, rival, 4.)
            differences.append((value - base) / m)
        records.append(dict(stock=stock, capital=capital, alpha=alpha, cost=cost,
                            rival=rival, no_split_optimal_production=n0,
                            threshold=max(0., differences[0]),
                            threshold_all_m=max(0., max(differences)),
                            maximum_threshold_gap=max(differences) - differences[0],
                            average_gains_by_extra_accounts=differences))
    return dict(conditions=len(records), records=records,
                max_two_account_threshold_gap=max(r["maximum_threshold_gap"] for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    start = time.monotonic()
    records, details = [], []
    initial_state = tuple(tuple(tuple(a) for a in creator) for creator in protocol["initial_state"])
    for index, cell in enumerate(protocol["base_grid"]):
        par = Parameters(**cell)
        for regime, costs in [("baseline", (1., 1.)), ("both_cost_half", (.5, .5)),
                              ("only_creator1_cost_half", (.5, 1.))]:
            solved, detail = solve_two_period(replace(par, costs=costs), initial_state)
            solved.update(cell=index, regime=regime)
            records.append(solved)
            details.append(detail)
            point = dict(completed=len(records), planned=len(protocol["base_grid"]) * 3,
                         elapsed_seconds=round(time.monotonic()-start, 3),
                         regime=regime, first_kind=solved["first_kind"], regret=solved["maximum_regret"])
            print(json.dumps(point), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(point) + "\n")
    changes = []
    for cell in range(len(protocol["base_grid"])):
        group = [r for r in records if r["cell"] == cell]
        base = group[0]
        for other in group[1:]:
            changes.append(dict(cell=cell, regime=other["regime"],
                                production_change=[other["first_actions"][i]["production"] -
                                                   base["first_actions"][i]["production"] for i in range(2)],
                                opening_change=[other["first_actions"][i]["opened"] -
                                                base["first_actions"][i]["opened"] for i in range(2)]))
    output = dict(utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=dict(conditions=len(records),
                               maximum_regret=max(r["maximum_regret"] for r in records),
                               terminal_subgames=sum(r["terminal_subgames"] for r in records),
                               terminal_mixed_subgames=sum(r["terminal_mixed_subgames"] for r in records),
                               first_mixed_conditions=sum(r["first_kind"] == "mixed" for r in records),
                               foresight_changes_first_action=sum(
                                   any(abs(a[k] - b[k]) > 1e-7 for a, b in zip(r["first_actions"], r["myopic_actions"])
                                       for k in ["production", "opened"]) for r in records)),
                  results=records, cost_comparisons=changes, threshold_audit=threshold_audit())
    (args.output / "dynamic_game_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    (args.output / "payoff_matrices_and_subgames.json").write_text(json.dumps(details) + "\n")
    print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
