"""Reproduce two exploratory integer-game strictness checks; no human data."""
import argparse
import json
from pathlib import Path

from finite_market import exact_score, solve_condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for cost in (.95, .5):
        condition = solve_condition("account_concave", .5, False, 12., cost, 1.)
        for equilibrium in condition["pure_equilibria_up_to_player_exchange"]:
            players = []
            for i in range(2):
                own, other = equilibrium["actions"][i], equilibrium["actions"][1-i]
                other_score = exact_score(other["n"], other["k"], .5, False)
                alternatives = []
                for n in range(1, condition["maximum_n_from_dominance"] + 1):
                    for k in range(1, n+1):
                        if (n,k) == (own["n"],own["k"]):
                            continue
                        score = exact_score(n,k,.5,False)
                        utility = 12*score/(score+other_score)-cost*n-(k-1)
                        alternatives.append((utility,n,k))
                best, n, k = max(alternatives)
                margin = equilibrium["payoffs"][i] - best
                players.append({"equilibrium_action":own,"best_alternative":{"n":n,"k":k},
                                "strict_payoff_margin":margin,"strict":margin>1e-10})
            results.append({"c":cost,"equilibrium":equilibrium["actions"],"players":players})
    args.output.write_text(json.dumps({"purpose":"Exploratory exact-integer strictness check", "conditions":results},indent=2))
    print(json.dumps(results),flush=True)


if __name__ == "__main__":
    main()
