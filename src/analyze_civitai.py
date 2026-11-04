#!/usr/bin/env python3
"""Describe a public Civitai snapshot and replay hypothetical allocation rules.

No exposure, quality, ownership, or causal effect is inferred from these data.
Run on the remote machine holding the metadata; export aggregates and figures.
"""
import argparse
import ast
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_prompt(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def stats_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def concentration(probabilities):
    p = np.sort(np.asarray(probabilities, dtype=float))
    m = len(p)
    return {
        "top_1pct_share": float(p[-max(1, math.ceil(m * .01)):].sum()),
        "top_10pct_share": float(p[-max(1, math.ceil(m * .1)):].sum()),
        "hhi": float(p @ p),
        "effective_accounts": float(1 / (p @ p)),
        "gini": float((2 * np.arange(1, m + 1) - m - 1) @ p / m),
    }


def constrained_equilibrium(alpha, c, tau, reward=1.0):
    a = reward / 4
    k = max(1.0, min(a * (1 - alpha) / tau, a / (c + tau)))
    return max(k, a * alpha / c), k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.bbox": "tight"})

    print("stage=load", flush=True)
    raw = pd.read_parquet(args.input, columns=["id", "username", "createdAt", "prompt", "stats", "nsfw"])
    frame = raw.drop_duplicates("id", keep="first").copy()
    frame["createdAt"] = pd.to_datetime(frame["createdAt"], utc=True, errors="coerce")
    author_ok = frame.username.notna() & frame.username.fillna("").str.strip().ne("")
    sfw = frame.loc[frame.nsfw.eq(False) & author_ok].copy()
    sfw["prompt_norm"] = sfw.prompt.map(normalize_prompt)
    parsed = sfw.stats.map(stats_object)
    sfw["reactions"] = parsed.map(lambda x: sum(x.get(key, 0) or 0 for key in
                                            ("cryCount", "laughCount", "likeCount", "heartCount")))
    counts = sfw.groupby("username").size().sort_values(ascending=False)
    valid = sfw.loc[sfw.prompt_norm.ne("")]
    within_groups = valid.groupby(["username", "prompt_norm"]).size()
    redundant_within = int((within_groups - 1).sum())
    prompt_authors = valid.groupby("prompt_norm").username.nunique()
    summary = {
        "data_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "unique_image_rows": len(frame),
        "duplicate_id_rows_removed": len(raw) - len(frame),
        "date_min": str(frame.createdAt.min()), "date_max": str(frame.createdAt.max()),
        "sfw_with_author_rows": len(sfw), "sfw_accounts": len(counts),
        "sfw_prompt_rows": len(valid), "sfw_missing_prompt_share": float(1 - len(valid) / len(sfw)),
        "account_record_count_quantiles": {str(q): float(counts.quantile(q)) for q in (.25, .5, .75, .9, .95, .99, 1)},
        "redundant_prompt_rows_global": len(valid) - valid.prompt_norm.nunique(),
        "redundant_prompt_rows_within_account": redundant_within,
        "within_account_prompt_redundancy_share": redundant_within / len(valid),
        "prompt_groups_seen_across_accounts": int(prompt_authors.gt(1).sum()),
        "positive_reaction_share": float(sfw.reactions.gt(0).mean()),
        "reaction_quantiles": {str(q): float(sfw.reactions.quantile(q)) for q in (.5, .75, .9, .99, 1)},
        "interpretation": "Counts describe this historical snapshot, not complete platform production. Prompt reuse is not an ownership, plagiarism, quality, or AI-use label.",
    }
    print(f"stage=described rows={len(sfw)} accounts={len(counts)}", flush=True)

    n = counts.to_numpy(dtype=float)
    replay = []
    for alpha in np.linspace(.1, 1, 19):
        scores = n ** alpha
        total = scores.sum()
        p = scores / total
        eligible = n >= 2
        # Fixed work, split between two nonempty accounts; retain integer pieces.
        split_score = np.floor(n[eligible] / 2) ** alpha + np.ceil(n[eligible] / 2) ** alpha
        gains = split_score / (total - scores[eligible] + split_score) - p[eligible]
        replay.append({"alpha": float(alpha), **concentration(p),
                       "two_account_eligible_accounts": int(eligible.sum()),
                       "unilateral_split_gain_median": float(np.median(gains)),
                       "unilateral_split_gain_p90": float(np.quantile(gains, .9)),
                       "unilateral_split_gain_max": float(gains.max())})
    summary["replay"] = replay
    summary["replay_interpretation"] = "Allocation of a fixed observed pool; not the real recommender and not a post-policy equilibrium. Split gains are fractions of a normalized reward pool before account costs."

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True)
    sorted_counts = np.sort(n)
    ax[0].step(sorted_counts, 1 - np.arange(len(n)) / len(n), where="post", color="#245a81")
    ax[0].set(xscale="log", yscale="log", xlabel="Records per account in the snapshot",
              ylabel="Fraction of accounts with at least x records", title="A  Observed account activity")
    bin_edges = [0, 1, 4, 9, 24, 99, np.inf]
    labels = ["1", "2-4", "5-9", "10-24", "25-99", "100+"]
    prompt_stats = valid.groupby("username").agg(rows=("id", "size"), unique=("prompt_norm", "nunique"))
    prompt_stats["bin"] = pd.cut(counts.reindex(prompt_stats.index), bin_edges, labels=labels)
    bins = prompt_stats.groupby("bin", observed=False)[["rows", "unique"]].sum()
    rates = 100 * (bins.rows - bins.unique) / bins.rows
    ax[1].bar(labels, rates, color="#c87537")
    ax[1].set(xlabel="Account record-count bin", ylabel="Redundant prompt rows (%)",
              title="B  Within-account exact prompt reuse")
    fig.savefig(args.output / "empirical_snapshot.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True)
    xs = [x["alpha"] for x in replay]
    ax[0].plot(xs, [100 * x["top_10pct_share"] for x in replay], color="#245a81", marker="o", markersize=3)
    ax[0].set(xlabel="Within-account quantity elasticity, alpha",
              ylabel="Share allocated to the top 10% of accounts (%)", title="A  Concentration under fixed-pool replay")
    ax[1].plot(xs, [1e4 * x["unilateral_split_gain_median"] for x in replay], color="#c87537", label="Median")
    ax[1].plot(xs, [1e4 * x["unilateral_split_gain_p90"] for x in replay], color="#245a81", linestyle="--", label="90th percentile")
    ax[1].set(xlabel="Within-account quantity elasticity, alpha", ylabel="Two-account gain (basis points of reward pool)",
              title="B  A unilateral identity-split deviation")
    ax[1].legend(frameon=False)
    fig.savefig(args.output / "mechanism_replay.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.0), constrained_layout=True)
    alphas = np.linspace(.02, 1, 250)
    for cost, color in zip((.18, .08, .03), ("#7a858c", "#245a81", "#c87537")):
        eq = [constrained_equilibrium(a, cost, .04) for a in alphas]
        ax[0].plot(alphas, [e[0] for e in eq], label=f"c/R = {cost}", color=color)
        ax[1].plot(alphas, [e[1] for e in eq], color=color)
    for a in ax:
        a.axvline(.84, color="black", linestyle=":", linewidth=.9)
        a.set_xlabel("Within-account quantity elasticity, alpha")
    ax[0].set(ylabel="Equilibrium content quantity, n", title="A  Quantity with n >= k >= 1")
    ax[1].set(ylabel="Equilibrium account count, k (relaxation)", title="B  Identity and production become complements")
    ax[0].legend(frameon=False)
    fig.savefig(args.output / "constrained_equilibria.png")
    plt.close(fig)
    summary["figure_3_assumptions"] = {"R": 1, "tau": .04, "costs": [.18, .08, .03], "model": "continuous n >= k >= 1"}
    summary["prompt_redundancy_bins"] = {str(label): {"rows": int(bins.loc[label, "rows"]),
                                                    "redundancy_share": float(rates.loc[label] / 100)}
                                          for label in labels}
    (args.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("stage=complete summary=analysis_summary.json figures=3", flush=True)


if __name__ == "__main__":
    main()
