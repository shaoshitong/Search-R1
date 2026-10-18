"""Descriptive interpolation of ordinary-round selection rates; no trainer access.

Run from the repository root with system Python (numpy/scipy/matplotlib).
Original JSON is read-only. Periodic oracle-selected observations are excluded.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit, logit

OUT = Path("reproduction/reports/h100-prefix5-ab200-20261018/ordinary-trend-fit")
SOURCE = OUT.parent / "selected-mixing-per-iteration-through042.json"


def predict(x, y, query, family, param):
    """Refit every fold; bounded predictions via logit link, except interpolation."""
    if family == "linear_interpolation":
        return np.interp(query, x, y)
    t, u = (x - 21.5) / 20.5, (query - 21.5) / 20.5
    z = logit(y)
    if family == "ridge_polynomial":
        degree, penalty = param
        X = np.polynomial.legendre.legvander(t, degree)
        Q = np.polynomial.legendre.legvander(u, degree)
        regularizer = np.eye(degree + 1) * penalty
        regularizer[0, 0] = 0
        return expit(Q @ np.linalg.solve(X.T @ X + regularizer, X.T @ z))
    if family == "smoothing_spline":
        return expit(make_smoothing_spline(t, z, lam=param)(u))
    if family == "gaussian_process":
        length, noise = param
        mean = np.mean(z)
        kernel = np.exp(-0.5 * ((t[:, None] - t[None, :]) / length) ** 2)
        kernel += (noise**2 + 1e-9) * np.eye(len(t))
        cross = np.exp(-0.5 * ((u[:, None] - t[None, :]) / length) ** 2)
        alpha = cho_solve(cho_factor(kernel, lower=True), z - mean)
        return expit(mean + cross @ alpha)
    raise ValueError(family)


def main():
    original = SOURCE.read_bytes()
    data = json.loads(original)
    rows = data["rows"]
    excluded = [r["outer"] for r in rows if r["periodic_actual_em_priority"]]
    assert excluded == [10, 20, 30, 40]
    assert all(r[arm]["mixed_fraction"] == 1 for r in rows
               if r["outer"] in excluded for arm in ("A", "B"))
    ordinary = [r for r in rows if r["outer"] not in excluded]
    x = np.array([r["outer"] for r in ordinary], dtype=float)
    missing = np.array(excluded, dtype=float)
    grid = np.linspace(1, 42, 411)
    candidates = {
        "linear_interpolation": [None],
        "ridge_polynomial": [(d, p) for d in range(1, 7)
                             for p in (0.001, 0.01, 0.1, 1.0, 10.0)],
        "smoothing_spline": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "gaussian_process": [(l, n) for l in (0.08, 0.15, 0.3, 0.6, 1.2)
                             for n in (0.03, 0.08, 0.15, 0.3)],
    }
    result = {
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "source_file": str(SOURCE),
        "excluded_periodic_oracle_rounds": excluded,
        "fit_observations_per_arm": len(x),
        "metric": "true mixed fraction of selected 512 question groups, not probe classification accuracy",
        "purpose": "Descriptive ordinary-round trend only; imputed periodic positions are estimates, not counterfactual evidence or measured selections.",
        "validation": "Nested refitting per leave-one-out fold over 36 interior ordinary observations; hyperparameters/families selected using same CV so errors are model-selection diagnostics, not independent performance estimates. Endpoints excluded to assess interpolation rather than extrapolation.",
        "bounds": "Logit response and sigmoid predictions for learned models; interpolation within measured probabilities. No extrapolation beyond round 42.",
        "arms": {},
    }
    for arm in ("A", "B"):
        y = np.array([r[arm]["mixed_fraction"] for r in ordinary])
        assert np.all((y > 0) & (y < 1))
        families = {}
        for family, params in candidates.items():
            trials = []
            for param in params:
                errors = []
                for i in range(1, len(x) - 1):
                    keep = np.arange(len(x)) != i
                    estimate = predict(x[keep], y[keep], x[i:i+1], family, param)[0]
                    errors.append(float(estimate - y[i]))
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                trials.append((rmse, param, float(np.mean(np.abs(errors)))))
            rmse, param, mae = min(trials, key=lambda v: v[0])
            imputed = predict(x, y, missing, family, param)
            curve = predict(x, y, grid, family, param)
            families[family] = {
                "hyperparameters": param, "cv_rmse": rmse, "cv_mae": mae,
                "imputed": [{"outer": int(i), "estimate": float(v), "measured": False}
                            for i, v in zip(missing, imputed)],
                "curve": [{"outer": float(i), "estimate": float(v)} for i, v in zip(grid, curve)],
            }
        selected = min(families, key=lambda f: families[f]["cv_rmse"])
        missing_map = {r["outer"]: r["estimate"] for r in families[selected]["imputed"]}
        filled = [{"outer": r["outer"], "value": missing_map.get(r["outer"], r[arm]["mixed_fraction"]),
                   "kind": "estimated" if r["outer"] in missing_map else "measured",
                   "original_value": r[arm]["mixed_fraction"]} for r in rows]
        result["arms"][arm] = {"selected_method": selected, "methods": families,
                               "filled_rows": filled}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fit-results.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    colors = {"A": "#2065b1", "B": "#de6c29"}
    for arm in ("A", "B"):
        a = result["arms"][arm]
        method = a["methods"][a["selected_method"]]
        ax.scatter(x, [100*r[arm]["mixed_fraction"] for r in ordinary],
                   s=22, color=colors[arm], alpha=.7, label=f"{arm}: observed ordinary rounds")
        ax.plot(grid, [100*r["estimate"] for r in method["curve"]], color=colors[arm],
                lw=2, label=f"{arm}: {a['selected_method'].replace('_', ' ')}")
        ax.scatter(missing, [100*r["estimate"] for r in method["imputed"]], marker="D", s=78,
                   facecolors="white", edgecolors=colors[arm], linewidths=2,
                   zorder=5, label=f"{arm}: estimated missing positions")
    for i in excluded:
        ax.axvline(i, color="#c4c9d0", lw=.8, linestyle=":", zorder=0)
    ax.set(xlabel="Outer iteration", ylabel="True mixed groups among selected 512 (%)",
           title="A/B ordinary-round trend: periodic 100% observations excluded",
           xlim=(.4, 42.6), ylim=(15, 90))
    ax.grid(axis="y", alpha=.18)
    ax.legend(ncol=2, loc="upper right", fontsize=9)
    fig.supxlabel("Diamonds are fitted estimates, not measurements. Raw observations remain unchanged.", fontsize=9)
    fig.savefig(OUT / "ordinary-trend.png", dpi=180)
    fig.savefig(OUT / "ordinary-trend.svg")
    plt.close(fig)

    lines = ["# A/B普通轮趋势拟合（截至第42轮）", "",
             "原始数据不变。排除第10/20/30/40轮实际EM优先选题产生的100%后，各组使用38个普通轮实测值。下列四点是普通轮趋势估计，不是实测值，也不能证明周期轮改用普通选题会得到这些数值。", "",
             "比较高斯过程、岭多项式、平滑样条和线性插值。对36个内部实测点逐点留一、每折重新拟合，按概率尺度RMSE选择参数和方法。该误差用于方法选择，不是独立泛化评测，也不是探针分类准确率。未外推第43轮以后。", "",
             "| 方法 | A留一RMSE（百分点） | B留一RMSE（百分点） |", "|---|---:|---:|"]
    for family in candidates:
        lines.append(f"| {family} | {100*result['arms']['A']['methods'][family]['cv_rmse']:.3f} | {100*result['arms']['B']['methods'][family]['cv_rmse']:.3f} |")
    lines += ["", "| 组 | 所选方法 | 参数 |", "|---|---|---|"]
    for arm, a in result["arms"].items():
        lines.append(f"| {arm} | {a['selected_method']} | {a['methods'][a['selected_method']]['hyperparameters']} |")
    lines += ["", "| 迭代 | A估计 | B估计 |", "|---|---:|---:|"]
    for j, i in enumerate(excluded):
        vals = [result["arms"][a]["methods"][result["arms"][a]["selected_method"]]["imputed"][j]["estimate"] for a in ("A", "B")]
        lines.append(f"| {i} | {100*vals[0]:.2f}% | {100*vals[1]:.2f}% |")
    lines += ["", "![拟合曲线](ordinary-trend.png)", "", "## 完整拟合版数据", "",
              "普通轮保持实测值，只替换四个周期位置；‘估计’不代表实际混合题数。", "",
              "| 迭代 | A | B | 类型 |", "|---|---:|---:|---|"]
    for a, b in zip(result["arms"]["A"]["filled_rows"], result["arms"]["B"]["filled_rows"]):
        lines.append(f"| {a['outer']} | {100*a['value']:.2f}% | {100*b['value']:.2f}% | {'估计' if a['kind']=='estimated' else '实测'} |")
    lines += ["", f"源数据SHA256：`{result['source_sha256']}`。", "",
              "复现：从仓库根目录运行 `python3 -m reproduction.operations.h100_reference_replica.fit_selected_mixing`。模型使用logit链接约束输出在0与1之间，未裁剪概率。所有拟合均为本地小指标分析，不访问训练或改动训练。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    with (OUT / "selected-mixing-through042-filled.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "A_mixed_percent", "B_mixed_percent", "is_imputed"])
        for a, b in zip(result["arms"]["A"]["filled_rows"], result["arms"]["B"]["filled_rows"]):
            writer.writerow([a["outer"], f"{100*a['value']:.6f}",
                             f"{100*b['value']:.6f}", int(a["kind"] == "estimated")])
    assert SOURCE.read_bytes() == original
    for a in result["arms"].values():
        assert sum(r["kind"] == "estimated" for r in a["filled_rows"]) == 4
        assert all(0 <= r["value"] <= 1 for r in a["filled_rows"])
    print(json.dumps({arm: {"method": a["selected_method"],
                            "rmse_pp": 100*a["methods"][a["selected_method"]]["cv_rmse"],
                            "imputed": a["methods"][a["selected_method"]]["imputed"]}
                      for arm, a in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
