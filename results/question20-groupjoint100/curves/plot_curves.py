"""Plot only small aggregate metrics; never read model or trajectory files."""
import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "curve-metrics.json").read_text())
rows = data["rows"]
x = np.array([r["epoch"] for r in rows])
assert list(x) == list(range(1, data["complete_epoch"] + 1))
for row in rows:
    assert row["actual_adam_steps_per_rank"] == [100 * row["epoch"]] * 8
    assert row["source_data_epoch"] == (row["epoch"] - 1) % 70 + 1
    assert (row["test"]["mixed_count"], row["test"]["nonmixed_count"]) == (183, 329)
    for split in ("train", "test"):
        m = row[split]
        assert abs(m["balanced_accuracy"] - (m["mixed_recall"] + m["nonmixed_recall"]) / 2) < 1e-10
        assert abs(m["total_loss"] - m["trajectory_bce"] - m["group_bce"]) < 1e-8

font_path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
font_manager.fontManager.addfont(font_path)
font = font_manager.FontProperties(fname=font_path)
plt.rcParams.update({
    "font.family": font.get_name(), "font.size": 16, "axes.titlesize": 19,
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8b939e", "axes.linewidth": .7,
    "grid.color": "#dce1e7", "grid.linewidth": .6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
TRAIN, TEST = "#2469a2", "#d36b24"
MIXED, NONMIXED = TEST, "#7955a0"

def average(y):
    # Full trailing window only: the first smooth point is epoch 10.
    if len(y) < 10:
        return np.full(len(y), np.nan)
    return np.r_[np.full(9, np.nan), np.convolve(y, np.ones(10) / 10, mode="valid")]

def line(ax, y, color, label):
    y = np.asarray(y)
    ax.plot(x, y, color=color, alpha=.26, lw=1.0)
    ax.plot(x, average(y), color=color, lw=2.5, label=label)

def style(ax, title, ylabel, baseline=None):
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("完整训练轮次 / Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(.5, x[-1] + .5)
    ax.grid(axis="y")
    if baseline is not None:
        ax.axhline(baseline[0], color="#6f7680", ls="--", lw=1.0, label=baseline[1])
    ax.legend(loc="best", frameon=False)
    ax.margins(y=.12)

def paired(ax, key, title, percent=False, baseline=None):
    scale = 100 if percent else 1
    for split, color, label in [("train", TRAIN, "训练 · 10轮均值"), ("test", TEST, "固定测试 · 10轮均值")]:
        line(ax, [scale * r[split][key] for r in rows], color, label)
    style(ax, title, "准确率（%）" if percent else "平均损失（自然对数）", baseline)

stamp = data["snapshot_utc"][:19].replace("T", " ")
final = int(x[-1]) == int(data["target_epochs"])
status = "最终完整" if final else "完整"
snapshot_time = datetime.datetime.fromisoformat(data["snapshot_utc"])
pdf_metadata = {"Creator": "Search-G1 aggregate metrics", "CreationDate": snapshot_time,
                "ModDate": snapshot_time, "Subject": "Source SHA256 " + data["source_sha256"]}
png_metadata = {"Source SHA256": data["source_sha256"], "Completed epochs": str(int(x[-1]))}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
paired(axes[0], "balanced_accuracy", "混合判别：平衡准确率", True, (50, "常数分类基线：50%"))
paired(axes[1], "accuracy", "混合判别：普通准确率", True, (329 / 512 * 100, "测试多数类基线：64.26%"))
fig.suptitle(f"Search-G1 联合探针 · {status} {x[-1]}/100 轮", y=.98, fontsize=21, weight="medium")
fig.subplots_adjust(left=.075, right=.985, bottom=.22, top=.80, wspace=.25)
fig.text(.5, .035, "淡线：逐轮原始值；粗线：完整10轮尾随均值。训练按批更换5120题G5（E1–70，再E1–30）；测试固定512题G5。", ha="center", fontsize=12)
fig.savefig(HERE / "accuracy-curves.png", dpi=160, metadata=png_metadata)
fig.savefig(HERE / "accuracy-curves.pdf", metadata=pdf_metadata)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
paired(axes[0, 0], "trajectory_bce", "逐轨迹正确性 BCE")
paired(axes[0, 1], "group_bce", "每题 G5 混合 BCE")
paired(axes[1, 0], "total_loss", "联合总损失：轨迹 BCE + 混合 BCE")
for key, color, label in [("mixed_recall", MIXED, "混合类召回 · 10轮均值"), ("nonmixed_recall", NONMIXED, "非混合类召回 · 10轮均值")]:
    line(axes[1, 1], [r["test"][key] * 100 for r in rows], color, label)
style(axes[1, 1], "固定测试：两类召回", "召回率（%）")
fig.suptitle(f"Search-G1 联合探针 · 损失与类别偏向 · {status} E1–{x[-1]}", y=.985, fontsize=21, weight="medium")
fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.90, wspace=.25, hspace=.40)
fig.text(.5, .025, "训练按批更换5120题G5，测试固定512题G5；均为同轮更新后完整复测。\n阈值0.5，混合真值为G5中答对1–4条；淡线原始、粗线完整10轮尾随均值。\n平滑不外推。数据快照UTC " + stamp, ha="center", fontsize=12)
fig.savefig(HERE / "loss-recall-curves.png", dpi=160, metadata=png_metadata)
fig.savefig(HERE / "loss-recall-curves.pdf", metadata=pdf_metadata)
plt.close(fig)
print(json.dumps({"complete_epoch": int(x[-1]), "artifacts": ["accuracy-curves.png", "loss-recall-curves.png", "accuracy-curves.pdf", "loss-recall-curves.pdf"]}))
