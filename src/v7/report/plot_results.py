"""Publication figures from aggregated V7 receipts; no raw observations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[3]
audit = root / "docs/V7_AUDIT"
out = Path(__file__).with_name("figures")
out.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"Hiragino Sans GB", "font.size":11,
    "axes.unicode_minus":False,"pdf.fonttype":42,"svg.fonttype":"path",
    "axes.spines.top":False,"axes.spines.right":False,
    "text.color":"#123047","axes.labelcolor":"#123047",
    "xtick.color":"#657783","ytick.color":"#123047",
    "axes.edgecolor":"#bac7cd","figure.facecolor":"white"})
navy,teal,gold,red = "#123047","#087f8c","#d9a441","#bc5a45"
def save(fig,name):
    fig.savefig(out/(name+".pdf"),bbox_inches="tight")
    fig.savefig(out/(name+".svg"),bbox_inches="tight")
    fig.savefig(out/(name+".png"),bbox_inches="tight",dpi=180)
    plt.close(fig)

# An exact symmetric, fixed-single-account slice, not an estimated response.
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
b=np.linspace(0,4,200)
for c,color in [(1,navy),(.5,teal)]:
    axes[0].plot(b,np.maximum(0,1.5/c-b),label=f"单位成本 c={c:g}",color=color,lw=2.3)
axes[0].set(xlabel="已有有效作品存量 B",ylabel="均衡新增生产 n*",title="A  旧存量挤出新增生产")
axes[0].legend(frameon=False,fontsize=9)
axes[0].grid(axis="y",alpha=.15)
axes[1].bar([0,1],[.5,11/6],width=.55,color=[gold,teal])
for i,v in enumerate([.5,11/6]):axes[1].text(i,v+.06,f"{v:.3f}",ha="center")
axes[1].set(xticks=[0,1],xticklabels=["只看当期","两期前瞻"],ylabel="第一期生产",title="B  严格可解两期示例")
axes[1].set_ylim(0,2.2)
save(fig,"fig01_stock_and_foresight")

x=json.loads((audit/"dynamic_game_summary.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
labels=["双方降本：作者1","双方降本：作者2","单方降本：作者1","单方降本：作者2"]
for ax,key,title in zip(axes,["production_change","opening_change"],["第一期生产变化","第一期开号变化"]):
    counts=[]
    for regime in ["both_cost_half","only_creator1_cost_half"]:
        group=[r for r in x["cost_comparisons"] if r["regime"]==regime]
        for i in [0,1]:counts.append([sum(r[key][i]>1e-7 for r in group),sum(abs(r[key][i])<=1e-7 for r in group),sum(r[key][i]<-1e-7 for r in group)])
    bottom=np.zeros(4)
    for k,label,color in zip(range(3),["增加","不变","减少"],[teal,"#dce5e7",red]):
        v=np.array([c[k] for c in counts]);ax.barh(range(4),v,left=bottom,color=color,label=label,height=.6)
        for i,n in enumerate(v):
            if n:ax.text(bottom[i]+n/2,i,str(n),ha="center",va="center",color="white" if k!=1 else navy)
        bottom+=v
    ax.set(yticks=range(4),yticklabels=labels,xticks=[0,4,8,12,16],xlabel="16 组制度条件中的数量",title=title)
    ax.invert_yaxis()
axes[0].legend(frameon=False,ncol=3,loc="lower left",bbox_to_anchor=(0,1.06),fontsize=9)
save(fig,"fig02_dynamic_costs")

e=json.loads((audit/"empirical_summary.json").read_text())
pooled={(r["model"],r["subset"]):r for r in e["pooled_models"]}
names=["no_state","account_only","work_only","dual_state"]
fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained",gridspec_kw={"width_ratios":[1,1.15]})
vals=[pooled[(n,"all")]["video_joint_log_loss"] for n in names]
axes[0].scatter(vals,range(4),s=65,c=["#a6b5bc",gold,navy,teal])
for i,v in enumerate(vals):axes[0].text(v+.017,i,f"{v:.4f}",va="center",fontsize=10)
axes[0].set(yticks=range(4),yticklabels=["无状态","仅账号信息","仅作品状态","双状态"],xlabel="留出期交叉熵（越低越好）",title="A  同一候选池的预测比较",xlim=(6.54,7.55))
axes[0].invert_yaxis()
axes[0].grid(axis="x",alpha=.15)
comp=next(r for r in e["comparisons"] if r["model"]=="dual_state" and r["reference"]=="work_only" and r["subset"]=="all" and r["evaluation"]=="joint")
delta=comp["delta"]
for y,key,label,color in [(0,"paired_date_bootstrap_95","按整日重采样",teal),(1,"origin_block_bootstrap_95","按四起点区块",gold)]:
    lo,hi=comp[key]
    axes[1].errorbar(delta,y,xerr=[[delta-lo],[hi-delta]],fmt="o",color=color,capsize=5,lw=2)
axes[1].axvline(0,ls="--",color=red)
axes[1].set(yticks=[0,1],yticklabels=["整日区间","起点区间"],xlabel="双状态 - 仅作品：损失差",title="B  账号信息的额外改善跨零",ylim=(-.6,1.6))
save(fig,"fig03_state_prediction")

fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout="constrained")
new=[pooled[(n,"new")]["video_conditional_log_loss"] for n in names]
axes[0].scatter(new,range(4),s=60,c=["#a6b5bc",gold,navy,teal])
for i,v in enumerate(new):axes[0].text(v+.002,i,f"{v:.4f}",va="center",fontsize=10)
axes[0].set(yticks=range(4),yticklabels=["无状态","仅账号信息","仅作品状态","双状态"],xlabel="新作品子池内条件交叉熵",title="A  新作仍可利用共享信息",xlim=(3.94,4.085))
axes[0].invert_yaxis()
for level,color in [("item",teal),("account",navy)]:
    series=[]
    for model in ["carry_forward","ar1","ar1_plus_observed_states_supply"]:
        rows=[next(z for z in t[level]["models"] if z["model"]==model) for t in e["observed_one_step_transition_predictions"]]
        series.append(sum(r["mse_log1p_exposure"]*r["test_pairs"] for r in rows)/sum(r["test_pairs"] for r in rows))
    axes[1].plot(range(3),series,"o-",color=color,label="作品" if level=="item" else "账号")
axes[1].set(xticks=range(3),xticklabels=["沿用前日","AR(1)","AR(1)+状态"],ylabel="log(1+曝光) 的 MSE",title="B  复杂一步预测未胜过前日值")
axes[1].legend(frameon=False)
save(fig,"fig04_new_and_transition")
print("Four V7 figures written",flush=True)
