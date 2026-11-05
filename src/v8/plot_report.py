"""Readable experiment figures; all numbers come from retained V7 aggregates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[2]
audit=root/"docs/V7_AUDIT"
out=Path(__file__).with_name("report")/"figures"
out.mkdir(parents=True,exist_ok=True)
e=json.loads((audit/"empirical_summary.json").read_text())
d=json.loads((audit/"empirical_new_work_group_diagnostic.json").read_text())
plt.rcParams.update({"font.family":"Hiragino Sans GB","font.size":11,
 "axes.unicode_minus":False,"svg.fonttype":"path","pdf.fonttype":42,
 "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white",
 "axes.edgecolor":"#b7c6cb","text.color":"#123047","axes.labelcolor":"#123047"})
navy,teal,gold,grey,red="#123047","#087f8c","#d9a441","#94a8b0","#bc5a45"
def save(fig,name):
 for suffix in ["svg","pdf","png"]:fig.savefig(out/(name+"."+suffix),bbox_inches="tight",dpi=180)
 plt.close(fig)
models=["no_state","account_only","work_only","dual_state"]
labels=["不使用历史","只用账号共享历史","只用作品自身历史","两类历史一起使用"]
lookup={(r["model"],r["subset"]):r for r in e["pooled_models"]}
fig,axes=plt.subplots(1,2,figsize=(10.3,3.9),layout="constrained",gridspec_kw={"width_ratios":[1.25,1]})
values=[lookup[(m,"all")]["video_joint_log_loss"] for m in models]
axes[0].scatter(values,range(4),s=70,c=[grey,gold,navy,teal])
for i,v in enumerate(values):axes[0].text(v+.018,i,f"{v:.4f}",va="center",fontsize=10)
axes[0].set(yticks=range(4),yticklabels=labels,xlim=(6.52,7.56),xlabel="同池预测损失（越低越好）",title="全体作品：自身历史贡献最大")
axes[0].invert_yaxis();axes[0].grid(axis="x",alpha=.15)
row=next(r for r in e["comparisons"] if r["model"]=="dual_state" and r["reference"]=="work_only" and r["subset"]=="all" and r["evaluation"]=="joint")
for y,key,label,col in [(0,"paired_date_bootstrap_95","按整日重采样",teal),(1,"origin_block_bootstrap_95","按测试窗重采样",gold)]:
 v=row["delta"];lo,hi=row[key]
 axes[1].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt="o",color=col,capsize=5,lw=2)
axes[1].axvline(0,color=red,ls="--",lw=1.5)
axes[1].set(yticks=[0,1],yticklabels=["整日区间","窗口区间"],ylim=(-.5,1.5),xlabel="加入账号信息后的损失差",title="额外账号信息：区间跨过零")
save(fig,"experiment_all_works")

fig,axes=plt.subplots(1,2,figsize=(10.3,3.9),layout="constrained",gridspec_kw={"width_ratios":[1.25,1]})
values=[lookup[(m,"new")]["video_conditional_log_loss"] for m in models]
axes[0].scatter(values,range(4),s=70,c=[grey,gold,navy,teal])
for i,v in enumerate(values):axes[0].text(v+.003,i,f"{v:.4f}",va="center",fontsize=10)
axes[0].set(yticks=range(4),yticklabels=labels,xlim=(3.935,4.105),xlabel="新作子池预测损失（越低越好）",title="新作品：共享信息提供帮助")
axes[0].invert_yaxis();axes[0].grid(axis="x",alpha=.15)
main=next(r for r in e["comparisons"] if r["model"]=="account_only" and r["reference"]=="no_state" and r["subset"]=="new" and r["evaluation"]=="conditional")
sub=next(r for r in d["date_intervals"] if r["group"]=="available" and r["metric"]=="within_nll")
sub_value=next(r for r in d["groups"] if r["model"]=="account_only" and r["group"]=="available")["within_group_conditional_log_loss"]-next(r for r in d["groups"] if r["model"]=="no_state" and r["group"]=="available")["within_group_conditional_log_loss"]
# This file's schema is fixed, and every plotted interval is an actual bootstrap.
ci=next(v for k,v in sub.items() if "95" in k and isinstance(v,list))
for y,value,interval,col in [(0,main["delta"],main["paired_date_bootstrap_95"],teal),(1,sub_value,ci,gold)]:
 lo,hi=interval;axes[1].errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt="o",color=col,capsize=5,lw=2)
axes[1].scatter([0],[2],c=grey,s=55)
axes[1].axvline(0,color=red,ls="--",lw=1.5)
axes[1].set(yticks=[0,1,2],yticklabels=["全部新作","有源历史组内*","无源历史组内*"],ylim=(-.5,2.5),xlabel="只用账号历史 - 不使用历史",title="改善也发生在组内")
axes[1].invert_yaxis()
save(fig,"experiment_new_works")
print("V8 experiment figures complete")
