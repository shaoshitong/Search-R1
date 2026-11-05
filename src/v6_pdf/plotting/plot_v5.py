"""Scientific figures from frozen V5 summaries; no raw-data acquisition.

Claude Code was attempted but returned Not logged in (exit 1); this source was
then written by Codex. See claude_plot_response.json and cc_execution_receipt.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np
import pymupdf

NAVY, TEAL, GOLD, RED, GRAY = "#123047", "#087F8C", "#D9A441", "#BC5A45", "#71808A"
BG, GRID = "#FFFFFF", "#E7E9E9"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[3] / "outputs/V5_AUDIT")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    font = font_manager.findfont(font_manager.FontProperties(family="Hiragino Sans GB"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Hiragino Sans GB", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10, "axes.unicode_minus": False,
        "figure.facecolor": BG, "axes.facecolor": BG, "text.color": NAVY,
        "axes.labelcolor": NAVY, "xtick.color": GRAY, "ytick.color": NAVY,
        "axes.edgecolor": "#B7C0C4", "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.facecolor": BG, "lines.linewidth": 1.7})
    sources, data = {}, {}
    for name in ("kuairand_cross_item_summary.json", "kuairand_diagnostics_and_acceptance.json",
                 "kuairec_new_upload_first_day.json", "weighted_integer_results.json"):
        path = args.data / name
        sources[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        data[name] = json.loads(path.read_text())
    summary = data["kuairand_cross_item_summary.json"]
    diag = data["kuairand_diagnostics_and_acceptance.json"]
    new = data["kuairec_new_upload_first_day.json"]
    theory = data["weighted_integer_results.json"]
    models = {m["model"]: m for m in summary["models"]}
    cases = {r["condition"]["case_id"]: r for r in theory["conditions"]}
    manifest = {"source_files": sources, "font": {"family": "Hiragino Sans GB", "path": font},
                "png_dpi": 300, "generator": "Codex direct matplotlib after actual Claude Code login failure", "figures": []}

    def clean(ax, axis="x"):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis=axis, color=GRID, linewidth=0.65, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)

    def save(fig, name, caption, plotted, source_names):
        stem={"old_work_loss_and_intervals":"fig01_oldwork", "cross_item_gain_concentration":"fig02_concentration",
              "new_work_loss_by_origin":"fig03_newwork", "new_work_seen_mass":"fig04_group_mass",
              "minimum_content_boundary":"fig05_minimum_content", "cost_boundary_counterexample":"fig06_integer_reversal"}[name]
        files = {}
        for ext in ("svg", "png"):
            path = args.out / f"{stem}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.08)
        # Hiragino is CFF-based; direct Type42 embedding produced broken PDF
        # glyphs in the actual QA render. SVG paths preserve all glyph outlines.
        svg_doc=pymupdf.open(stream=(args.out/f"{stem}.svg").read_bytes(),filetype="svg")
        (args.out/f"{stem}.pdf").write_bytes(svg_doc.convert_to_pdf())
        svg_doc.close()
        for ext in ("pdf", "svg", "png"):
            path=args.out/f"{stem}.{ext}"
            files[ext] = {"name": path.name, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest["figures"].append({"name": stem, "semantic_name":name, "files": files, "size_inches_before_tight_crop": fig.get_size_inches().tolist(),
                                   "caption_zh": caption, "source_names": source_names, "plotted_values": plotted})
        plt.close(fig)
        print(json.dumps({"figure": name, "files": [x["name"] for x in files.values()]}, ensure_ascii=False), flush=True)

    # 1. Distinguish absolute predictive loss from genuine paired intervals.
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.8), gridspec_kw={"width_ratios": [1.05, 1], "wspace": .38})
    fig.subplots_adjust(left=.24, right=.98, top=.88, bottom=.17)
    names = ["uniform_author", "uniform_video_quantity", "source_fitted_quantity_demand_age", "A_account_transfer", "B_self_item_memory"]
    labels = ["均匀账号", "按作品数量", "内容基线", "A→B 跨作品", "B自身记忆\n（更多信息）"]
    values = [models[x]["test_author_loss"] for x in names]
    yy = np.arange(len(names))[::-1]
    axes[0].scatter(values, yy, s=42, c=[GRAY, GRAY, GOLD, TEAL, NAVY], zorder=3)
    for x, y in zip(values, yy): axes[0].text(x+.045, y, f"{x:.3f}", va="center", fontsize=10)
    axes[0].set(yticks=yy, yticklabels=labels, xlim=(5.15, 7.13), ylim=(-.6, 4.6), xlabel="测试损失 / nats")
    axes[0].set_title("A  旧作品预测", loc="left", pad=13)
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(.5)); clean(axes[0])
    contrast_labels = ["A跨作品 - 内容基线", "A数量修订 - 线性数量", "B自身记忆 - A跨作品"]
    contrasts = summary["comparisons"]
    for y, row, label, color in zip([2, 1, 0], contrasts, contrast_labels, [TEAL, GOLD, NAVY]):
        x = row["test_delta_author_loss"]; lo, hi = row["conditional_account_bootstrap_95_interval"]
        axes[1].errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=color, capsize=3, markersize=5, zorder=3)
        axes[1].text(-1.43, y+.25, label, fontsize=10, va="bottom")
    axes[1].axvline(0, color=RED, linestyle=(0,(3,3)), linewidth=1)
    axes[1].set(xlim=(-1.46,.18), ylim=(-.55,2.8), yticks=[], xlabel="配对损失差 / nats")
    axes[1].set_title("B  条件95%区间", loc="left", pad=13)
    clean(axes[1])
    save(fig,"old_work_loss_and_intervals","旧作品测试损失及真实配对区间。区间为条件账号bootstrap；B自身记忆使用更多信息，不能当作同信息机制比较。", {"losses":dict(zip(names,values)),"comparisons":contrasts}, ["kuairand_cross_item_summary.json"])

    # 2. Post hoc concentration: no confidence bars were supplied for these values.
    influence=diag["influence"]
    gains=[summary["comparisons"][0]["test_delta_author_loss"]]+[influence["drop_highest_exposure_accounts"][str(k)] for k in (1,5,10)]
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.8),gridspec_kw={"width_ratios":[1.25,1],"wspace":.65})
    fig.subplots_adjust(left=.22,right=.97,top=.86,bottom=.19)
    y=np.arange(4)[::-1]
    axes[0].barh(y,gains,color=[TEAL if x<0 else RED for x in gains],height=.48,zorder=3)
    axes[0].axvline(0,color=GRAY,linewidth=.8)
    for x,t in zip(gains,y): axes[0].text(x+.0015 if x>=0 else x/2,t,f"{x:+.3f}",ha="left" if x>=0 else "center",va="center",fontsize=10,color=NAVY if x>=0 else "white")
    axes[0].set(yticks=y,yticklabels=["全部账号","剔除最高曝光1个","剔除最高曝光5个","剔除最高曝光10个"],xlim=(-.055,.031),xlabel="A跨作品 - 内容基线")
    axes[0].set_title("A  增益集中诊断",loc="left",pad=13); clean(axes[0])
    counts=[influence["accounts_with_negative_paired_loss_numerator"],influence["accounts_with_positive_paired_loss_numerator"]]
    axes[1].barh([1,0],counts,color=[TEAL,RED],height=.42,zorder=3)
    for y,x in zip([1,0],counts):axes[1].text(x+12,y,str(x),va="center",fontsize=11,color=NAVY)
    axes[1].set(yticks=[1,0],yticklabels=["损失降低","损失升高"],xlim=(0,max(counts)*1.22),xlabel="有测试曝光的账号数")
    axes[1].set_title("B  账号层面方向",loc="left",pad=13);clean(axes[1])
    save(fig,"cross_item_gain_concentration","事后、固定预测的集中度诊断；未重拟合。移除最高曝光账号后增益改变方向，不能据账号数量解读因果收益。",{"paired_deltas":gains,"negative_positive_account_counts":counts,"post_hoc":influence["post_hoc"]},["kuairand_cross_item_summary.json","kuairand_diagnostics_and_acceptance.json"])

    pooled=new["pooled_models"]
    # 3. Pooled loss plus four actual origin contrasts; no fabricated error bars.
    keys=["duration_base","history_H","catalogue_stock_Q","stock_flow_QF","history_stock_flow_HQF"]
    labels=["时长基线","历史 H","库存 Q","库存+发布 QF","历史+QF"]
    pooled_losses=[pooled[k]["joint_log_loss"] for k in keys]
    origin_ids=[o["origin"] for o in new["origins"]]
    origin_deltas=[]
    for o in new["origins"]:
        by_name={m["model"]:m["test"]["joint_log_loss"] for m in o["models"]}
        origin_deltas.append(by_name["history_stock_flow_HQF"]-by_name["history_H"])
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.8),gridspec_kw={"width_ratios":[1.1,1],"wspace":.5})
    fig.subplots_adjust(left=.20,right=.98,bottom=.18,top=.84)
    y=np.arange(len(keys))[::-1]
    axes[0].scatter(pooled_losses,y,s=43,c=[GRAY,NAVY,GOLD,GOLD,TEAL],zorder=3)
    for v,t in zip(pooled_losses,y):axes[0].text(v+.015,t,f"{v:.3f}",va="center",fontsize=10)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(4.30,4.92),ylim=(-.6,4.6),xlabel="联合池损失 / nats")
    axes[0].set_title("A  汇总测试损失",loc="left",pad=13);clean(axes[0])
    yy=np.arange(len(origin_ids))[::-1]
    axes[1].barh(yy,origin_deltas,color=TEAL,height=.42,zorder=3)
    for v,t in zip(origin_deltas,yy):axes[1].text(v-.012,t,f"{v:+.3f}",ha="right",va="center",fontsize=10)
    axes[1].axvline(0,color=GRAY,lw=.8)
    axes[1].set(yticks=yy,yticklabels=origin_ids,xlim=(-.65,.025),xlabel="HQF - H 损失差")
    axes[1].set_title("B  四个滚动起点",loc="left",pad=13);clean(axes[1])
    save(fig,"new_work_loss_by_origin","新作首日联合池的汇总测试损失与各起点HQF-H差值，越低越好；点和条均为实际损失，没有人为构造误差条。",{"pooled_losses":dict(zip(keys,pooled_losses)),"origins":origin_ids,"origin_HQF_minus_H":origin_deltas},["kuairec_new_upload_first_day.json"])

    # 4. Exact log-score decomposition; never add uncertainty not supplied for parts.
    keys=["history_H","catalogue_stock_Q","stock_flow_QF","history_stock_flow_HQF"]
    labels=["历史 H","库存 Q","库存+发布 QF","历史+QF"]
    masses=[pooled[k]["seen"]["predicted_group_exposure_mass"] for k in keys]
    actual=pooled["history_H"]["seen"]["observed_group_exposure_mass"]
    def decomposition(model):
        m=pooled[model];total=m["joint_exposures"]
        within=sum(m[g]["joint_exposures"]*m[g]["within_group_conditional_log_loss"] for g in ("seen","unseen"))/total
        between=sum(m[g]["joint_exposures"]*(m[g]["joint_pool_log_loss_contribution_per_group_exposure"]-m[g]["within_group_conditional_log_loss"]) for g in ("seen","unseen"))/total
        return np.array([between,within,m["joint_log_loss"]])
    parts={name:decomposition(name) for name in ("history_H","history_stock_flow_HQF")}
    delta=parts["history_stock_flow_HQF"]-parts["history_H"]
    identity_error=float(delta[0]+delta[1]-delta[2])
    if abs(identity_error)>1e-9:raise ValueError("Log-score decomposition does not reconstruct total")
    fig,axes=plt.subplots(1,2,figsize=(6.4,3.8),gridspec_kw={"width_ratios":[1,1],"wspace":.65})
    fig.subplots_adjust(left=.20,right=.98,bottom=.18,top=.81)
    y=np.arange(len(keys))[::-1]
    axes[0].barh(y,masses,color=[NAVY,GOLD,GOLD,TEAL],height=.45,zorder=3)
    for v,t in zip(masses,y):axes[0].text(v+.015,t,f"{v:.1%}",va="center",fontsize=10)
    axes[0].axvline(actual,color=RED,linewidth=1.3,linestyle=(0,(4,3)))
    axes[0].text(actual,3.68,f"实际 {actual:.1%}",ha="center",fontsize=10,color=RED)
    axes[0].set(yticks=y,yticklabels=labels,xlim=(0,.74),ylim=(-.65,4.2),xlabel="见过账号的曝光份额")
    axes[0].xaxis.set_major_formatter(ticker.PercentFormatter(1,decimals=0))
    axes[0].set_title("A  seen组份额",loc="left",pad=13);clean(axes[0])
    axes[1].barh([2,1,0],delta,color=[TEAL,RED,NAVY],height=.4,zorder=3)
    for v,t in zip(delta,[2,1,0]):axes[1].text(v+.003 if v>=0 else v/2,t,f"{v:+.4f}",ha="left" if v>=0 else "center",va="center",fontsize=10,color=NAVY if v>=0 else "white")
    axes[1].axvline(0,color=GRAY,lw=.8)
    axes[1].set(yticks=[2,1,0],yticklabels=["组间份额","组内条件项","合计"],xlim=(-.26,.065),xlabel="HQF - H 损失差")
    axes[1].set_title("B  恒等分解（无CI）",loc="left",pad=13);clean(axes[1])
    save(fig,"new_work_seen_mass","训练期见过账号的联合池曝光份额，及HQF-H损失差的精确恒等分解。分解没有新增CI或统计检验；组间份额改善不等于组内排序改善，unseen不代表新注册。",{"predicted_mass":dict(zip(keys,masses)),"observed_seen_mass":actual,"decomposition_HQF_minus_H":{"between_group":float(delta[0]),"within_group":float(delta[1]),"total":float(delta[2]),"identity_error":identity_error},"seen_test_works":new["test_seen_works"],"unseen_test_works":new["test_unseen_works"]},["kuairec_new_upload_first_day.json"])

    # 5. This is an analytical model illustration with parameters read from case 12.
    case=cases[12];p=case["condition"];b=case["continuous_benchmark"]
    eta=p["ratio"]**(1/(1-p["alpha"]))
    ks=np.arange(2,7,dtype=float);lower=ks;upper=ks-1+1/eta
    fig,ax=plt.subplots(figsize=(6.4,3.7));fig.subplots_adjust(left=.12,right=.96,bottom=.16,top=.80)
    # Use explicit pastel fills and in-range geometry: SVG-to-PDF readers vary
    # in opacity and clipPath support, while these opaque paths are portable.
    ax.fill_between(ks,1.8,lower,color="#FAF3F0")
    ax.fill_between(ks,lower,upper,color="#F9F0E1")
    ax.fill_between(ks,upper,9.2,color="#EEF6F7")
    ax.plot(ks,lower,"o-",color=NAVY,markerfacecolor="white",label=r"总量下界 $n=k$")
    ax.plot(ks,upper,"s-",color=GOLD,markerfacecolor="white",label=r"内点边界 $n=k-1+1/\eta$")
    cx,cy=b["k"],b["n"][0];eq=case["pure_equilibria"][0]["actions"][0];ix,iy=eq["k"],eq["n"]
    ax.scatter(cx,cy,s=55,c=RED,marker="o",zorder=5)
    ax.scatter(ix,iy,s=55,c=TEAL,marker="D",zorder=5)
    box={"facecolor":"white","edgecolor":"none","alpha":.93,"pad":2}
    ax.annotate(f"连续候选\nk={cx:.2f}, n={cy:.0f}",(cx,cy),(3.45,2.35),fontsize=10,bbox=box,arrowprops={"arrowstyle":"-","color":RED,"lw":.8})
    ax.annotate(f"整数均衡\nk={ix}, n={iy}",(ix,iy),(2.05,5.1),fontsize=10,bbox=box,arrowprops={"arrowstyle":"-","color":TEAL,"lw":.8})
    ax.text(4.6,3.2,"总量不可行",color=RED,fontsize=10.5)
    ax.text(4.4,5.4,"新号最低件数绑定",color="#956D17",fontsize=10.5)
    ax.text(3.6,8.2,"加权内点满足最低量",color=TEAL,fontsize=10.5)
    ax.set(xticks=ks,xlim=(1.9,6.15),ylim=(1.8,9.2),xlabel="账号数 k",ylabel="总作品数 n")
    ax.legend(loc="lower left",bbox_to_anchor=(-.01,1.01),ncol=1,frameon=False)
    ax.set_title("模型示例：最低内容约束比 n≥k 更强",loc="left",y=1.31,pad=0);clean(ax,"y")
    save(fig,"minimum_content_boundary","模型示例，参数完全来自预定case12：新号每号至少1件时，固定整数k的内点要求n≥k-1+1/eta。连续候选点属于松弛，不是整数观察。",{"case_id":12,"parameters":p,"eta":eta,"boundary_k":ks.tolist(),"boundary_n":upper.tolist(),"continuous_point":{"k":cx,"n":cy},"finite_point":eq},["weighted_integer_results.json"])

    # 6. Both endpoints are strict finite pure equilibria.
    pair=[cases[151],cases[150]];x=np.arange(2)
    cc=[z["continuous_benchmark"]["k"] for z in pair]
    k1=[z["pure_equilibria"][0]["actions"][0]["k"] for z in pair]
    k2=[z["pure_equilibria"][0]["actions"][1]["k"] for z in pair]
    c1=[z["condition"]["c"][0] for z in pair]
    fig,ax=plt.subplots(figsize=(6.4,3.8));fig.subplots_adjust(left=.12,right=.96,bottom=.20,top=.76)
    ax.plot(x,cc,"o--",color=GRAY,markerfacecolor="white",markersize=8,label="连续共同 k",zorder=2)
    ax.plot(x,k1,"D-",color=TEAL,markersize=6,label="整数：创作者1",zorder=4)
    ax.plot(x,k2,"s-",color=NAVY,markersize=4.5,label="整数：创作者2",zorder=3)
    ax.text(.55,3.18,f"{k2[0]} → {k2[1]}",color=NAVY,fontsize=11)
    ax.text(.51,2.56,f"{cc[0]:.0f} → {cc[1]:.3f}",color=GRAY,fontsize=11)
    ax.text(.47,1.24,f"{k1[0]} → {k1[1]}",color=TEAL,fontsize=11)
    ax.set(xticks=x,xticklabels=[f"$c_1={v:g}$" for v in c1],xlim=(-.15,1.15),ylim=(.7,3.5),yticks=[1,2,3],ylabel="账号数 k",xlabel="创作者1的生产成本下降 →")
    ax.legend(loc="lower left",bbox_to_anchor=(-.015,1.04),ncol=3,frameon=False,columnspacing=1.1,handlelength=1.6)
    ax.set_title("模型示例：物理边界下可出现相反方向",loc="left",y=1.29,pad=0);clean(ax,"y")
    save(fig,"cost_boundary_counterexample","预定case151→150：c1从2降至1，两个精确整数端点均为严格纯均衡。连续候选违反最低内容要求，不能将该反例称为对连续内点定理的推翻。",{"case_ids":[151,150],"conditions":[z["condition"] for z in pair],"c1":c1,"continuous_k":cc,"finite_k1":k1,"finite_k2":k2,"strict_margins":[z["pure_equilibria"][0]["strict_margins"] for z in pair]},["weighted_integer_results.json"])

    (args.out/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"complete":True,"figures":len(manifest["figures"]),"output":str(args.out.resolve())}),flush=True)


if __name__=="__main__":main()
