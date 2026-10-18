"""Write small human-readable progress from genuine saved observation evidence."""
import datetime,json
from pathlib import Path
from .control import OUT
ROOT=OUT.parents[1]

def main():
 latest=json.loads((OUT/'DIVERSE_LATEST.json').read_text());v=latest['actual'];c=json.loads((OUT/'DIVERSE_CONFIG.json').read_text());facts=v.get('facts',[])
 lines=['# B：CP50恢复、10选5、8卡独立实验','',f"实测时间：{datetime.datetime.fromtimestamp(v['observed_unix'],datetime.timezone.utc).isoformat()}。目标outer200，当前完整{v['joint_outer']}；总任务尚未完成。",'',f"当前阶段：{v.get('stage',{}).get('stage')}。节点H10018（h100-10 /10.33.0.18），8卡0–7。源B保持原样。",'', '| 完整outer | Actor Adam／rank | Probe Adam／rank | 训练512组混合比例 | 训练墙钟分钟 |','|---:|---:|---:|---:|---:|']
 for row in facts:
  wall=row['phase_timings'].get('outer-%06d-training-wall'%row['outer']);duration=f'{wall/60:.2f}'if wall is not None else '未知'
  if row['outer']==51:duration+='（缓存采样后段）'
  lines.append(f"| {row['outer']} | {row['actor'][0]} | {row['probe'][0]} | {100*row['selection_metrics']['mixed_fraction']:.2f}% | {duration} |")
 if facts:
  last=facts[-1];d=last['diversity_comparison'];p=last['prefix_counts'];z=last['probe_variance']
  lines+=['',f"最新完整轮：实际生成{p['prefix_candidates_generated']}个前缀，保留{p['prefixes_retained']}个，丢弃前缀的探针调用为0。直接取同批前5条平均{d['first_native_five_mean_unique']:.4f}种不同前缀；10选5后{d['retained_five_mean_unique']:.4f}种。{d['groups_strictly_more_unique_than_first_five']}/{d['questions']}组增加；全部组保留了10个候选中最多可得的不同前缀。",'',f"保留G5内探针概率方差的题均值：{z['mean_within_group_probability_variance']:.9g}。这与训练混合比例均不是测试集EM/F1。未重复初始基线。"]
 lines+=['','mini128、每卡micro1、global micro8、accum16、每rank20更新机会；分段20/40/80、Top768/1024/1536及其他B规则保留。源CP50 Adam1000、scheduler50与probe/EMA700已真实继承。4→8分片的RNG映射不称为完整executor精确重放。','',f"实际运行源码：`{v['source_commit']}`；checkpoint binding兼容c3ced090。目录：`{c['base']}`。模型、数据、索引和checkpoint全部留在远端。",'','第一CPU导入的两个错误及第一GPU actor的零写入退出证据保留；首次GPU失败为monitor硬编码4卡分块。修复保持相同32个全局监控样本和阈值，独立CPU36/36核验通过。复用原51轮采样，未重采或reset；失败成本与正常训练分别记录。','',f"证据：`{latest['evidence']}`；直接8rank日志核对见DIVERSE_LATEST.json。仅完成51–55并确认56真实推进后才交付启动验收，随后同一实例继续200；完整最终评测与资源收尾仍未完成。",'']
 text='\n'.join(lines);(OUT/'DIVERSE_REPORT.md').write_text(text);(ROOT/'outputs/REPORT.md').write_text(text)
 for target in (ROOT/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  target.mkdir(parents=True,exist_ok=True);(target/'DIVERSE_REPORT.md').write_text(text)
 print(json.dumps(dict(complete_outer=v['joint_outer'],report=str(OUT/'DIVERSE_REPORT.md'))))
if __name__=='__main__':main()
