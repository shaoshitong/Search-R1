"""Small current report; original failed/queued evidence remains historical."""
import datetime,json
from pathlib import Path
from .control import OUT

def main():
 v=json.loads((OUT/'SHARED_RESUME64_LATEST.json').read_text());root=OUT.parents[1]
 rows=v.get('facts',[]);stage=v.get('stage')or{};health=v.get('actual_node_health')or{}
 cache=OUT/'ITERATION_WALL_TOTALS.json'
 totals={r['outer']:r for r in json.loads(cache.read_text()).get('rows',[])}if cache.exists()else{}
 lines=['# B 10选5：H1005共享64轮恢复','',f"实测UTC：{datetime.datetime.fromtimestamp(v['observed_unix'],datetime.timezone.utc).isoformat()}。完整outer {v['complete_outer']}/200，当前{stage.get('stage')}；总任务尚未完成。",'',f"同出生身份存活：{health.get('same_birth')}；cleanup：{v.get('cleanup')}。H1005/10.33.0.5，8卡原有任务共卡运行。只取消了本任务原排队3674；GPU Slurm预约数0，原外来进程不停止。",'',f"运行目录：`{v['receipt']['command'][-1]}`。真实存储为保留H10018的NFS导出，未通过Mac传输模型、索引或训练资产。",'','恢复继承CP64：8rank actor1280、scheduler64、probe raw/EMA/Adam880、RNG及已预留65轮候选池。8卡micro1/accum16、每题实际采10前缀选5且仅5个过探针、原B其余目标和预算保留。未清零或从基座重开；不声称完整executor逐位重放。','', '| 新完整轮 | Actor Adam/rank | Probe Adam/rank | 最终512组混奖 | 整轮分钟 |','|---:|---:|---:|---:|---:|']
 lines[-2:] = ['| 新完整轮 | 类型 | Actor Adam/rank | Probe Adam/rank | 最终512组混奖 | 训练分钟 | 迭代总分钟 |','|---:|---|---:|---:|---:|---:|---:|']
 for r in rows:
  wall=r['phase_timings'].get('outer-%06d-training-wall'%r['outer']);duration=f'{wall/60:.2f}'if wall is not None else'未知'
  total=totals.get(r['outer'],{}).get('total_wall_seconds_including_qa_grounding_storage');total_duration=f'{total/60:.2f}'if total is not None else'待核对'
  kind='周期全池'if any('complete-unselected-pool'in k for k in r['phase_timings'])else'普通'
  lines.append(f"|{r['outer']}|{kind}|{r['actor'][0]}|{r['probe'][0]}|{100*r['selection_metrics']['mixed_fraction']:.2f}%|{duration}|{total_duration}|")
 lines+=['','周期全池轮使用全候选的实际结果选组，其训练组混奖比例与普通轮分开看，不能当作测试集EM/F1。']
 lines+=['','训练分钟取原生training-wall；迭代总分钟取原生iteration-wall-time，包含本轮原定QA、grounding及存储处理。缺少总计实读时显示待核对，不以训练耗时替代；训练200后的最终51713评测另行计时。']
 active=v.get('active_actor',[])
 if active:lines+=['',f"当前直接rank记录：{[(r['rank'],r['writes'],r['opportunities'],r['last_adam'])for r in active]}。实际写入单列，不把机会当写入。"]
 lines+=['','此前45GB空余及等待整节点属于助手额外加入的限制，已按最新人工共卡指令撤回；实际运行使用既有共享配置与UUID租约。第一SSH启动退出、systemd启动因NFS设备号误报而退出的证据保留，均0个actor写入。当前独立服务托管，不借用其他任务cgroup。','',f"运行源码：`{v['receipt']['source_commit']}`，科学checkpoint binding保留c3ced090。新NFS适配仅允许来自精确retained18导出且inode/大小/数据mtime完全相同的st_dev映射，原全SHA与28片顺序不变；明确未声称新的字节SHA读回。",'','51–64轮普通轮（排除周期60）实际训练混奖平均48.18%，参考B为41.56%，提高6.63个百分点。训练组混奖比较不能证明测试EM/F1提升；8卡/4卡、后续采样分化及10候选生成成本差异单列。','',f"直接证据：`{v['evidence']}`。恢复验收须65–69完整且70真实继续，随后同一实例保持200总预算。5秒整卡显存/利用率记录远端日志，不能用Torch峰值或共享SM利用率推称本任务独占资源/计费。",'']
 text='\n'.join(lines)
 for p in (OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):(p/'SHARED_RECOVERY64_REPORT.md').write_text(text)
 (root/'outputs/REPORT.md').write_text(text)
 print(json.dumps(dict(complete=v['complete_outer'],stage=stage.get('stage'))))

if __name__=='__main__':main()
