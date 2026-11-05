# V7 实证运行入口

本目录只有源码/协议，不含原始或逐行数组。文档冻结方案在 `docs/V7_EMPIRICAL_PROTOCOL.md` 与 `docs/V7_OPENCHANNELSTATS_PROTOCOL.md`。

## KuaiRec

远端 `/data/shaoshitong/ai_creator_game_20261104/v7/empirical/`：

- `analyze_states.py`：4 起点×双哈希方向，3/9/9/27 设置四评分族，一步观测曝光预测。
- `run_remote.py RESEARCH_ROOT`：nice10、单线程，保存每轮真实退出。run01 exit0。
- `verify_results.py RESEARCH_ROOT`：从保存预测重构指标、哈希与 bootstrap 有效权重验收，不重新拟合。
- `diagnose_new_groups.py RESEARCH_ROOT`：结果后新作来源历史可用性分解，固定预测、无重拟合。

## 公开 YouTube 小源

独立远端 `v7/openchannelstats_empirical/`：

- `analyze_openchannelstats.py` / `openchannelstats_protocol.json`：独立冻结的公开 opt-in impressions 预测。
- `run_openchannelstats.py RESEARCH_ROOT`：真实退出/log；run01 exit0。
- `verify_openchannelstats.py RESEARCH_ROOT`：输入SHA、预测重构与固定评分移除频道诊断。

两种 `RESEARCH_ROOT` 均为 `/data/shaoshitong/ai_creator_game_20261104`，Python 为 `/home/shaoshitong/anaconda3/bin/python`，只 CPU，无 GPU/媒体/权重。已有结果不需重复运行。原始来源文件的获取入口和 SHA 在 source_discovery 采集审计；本目录不处理账号登录或凭据。

`/data` 是本机 ext4，不是共享挂载。HKUST2 独立副本与文件 SHA 另记复制审计。源码/文档/小聚合由 root 统一提交推送，本子任务不自行 git 提交。

曝光残差记忆与理论旧内容存量单位不同；公共 h>0 只对应评分形式，不等于已识别非负资本。预测在测试内逐日用已发生反馈更新，不是固定状态的多步结构模拟。全部因果、owner、真实冷启动与 AI 成本限制见报告。
