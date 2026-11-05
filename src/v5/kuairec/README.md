# V5 KuaiRec 独立复现入口

源码与小型审计可本地保存。原始 CSV、逐作品预测和中间分割留在 hkust3：

`/data/shaoshitong/ai_creator_game_20261104/v5/kuairec/`

源数据是上级根目录下 `v3/kuairec/item_daily_features.csv`，不得覆盖 v1/v2/v3。
运行环境为 `/home/shaoshitong/anaconda3/bin/python`，使用 numpy/pandas/scipy/pyarrow。
本任务是 CPU 元数据分析，不启动 GPU 作业、不下载媒体。

## 已完成的入口

- `PROTOCOL.md`、`protocol.json`：旧作品迁移冻结方案。不要事后改写；更改方案须新文件记录。
- `analyze_transfer.py`：46 个账号 A/B 跨旧作品历史迁移、999 次同层身份置换、四个固定起点。
- `run_remote.py RESEARCH_ROOT`：捕获真实进程退出码和日志。已完成 `audit/run_01_exit.json`，exit 0。
- `audit_new_uploads.py`：不重新拟合，仅审计新上传后缺失作品日。
- `NEW_UPLOAD_PROTOCOL.md`、`new_upload_protocol.json`：旧作品分析结束后另行冻结的新作首日补充。
- `analyze_new_uploads.py`：四窗共同自然日首日池的有限网格预测和 seen/unseen 联合及组内条件指标。
- `run_new_upload.py RESEARCH_ROOT`：每次保留独立 attempt 日志与真实退出。attempt01 为整数负指数 dtype 错误，attempt02 仅修复浮点转换，exit 0；未变更协议。
- `verify_acceptance.py RESEARCH_ROOT`：不重新拟合，核对输入/协议/执行源码哈希、真实 exit、四窗口和首日实际候选数；已 exit 0。

在目标远端部署代码后，入口参数 `RESEARCH_ROOT` 为
`/data/shaoshitong/ai_creator_game_20261104`。例如：

```bash
/home/shaoshitong/anaconda3/bin/python /data/shaoshitong/ai_creator_game_20261104/v5/kuairec/code/verify_acceptance.py /data/shaoshitong/ai_creator_game_20261104
```

已有结果无需重复运行。主报告与补充报告在本轮 `outputs/V5_KUAIREC_VALIDATION.md`、
`outputs/V5_KUAIREC_NEW_UPLOAD_SUPPLEMENT.md`；小审计 `outputs/V5_AUDIT/kuairec_*`。
原始输入 SHA256 `45943d63c44652b6403f3a4f78c7225e1afe7916bab17d9a674d7979245e085b`。

## 解释边界

`show_cnt` 为曝光，不是质量或收益。主检验的共同旧作池使用未来行存在信息做样本选择。
补充只在实际观察到上传自然日行的视频池内评价，不补缺行，不意味着完整平台新作池。
Q/F 为截至训练末实际已出现目录的代理，H 为训练旧作曝光率；不观测完整生产史、
经济主体 owner、注册时间、AI 使用成本或外生排序变化。预测改进不等于因果或均衡验证。
