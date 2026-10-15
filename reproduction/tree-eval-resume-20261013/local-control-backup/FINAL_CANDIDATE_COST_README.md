# 训练末轮候选成本补充入口

`finalize_nested_candidate_costs.py` 是纯本地入口，不导入 SSH helper，不读取 PT、模型、原始 rollout 或远端文件。它复用已发布 appender/candidate collector 的本地身份验证函数，其源码 SHA 固定在 `DEPENDENCIES`。旧入口和旧产物不改。

仅在真实 FULL/native/CUM200、已收集 candidate200、其 index/registry 全部存在后执行。不存在时返回 PENDING 且不写输出；身份、SHA、分页、计数或覆盖冲突明确报错。无需训练201，保存的 driver cursor201只证明下一保存位置。8rank Adam来自实际 writer/FULL/CUM计数，不硬编码4000；scheduler必须实际200。

从工作区根目录调用，两个 SHA 必须取实际不可变 CUM200及包含candidate200的 registry快照，不能预填未来值：

```sh
python3 -B work/tree-speed-repair-20261008/nested-final-candidate-cost-fix-source/reproduction/tree-eval-resume-20261013/local-control-backup/finalize_nested_candidate_costs.py \
  --base outputs/tree-nested-branch-handoff-20261013 \
  --cumulative-sha256 ACTUAL_CUM200_RAW_FILE_SHA256 \
  --registry-sha256 ACTUAL_REGISTRY_WITH_CANDIDATE200_SHA256
```

先逐字节绑定 CUM200及其全部字段分片和两镜像，确认原candidate结局覆盖1..199、只缺200。再核对CUM绑定的book、FULL/poll/native/resource200与现成candidate200来源pin。仅在内存补末轮一次；原sampling、generation/prefill/retrieval/scoring、outer/dev/driver CPU、共享容量上界和历史失败数据原样沿用，候选子集/fallback不再相加。末轮单项scalar为null时仍保留null和missing200；源candidate中的partial known sums另保留，不改原aggregate known_sum语义。

新输出为 `FINAL_CANDIDATE_COST_THROUGH200.json/md`、`final_candidate_cost_parts/*.json` 和 `FINAL_CANDIDATE_COST_THROUGH200_LOCAL_RECEIPT.json`。每个文件最多100000字节（比100KiB更严格）。完整dict/list字段按实际大小分页；若一个嵌套容器超过预算，也完整分页，`original_field`使用点路径或`[index]`路径。必须按receipt的全部`output_refs`核对SHA/bytes与两镜像，不能固定分片数。可用 `LocalInputs.expand(value, field, 'final_candidate_scalar_field_part_v1')` 动态还原新FINAL字段；旧CUM使用默认schema。

所有已有输出先做不覆盖检查；receipt最后写入，并进行两次全部输出字节读回。重复调用仍重新验证原始FULL/native/CUM/index/candidate，并仅从原CUM200在内存重建预期结果以比较字节；不把已有FINAL作为累计输入。receipt必须逐字节等于完整重建的schema、输入和输出manifest，FINAL JSON/MD根文件及递归分页闭包必须全部存在且与manifest精确一致，删减manifest不能通过幂等检查。缺少原始输入但已有receipt会明确报错。

CUM、分片、registry、candidate、receipt及复用本地helper读取的所有JSON层级均拒绝重复对象key，包括SHA/bytes正确的JSON分片；仅在本进程导入的隔离helper模块绑定strict decoder，不修改旧helper文件或全局标准库。

该产物仅证明training candidate元数据闭合，不代表完整goal、51713评测、HF冷重载、资源清理或服务终态CPU验收。CUM中的service CPU null保持null；DP padding分摊、独立结果wall、FLOPs、真实GPU计费与能耗不推算。
