# 第57批export拒绝的确切原因

固定100批export preflight在56批/7,168题之后，于 `inputs/shard-00/batch-000390.json` 被拒绝。此时没有archive或HF上传。本报告只诊断metadata，export修复和新attempt由event操作员独占。

源端实际小型读取证明：当前cached audit与冻结计划的audit整个对象相同，status仍为 `EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND`；proof文件仍为2,625 bytes、SHA `6c607c3606cd0dbccce26530387f82c7ee3a22d3c9088ab3c7bdddd577a9527b`，verified epoch仍为1792541200.9112196。没有audit更新时间变化，也没有该proof/fstat绑定变化的证据。

两个本来不同的口径被export消费者直接比较：

| 定义 | SHA256 |
|---|---|
| 原input行顺序生成640个own IDs | `bfb02f10f574244951c3658ca40b9088f7df6da374805ca1df7b3b944acf0375` |
| 相同640 IDs做lexicographic sort后 | `f71ac3156a4df01266011ddc06509ea03e67c07eaf56536d49fda436ace91770` |

peer原auditor `audit_data.py:367`定义 `actual_trajectory_ids_sha256=SHA(canonical(sorted(labels)))`；计划生成器 `peer_newseed_delta.py`则按input行顺序积累 `own_tids`，将其digest写入 `expected_own_G5_IDs_sha256`。两者各自正确。旧generic exporter直接比较二者，因而错误拒绝。

本批128题的candidate ordinal为99840至100094、步长2；`c%04d`只是最小宽度，超过99999后成为六位数，因此字典序把 `c100000` 放到 `c99840` 前面。这解释了前56批未触发该条件。使用源端实测的ordinal范围，在本地仅重建ID字符串，即可精确复现上述两个SHA；没有读取任何payload、EM或轨迹内容。

正确修复应保留两种显式验收：真实input生成的ordered IDs对计划ordered digest；相同IDs排序后的digest对独立auditor proof。当前proof描述符和fstat仍必须对原冻结proof验收，不能放松。应加入跨99999→100000的fixture。

原权威计划SHA `00e5021284cb8f6614670452b749c7add10c028e50578822432600762074662a`保持有效且未修改，成员仍为原100批；不需要重新freeze、扩大成员、重启peer或重做源audit。新export attempt需要发布新消费者源码并保留首个失败attempt。详细小型证据为同目录 `frozen100-digest-order-root-cause.json`。
