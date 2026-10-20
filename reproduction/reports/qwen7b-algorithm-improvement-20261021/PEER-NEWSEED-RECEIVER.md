# 新 seed G5 接收端：代码就绪，尚未传输

实现：`reproduction/operations/qwen7b_warmup/peer_newseed_receive.py`。此单文件在 HKUST2 执行 receive，在 HKUST3 / HPC3 运行对应角色；HKUST2 只需既有 requests，其他角色只用 Python 标准库。代码 SHA 绑定不可变 code/<SHA> 目录，用 4096B 分块上传，命令均小于16KB。

## 启动入口（发布后，由控制端传入 RAM 凭据）

在 HKUST2 执行：

```sh
python3 /immutable/code/peer_newseed_receive.py --action receive \
  --base /data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021/newseed-20261021-185747-445fc691 \
  --snapshot newseed-20261021-185747-445fc691 \
  --prefix peer-newseed-to-hpc3-20261021/newseed-20261021-185747-445fc691 \
  --catalog-commit <真实catalog提交40位SHA>
```

一行 JSON 从 stdin 进入，键仅 `hf_token` 与 `hpc3_password`。不要放 shell 命令、文件、日志或 argv。可选 `--hf-proxy http://127.0.0.1:17897`（也允许17894/17898），只选已实测的 HKUST2 自有出口。默认 HKUST2 直连，明确关闭 requests 环境代理继承，不恢复 Mac 18092/7892。

既有 source/HF catalog 完成前不得启动；该模块没有轮询等待伪造 catalog，也不宣称上游已完成。source/HF 真正完成后每一组从 pinned catalog commit 的 paths-info 核对 LFS SHA 与大小，再下载 Range 断点，完整 SHA 通过后才转发。

## 实际路线与产物

- HKUST2 `/data/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021/<snapshot>`，全局 receiver.lock、不可变 receive-intent/catalog binding、真实字节 progress、失败只记录 type。
- HKUST2 本机 rsync 至 HKUST3 `/backup/shaoshitong/qwen7b-peer-newseed-to-hpc3-20261021/<snapshot>`。rsync receiver 用 flock 防并发写同一组，归档 SHA 通过才转下一跳。
- HKUST3 使用既有 sshpass、mou176、公开 known_hosts，直接校园 SSH / rsync 至 HPC3。StrictHostKeyChecking=yes，不复制私钥、不更改服务。
- HPC3 独立目标 `/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/<snapshot>/raw`。归档留在同 snapshot 的 archives，源/HKUST2/HKUST3 副本都保留。
- 仅接收精确成员表中的普通 raw 文件；拒绝绝对路径、..、非规范路径、链接、特殊文件、tensor/model 后缀、重复或额外成员。逐文件 SHA，随后独立 fresh 全量读回；最后同时 fresh 所有归档和所有解包文件才写 complete.json。
- complete 状态为 PEER_NEWSEED_RAW_ALL_ARCHIVES_AND_FILES_FRESH_SHA256_VERIFIED，只证明该封存快照迁移，不冒称全169028源采样完成，也不冒称已建立 hidden feature bank。原 B/bank 没有写入。

## 已完成验证

本地11项 CPU tests 通过（0.026s）：catalog/LFS/source绑定、G5计数、断点及拒绝被忽略Range、symlink/path traversal/tar links/重复/额外/缺失成员拒绝、真实合成归档接受＋全文件fresh校验＋篡改拒绝、rsync SSH argv 不重复 host、凭据无 argv、所有代码命令<16KB。

只读真实路线已成功：HKUST2 xielab2 → HKUST3 xielab3 → HPC3 ACD-Login-1/uid21930。HKUST2 rsync/flock/requests2.31 可用、余2.492TB；HKUST3 rsync/flock/sshpass 可用、余9.880TB。证据为 actual-launches/peer-newseed-receiver-readonly-route.json。未执行远端写入或真实数据下载。

HPC3 默认 Python3.6，target-init/accept/finalize 只用3.6标准库；shlex.join 位于只在 HKUST2/3 调用的安装/传输路径。实际远端 tiny fixture 接受和 readback 在发布后进行；本地测试不代替此实际证明。

## 2026-10-21 实际标准库资格补记

发布 commit755ed4ba3eef237ab2244c09c17cdac3966a43fc 后，在 HPC3 ACD-Login 的实际 Python3.6.8 上执行9项目标端测试，0失败/0错误，包含真实合成 target_accept、全部文件独立SHA读回和篡改拒绝。另两项仅测试HKUST2/3 shlex.join传输路径，已在本地13项联合测试通过，未错误地要求HPC3运行它们。

独立共享fixture：`/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-receiver-stdlib-fixture-20261021/755ed4ba3eef237ab2244c09c17cdac3966a43fc`。测试拦截了唯一一次合成complete写入，第二次fresh读回确认 synthetic-tmp为空、没有任何持久complete文件；该fixture绝不作为真实源/HF catalog或迁移验收。qualification.json SHA256为1dbd188cd48a1ba656aaa96760806834f635a786347fc07fbb4499eb0c941b3d，receiver源码仍为3d9b43e03217e4cdb1d28491c198b4ef19950ad873fad5632fab0ddba6ab3cc9。

HKUST2/3新SSH连接出现间歇性TCP22超时，也观察过四条别名/直连路径同时成功，故接收代码部署改用本任务短时ControlMaster复用，结束只关闭本任务master；没有改动Mac网络/代理配置。真实HF catalog未出前仍不启动receive。
