# HKUST3 旧 reserve checkpoint 精确退役

这是隔离交付给主线发布、执行的最小工具。编写和测试期间没有连接或修改远端，也没有修改实验 work checkout。

## 固定范围

- 仅允许 `xielab3` 上 UID `1002` 执行。
- 固定 payload：`/backup/shaoshitong/qprobe-old-original-reserve-20261016/payload`。
- `retirement-plan.json` 的完整字节 SHA256 必须为 `abb9f58068655dcc11f44607ea8c5379cefdbd97e0c6e4dc306c94da9f19834c`。
- 只删除清单中 45 个 `.pt` / `.safetensors`：30 个旧 actor checkpoint / training state，15 个配对恢复 boundary / RNG state。
- 45 项总逻辑字节为 `115049176725`，总 `st_blocks * 512` 为 `115049357312`；全部独立 inode、nlink=1。
- 剩余 73 项逐项保护，尤其两个 TOPD Qwen 原始模型资产副本。保留所有目录、metadata、旧 reward `.pkl`、日志和评测。
- 主线已实测当前 Prefix20 的 fresh actor、epoch68 head 和 checkpoint 输出路径不引用这个 reserve；其小型证据完整 SHA 也固定在脚本中。

工具没有 root、plan、resume 参数，没有 SSH、Slurm、子进程、模型加载或目录递归删除功能。

## 执行流程

先由主线把这三个文件一起发布到 HK3，核对 publication manifest 的 SHA：

1. `retire_once.py`
2. `retirement-plan.json`
3. `hpc3-nonreference-evidence.json`

在主线当前训练引用复核仍成立时，使用 HK3 的 Python 执行：

```sh
python3 -B /absolute/published/directory/retire_once.py
python3 -B /absolute/published/directory/retire_once.py --execute
```

第一条只读，第二条执行用户已授权的固定范围。主线负责发布、执行和实际空间观测，本子任务没有执行这两条远端命令。

执行前再次核对固定 owner、所有路径无 symlink、118 项成员集合完整、全部候选和保护项的 dev/inode/nlink/size/blocks/mtime/UID/GID/mode。检查 `/proc/locks`、七个已记录 reserve / durability 进程的 PID + start ticks，以及当前 UID 可访问进程的 cwd / fd / mmap；可能的 checkpoint consumer 不可读时拒绝。不可访问的 sshd、systemd、sd-pam 服务详情只记录限制，之前实测的五个权限例外属于这些服务。

预检通过后独占创建以下目录和 claim，任何已存在状态均拒绝：

`/backup/shaoshitong/qprobe-old-original-reserve-20261016/control/checkpoint-retire45-abb9f5806865`

claim 和目录先 fsync。每项先把包含文件身份的 unlink intent 写入 journal 并 fsync，再次核对文件身份，以已打开的父目录 fd 执行 unlink，fsync 父目录，最后持久记录 done。结束后分别进行两次 absent 扫描，中间复核全部 73 个保护项及剩余成员集合，再独占发布 complete receipt。

## 不明结果与限制

发生异常时保留 claim / journal，并尽力写入不可覆盖的 `failure.json`。即使只删除了一项、未写完 claim，或 unlink 已发生而 fsync / done 失败，重跑也会拒绝；不能删除 claim 来重试。主线必须读取 journal、fresh stat / absent 状态后单独处理。`removed_count_observed` 只统计已完成 unlink + 父目录 fsync 的项，不能把未记 done 的 intent 当作文件仍在。

本工具不重读 115GB 内容；完整历史 SHA 已在计划中，当前准入使用两次 inventory 的固定身份，执行时再次 fresh stat。receipt 的 allocated bytes 是被退役文件的原 `st_blocks` 合计，实际文件系统可用空间变化需由主线另行测量。

## 本地验证

```sh
python3 -B -m unittest discover -s /absolute/path/to/drafts/hk3_retire -p 'test_*.py' -v
```

13 个测试通过：生产计划边界和 TOPD 保护、无通用删除入口、inode 替换、硬链增加、symlink、锁、进程 birth、已记录进程 metadata 变化、活文件描述符、计划篡改、部分失败不重放、既有 unknown claim、成功后的保护项与再次执行拒绝。所有写删仅发生在测试临时目录中的小文件，尚未完成远端实际执行验证。
