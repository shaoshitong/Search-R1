# 参考 B 在 HPC3 的独立部署：已准备，未训练

用户指令（2026-10-19，Asia/Shanghai）：查看线程 `01a0fcbc-0b46-7502-8b66-d8b21a0e032d`，按其代码、数据、逻辑及探针准备本线程环境，**先部署，不运行**。随后明确选择：**官方初始 actor＋同一个第5 epoch EMA 探针**。用户进一步确认优先复用本线程之前的环境。

## 当前状态

- **代码、路径映射、Python入口、依赖层和缓存已部署；未提交 Slurm，未启动训练或 rollout，未创建 actor/probe optimizer，未导入任何旧在线 checkpoint。**
- 目标：`/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019`。
- Python入口：目标下 `env/bin/python`；只做CPU检查的入口是 `env/bin/python-cpu-check`。
- 配置：`preparation/prepared-config.json`，保留 `preparation_only=true`、`training_authorized=false`。初始 actor outer=0、online probe Adam=0 是计划值，尚未执行初始化。
- 引用线程标题为“U-验证 Tree-GRPO 效果”，但当前实际运行的是下面的 B / diverse10to5 方案，不能按标题推断为早期 Tree-GRPO。
- 其当前8-rank拓扑作为参考记录，没有申请8卡。实际 GPU UUID、资源数、Slurm及HPC3执行命名空间绑定都留到明确启动指令后处理。本次不声称已经通过GPU/DDP或训练启动验证。

## 已核实的参考来源

实际运行源码 `bc8b16728b689ccbb97457e06f2ef0a65cb39fa7`；配置 SHA `0d4ab4c0bba08f8244afa5d3fce0bcf4ab97450b8fa93c218340da416517ae88`。对其实际远端 repo 中90个科学相关源码文件逐个读取 SHA，与本线程副本完全一致。复制的是不可变运行版本，不是另一线程不断更新的本地 HEAD。

独立分支 `codex/reference-b-hpc3-prepared-20261019` 从上述提交建立，只新增本次 operations 部署目录。运行部署版本 `92c598f8cd35b8e85662ce2290d358ce350b80b3`；1547个文件在 HPC3 全部 SHA 通过。另一线程在 H1005 的实例、代码、锁和数据没有被修改。

## 与本线程上一轮实验的区别

| 项目 | 上一轮已完成实验 | 本次参考方案 |
|---|---|---|
|每题 Prefix采样|5条|先采10条，按token差异确定性保留5条|
|Prefix去重/差异选择|无上述10选5步骤|位置差异＋bigram Jaccard，farthest-first；不使用EM或探针预测|
|普通轮探针|两次P10之间冻结|本轮512个原始G5，Mixed类别平衡重采样，10次实际Adam/rank|
|P10探针|后期3epoch/150次更新|2560题G5，一次平衡采样pass，50次更新/rank|
|组Mixed BCE权重|1|5，逐轨迹BCE权重仍为1|
|用于选题的权重|原方案当前探针|raw Adam更新；EMA0.99用于选题|
|P20/P40/P80普通轮续写题数|统一1024|依次768/1024/1536，之后真实Mixed优先选512|

Prefix长度阶段保持1–50为5，51–100为20，101–150为40，151–200为80。短回答仅在探针特征中padding/mask，原轨迹token/logprob/位置不重采样。前50轮沿用源代码较早的选题路径，不能声称全程都有768/1024/1536池。参考线程历史上在CP50引入后段协议；本次从official0开始是新计划实验，并非它的历史恢复重放。

探针损失仍根据同一G5的 strict EM构造 `0<K<5` 标签，使用逐轨迹 BCE 加5倍组Mixed BCE；q仍为 `1-prod(p)-prod(1-p)`。P10之外的每轮在线拟合是本次相对上一轮的重要变化。200只保存/评测，不再拟合。原actor训练目标、strict scorer、BF16计算及FP32master/Adam保持参考代码。

## 数据与权重

官方actor revision：`aa8e72537993ba99e69dfaafa59ed015b17504d1`。已有模型路径：
`/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725`。

公共manifest：`0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24`。train169028、固定dev512、final51713及E5/fullWiki2018检索资产复用远端已有文件；新配置未改变划分。

EMA5来源run `8af926a49630428097b94027864da68b`。H100端文件SHA为 `06ffd8c21b6208ff5a0beb11cded62da7b9551c3e1357a1d7321e69eb06d994e`，HPC3现存文件SHA为 `0bf01eddd3cdaec02f91ce8a09d8c5d49d795428e634972791c99179c862d9aa`。完整源epoch/run/binding提取检查通过；**14个EMA参数张量和2个非持久缓冲的shape、dtype、逐字节SHA全部一致**，不是凭run名称推断权重相同。文件的序列化SHA不同已明确保留，没有伪称文件字节相同。

HPC3探针文件：
`/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017/reference-assets/warmup-epoch5-ema.pt`。

奖励Grounding round0与选题探针是两类资产。真实版本同为 `6de29bcb7be741cfa1bfacdf9e34efc9:0:0dcad8285ebe4547b5999a21e6b4e8b9`；两端实际head SHA一致：boundary `f19abd7995ff64a0ba0aaa13750c121a9ef38621bf7cfc0554152bec928b4af5`，commit `75cdfdd41ad57ade7f4e080eef5ab37a2be4c89417424218dbb8f87d96d13e7c`。保留参考代码50/100/150刷新及下一轮采用的生命周期。

数据小文件、来源池、固定题目身份和校准文件fresh SHA已核对；大模型、索引和语料复用此前本线程实际完成200轮及完整评测所用的远端资产，此次仅复查存在与路径，不重复全量hash或复制。模型、tensor、原始rollout、环境包都未经过Mac中转。

## 环境与检查范围

复用上一实验实际使用的共享基础包，独立代码、配置、sklearn overlay副本、FAISS GPU库副本和缓存。主依赖与参考线程实测版本完全一致：

| 包 | 版本 |
|---|---|
|torch|2.4.0|
|transformers|4.46.3|
|vllm|0.6.3|
|datasets|3.2.0|
|safetensors|0.8.0|
|numpy|1.26.4|

CPU检查通过实际导入torch/transformers/vLLM/datasets/safetensors/FAISS/flash_attn/sklearn/joblib；验证8-rank数学分配、20个actor更新机会、类别平衡、10选5差异算法、Prefix阶段、训练禁用标志，以及上述完整EMA读取。两次独立检查和工具补齐后的最终检查均成功退出，CUDA未初始化。运行Python还验证外部传入Prefix40不会被入口固定覆盖成5。

基础Python原来未包含pip；本次从已有远端环境离线复制pip25.0到**私有overlay**，未重新安装科学依赖，也未改动共享包。`pip check`实际返回1：torch2.4.0元数据要求nvidia-cublas-cu12==12.1.3.1，现有运行环境为12.4.5.8。用户已要求沿用此前实际跑通的环境，因此保留现状并记录该既有依赖声明差异，**不把pip check写成通过，也不为清除告警而降级cuBLAS**。

登录节点缺少真实GPU驱动库时，vLLM可出现 `libcuda.so.1`扩展导入提示，另有缺失可选 `vllm._version`提交号的警告。CPU检查成功不证明GPU扩展、通信或训练可用；本次未申请GPU来做运行验证，未用stub伪造GPU通过。

检查收据：本线程 `outputs/reference-b-deployment-20261019/inspection-1.json`、`inspection-2.json`、`inspection-final.json`、`environment-final-notes.json`。远端收据位于目标 `preparation/`。源码传输和环境准备结果经过远端文件重读核验；一次控制读取超时与一次系统Python3.6控制脚本参数不支持均已修正，未据此重启训练或重复创建环境。

## 后续启动边界

这次完成的是源码、依赖和资产部署。没有Slurm提交、模型推理、rollout、optimizer更新或旧checkpoint恢复。初始actor及EMA5只准备引用，不预先训练。未来按用户指令绑定实际资源和HPC3执行入口后，再验证真实GPU流程；**不要因为旧goal、参考线程状态或本报告而自动开跑**。
