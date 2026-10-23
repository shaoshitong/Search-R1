# 7B Easy2560 四卡训练：整轮速度归因

最新实测（2026-10-23 09:29CST）：新正式四卡作业750491已完整完成并封存71轮，
第72轮已保存512道候选。第71轮整轮壁钟为1410.022秒（23分30秒），
旧65—69轮均值1846.455秒（30分46秒）：少23.636%时间、约1.3095倍速度。
两个实际before-generation边界之间包含完整checkpoint写入、封存和轮间工作，
不含启动恢复；这是不同policy/数据下的新一轮观察，五轮稳定性及最终效果仍待验收。
候选填充652.194秒，比旧均值1075.965秒少39.385%；actor158.451秒，
adv211.025秒、checkpoint写入153.001秒。分段日志不能相加替代整轮壁钟。
见[第71轮整轮证据](evidence/combined-first-complete-outer71-summary.json)。

此前正式750029已完成并封存70轮，四卡各1400次
Adam及scheduler70；按已有授权正常取消后已确认终态。组合版使用runtime
`dde1452d2ec65fb3f4a0464d4beacf3b8136414c`、独立双卡检索750471，现有CPU资源
已读回CP70完整状态；新四卡作业750491已在ACD1-14运行，实际四卡
model/Adam/scheduler/RNG/driver/重建数据流均已读回一致。第71轮四卡各新增20次Adam、
累计1420次、scheduler71；监控额外前向均0，实际FA3为Q28/KV4，五段候选仅一次权重同步。
此前提前停止在异步检索下
使512题候选158.192→94.030秒（少40.56%），但正式同类串行检索下仅
180.867→163.918秒（少9.37%）。两组都少约68%raw token，串行组decode
调用数却基本不降。准备好的单独search-stop目标已被取代；当前切换的是
异步代理、FAISS合批与search-only stop组合版，已取得上面的首个完整outer整轮收益。
见[CP70切换记录](evidence/combined-cp70-transition-summary.json)。


旧作业 `749598` 在 `ACD1-14` 四张H100上运行源码
`c7976da3afc863a1c20b99daa7cc99de3bb61f15`，现已完整完成并封存第60轮，
四卡各累计1200次Adam、scheduler60。第55—59轮完整壁钟均值1741.910秒，
即29分02秒的性能历史保持有效。完整第60轮断点后按已有授权取消旧作业，
唯一续训作业 `750029` 已于2026-10-23 03:33:27CST在ACD1-14分配四张H100、
48CPU、512G；03:35实测为RUNNING，去除剩余梯度日志扫描的runtime
`2d5183455551145c52d1783a38663e2fcf170ceb` 身份匹配。检索745198继续运行。
四卡实际GPU恢复已通过，第61—68轮连续完整更新并分别保存完整断点。
最新CP68已封存，四卡各累计1360次真实Adam，scheduler68；第69轮已保存1024道候选题。
恢复后五轮跟进已验收，61—65阶段每卡新增100次Adam、累计1300、scheduler65。
五轮完整壁钟均值1789.972171秒，即29分49.97秒；目标200轮、固定验证、native51713、
完整归档和最终实际四卡恢复仍未完成，不允许update201。见
[恢复后五个完整轮次](evidence/leanmetrics-five-complete-outers-summary.json)。

## 为什么局部接近两倍，整轮没有接近两倍

独立固定工作量实验使用同一CP50、同一第51轮512×5条已有轨迹、原有四卡布局，
每卡实际20次Adam和80次训练前向。旧实现266.079秒，去掉回滚快照、重复检查和
额外监控前向后140.791秒，再使用紧凑GQA后134.232秒。1.982倍是这部分工作的
速度比，未包含候选生成、优势计算和完整断点保存。

正式训练第52→53轮，actor更新288.139→165.714秒，确实减少42.49%；候选生成
1093.305→1141.550秒。远端保存的实际原始生成token为4,870,541→5,122,972，
增加5.18%，所以这两个候选阶段工作量不同。不能用相邻轮候选时间直接判定FA3
内核退化，也不能把已裁剪保留的policy token当作GPU实际生成token。

候选小batch也没有必然的FA3收益。同引擎容量、相同prompt、固定128个greedy token
的独立测试中，active batch16时FA3比XFormers慢18.6%；batch32和64分别少用约6.3%
和14.0%时间。这是限定形状测试，没有正式训练batch直方图，不能套用到全部rollout。

## 已完成轮次的实测

| 完整轮次 | 候选生成 | Actor更新 | 含完整断点写入／封存和轮间工作的整轮时间 |
| --- | ---: | ---: | ---: |
| 52，旧实现 | 1093.305秒 | 288.139秒 | 1991.125秒（33分11秒） |
| 53，lean实现 | 1141.550秒 | 165.714秒 | 1925.389秒（32分05秒） |
| 55，lean＋紧凑GQA＋native前缀解码 | 998.870秒 | 157.305秒 | 1756.665秒（29分17秒） |
| 56，同上 | 1016.930秒 | 145.983秒 | 1775.756秒（29分36秒） |
| 57，同上 | 944.780秒 | 144.034秒 | 1679.849秒（28分00秒） |
| 58，同上 | 946.155秒 | 148.111秒 | 1691.419秒（28分11秒） |
| 59，同上 | 1049.086秒 | 154.797秒 | 1805.860秒（30分06秒） |
| 61，新2d版本去剩余梯度日志扫描 | 1029.825秒 | 165.098秒 | 1793.499秒（29分53.5秒） |
| 62，同上 | 1004.599秒 | 148.878秒 | 1763.475秒（29分23.5秒） |
| 63，同上 | 1015.511秒 | 145.977秒 | 1759.587秒（29分19.6秒） |
| 64，同上 | 1031.303秒 | 148.158秒 | 1783.906秒（29分43.9秒） |
| 65，同上 | 1082.992秒 | 153.460秒 | 1849.393秒（30分49.4秒） |
| 68，当前2d版本 | 1072.590秒 | 151.640秒 | 1836.716秒（30分36.7秒） |
| 61—65均值 | 1032.846秒 | 152.314秒 | 1789.972秒（29分49.97秒） |

第55轮比第53轮整轮少8.76%，比第52轮少11.78%。候选占第55轮整轮约56.9%，
actor更新约9.0%；优势计算202.775秒、checkpoint写入162.291秒也仍在整轮里。
这些是实际轮次观测，问题、policy和节点可能不同，不能将差值当作严格同输入因果实验。
第55、56两轮完整均值1766.211秒，即29分26秒。第56轮选中485／512道混合题，
比例94.73%；四卡实际Adam累计1100→1120，scheduler各前进一次。第56轮完整
checkpoint为91,405,594,385字节，manifest SHA为
`21454de9cf93cf9a7e45907e273b74522b0cf52fecaec75a5b0ef7f0fd3d0d04`。

第57轮四卡各完成20次Adam，累计1140，完整断点91,405,594,386字节已封存，
随后第58轮开始。第55—57轮整轮平均1737.423秒，即28分57秒，比旧第52轮观测
耗时少12.74%；仍不将不同轮问题和policy下的差值当作同输入因果收益。第57轮
选中443／512道混合题（86.52%），训练EM日志四舍五入为58.1%，reward均值0.753。

第58轮同样完整结束，四卡累计1160次Adam、scheduler58，完整断点
91,405,594,898字节，manifest SHA为
`c2a5b82307b918eee3bfee8e1d95d918e7ab56d3d1d6156e7313e86d85bc217a`。
第58轮选中480／512道混合题（93.75%）。随后第59轮四卡各完成20次真实更新，
累计1160→1180，scheduler各推进一次，0额外monitor forward；完整断点
91,405,595,410字节已封存，manifest SHA为
`e412b2a50829fea49347f71a313206485ceb4cee5a82f33ea674be790f20ed70`。
第59轮选中490／512道混合题（95.70%），训练EM日志四舍五入为53.9%，reward
均值0.697。第60轮随后也完整完成：四卡各累计1200次Adam、scheduler60，
完整断点91,405,595,410字节已封存，manifest SHA为
`b1dbae539f4a1e1513293b930009bac19f84a83358c90d3d427958860ee31ed5`。

第55—59轮完整均值按连续生成边界计算：
`(mtime60 - mtime55) / 5 = (1792694894.188066 - 1792686184.637950) / 5`
`= 1741.910秒`，即29分02秒，较旧第52轮观测耗时少12.52%。这个口径包括完整
断点写入、封存和轮间工作，不能用日志里的`candidate_fill + timing_s/step`替代；
仍不将不同轮问题和policy下的差值当作同输入因果收益。这是旧749598在完成60轮
后取消前的历史计时，不能当作750029新runtime的同输入速度对照。见
[五轮完整计时与第59轮完整断点](evidence/native-prefix-five-complete-outers-summary.json)。

新750029首个完整第61轮壁钟为
`1792700013.084385 - 1792698219.585013 = 1793.499372秒`，即29分53.5秒；
边界来自第61与第62轮实际`0-before-generation.pt`。候选1029.824673秒，
actor165.098秒，adv206.271秒，checkpoint写入158.326秒。四卡各完成20次Adam，
累计1220、scheduler61，额外monitor forward为0；post norm为null／measured=false。
候选五阶段同步仍为`[1,0,0,0,0]`，实际FA3执行有记录。完整23个checkpoint文件
加7个边界文件共91,405,595,915字节已封存，manifest SHA为
`05272cf13d4a69ae4fababbfeb69143f5d0aaad6dd0c7db59bcf7fcd23739ef6`。
选题混合464／512（90.625%），训练EM日志约56.3%、reward0.732，均不是固定
验证集结果。这是首轮历史记录，后续62—65已完整完成。首轮不同数据和policy，
不能证明同输入因果提速，更不能把最后约3秒clip组件补丁称为整轮倍增。见
[新版本首个完整第61轮](evidence/leanmetrics-first-complete-outer61-summary.json)。

当前恢复后五轮61—65已连续完整并sealed，第66轮实际生成保存512题，
`post_restore_five_outers_accepted=true`。四卡各新增100次Adam，累计1300／
scheduler65；五轮均无额外monitor forward，最新write的post norm均为
null／measured=false。每轮候选五阶段同步均为`[1,0,0,0,0]`，候选及actor实际
FA3、紧凑Q28／KV4均有实测记录。CP65的23个checkpoint文件和7个边界文件
共91,405,597,196字节已封存，manifest SHA为
`1bb578ccd19217265e6046656920a30dac972d668b5f7bc5574e02a025a5501c`。

五轮完整均值1789.972171秒（29分49.97秒），候选均值1032.846189秒
（17分12.85秒，占57.7%），actor均值152.3142秒（2分32.31秒，占8.51%）。
这比旧55—59轮历史均值29分02秒略长约48秒，policy和数据不同，不构成最后
约3秒clip补丁的整轮因果提升或退化证明。约3秒组件收益不应表述为整轮倍增。

| 轮次 | 选中混合题／512 | 训练EM日志 | 训练reward均值 |
| --- | ---: | ---: | ---: |
| 61 | 464 | 56.3% | 0.732 |
| 62 | 472 | 55.3% | 0.731 |
| 63 | 467 | 57.1% | 0.747 |
| 64 | 469 | 56.7% | 0.751 |
| 65 | 487 | 54.3% | 0.712 |

五轮选题混合比例加权为92.1484%。这些均为训练数据统计，不能称作固定验证集
准确率或最终质量。五轮恢复跟进通过不代表整个200轮目标完成。见
[五轮完整更新、计时与第66轮实际生成](evidence/leanmetrics-five-complete-outers-summary.json)。

发现的另一个CPU开销是反复经过Hugging Face层进行前缀解码。对同一1024条已有
action交替运行三次，HF解码8.5425ms/action，native backend解码0.4803ms/action，
CPU组件速度比17.787倍。37,907条实际action的截断和保留token全部一致，48项行为
测试通过。这个17.787倍不能称为整轮训练提速。

生产第55轮实际报告确认：候选prefill／decode和actor前向／反向均执行FA3，
actor为Q28／KV4；每卡五个候选阶段的权重同步次数为 `[1,0,0,0,0]`；每卡额外
监控前向为0。没有在健康生产进程里增加新的性能监控前向。

## 继续定位候选阶段：串行单条检索

从61—65轮现有日志进一步拆分，整轮均值1789.97秒中：候选1032.85秒、
actor152.31秒、adv210.54秒、checkpoint写入154.92秒；其余239.36秒尚未
独立归因，包含封存和轮间工作，不能全部称为CPU或磁盘耗时。候选的异步phase
本身892.38秒，也包含生成、工具等待、CPU处理和模型转换，并非纯GPU内核时间。
每轮生成12,800条轨迹、约39,155次生成请求和26,351次逻辑检索；实际用于
actor更新的是选中的2560条轨迹。见
[现有日志分段计时](evidence/leanmetrics-phase-breakdown61-65.json)。

源码存在明确的串行链路：客户端每次`queries=[query]`，本地单event loop代理
同步等待HTTP，远端也在主线程同步执行encoder／FAISS。`retrieval_batch_size=512`
是上限，不能把单条请求自动聚成512条。16个客户端检索线程并未形成GPU批量检索；
远端每次还执行原始`empty_cache()`。五块候选依次处理，块内轨迹则独立异步推进，
不能误称为每次搜索都做四卡全局barrier。

只读现有`/health`三次，没有添加检索请求或GPU工作。连续85.95秒的自然训练窗口中，
1434次逻辑检索有473次缓存命中，961次实际后端调用累计39.796秒，即平均
41.41毫秒／次、串行后端忙碌时间占窗口46.3%。上游请求增量与本作业代理cache
miss增量均为961；两个子窗口分别约72.8%和32.5%，说明随生成交错而变化。
这证明串行检索是实际的大项候选，尚不能把46.3%称为整轮可消除的耗时或GPU利用率。
下一步独立重放相同已保存查询，测小批检索与单条检索，并分别核对ID、顺序和分数。
健康训练与原检索服务保持运行。见
[自然训练计数差值](evidence/leanmetrics-natural-retrieval-window-summary.json)。

独立作业750360在ACD1-20两张H100完成，Slurm `COMPLETED / 0:0`，用时2分07秒。
使用同一128条已保存查询、原E5和21,015,324条Flat索引、原四逻辑shard映射
`[0,1,0,1]`；两轮按batch大小正反顺序计时。没有调用正式检索服务、加载actor
或修改训练。128条完整编码＋FAISS的中位耗时分别为：batch1 5.257秒、batch8
0.760秒、batch16 0.533秒、batch32 0.259秒。全部12组的384个top3文档ID与顺序
一致；批量编码后的最大score差约0.000499。

其中单条基线的encoder均值约1.031秒、FAISS约4.221秒，说明此组件实测的大项是
FAISS逐条扫描。复用单条encoder向量的FAISS-only对照中，batch16相对单条为
8.65倍，最大score差1.79e-7；batch32为14.67倍、最大差1.07e-6，文档顺序均相同。
这仍是128条查询的组件实验，不是全量数值保证，也不是整轮20倍。下一次扩大至
512条，并实际测“E5逐条编码＋仅FAISS合批”，避免批量E5自身的数值变化。正式服务
保持原样。见[独立检索分段结果](evidence/retrieval-batching-r1-final-summary.json)。

后续保守对照750363也已在ACD1-4两张H100完成，`COMPLETED / 0:0`，用时4分49秒。
512条查询的E5均保持逐条`encode([query])`，只合并CPU float32向量进行FAISS。
两轮正反顺序、全部12组都保持1536个top3文档ID和顺序相同。

| FAISS batch | 512条的完整编码＋检索中位耗时 | 相对单条组件速度比 | 最大score差 |
| --- | ---: | ---: | ---: |
| 1 | 21.363秒 | 1 | 0 |
| 8 | 5.570秒 | 3.84倍 | 2.38e-7 |
| 16 | 4.650秒 | 4.59倍 | 2.38e-7 |
| 32 | 3.783秒 | 5.65倍 | 1.13e-6 |

当前客户端同时最多16个检索请求，因此优先实验batch上限16；不能假设生产能填满32。
这个对照没有HTTP、排队、corpus查文档或actor生成，也不能证明所有未来查询的ID永远
一致。下一步需要私有异步服务／代理及同一policy的完整候选阶段对照。默认batch16、
最多等待2ms的独立队列组件已准备并通过2个fake异步测试，尚未接入正式服务或训练。
已验证的组件收益不会直接乘到整轮时间上，也不会据此立即取消健康allocation。见
[512条逐条E5保守实测](evidence/retrieval-batching-r2-final-summary.json)。

## CPU回滚快照的独立实际耗时

作业 `749739` 在独立的 `ACD1-25` 四卡节点完成，Slurm `COMPLETED / 0:0`。
使用同一CP50和第51轮已有数据，保留旧版全部训练工作；只在该诊断进程中包裹原
快照构造函数计时，原始CPU复制逻辑不变。四卡均实际完成20次快照、20次Adam、
80次训练前向和旧版21次监控前向。正式训练进程未改动。

| Rank | 20次快照构造耗时 | 每轮复制参数＋Adam字节 |
| --- | ---: | ---: |
| 0 | 121.770秒 | 456,936,993,040 |
| 1 | 122.764秒 | 456,936,993,040 |
| 2 | 122.778秒 | 456,936,993,040 |
| 3 | 118.969秒 | 456,936,993,040 |

每次复制22,846,849,652字节，即约22.85GB；20次约456.94GB／卡／轮。计时外的
预先CUDA同步累计每卡只有1.5–1.7毫秒。四卡并行，不能把四行耗时相加当作整轮
壁钟耗时；也没有测量异常回滚或快照释放。此重复旧版测试的全部固定工作耗时
273.788秒，节点和计时包装与原266.079秒实验不同，不能精确相减分摊其他检查
的独立耗时。直接测得的119–123秒说明CPU快照确实是旧版更新阶段的大项。

## 剩余梯度日志扫描：实际恢复与五轮跟进已通过

复核发现旧lean虽已关闭所有有限性检查，仍为日志计算max_abs并重扫裁剪后全部
梯度范数。源码`c14bedd9e897e4bed51de9c693a1ef29302b436d`让lean跳过这些
测量，保留原FP64 norm、裁剪系数与逐元素cast／multiply／copy顺序。未测的
post norm记录为null和measured=false，CPU证据reader也不再冒称它是实测值。
默认legacy行为不变。59项focused测试通过，含20次真实toy Adam更新逐位一致。

独立单卡作业749784在H100完成，退出0。使用旧实测报告中的29个本地分片尺寸、
共1,903,904,128个FP32梯度元素（7,615,616,512字节），生成合成梯度，不加载
权重或checkpoint。新旧裁剪对全部29个分片逐位一致；6次交替测量中位wall从
0.199648秒降到0.087261秒。20次clip的组件投影节省2.248秒，**不是整轮提速**。
正式训练有CPU offload，因此这个单卡CUDA合成梯度实验当时不能代替实际FSDP
梯度放置与通信测量；随后复用现有native4脚本，补齐了实际梯度配对证据。

最终独立四卡配对作业749913在ACD1-6完成，Slurm `COMPLETED / 0:0`，elapsed
5分02秒。使用相同CP50和第51轮512×5条已有轨迹，每卡真实完成20次Adam和
80次训练前向、0次额外monitor forward。四卡梯度实测均为各自`cuda:rank`上的
29个FP32 tensor，共1,903,904,128个元素、7,615,616,512字节。每卡20次配对中，
**全部完整梯度和裁剪前norm均逐位一致**；新裁剪结果随后用于真实Adam更新。

| Rank | 旧clip：20次body总耗时 | 新clip：20次body总耗时 | 组件减少 |
| --- | ---: | ---: | ---: |
| 0 | 5.008秒 | 2.087秒 | 2.921秒 |
| 1 | 5.014秒 | 2.034秒 | 2.980秒 |
| 2 | 5.017秒 | 2.080秒 | 2.936秒 |
| 3 | 5.017秒 | 2.007秒 | 3.010秒 |

clip body约2.40—2.50倍，但正式每outer只有20次clip，对应约3秒／outer的组件
收益量级，**不是整轮2.5倍**。计时包括clip调用及末尾CUDA同步，不含预同步、
备份／恢复复制和完整梯度比较；预同步另计，旧版每卡20次累计0.084—0.086秒，
新版0.180—0.183秒。四卡时间不能相加当wall；也不能将包含双clip与完整梯度克隆的
142.392秒诊断总时间当作生产提速。此证据证明已测20次真实梯度的裁剪结果一致，
不是两个独立完整训练的全部Adam状态或最终QA质量相等的证明。见
[实际四卡全梯度配对与计时](evidence/clip-native4-paired-final-summary.json)。

下一版源码`2d5183455551145c52d1783a38663e2fcf170ceb`及独立运行环境已准备。
完整60断点后，按人类2026-10-22 13:34:40UTC的既有条件授权，于03:17:44CST
取消旧749598；03:18:35回读终态为`CANCELLED by14090 / 0:0`。随后既有CPU
恢复读取从03:18:51运行至03:23:30，约279秒，通过
`EASY2560_NATIVE4_FULL_SHA_CPU_STATES_PASS`，receipt SHA为
`6651c5ff154eb43fb644e3bc754dbf2536bf67a6e77360b9d09955b81de51cd5`。
完整模型／Adam／scheduler／RNG／数据边界保留，不从零训练或重置Adam；CPU
读核验通过不等于新四卡已实际恢复。

唯一新作业750029已于2026-10-23 03:33:27CST在ACD1-14分配四张H100、
48CPU、512G，分区emergency_acd、7天。03:35实测RUNNING且runtime2d身份匹配，
随后实际native4恢复的combined结果已为true：四卡模型集体回读一致，Adam／
scheduler／RNG逐卡回读全部一致，rank0逐个核对339个model tensor，观察到四个
不同CUDA UUID。各卡恢复Adam1200／scheduler60，driver和重建数据流游标均为
153600且dataequal。随后61—65已连续完整更新并封存，第66轮已实际生成保存
512题，五轮跟进验收通过。恢复及跟进通过不证明最终QA质量或同输入整轮提速。见
[实际CP60四卡恢复](evidence/leanmetrics-actual-cp60-native4-restore-summary.json)。
检索745198保持运行；
独立绑定750029的100／150／200归档pair，CPU core640634／companion640635
同birth身份实际live且轮询推进，无错误。见
[完整断点切换进展](evidence/leanmetrics-transition-progress.json)与
[目标环境准备回执](evidence/leanmetrics-target-preparation-1792690562356391000.json)。

最后补丁只有约3秒／轮的组件收益，不值得为此失去已有allocation再排队；
**这次切换时机判断错误**。完整状态已保留，但不能用局部clip的2.5倍掩盖实际
排队代价。新版本五轮实际完整耗时已测得，但没有同输入整轮因果提速结果。

一次SSH观测通道超时／认证前断连已经保存原始小型错误回执；随后原固定路由恢复，
当时749812仍为PENDING(Resources)，未观察到诊断或生产作业故障。没有据观测超时
取消、重启或重复提交作业，也未修改全局SSH配置和凭据。

为避免仅因排队重复提交，又读取旧四卡诊断749089／749739的Slurm资源记录。
两者实际成功完成，elapsed分别6分38秒／6分49秒，均申请4GPU／48CPU／512GB；
主job、batch、extern的MaxRSS字段全部为空。当时749812也申请该资源，仍为
PENDING(Resources)。空RSS不是0，无法据此证明512GB明显过大或是唯一排队原因，
因此没有盲目降低内存、重提或取消现有作业。见
[调度与内存记录](evidence/clip-native4-resource-usage-1792691692195606000.json)。

进一步读取具体排队限制发现：749812预计05:01才可在ACD1-8启动，而被排除的
ACD1-13当前调度显示空余4GPU、67CPU及足够512G内存。旧749590失败是一次分配
4GPU却只看见3GPU的真实异常，不能证明节点永久故障；actor-only同样需要真实
四卡，若异常重现会在set_device／NCCL失败。仅对现有独立诊断749812清空
ExcNodeList，命令返回0，随后回读为(null)。没有重提作业，没有改变4GPU／48CPU／
512G／30分钟和CP50／selection51／20次真实更新；生产749598不受此变更影响。
这次调度更新尚不证明CUDA四卡或数值验收，需继续读取实际测试输出。精确操作与
回执见[调度更新](evidence/clip-native4-historical-node-update-1792693491575892000.json)。
Slurm清空该字段的语法来自[官方scontrol说明](https://slurm.schedmd.com/scontrol.html)。

解除排除后，749812确实调度到ACD1-13，但25秒后FAILED／1:0。四rank均在
CUDA初始化报unknown error，available devices变为zero，actual_clipping_calls=0，
未执行任何Adam／配对裁剪。这不能冒称裁剪数值失败，也不能认定本次仍是旧的
三卡可见模式；只证明该次allocation无法初始化CUDA。完整失败root与小型报告
保留。随后只恢复这个真实失败的独立诊断，新job749913、新root paired-r2；
恢复默认排除ACD1-13，复用相同2d51834源码、CP50、selection51、四卡和20次实际
更新，首次实际回读为PENDING(Resources)。没有重启或热改生产749598／检索745198。
见[真实失败与唯一恢复记录](evidence/clip-native4-node-failure-and-recovery-summary.json)。
恢复作业749913已完成上述实际四卡配对验收；749812仅保留为历史失败证据。

随后独立节点健康短测750051在ACD1-13真实`FAILED / 1:0`，用时8秒，四rank
均读到`cudaDeviceCount=3`，虽然`CUDA_VISIBLE_DEVICES=0,1,2,3`。没有进入
model／forward／NCCL；此结果保留，不改变正式750029对该节点的排除。
其余健康节点实测最多3张空GPU，没有可同时提供4张空GPU的健康节点，因而没有
降低CPU／内存请求制造无用改动。见
[节点13实际健康短测](evidence/node13-health-final-summary.json)。

## 数值与证据范围

前缀解码优化不改模型前向、loss、reward、采样和截断逻辑。紧凑GQA已有独立输出／
梯度对SDPA验证；BF16多步训练不宣称逐位一致。固定20步实验中的参数／Adam是
小型采样比较，不能扩充为全部参数／Adam或最终reward完全相等。实际恢复验证与
跨内核训练数值比较是两个不同证据范围。

为量化重复运行本身的差异，复用了本次legacy测量与原legacy的小型rank报告，
没有新跑GPU：四卡初始参数／Adam样本、RNG、数据及80次前向token数一致，
20步后比较如下。两种比较都只覆盖报告样本。

| 20步后样本指标 | 旧legacy→lean＋compact | 旧legacy→同legacy重复 |
| --- | ---: | ---: |
| LP样本相对L2，四rank范围 | 0.04864–0.06627 | 0.05199–0.08124 |
| Policy loss相对L2，四rank范围 | 0.00325–0.00509 | 0.00323–0.00485 |
| Gradient norm相对L2 | 0.07924 | 0.05119 |
| 最终参数样本最大绝对差 | 2.4984e-5 | 2.2870e-5 |

这说明跨作业／节点和计时包装下，同旧实现重复20步也不是逐位复现；不能把之前
所有差异单独归因compact。但这不是完整梯度、完整Adam或最终QA等价的证明，
也不能用小型参数样本误差替代完整张量误差。原始Adam报告只有样本hash，无法
从hash不同计算moment的数值误差大小。

原始小型证据：

- [正式第55轮完整状态与计时](evidence/native-prefix-deployment-and-first-outer-evidence.json)
- [两轮完整计时与第56轮完整断点](evidence/native-prefix-two-complete-outers-summary.json)
- [三轮完整计时与第57轮完整断点](evidence/native-prefix-three-complete-outers-summary.json)
- [四轮完整计时与第58轮完整断点](evidence/native-prefix-four-complete-outers-summary.json)
- [五轮完整计时与第59轮完整断点](evidence/native-prefix-five-complete-outers-summary.json)
- [实际四卡全梯度配对与计时](evidence/clip-native4-paired-final-summary.json)
- [完整60断点与续训排队状态](evidence/leanmetrics-transition-progress.json)
- [实际CP60四卡恢复](evidence/leanmetrics-actual-cp60-native4-restore-summary.json)
- [新版本首个完整第61轮](evidence/leanmetrics-first-complete-outer61-summary.json)
- [新版本五个完整轮次与第66轮实际生成](evidence/leanmetrics-five-complete-outers-summary.json)
- [节点13实际健康短测](evidence/node13-health-final-summary.json)
- [此前候选实际工作量与内核形状测试](evidence/fa3-production-speed-cause-20261022.json)
- [四卡20次回滚快照的独立计时](evidence/rollback-snapshot-measured-summary.json)
- [同legacy重复20步的样本差异](evidence/rollback-snapshot-legacy-repeatability-summary-20261023.json)
- [原legacy与lean＋compact的样本差异](evidence/fastpath-native4-comparison-1792678397691460000.json)

checkpoint、optimizer、完整rollout和模型始终保留远端，本地只保存代码与小型审计结果。


## 私有完整候选对照的启动记录（2026-10-23）

正式750029及检索745198持续运行。私有双卡服务750396在ACD1-4加载原始
21,015,324条索引与文档后READY，保留逐题E5，仅将FAISS向量合批16。
固定的HF权重为当时最新完整CP66，五个父题chunk也已hardlink固定；没有复制
optimizer，也不在Mac存放权重或raw轨迹。独立四卡A/B使用相同policy和父题，
顺序冷缓存运行旧串行proxy和新async proxy，最终各生成2560×5条真实轨迹。

三个独立测试入口错误均保留了真实证据：750398的Ray空闲worker预启导致线程
耗尽；750418的worker PYTHONPATH遗漏driver依赖overlay；750419已实际完成
四卡引擎初始化，随后输入指纹记录中的NumPy数组无法JSON序列化。修复分别
复用正式NPROC8192并减少空闲线程、完整继承driver导入路径、仅在记录时转换
NumPy值。三次均未产生actor/optimizer更新，没有重启正式训练。

源码bf15a8c的独立750423已分配ACD1-32四卡继续原对照。此记录不是候选完整
阶段或正式整轮提速验收；需实际完成两组轨迹，并同时报告生成token、请求、
检索动作与工作量差异。

## 私有startup512检索合批实测（2026-10-23）

750423的startup两组均实际完成并退出0。两组使用相同512道父题、2560条轨迹输入，
输入指纹一致，均从空的LRU4096缓存开始。生成阶段从179.024秒降至165.799秒，
省13.225秒，耗时减少 **7.39%**（1.080倍）；包含输入读取、评分和输出保存的
pipeline从205.888秒降至193.017秒，耗时减少6.25%。小型原始量化摘要见
[startup512检索合批摘要](candidate-startup512-retrieval-batching-summary.json)。

async组实际生成token为1,072,779，对照为1,056,148，增加1.57%；prompt token
增加1.70%，生成请求7890→7942，逻辑检索5330→5381。异步调度下生成轨迹和
工作量已有差异，因此这是同输入的真实流水线实测，不能称为逐token相同的固定
工作量对比；该startup也尚未执行完整2560候选中的Top512选择。

实际FAISS平均batch为3688／2442＝ **1.51**，距离配置上限16仍有差距。服务
`max_batch_input_queries`读值为累计highwater11；它不是可相加的计数，后续不能
拿其delta推断某个新variant的最大batch。两组均无失败、无缓存淘汰；缓存命中
1753→1693，命中率32.89%→31.46%（减少1.43个百分点），后端请求反而从3577
增加到3688（增加3.10%）。async完成缓存有3615项，后端调用多出73次，符合
保留“不合并inflight”时允许重复后端工作的语义。

后端计时delta虽然为154.523→113.790秒，但旧fallback包含文档读取，新batch
执行计时不包含文档读取／格式化及请求排队，两者不能直接作为严谨的后端速度比。
7.39%是本次一个startup块的生成省时，不能写成正式训练整轮同幅提速。

截至原始观测时间1792711113.4979742，完整五块对照尚未完成：full-legacy已实际
完成四卡引擎初始化，full-async尚无完成结果。16ms等待窗口的独立startup对照已
准备；本文未核验其实际排队、运行或完成状态，不作推断。正式750029与检索
745198没有因本次证据整理而改动。


### Actual generated tokens versus retained policy tokens

The formal `raw_generated_policy_tokens` label is misleading: `mixed_runtime.py`
sums the final sampling mask, which contains only retained model prefixes. The
private benchmark separately counts original RequestOutput tokens and retained
model tokens. Startup legacy generated1,056,148 raw and retained333,199; async
generated1,072,779 and retained334,217. This is a statistics-definition difference
between reports, not evidence that the benchmark uses a larger max_tokens or
a different rollout budget. Both run the original500-token turn requests and
4096 total budget; the parent sampling configuration and parser are shared.

The unused tail remains real GPU work in both pipelines. A private optional
search-boundary stopping experiment is being implemented, not deployed to formal
training. It must preserve the complete boundary token and original logprobs.
The original parser prefers any later `</search>` over `</answer>`, so the first
version may stop only at a confirmed search boundary; stopping on an early answer
would change that rule. EOS/generation-config provenance must remain truthful.
The generated/retained ratio does not imply the same training-outer speed ratio.
See [actual token counters](evidence/candidate-raw-retained-token-accounting.json).


## 私有完整2560候选检索合批实测完成（2026-10-23）

后续实测确认750423在ACD1-32完成，Slurm为 **COMPLETED，退出码0:0**，
总作业时间44分41秒；两组完整五块均各完成2560道父题、12,800条真实轨迹。
完成观测时间为2026-10-23 07:54:22（CST），取代上文较早快照中的“五块尚未完成”。
详细计数、输入指纹及实际Slurm行见
[完整2560候选最终摘要](evidence/candidate-retrieval-full2560-final-summary.json)。

生成阶段893.886→841.931秒，省51.955秒，耗时减少 **5.8123%**；包含输入读取、
评分和输出保存的pipeline为1032.909→980.517秒，省52.392秒，耗时减少
**5.0723%**。async实际FAISS平均batch为18,320／12,335＝ **1.4852**；
两组proxy的failed_requests与backend_exceptions均为0。

完整两组使用相同父题和固定policy，输入指纹相同，仍产生了不同的随机轨迹：
raw生成token为5,343,550→5,358,907（增加 **0.2874%**），生成请求39,536→39,617
（增加 **0.2049%**），prompt token增加0.4216%，逻辑检索26,724→26,808。
缓存命中率32.62%→31.66%，命中数8718→8488，后端请求18,006→18,320。
因此这里报告候选阶段的实测耗时变化，不把它推成正式训练整轮的因果加速倍率。

Easy选中的混合组分别为438／512（85.55%）与422／512（82.42%）。这是上述
两次随机生成的选题质量样本，既不是固定验证集准确率，也不能据此断言最终模型
性能下降或质量等价；本实验没有执行actor训练和最终51,713题验证。

另有[真实token的CPU search-stop回放](evidence/search-action-stop-saved-cpu-replay-summary.json)：
15,832个实际action、591,019个保留LP精确一致，可避免原raw token的68.39%；
这是固定已存token的CPU结果，独立GPU实验750449当时仍待完成，不是GPU提速验收。

## 16毫秒合批窗口的实际候选测试

独立作业750433在ACD1-20四张H100完成，`COMPLETED / 0:0`，共8分36秒。
两个fresh进程读取同一policy66、同一512题及相同采样配置，均完成2560条轨迹；
仅第二组使用16ms等待窗口的私有异步检索。候选生成174.703→159.506秒，
实测少8.699%；含输入读写和评分200.535→186.421秒，少7.038%。实际FAISS
平均batch1.974，最大13；raw token为1,051,668→1,049,603，少0.196%，
prompt token多0.434%，逻辑检索5288→5319，后端失败均为0。

等待16ms使平均合批量比之前2ms约1.49有所提高，但仍远小于上限16。
这是512题启动块测量，节点和轨迹与2ms实验不同，不能将其外推为正式整轮收益，
也不能宣布16ms相对2ms的严格因果增益。正式训练未改动。见
[16ms完整结果摘要](evidence/candidate-retrieval-wait16-completed-summary.json)。

## Search提前停止的显式正式配置（尚未切换）

新增可选 `fa3-lean-search-stop-v1`，继承旧lean全部设置，仅增加
`actor_rollout_ref.rollout.search_action_boundary_stop=true`。现有prepare、
CPU恢复、四卡提交入口透传此profile；旧默认保持。wrapper安装在每个vLLM
实例构造后，正常FSDP权重同步或KV cache释放不会重建该实例。仅在现有phase
日志附加轻量停止计数，无额外forward、梯度扫描或监控进程。

共23项独立CPU测试已通过，包括实际启动argv除新flag外一致、新旧profile、
四卡/200轮/batch参数、原StopChecker优先级和同一恢复入口透传。本机没有
Hydra/CUDA，以上不是正式GPU新profile验收；当前750449独立四卡实测仍待完成。

## Search提前停止的实际四卡测速已完成

独立750449在ACD1-20四张H100完成，`COMPLETED / 0:0`，共6分12秒。
两组用同一冻结policy66、相同512题、相同采样与async2ms检索配置，均
实际完成2560条轨迹，四rank均有FA3 prefill/decode调用。唯一开关是search边界停止。

- 候选生成158.192→94.030秒，少40.559%，速度比1.682。
- 包含输入读写、评分、保存184.048→120.846秒，少34.340%。
- raw token1,059,962→338,192，少68.094%；实际保留335,198→335,494。
- prompt token多0.225%，生成请求多0.291%，逻辑检索5344→5367。
- 四rank实际search停止1329/1342/1365/1332次，共5368次；无模糊前缀、
  确认拒绝或未释放的sequence filter。LP/token未改写；原StopChecker先执行。

CPU旧token回放的逐项精确性与此处实际GPU速度是两份不同证据。新采样会改变
异步调度及后续随机数消费，所以不承诺两次随机rollout逐条相同。这里证明
候选块上的明显观测收益，未证明正式整轮40.56%提速或最终准确率相同。
正式续训保留现有745198检索配置，整轮收益须在恢复后的完整outer实测。见
[实际四卡停止计数和完整A/B](evidence/search-action-stop-live-native4-summary.json)。

## 为什么不能只把提前停止直接搬进当前串行检索训练

匹配生产串行代理的独立750454也已完成：ACD1-20四H100，
`COMPLETED / 0:0`，共7分46秒。相同policy66和512题，两组都完成2560轨迹。
候选180.8668→163.9175秒，少9.371%；含读写评分207.7075→190.8220秒，
少8.130%。raw token1,050,464→335,516，少68.060%；保留内容长度接近。
全部四rank实际FA3，search-stop共5344次，无候选错误。

与另一次async2ms实验的40.559%候选降幅相比，串行组的decode调用数几乎
未变（增加0.101%），async组则下降69.444%。这支持串行检索等待／背压限制
活跃生成、吞掉提前停止收益的判断；没有把独立两次调度的差额当作严格单机制
因果证明。

因此先前已prepare但尚未提交的stop-only r1明确作废为未激活历史。保持健康
750029/745198，组合接入已测async CPU代理和单条E5＋FAISS16/2ms服务，
再在后续实际恢复后的完整outer测壁钟，避免再次为小组件收益交出健康allocation。
见[两次实际pair完整口径](evidence/search-action-stop-serial-vs-async-summary.json)。

## Combined production integration (formal CUDA restore and whole-outer timing pending)

The explicit fa3-lean-batched-search-stop-v1 profile inherits the measured
search-stop settings and adds only the async CPU retrieval proxy. The candidate
benchmark and production now share the same independent-opener HTTP and owner-loop
completed-cache implementations. Their three implementation ASTs match the
measured candidate code; no in-flight request merging or GPU work runs on HTTP threads.

A separate two-H1007day service reuses the existing environment/E5/index/corpus,
keeps singleton E5, and batches only FAISS16/2ms. It publishes actual stage64 and
READY evidence with truthful rebatching=True. Existing745198 and default profiles
remain unchanged; the existing continuation entry accepts an explicit new receipt.

The28 relevant CPU tests passed. These are wiring tests, not formal CUDA restore
or complete training-outer speed acceptance. Forward/loss/reward/sampling/four-GPU
geometry/200-no201 are unchanged.

## Combined target prepared, old production continues

Actual seven-day2H100 service750471 onACD1-4 is READY at08:26CST, with
64 original stage queries completed and64 rows each of document IDs, scores
and document SHA. Singleton E5 and true FAISS16/2ms settings are recorded.
Both new service and training immutable runtime dde1452 have all2007 source
files/26,475,549 bytes SHA-verified remotely. The combined target passed the
existing prepare path with explicit750471 receipt at08:27CST.

No source stop or new training submission has occurred. Production750029
sealed69 and saved1536 questions of70; preserve its current work until a
complete sealed boundary>=70. Whole-outer gain, actual new native4 restore
and five completed follow-up outers remain pending. Existing745198 and archive
consumers remain untouched. See
[prepared target and actual service result](evidence/combined-batch-search-stop-prepared-summary.json).
