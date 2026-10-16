"""Curate QDP settings, warmups, diagnostics and execution attempts separately."""
import json, hashlib, re, datetime
from pathlib import Path
from collections import Counter
def redact(s):
 return re.sub(r'hf_[A-Za-z0-9]+','[HF_TOKEN_REDACTED]',s)

D=Path(__file__).resolve().parent
C=Path('/Users/shaoshitong/Documents/Codex/2026-10-02')
S=Path('/Users/shaoshitong/Documents/Search-G1')
P=S/'preparation'
G=C/'users-shaoshitong-documents-remote-codex-control'/'work'
T=C/'users-shaoshitong-documents-remote-codex-control-2'/'outputs/search-g1-prefix1024-diversity512'
O=C/'users-shaoshitong-documents-remote-codex-control-5'/'outputs'
CO=C/'users-shaoshitong-documents-remote-codex-control-6'/'outputs'
R=S/'qprobe-prefix20-online1000-h1008-20261016/results'
SESS={'G1':'01a0fb1b-d3ff-74b3-aa13-49ca57871f92','Gi':'01a0fcbb-863f-7450-926d-db537f648265',
 'Tree':'01a0fcbc-0b46-7502-8b66-d8b21a0e032d','TOPD':'01a0fcbe-15f8-7600-ae84-9487d924560a',
 'Co':'01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef','design':'01a11c5d-303b-7441-8db7-c645df252713'}
READ_CACHE={}
def raw_once(p):
 p=Path(p)
 if str(p) not in READ_CACHE:READ_CACHE[str(p)]=p.read_bytes()
 return READ_CACHE[str(p)]
def load(p):return json.loads(raw_once(p))
def source(p):
 p=Path(p)
 if not p.exists():return {'path':str(p),'exists':False}
 raw=raw_once(p);return {'path':str(p),'exists':True,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

GI_STATE=load(G/'search-g1-actorhidden-period20-state.json')
TOP_STATE=load(O/'STATE.json')['dynrpo_native_qwen_hpc3']
TREE_REWARDS=load(T/'TRAINING_REWARD_HISTORY.json')
def find_outer_values(v):
 if isinstance(v,dict):
  for k,x in v.items():
   if k=='outer' and isinstance(x,int):yield x
   if isinstance(x,(dict,list)):yield from find_outer_values(x)
 elif isinstance(v,list):
  for x in v:yield from find_outer_values(x)
GI_COMPLETE=GI_STATE['complete_outer']
TOP_COMPLETE=TOP_STATE['completed_outer']
TREE_COMPLETE=max(find_outer_values(TREE_REWARDS))
experiments=[]
def add(id,owner,name,kind,method,progress,result,paths,identity='',caveat=''):
 experiments.append(dict(id=id,owner=owner,session_id=SESS.get(owner),name=name,kind=kind,method=method,
 progress=progress,result=result,identity=identity,caveat=caveat,sources=[source(p) for p in paths]))

add('D01','design','Question-only QDP 原始设计与探索比例消融','方案／CPU实现',
 '冻结官方模型第27层末有效question-token表示；LN→MLP64→K=0..5六分类；CE6+0.1均值MSE+0.1归一化样本std MSE；按期望样本std选题。E=0/64/128/256四档，入训均512×5。',
 '已形成方案、实现与CPU检查；不能把四档都称为已跑完的RL对照。',
 '已运行的E=64和E=0分支分别见下；没有找到E=128/256独立正式训练结果。',
 [S/'qprobe-teacher-warmup-a800/docs/question-diversity-probe.md',D/'01a11c5d-303b-7441-8db7-c645df252713-chronology.json'],
 '原实现734f05ead727430e96178f7aa23ddd0c57d38226；branch codex/search-g1-question-diversity-probe')
add('D02','Co','Pairwise ranking／RankNet 增补建议','未执行方案',
 '若std拟合改善而Top命中率未改善，再加入同policy高std题应排在低std题之前的排序loss。',
 '找到建议文本；未找到独立run或实际loss启用凭证。','不能当作已做过的消融。',
 [D/(SESS['Co']+'-chronology.json')])

add('C01','Co','HPC3 小MLP 原始预热','warmup',
 '5120题×5 frozen-teacher；MLP64，135622参数；原六分类＋均值/std辅助；2epoch，batch256。',
 '40次HeadAdam完成；actor更新0。','同一缓存可复用，但不同loss的head不能冒称同一warmup。',
 [CO/'qprobe-warmup-hpc3/handoff.json'], 'run c3856315d3f345d7920ebf5725a083b4；恢复后主job719225')
add('C02','Co','HPC3 原始MLP QDP 正式RL','RL',
 'Top448＋随机64；每outer入训512×5；在线recent16 replay，head每outer2epoch。',
 '完成117/200后按人工新loss指令停止。','固定512：iter50 35.9375%（184）；iter100 41.2109375%（211）。',
 [CO/'qprobe-warmup-hpc3/handoff.json',D/(SESS['Co']+'-chronology.json')],
 'job719225；run c3856315d3f345d7920ebf5725a083b4','人工更换设置，不标为数值崩溃；早期启动失败见附录。')
add('C03','Co','HPC3 σ3-loss 预热','warmup',
 '相同MLP64；CE3＋1×normalized σ MSE＋0.1 CE6，取消mean MSE；5120×5，2epoch。',
 '40次HeadAdam完成。','warmup总loss1.1206166；不是QA EM。',
 [CO/'qprobe-sigma3-loss-20261015/handoff.json'], 'job721485；run70827e85fe584f0c84453aae907918a4')
add('C04','Co','HPC3 σ3-loss 正式RL','RL',
 'λσ=1，Top448＋随机64；recent16×512组，2epoch，head batch256；其余保留Search-G1 shaped reward。',
 '200/200训练与51713题评分完成；清理／归档验收尚未全部通过。',
 '固定512 iter200 41.9921875%（215）；终评20949/51713＝40.5101%，七集等权32.9841%。',
 [CO/'qprobe-sigma3-loss-20261015/handoff.json',D.parent/'session-refresh-20261017-pinned/jxu957-current.json',D.parent/'session-refresh-20261017-pinned/all-seven-results.json'],
 'job721485；run70827e85fe584f0c84453aae907918a4',
 '包装器报trainer left owned subprocesses；评分完成不等于全部收尾验收。handoff旧进度已由实际终评读回覆盖。')

add('A01','G1','A800 原始teacher warmup','warmup',
 '冻结teacher5120×5；question-only MLP六分类＋均值/std；2epoch／40HeadAdam。',
 '首次采样快照契约失败；修复与缓存复用后的实际warmup完成。','用于后续micro2／micro4／async分支；warmup不算actor outer。',
 [P/'qprobe-teacher-warmup-a800-deployment.json',P/'qprobe-micro4-superseded00028-terminal.json',P/'qprobe-async-top512-terminal-authority-before-fullouter-restore.json'])
add('A02','G1','A800 question-only QDP micro2','RL／已停止',
 '四卡actor micro2版本；原六分类QDP；不是后来micro4正式新run。',
 '1完整outer；第二轮6次部分Adam；合计26次／rank，人工换micro4。',
 '首轮reward0.2495036；无可冒称的200轮结果。',
 [P/'qprobe-micro4-superseded00028-terminal.json'],'run00028e46b70b40a2ba6980b1c988f92c','人工改micro配置；不把部分outer算完整。')
add('A03','G1','A800 question-only QDP micro4','RL／故障退出',
 '四卡micro4／global16；保持mini128及20Adam/outer；原六分类head。',
 '完成4个outer，随后故障；旧run保留。','未找到该run可用的固定512里程碑成绩。',
 [P/'qprobe-micro4-terminal-authority-before-async-top512.json'],'run34fffb9fe8ad4d75b54b9c9747c256fb；base d8569124…')
add('A04','G1','A800 async Top512 原始MLP','RL／恢复后已停止',
 '候选5120→Top512，无随机64；整outer采样权重同步、逐轨迹状态机；保留原六分类loss。',
 '首执行完成5后native校准identity故障；同run恢复链后达到38完整outer，再按人工换loss。',
 '固定512 iter5 EM24.609375%；32个普通outer计时曾被独立分析。',
 [P/'qprobe-async-top512-terminal-authority-before-fullouter-restore.json',P/'qprobe-async-restore5-terminal-metadata.json',D/(SESS['G1']+'-chronology.json')],
 'run1f34290a70454871aea9d368dc8a0b36；a365→0065（被替代修复）→af113恢复',
 '恢复与目标回放不是新算法；原5轮checkpoint、继承计数与新写入分开。')
add('A05','G1','A800 σ3-loss λσ=3 预热','warmup',
 'CE3＋3×normalized σ MSE＋0.1 CE6；相同小MLP；5120×5，2epoch40更新。',
 '真实warmup完成。','总loss约1.6003；不能与Co的λσ=1直接称同设置。',
 [P/'qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json',S/'repo/reproduction/reports/qprobe-sigma3-f847-hour01.json'])
add('A06','G1','A800 σ3-loss λσ=3 正式RL','RL／已停止',
 'Top512，无探索；teacher-v4修复后正式训练。','至少3完整outer，下一outer有部分更新；后续人工切10epoch warmup。',
 '无完整固定512里程碑可据此排名。',
 [P/'qprobe-warmup10-oldf847-terminal-cleanup-accepted-archive.json',D/(SESS['G1']+'-chronology.json')],
 'runf84749ca43be42bd91f2bc63d73347dd；base86cf…','前两个失败namespace/c528与1573单列为执行失败。')
add('A07','G1','A800 σ3-loss warmup10','warmup',
 '沿用小MLP／σ3监督，把同5120题预热延长到10epoch／200HeadAdam。',
 '10/10完成。','训练题二分类accuracy69.86%；不能当QA准确率。',
 [P/'qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json',S/'repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.md'],
 'runf4fb19b1fede4dffaac823b8937f35cf')
add('A08','G1','A800 σ3 warmup10 后正式RL','RL／已停止',
 '与A07同run，预热后从官方actor开始。','2完整outer、40成功Adam／rank、0skip；人工换架构，资源清理验收通过。',
 '无该分支固定512里程碑。',[P/'qprobe-warmup10-f4fb-terminal-cleanup-accepted-archive.json'],
 'runf4fb19b1fede4dffaac823b8937f35cf','不是训练故障，不能误报未启动或仍运行。')
add('A09','G1','全question-token MLP128＋4-head attention warmup10','warmup',
 '全部有效question token→MLP128→单层4-head attention→补到244后flatten；BCE(mixed)+0.3 normalized std MSE。',
 '10epoch／200更新，exit0；没有正式actor训练。',
 'loss0.765503→0.587306；训练accuracy41.54→72.23%；Top512真实混合39.26→79.69%，epoch8曾84.38%。',
 [S/'repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.md',S/'repo/reproduction/reports/qprobe-token-binary-warmup10-fulltoken-results.json'],
 '完成base494eafd18b004cb980121e46b188db44',
 '不是完整Qwen block。epoch8权重没单独保存；两次失败/缓存复用在附录。')
add('A10','G1','完整原生Qwen最后block question-only warmup10','warmup',
 '冻结encoder layer27所有question hidden→完整复制原生layer35→masked flatten244→Linear2；BCE mixed+0.3 normalized std MSE。',
 '10epoch／200HeadAdam完成。','训练accuracy99.8828125%（5114/5120），BCE0.0366394；Top512全mixed。',
 [S/'repo/reproduction/reports/qprobe-native-qwen-warmup10-096c-training-scalars.json',P/'qprobe-teacher-warmup-a800-progress-handoff.json'],
 'warmupbase096c8297cba74f8089063e763686671e；Head SHA9b32f40e…',
 '这是同一训练题拟合，不是泛化；目标文件名有warmup20，不得覆盖实际10epoch/200计数。')
add('A11','G1','原生Qwen question-only 正式RL','RL／已停止',
 'Top512；每outer head2epoch／recent8 replay、指数年龄权；只导入独立探针，不改actor初始官方权重。',
 '1完整outer＋第二轮10部分Adam，总30／rank；人工停止后新512诊断。',
 '单轮主流程约21分30秒；无5轮QA EM可报。',
 [P/'qprobe-teacher-warmup-a800-progress-handoff.json',P/'qprobe-native-human-stop-new512-diagnostic-20261016.json'],
 'run509c391b7e5c4174bfb55f724f41a3d6；execution60fca3cc2edd4dedb3773230b91f34ea',
 '多个启动修复只复用同一已完成warmup，不能每次算新10epoch实验。')
add('A12','G1','Head200 CPU复现＋全新512题G5泛化诊断','诊断',
 '独立复现旧Head200训练集表现、比对formal初始权重、同题G5标签审计；新512题不与预热/正式候选重叠；不训练actor/head。',
 '新512×5真实轨迹全部完成。','旧训练99.88%；新题mixed分类58.7891%（301/512），低于总预测非mixed基线64.2578%。',
 [P/'qprobe-native-new512-cpu-weight-compare-20261016.json',P/'qprobe-teacher-warmup-a800-progress-handoff.json'],
 '新512base16034e59031e4f21a323ebb2556bef7f',
 '分类目标是G5是否混合，不是单轨迹QA EM；std输出全0的诊断亦保留。')
add('A13','G1','A800 question＋Prefix5 单轨迹BCE，balanced warmup20','warmup／未跑满',
 '原生Qwen block＋flattenLinear1；每条轨迹自己最终EM；batch256强制128对128错有放回，lr3e-4；20epoch目标2000Adam。',
 '实际6完整epoch＋第7轮91次，合计691Adam后人工停止；旧不平衡fit从未启动。',
 '新题轨迹判别balanced accuracy最好epoch4 58.4280%，epoch6 51.7827%。',
 [P/'qprobe-prefix-correctness-balanced-warmup20-human-intent-20261016.json',P/'qprobe-prefix-correctness-balanced20-deployment-20261016.json',P/'qprobe-teacher-warmup-a800-progress-handoff.json'],
 'baseaa4a9ebd63ac4683985bf74a748dcfb2','文件旧名prefix20，但实际policy prefix是5；训练分类指标不是QA EM。')

add('H01','Gi','H100 三个原生Qwen block，σ3 warmup2','warmup',
 'question全token表示；3个完整Qwen decoder block；CE3+1 normalized σ MSE+0.1 CE6；5120×5；2epoch40HeadAdam。',
 '真实warmup完成，仅保存最终统计，没有逐mini曲线。',
 '总loss1.23408，对照常数1.23985；CE3 .90922，std RMSE .23872，改善很小。',
 [G/'qprobe-state-pre-binary-normalization-20261015.json',G/'qprobe-stdloss3399-terminal-archive-v1.json'],
 'jobs3399/3400；后者runc025948ebc0947f69d86af84871a9237')
add('H02','Gi','H100 三block σ3 正式RL','RL／故障链后改设置',
 'Top448＋随机64；actor8H100，probe单GPU7更新；frozen官方question编码。',
 '该阶段完成首5轮及dev5；3399/3400均留有故障终态。',
 '固定512 dev5 EM26.3671875%（135/512）。',
 [G/'qprobe-state-pre-binary-normalization-20261015.json',G/'qprobe3400-terminal-and-warmup10-fresh-submit-v2.json'],
 '3396/3397等初始启动→3399→3400','初始六分类计划后来被σ3替代；未把仅计划的配置冒称独立完成。')
add('H03','Gi','H100 三block mixed-BCE warmup10','warmup',
 'BCE(mixed)+0.1 normalized sample-variance MSE；注意辅助项为variance，不是A800的0.3 std；5120固定题。',
 '10epoch200HeadAdam完成，teacher mixed1893/5120。',
 '最终BCE约.5146、总loss.5267、训练accuracy77.13%、Top448混合92.41%。',
 [G/'qprobe3402-warmup10-training-curve-summary-v1.json'],
 'job3402；runf77c3245dcfe49a88a5a99b1dec161e6')
add('H04','Gi','H100 三block mixed-BCE 正式RL','RL／已停止',
 'actor八卡，probe GPU7单卡FP32/micro4；replay每outer2epoch；第14轮96HeadAdam，6144串行micro反传。',
 '停止前账本口径不齐：18轮完整耦合已严证、19轮console、actor更新达20而probe19；不计为20完整耦合outer。',
 '探针更新普通轮平均约6分29秒；第14轮外层523.5秒。',
 [G/'qprobe-h1008-transformer3-state.json',G/'qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json',D.parent/'h100-qprobe-vs-mixed-latency-20261016/REPORT.md'],
 'job3402；runf77c3245dcfe49a88a5a99b1dec161e6','更换前缀设置后人工停止；未把摘要的19/20口径混为完成轮数。')
add('H05','Gi','Question-only vs Question＋Prefix5 原拟议AB','未执行方案',
 '两臂比较问题表征与问题＋5生成token；原拟10epoch。','随后人工改为Question＋10/20两臂；未发现该AB的完整GPU训练产物。',
 '不能补造两臂准确率。',[G/'qprobe3402-prefix5-stop-summary-20261015T174356636271Z.json',D/(SESS['Gi']+'-chronology.json')])

for n,acc,train,bce in [(10,53.613533,96.21484375,2.3099149),(20,54.0186526,95.6796875,2.8185778)]:
 add(f'P{1 if n==10 else 2:02}','Gi',f'Question＋Prefix{n} balanced warmup20','warmup／消融臂',
  f'question＋本轨迹前{n}个policy token；冻结encoder27→原生block35＋flattenLinear1；单轨迹BCE，exact1:1采样；每臂4H100/global256。',
  '20epoch／每rank2000HeadAdam；actor0；exit0。',
  f'epoch20原始train accuracy{train:.4f}%；原始test balanced accuracy{acc:.4f}%；test BCE{bce:.4f}。',
  [S/'qprobe-prefix10-20-h1008-20261016/results/prefix10-20-h1008-20261016/FINAL.json'],
  'job3405；base94cc1d4','raw与prior-corrected指标分开，不报告成QA EM。')
for n,acc,auroc in [(10,54.269645,.5996751),(20,53.033009,.5832956),(40,52.444618,.5786141),(80,56.728612,.7042186)]:
 add({10:'P03',20:'P04',40:'P05',80:'P06'}[n],'Gi',f'Prefix-only{n} balanced warmup20','warmup／消融臂',
  f'encoder仍看到完整question上下文，但probe只收到前{n}个生成位置hidden；原生block＋flattenLinear1；每臂2H100/global256；单轨迹BCE。',
  '20epoch／每rank2000HeadAdam；actor0；exit0。',
  f'epoch20原始test balanced accuracy{acc:.4f}%，AUROC{auroc:.4f}。',
  [S/'qprobe-prefix-only-four-h1008-20261016/results/prefix-only-four-h1008-20261016/FINAL.json'],
  'job3406；base58d687c','不能把prefix-only说成无问题信息；其hidden已条件于question。')
add('P07','Gi','Prefix-only20 在线换题 warmup1000','warmup／提前换协议',
 '冻结actor，每epoch5120×5新真实轨迹、仅训probe；全pool轮转；实际global256/local32、100Adam/epoch。LR从3e-4在CP7后改3e-5。',
 '账本停在71epoch／7100Adam每rank；第70轮已有独立不可变CP与全SHA核验；不是1000轮完成。',
 'epoch70派生G5 mixed test balanced accuracy58.5987%、普通accuracy59.5703%；epoch68权重被导出给Co。',
 [G/'prefix20-online1000-state.json',R/'prefix20-online1000-lr3e5/MIXED-G5-THROUGH70-SUMMARY.json'],
 'run8a6b24dc19664cb3a3fb97101f21c54e；jobs3407→3408→3409',
 '实际batch不是原计划25600；CP7实际LR3e-4→3e-5，不是3e-5→3e-6。70与停训时71口径分开。')
add('P08','Gi','Question＋Prefix20 replay70','warmup／数据协议消融',
 '从旧71轮Head/Adam状态转入question＋20表示，lr3e-6，回放70份已有G5数据；每iteration100HeadAdam。',
 '70/70完成；前6轮只有test，后64轮有配对train/test；14100累计Adam／rank；exit0及CP验收通过。',
 '最终train mixed balanced73.9199%、test60.9107%；最好test64.9410%在iteration30。',
 [R/'question20-replay70/FINAL-ACCEPTANCE-SUMMARY.json',R/'question20-replay70/BALANCED-ACCURACY-SUMMARY.json',G/'prefix20-question-replay70-state.json'],
 'run80cb2d8592304bc197614be9b04911c9；jobs3410→3411','这不是fresh actor RL，也不是70个新outer。')
add('P09','Gi','Question＋Prefix20 joint BCE100','warmup／监督消融',
 'fresh probe；逐轨迹EM BCE＋同题G5 mixed BCE，权重1:1；h=1−∏p−∏(1−p)；lr3e-5 cosine→3e-6。',
 '100/100epoch，每rank10000HeadAdam，actor0；10/20/…/100保存；最终验收通过。',
 '最好观测test mixed balanced62.9619%在epoch45，但无该权重；最好实际保留CP是epoch100，59.8759%。',
 [R/'question20-groupjoint100/FINAL-ACCEPTANCE.json',R/'question20-groupjoint100/FINAL-METRICS-100.json'],
 'job3412；runcee20bcd24194b79a9a8a88809bd5886；HeadSHA8df6bd44…',
 'epoch68 HF权重来自P07，不是本实验；这个区别会影响后续初始化对比。')

add('L01','Gi','Question＋Prefix20 冻结encoder Top512 RL','RL／已停止',
 'P09实际保留CP100初始化head；officialactor fresh；5120×5前缀→预测Top512→续写原前缀；选中2560轨迹jointBCE训head1epoch/lr1e-6，每outer10HeadAdam。',
 '21完整outer后人工停止；无偏随机候选未进入head训练引发选择偏差担忧。',
 '最后selected trainEM7.34375%；固定512 iter5 EM24.21875%（124）。',
 [D.parent/'session-refresh-20261017-pinned/h100-8-current.json',G/'prefix-selection200-resumed6-10-runtime-summary.json'],
 'run5cba22ca409a4571a1e6b87d89c86b09；jobs3413/3414/3415/3419/3420恢复链',
 '输入是独立冻结官方encoder hidden，不能说成当时actor hidden；QA曾有Triton布局编译失败。')
add('L02','Gi','当前actor-hidden Prefix20＋period20 RL','RL／进行中',
 '当前actor layer27捕获question＋20 hidden，采样21丢第21以对齐；5120候选Top512；每20outer才用全5120×5训probe1epoch（100HeadAdam），其余冻结。',
 f'本次本地实际状态记录{GI_COMPLETE}完整outer；head更新账本{GI_STATE.get("probe_adam_per_rank")}；早前远端冻结审计严证8。',
 '固定512 iter5 EM23.6328125%（121）。',
 [G/'search-g1-actorhidden-period20-state.json',G/'actorhidden-period20-first-five-actual-facts.json',D.parent/'session-refresh-20261017-pinned/h100-8-current.json'],
 'runa008d7ce983347bbb6e81051b25a7c4e；3423→3424→3425',
 '前两次分别存到1/2轮后storage校验失败，恢复属于同run，不是三个科学实验。')
add('L03','Tree','Prefix Top1024→真实diversity Top512 RL','RL／进行中',
 'P09 CP100 head；先预测Top1024并完整续写，再按实际hard-EM population variance挑512入训actor；head用全部1024×5标签，20更新/outer。',
 f'本次本地原始reward回执已到{TREE_COMPLETE}；早前冻结远端审计严证32。',
 '固定512 iter5 EM24.609375%（126）；训练EM因真实diversity二次筛选而不可当测试EM。',
 [T/'CURRENT_CONFIG.json',T/'TRAINING_REWARD_HISTORY.json',T/'STATUS.json',T/'TERMINAL3421_SMALL.json'],
 'runf3ee1435e5014ba1969387c91878893a；job3422；之前3421存到5',
 '与Gi单纯预测Top512不同；已发生恢复/评测Triton故障，保留原失败证据。')
add('L04','TOPD','HPC3 独立原生Qwen QDP warmup10／每outer刷新原方案','warmup／被新协议替代',
 '独立head/Adam；question-only原生Qwen最后block，BCE mixed+std；原计划warmup10后每outer2epoch/replay8，Top512。',
 '新5120特征准备/初始warmup有执行；旧正式job723510零actor更新时被后续period25设置替代。',
 '不能报成一个已完成200轮的TOP-D蒸馏实验。',
 [O/'dynrpo-independent-20261015/deployment-plan.json',O/'dynrpo-independent-20261015/hpc3-native-submission.json',O/'dynrpo-independent-20261015/periodic25-change.json'],
 '723434准备；723510旧正式取消；base native-thread5-20261015',
 '这是TOP-D会话承载的Search-G1 QDP，名称不代表使用TOP-D loss；原方案与正式period25分开。')
add('L05','TOPD','Periodic25 原生Qwen QDP 初始及周期warmup','warmup／周期refit',
 '0/25/50/75/100/125/150/175边界用当前student5120×5重新标注，head20epoch；不重置block/readout/Adam；200只存评测不refit。',
 '边界0/25/50完成；每次400HeadAdam，累计1200。','后续新边界是同一RL的refit阶段，不是独立fresh模型实验。',
 [O/'dynrpo-independent-20261015/periodic25-change.json',O/'dynrpo-periodic25-20261016-r16/refit25-latest-sample.json',O/'STATE.json'],
 'run1cd20aedc5e94511be8ea8b7db76933c；currentjob725286')
add('L06','TOPD','Periodic25 原生Qwen QDP 正式RL','RL／恢复中继续',
 'Top512/no random；fourGPU；question-only head只按period25刷新，actor每outer20Adam。',
 f'本地STATE最近记录{TOP_COMPLETE}完整outer；此前原始远端冻结读回68。',
 '固定512 iter5 EM25.390625%（130）；iter50 36.9140625%（189）。',
 [O/'STATE.json',O/'dynrpo-periodic25-20261016-r16/latest-sample.json',D.parent/'session-refresh-20261017-pinned/mou176-current.json'],
 'canonical1–5来自r13，6–10来自r15，11+来自r16；currentjob725286',
 'r15曾完成但未保存的outer11被放弃，从CP10重做；不累计两次11。17个部署目录不是17种算法。')
add('L07','Co','HPC3 Prefix-only20＋全对组虚拟0 RL','RL／多次启动失败后新提交',
 '导入P07 epoch68，而非P09 CP100；head只接20生成位置；逐轨迹BCE，不含group joint BCE；selected512×5每outer训head1epoch/lr1e-6；只EM全对组加虚拟0。',
 '当前727586仅有提交凭证，完整outer0；此前727414真实payload失败，不能算训练已完成。',
 '没有该新run可报告的固定5轮EM；旧dev0不是训练后成绩。',
 [CO/'qprobe-prefix20-hpc3-4gpu-20261016/handoff.json',CO/'qprobe-prefix20-hpc3-4gpu-20261016/727414-actual-payload-failure-and-zero-write-evidence-1792170376.json'],
 '727586；base qprobe-teacher-warmup200-prefix20-joint4-recovery727414-20261017',
 '名称含joint4，但实际不是逐轨迹BCE＋groupBCE联合监督；虚拟0是另一算法改变。')

add('N01','G1','Prefix5 scale20 teacher数据／特征构建','数据构建',
 'frozen官方teacher；20×5120题／102400题／512000条轨迹；第一份旧5120复用，额外97280题；保存prefix5与各自最终EM，不更新actor/head。',
 '200chunk完成；最终数据CPU审计和HF追加验收通过。','此数据阶段不是200outer RL，也不是200epoch probe拟合。',
 [P/'qprobe-teacher-warmup-a800-progress-handoff.json',P/'qprobe-prefix-scale20-final-audit-1792141000881306000.json'],
 'datasetbuildbase0850db2977bd4801b9ced300b0ab387f；run62407ad0ed404586998439ca1f55b902',
 '前期c6缓存scope adapter失败与修复单列；历史prefix20数据追加后统一裁到5。')
add('N02','G1','Question＋Prefix20 EMA joint200 原方案','已实现CPU／未GPU执行方案',
 'question＋20；轨迹BCE＋groupBCE 1:1；EMA0.99评估；目标200epoch。',
 'CPU核心已实现和发布；合并数据只有新prefix5，后由人工改为全部Prefix5；无该20-token独立GPU训练run。',
 '无完成epoch／测试准确率。',[P/'qprobe-teacher-warmup-a800-progress-handoff.json'],
 'current_question20_ema_joint_warmup','不能用后续5-token成绩填到20-token方案下。')
add('N03','G1','Question＋Prefix5 EMA0.99 joint200 全量warmup','warmup／进行中',
 '合并去重后445440个G5事件组／2227200轨迹；全部prefix5；nativeblock＋flattenLinear1；轨迹BCE＋mixed-groupBCE权重1:1，EMA0.99；每epoch全数据一遍。',
 '4完整epoch，epoch5活动；global256改1024时从CP2完整状态恢复；actor更新0。',
 'epoch4 EMA：train mixed balanced约84.75%，test约56.57%；单轨迹train/test accuracy约83.50/77.03%；泛化gap明显。',
 [P/'qprobe-teacher-warmup-a800-progress-handoff.json',P/'question5-global1024-resume-deployment-20261016.json',P/'question5-global1024-epoch4-training-window-publication-20261017.json'],
 'run8af926a49630428097b94027864da68b；executionbcfa1de0a90a4fcda70263a75153fcee',
 '445440是采样事件组，不等于445440个唯一问题；离线probe分类指标不进QA榜。')
add('N04','G1','A800 actor-hidden Prefix5＋period20 新正式RL','准备／尚未开始训练',
 '参考Gi actorhidden20；改question＋5；四卡5120→Top512；导入切换时实际已完成EMA head、fresh在线Adam与officialactor。',
 '截至本次本地记录，新v2源码已stage、CPU prepare待完成；completeouter0；旧EMA warmup尚未停止。',
 '无新actor训练／dev5成绩。',[P/'a800-actorhidden-prefix5-human-intent-20261017.json',P/'qprobe-teacher-warmup-a800-progress-handoff.json'],
 '原准备run8e6ced4e…因rotary buffer loader假设失败；新v2 run411c44aad7ba4e8c90f5fbec9e5ecbaa',
 '实际current_authoritative以新v2为准；只是prepare尝试，不把它叫新RL已经跑起来。')

# Execution attempts. These are separate from settings/stages above.
attempts=[]
def attempt(owner,label,status,reason,paths,progress=''):
 attempts.append(dict(owner=owner,session_id=SESS.get(owner),attempt=label,status=status,reason=reason,
   progress=progress,sources=[source(p) for p in paths]))

g1_notes={
 'qprobe-teacher-warmup-a800-deployment.json':('失败','首批teacher QPROBE_POLICY_SNAPSHOT_CHANGED_DURING_GENERATION；不能据此声称actor漂移','formal0'),
 'qprobe-teacher-warmup-a800-snapshot-repair-deployment.json':('启动失败／后续替代','c52b preclaim/空闲GPU/端口门禁，保留原始原因未篡改','formal0'),
 'qprobe-a800-flex-deployment.json':('人工换micro配置','micro2已写26Adam／rank；其中仅1完整outer','1＋部分6Adam'),
 'qprobe-a800-micro4-deployment.json':('故障退出','实际FAILED_34FF；多项failure_diagnostics保存，细分原因见终态原文件','4完整outer'),
 'qprobe-a800-async-top512-deployment.json':('故障退出','5轮后native calibration identity绑定失败','5完整outer'),
 'qprobe-a800-async-top512-restore5-deployment.json':('修复候选被替代','0065 restore候选不等于新fresh模型实验；以后续fullouter恢复为准','父CP5'),
 'qprobe-a800-async-fullouter-restore5-deployment.json':('同run恢复／后人工换loss','完整状态恢复CP5，后原MLP链达到38轮','累计38'),
 'qprobe-a800-sigma3-loss-deployment.json':('启动失败','c528 preclaim；无trainer／formal更新','0'),
 'qprobe-a800-sigma3-fresh-start-fix-deployment.json':('启动失败','1573 teacher cache namespace mismatch','0'),
 'qprobe-a800-sigma3-teacher-v4-deployment.json':('人工换warmup10','f847 teacher-v4实际warmup与3轮正式已执行','≥3完整＋部分'),
 'qprobe-a800-sigma3-warmup10-deployment.json':('人工换架构／清理通过','f4fb：warmup200HeadAdam，正式2完整outer后停止','2outer/40Adam'),
 'qprobe-token-binary-warmup10-deployment.json':('打包失败','第一次源码打包失败，不能当Head已训练','Head0'),
 'qprobe-token-binary-headfit-resume-deployment.json':('完成','前一a799 encoder完成但allocator-zero检查失败；494复用缓存freshHead完成10epoch','200HeadAdam'),
 'qprobe-qwen-lastblock-warmup10-deployment.json':('启动失败','c83d native metadata JSON键绑定错误，Head未训练','Head0'),
 'qprobe-qwen-json-binding-repair-deployment.json':('完成','096c修复后完整Qwenblock warmup10完成','200HeadAdam'),
 'qprobe-qwen-native-formal-deployment.json':('CPU prepare失败','aa975，未启动actor/encoder/数据复制；原warmup保留','actor0'),
 'qprobe-qwen-native-formal-cpu-repair-deployment.json':('CPU prepare失败','002a修复候选再次prepare失败；实际原因保存于failure-preserved','actor0'),
 'qprobe-qwen-native-formal-cpu-repair-v2-deployment.json':('启动失败','92bb完整bank已合格，但prelaunch/入口失败；缓存不重做','actor0'),
 'qprobe-qwen-native-driver-process-fix-deployment.json':('启动失败','ff1196 driver/process及bank契约问题；保留失败并过渡v2','actor0'),
 'qprobe-qwen-native-driver-process-fix-v2-deployment.json':('启动失败','4fd508正式启动失败，exact argv终态诊断保留','actor0'),
 'qprobe-qwen-native-selection-metrics-fix-deployment.json':('人工停止','成功开始原生formal；1完整outer＋10部分Adam，非30outer','1outer/30Adam总'),
 'qprobe-prefix-correctness-warmup20-deployment-20261016.json':('旧采样协议未拟合','unbalanced fit未启动，被balanced sampler替代；encoder产物复用','Head0'),
 'qprobe-prefix-correctness-balanced20-deployment-20261016.json':('人工停止／未跑满','6epoch＋91次部分更新；保留partial state','691HeadAdam'),
}
for n,(st,reason,pr) in g1_notes.items():
 p=P/n
 if not p.exists():continue
 d=load(p); label=d.get('base',n).split('/')[-1]
 attempt('G1',label,st,reason,[p],pr)
for n,st,reason in [
 ('qprobe-token-binary-a799-failed-encoder-complete-archive.json','encoder后失败','全token特征5120完成；allocator-zero断言退出，Head0；494后续复用'),
 ('qprobe-prefix-scale20-source-publication-20261016.json','数据链CPU修复','原c6缓存scope adapter失败；修复后0850新执行完成scale20，不重复计teacher数据'),
 ('question5-global1024-resume-deployment-20261016.json','同run恢复','global256→1024，从CP2载入Head/Adam/EMA/RNG；非重新200epoch'),
 ('a800-actorhidden-prefix5-human-intent-20261017.json','准备失败后v2','8e6c原prepare loader假设错误；v2 411c准备中，未训练actor')]:
 if (P/n).exists():attempt('G1',n,st,reason,[P/n])

coh=load(CO/'qprobe-warmup-hpc3/handoff.json')
for f in coh.get('failed_jobs',[]):
 # failed_jobs has appeared both as a list and a job-keyed dictionary.
 if isinstance(f,str):
  obj=coh['failed_jobs'][f];label=f
 else:obj=f;label=str(f.get('job',f.get('job_id','unknown')))
 attempt('Co',label,'故障退出／历史',redact(json.dumps(obj,ensure_ascii=False))[:2300],[CO/'qprobe-warmup-hpc3/handoff.json'])
for j in ['718650','719225','721449','721485']:
 attempt('Co','HPC3 '+j, {'718650':'早期执行／随后恢复','719225':'117轮后人工停止','721449':'启动失败','721485':'训练及评分完成，收尾异常'}[j],
  {'718650':'原MLP恢复链，勿与719225重复算独立设置','719225':'实际主run c385，切换新loss','721449':'原启动失败保留；721485承接新σ3设置','721485':'200outer及51713评分已完成；原始归档/cleanup故障仍未全部验收'}[j],
  [CO/('qprobe-warmup-hpc3/handoff.json' if j in ['718650','719225'] else 'qprobe-sigma3-loss-20261015/handoff.json')])
for j,reason in [('726927','pristine启动失败；初始环境/门禁诊断保留'),('727000','启动失败；shared JIT相关诊断'),('727050','启动失败；cross-backend numeric gate相关诊断'),('727148','缺joblib等启动依赖失败'),('727251','Ray线程资源/启动门禁失败；后移除额外smoke后重试'),('727414','真实payload失败；错误base/协调器绑定已留证；无完整outer'),('727586','纠正payload后单次提交，未读到完整outer')]:
 fs=[p for p in (CO/'qprobe-prefix20-hpc3-4gpu-20261016').glob('*.json') if j in p.name and re.search('fail|submit|terminal|payload|recovery|pristine',p.name)]
 attempt('Co','Prefix20 '+j,'提交待实测' if j=='727586' else '故障退出',reason,
   [CO/'qprobe-prefix20-hpc3-4gpu-20261016/handoff.json']+fs[:3], '0完整outer')

topdirs=sorted(O.glob('dynrpo-periodic25-20261016*'),key=lambda p:int(p.name.rsplit('-r',1)[1]) if '-r' in p.name else 0)
topreasons={0:'GPU preflight失败',1:'模型初始化失败；详细诊断见model-initial-failure',2:'retriever readiness退出',3:'retriever／校准失败，附cleanup分区提交失败与最小资源重试',4:'overlay差异／prebind失败',5:'native process birth门禁失败',6:'trainer启动失败',7:'GPU采样BlockingIOError资源暂不可用',8:'trainer启动失败',9:'model coexistence门禁失败',10:'trainer失败',11:'trainer失败',12:'KeyError selected_predicted_diversity，正式outer0',13:'完成1–5；后trainer故障，从CP5恢复',14:'CP5恢复尝试未成功；之后转r15',15:'完成6–10可恢复；11虽完成未存；12前Ray线程池资源不足，cleanup又ENOSYS38',16:'从CP10恢复，重做11；当前实际推进中'}
for folder in topdirs:
 p=folder/'formal-submission.json'
 if not p.exists():continue
 d=load(p);ds=[]
 for line in d.get('stdout','').splitlines():
  try:ds.append(json.loads(line))
  except:pass
 job=next((x.get('job') for x in ds if isinstance(x,dict) and x.get('job')),None)
 num=int(folder.name.rsplit('-r',1)[1]) if '-r' in folder.name else 0
 fs=[p]+list(folder.glob('*failure*.json'))+list(folder.glob('*terminal*.json'))+list(folder.glob('latest-sample.json'))
 attempt('TOPD',f'Periodic25 r{num} / job{job}','恢复后进行中' if num==16 else '失败或被恢复替代',topreasons[num],fs,
  {13:'canonical1–5',15:'canonical6–10；11舍弃',16:f'canonical11以后；本次本地{TOP_COMPLETE}'}.get(num,'完整outer0／见原receipt'))
for j,reason in [('723411','独立native方案的准备尝试；actual native前置准备'),('723434','独立native准备job；新5120特征与warmup'),('723510','旧每outer2epoch正式方案，0actor时被人工改period25取消'),('723619','初始warmup10→20准备延续'),('723670','period25 preparation提交'),('724148','r3清理debug分区拒绝后改acd_u最小资源'),('724250','辅助收尾／检查作业，不是独立RL'),('724403','r7失败后的cleanup'),('725032','r16 CPU恢复检查job，未训actor')]:
 attempt('TOPD',j,'准备／辅助／被替代',reason,[O/'dynrpo-independent-20261015/periodic25-change.json',D/(SESS['TOPD']+'-history-hits.json')])

gijobs={
 '3396':('初始三block重连核查只返回Invalid job id；原native终态缺失，不能确定数值失败原因','qprobe3-job3396-reconnect-terminal-v1.json'),
 '3397':('旧启动失败，原terminal保留','qprobe-stdloss-old-terminal-check-v4.json'),
 '3399':('σ3三block warmup2链故障退出','qprobe-stdloss3399-terminal-archive-v1.json'),
 '3400':('首5轮及dev5后失败；被warmup10 fresh方案替代','qprobe3400-terminal-and-warmup10-fresh-submit-v2.json'),
 '3401':('新warmup10原尝试，trainer/owned cleanup异常，formal验收未完成','qprobe3401-binary-replacement-final-actual-v1.json'),
 '3402':('binary warmup10成功，后人工转prefix试验；actor/probe计数不齐','qprobe-h1008-transformer3-state.json'),
 '3403':('旧question-only／prefix5 AB计划被10/20方案替代，不能视作完成','prefix10-20-warmup20-state.json'),
 '3404':('10/20初始准备退出，后3405完成','prefix3404-terminal-resource-1792088260549484000.json'),
 '3405':('Question+10/20两臂20epoch完成','prefix10-20-warmup20-state.json'),
 '3406':('Prefix-only10/20/40/80四臂20epoch完成','prefix20-online1000-state.json'),
 '3407':('online1000第2轮frozen rollout失败：GPU UUID被当整数physical_device_id解析，ValueError；不是Head loss故障','prefix20-online-terminal3407-20261015T194713083944Z.json'),
 '3408':('完成7epoch后人工LR降10倍，从CP7恢复','prefix20-online1000-state.json'),
 '3409':('继承至71epoch后人工换question+20 replay协议','prefix20-online1000-state.json'),
 '3410':('replay最初6轮仅test，人工重启补配对train/test','prefix20-question-replay70-state.json'),
 '3411':('配对replay完成70/70，继承原6轮，不重复计','prefix20-question-replay70-state.json'),
 '3412':('jointBCE100完成验收','question20-groupjoint100-state.json'),
 '3413':('Prefix20正式初始执行，后恢复；不得与3414等合并称独立算法','prefix-selection200-state.json'),
 '3414':('Prefix20早期恢复尝试；真实细分原因见保存的恢复记录','prefix-selection200-state.json'),
 '3415':('Prefix20 resume2已提交；后续恢复到3420','prefix-selection200-resume2-submit-20261016T121739966237Z-actual.json'),
 '3419':('Prefix20后续恢复；当前链随后3420','prefix-selection200-state.json'),
 '3420':('继承CP，累计21完整outer，后人工停止','prefix-selection200-resumed6-10-runtime-summary.json'),
 '3423':('actorhidden存到1轮；legacy每outer10head storage断言错误','search-g1-actorhidden-period20-state.json'),
 '3424':('actorhidden存到2轮；prune把导入history误认owned路径','search-g1-actorhidden-period20-state.json'),
 '3425':(f'同run恢复继续；本次本地{GI_COMPLETE}完整outer','search-g1-actorhidden-period20-state.json')}
for job,(reason,n) in gijobs.items():
 p=G/n
 if not p.exists():
  # Do not invent a receipt for a job merely mentioned in a previous summary.
  candidates=list(G.glob('*prefix*selection*state*.json')) if job in ['3413','3414','3419'] else []
  ps=candidates[:1]+[D/(SESS['Gi']+'-history-hits.json')]
 else:ps=[p]
 attempt('Gi','H100 '+job,'完成／恢复／故障，见说明',reason,ps)
attempt('Tree','H1003421','评测故障／从CP5恢复','首5训练完整；QA5 Triton mma-layout转换断言失败，checkpoint保留；不是actor第五轮失败',[T/'TERMINAL3421_SMALL.json'])
attempt('Tree','H1003422','同run恢复后进行中',f'恢复链prefix1024-diversity512；本次本地原始reward回执{TREE_COMPLETE}',[T/'CURRENT_CONFIG.json',T/'STATUS.json',T/'TRAINING_REWARD_HISTORY.json'])

# Exhaustively retain every canonical A800/reference deployment handle found locally.
# A handle proves preparation identity, not successful execution. Do not turn unknown
# preparation-only handles into invented failed or running training jobs.
covered={s['path'] for a in attempts for s in a['sources']}
canonical=[]
notes={
 'a800-actorhidden-prefix5-deployment-20261017.json':('CPU prepare失败','原8e6c：Transformers4.46.3 rotary buffers契约假设错误，训练0；v2重新准备'),
 'a800-actorhidden-prefix5-v2-deployment-20261017.json':('v2准备中','新411c，exact源码staged；本次记录尚无actor完整outer'),
 'qprobe-prefix-scale20-failed-cache-scope-deployment-20261016.json':('数据准备失败','原c6 scope adapter问题，0新Head/actor更新；0850修复独立执行'),
 'qprobe-prefix-scale20-dataset-deployment-20261016.json':('数据构建完成','0850 scale20数据构建，不是RL'),
 'qprobe-native-head200-control-deployment-20261016.json':('诊断完成；中间入口故障保留','原warmupHead200 CPU复现及tensor绑定；没有训练更新'),
 'qprobe-native-new512-diagnostic-deployment-20261016.json':('诊断完成','新512×5真实rollout，旧Head200 mixed accuracy58.7891%'),
 'qprobe-hf-trajectory-deployment-20261016.json':('资产导出／上传','旧teacher真实轨迹打包与HF发布；不算新probe fit'),
 'qprobe-hf-scale20-prefix5-deployment-20261016.json':('资产导出／上传','scale20 prefix5追加HF；不算新probe fit'),
 'question5-ema-joint200-deployment-20261016.json':('准备handle；无完整epoch凭证','db413最初EMA方案准备入口；后由slim/direct/cache入口替代，不冒称训练成功'),
 'question5-ema-joint200-slim-deployment-20261016.json':('准备协议被替代','f4aa slim路径；等待/数据路由被HF-direct取代；无该handle完成epoch凭证'),
 'question5-hf-direct-deployment-20261016.json':('特征缓存完成／训练前修复','d7dd复用/生成全量prefix5缓存；原归档/源码核验修复后直接复用缓存，不重新采样'),
 'question5-cached-sourcefix-deployment-20261016.json':('训练入口修复尝试','4347直接复用已封存缓存；随后procguard新handle8af接管，原失败或退出状态见index，不猜数值根因'),
 'question5-cached-sourcefix-procguard-deployment-20261016.json':('实际训练执行','8af初始execution091f；完成epoch1/2，后按批量变更要求完整恢复'),
 'question5-largebatch-resume-deployment-20261016.json':('技术恢复尝试','203d largebatch入口；pidfd Python封装兼容性阻塞，真实stop失败before-signal已保留；非训练数值故障'),
 'question5-largebatch-pidfd-resume-deployment-20261016.json':('技术恢复尝试','141a pidfd兼容入口；后由global1024同run执行bcfa替代，不累计成独立epoch实验'),
 'h100-question5-reference5-deployment-20261016.json':('参考缓存准备／对齐','4750 H100 reference5特征/格式对齐；没有在此handle发现独立actor RL结果'),
}
for p in sorted(P.glob('*.json')):
 if not re.search(r'deployment(?:-2026101[67])?\.json$',p.name):continue
 if not re.search(r'qprobe|question5|question20|h100-question5|actorhidden',p.name):continue
 canonical.append(p)
 if str(p) in covered:continue
 obj=load(p)
 st,reason=notes.get(p.name,('准备／执行状态不据handle推断','有独立canonical部署身份，但没有足够终态证据；保留源文件，不能宣称训练完成或数值失败'))
 attempt('G1',obj.get('base',p.name).split('/')[-1],st,reason,[p,P/'qprobe-teacher-warmup-a800-progress-handoff.json'])
 covered.add(str(p))
coverage={'canonical_handles_found':len(canonical),'canonical_handles_covered':sum(str(p) in covered for p in canonical),
 'all_canonical_handles_covered':all(str(p) in covered for p in canonical),
 'scope':'canonical deployment handles in Search-G1/preparation; stage/observe/claim receipts are indexed separately, not independent experiments'}
(D/'canonical-handle-coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')

audit=load(D/'full-history-scan-audit.json');pins=load(D/'pinned-sessions.json')
summary={'scope':'All 16 pinned histories, bounded immutable-at-scan prefix plus local actual experiment receipts. No original agent messages; no training changes; no new full rollouts or model weights downloaded.',
 'history_scan_completed_utc':audit['completed_utc'],'catalog_build_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'pinned_sessions':len(audit['sources']),'history_records':sum(s['total_records'] for s in audit['sources']),
 'history_bytes':sum(s['scanned_prefix_bytes'] for s in audit['sources']),
 'experiment_stage_records':len(experiments),'kinds':dict(Counter(e['kind'] for e in experiments)),
 'curated_execution_attempt_records':len(attempts),
 'limitation':'Chronology is complete over scan-start snapshots, not a claim that later live messages have been scanned. Source excerpts capped at18000 chars; original byte offset and SHA retained. Failure evidence includes unresolved causes, clearly distinguished from independently diagnosed root causes. Progress values have per-source dates and are not a new all-node live audit.'}
for name,obj in [('experiments.json',experiments),('execution-attempts.json',attempts),('inventory-summary.json',summary)]:
 (D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def link(path,label=None):return '['+(label or Path(path).name)+']('+str(path)+')'
def clean(s):return str(s).replace('|','／').replace('\n',' ')
lines=['# Question diversity probing 全历史实验清单','',
 f'历史完整扫描：16个置顶会话，{summary["history_records"]:,}条JSONL记录，{summary["history_bytes"]/1e9:.3f}GB原始历史；扫描完成于2026-10-17 01:27:44（北京时间）。本地实验receipt补查至本报告生成时。',
 '',f'本清单共 **{len(experiments)}项设置／阶段／诊断记录**，另列 **{len(attempts)}条执行、恢复、准备或辅助尝试记录**。前者不是{len(experiments)}个独立RL实验；同一设置的warmup和正式RL有继承关系，恢复job也不是新算法。',
 '', '读取了完整会话快照，而不是仅最近10轮。历史中的goal反复回放已去重；内部工具、父子agent与历史方案没有当作新人工授权。没有向原agent发消息，没有改变训练，也没有下载权重、tensor或完整rollout。',
 '', '## 指标口径','',
 '- **QA EM**：模型实际回答的严格正确率；固定512题QA评测与51713题终评分别标明。',
 '- **mixed分类accuracy／balanced accuracy**：预测同题五条轨迹是否有对有错；不是QA EM。',
 '- **轨迹correctness分类accuracy**：从question/prefix预测其自己的最终EM；也不是模型推理QA EM。',
 '- warmup的epoch／HeadAdam、正式actor outer／Adam分别记；部分outer不计完整outer。',
 '- Prefix-only的hidden仍含question上下文；“只接生成位置”不等于“没有问题信息”。',
 '- 同题G5样本std采用ddof=1时只有0、sqrt(.2)、sqrt(.3)；Tree二次挑题的population variance与其相差常数因子，不能把训练代码里的variance误写成std。',
 '', '## 全部设置与阶段','']
groups=[('D','原始设计与未执行消融'),('C','Co会话：小MLP与σ3'),('A','Search-G1会话：MLP、完整Qwen及Prefix5'),('H','Gi会话：三block与早期warmup'),('P','Gi会话：前缀长度、在线数据及联合监督warmup'),('L','后续正式选题RL'),('N','扩大数据、EMA与A800新方案')]
for prefix,title in groups:
 lines += ['### '+title,'','| ID | 实验／阶段 | 实际进度 | 结果／异常 |','|---|---|---|---|']
 for e in experiments:
  if e['id'].startswith(prefix):lines.append('| '+' | '.join(map(clean,[e['id'],e['name'],e['progress'],e['result']]))+' |')
 lines.append('')
 for e in experiments:
  if not e['id'].startswith(prefix):continue
  lines += [f'**{e["id"]} — {e["name"]}**', '',e['method'],'',
    '身份：'+(e['identity'] or '见下方实际证据；未虚构run identity')+'。所属会话：`'+str(e['session_id'])+'`。','']
  if e['caveat']:lines += ['注意：'+e['caveat'],'']
  lines += ['证据：'+' · '.join(link(s['path']) for s in e['sources'] if s['exists']),'']
lines += ['## 失败、恢复、停止及准备尝试附录','',
 '这张表保留失败和未完成执行。多个部署可能共用一个run；没有明确退出原因的地方保留“不确定／见原终态”，不根据超参数猜测失败原因。部分辅助job只做CPU准备、清理或归档。','']
for owner in ['G1','Gi','Tree','TOPD','Co']:
 lines += ['### '+owner+'会话执行尝试','','| 执行标识 | 状态 | 实际情况／故障 | 证据 |','|---|---|---|---|']
 for a in attempts:
  if a['owner']!=owner:continue
  reason=a['reason']+(('；'+a['progress']) if a['progress'] else '')
  lines.append('| '+' | '.join([clean(a['attempt']),clean(a['status']),clean(reason),' · '.join(link(s['path']) for s in a['sources'] if s['exists'])])+' |')
 lines.append('')
lines += ['## 继承关系与容易混淆的记录','',
 '1. **A800小MLP链**：原teacher warmup→micro2→fresh micro4→fresh async Top512；async首5轮故障后同run恢复到38，再换σ3。不能把4轮+5轮+38轮相加为一个完整模型。',
 '2. **损失变化**：Co的σ3 λσ=1；A800改为λσ=3；H100三block的binary辅助是0.1 normalized variance MSE；A800binary辅助是0.3 normalized std MSE。它们不是同一loss。',
 '3. **Head200过拟合证据**：同训练题99.88%，新512题mixed分类58.79%；这是泛化失败，正式训练是否QA退化另看QA输出。',
 '4. **epoch68与epoch100不同来源**：Co Prefix20初始化用Prefix-only在线换题epoch68；Gi/Tree后续question+20初始化用jointBCE100实际保留的epoch100。epoch45更高但未留权重，不能声称用它初始化。',
 '5. **最新在线方案**：Gi actor-hidden period20用所有5120候选的标签周期refit；Tree先续写1024再按实际diversity选512；Co新Prefix20只用选中512拟合且另加all-correct虚拟0。',
 '6. **period25**由TOP-D会话运行，但算法不是TOP-D蒸馏；当前是独立原生Qwen QDP。',
 '7. **Question+5 EMA**每epoch遍历445440 G5事件组（2227200轨迹），不是每epoch5120题；actor更新0。',
 '8. Search-G1原来的闭卷探针/证据依赖探针属于reward shaping；QDP是新增选题probe。纯Search-G1旧reward探针校准、硬件smoke、普通mixed-first全rollout、递归Tree等不冒称QDP消融。',
 '', '## 全部置顶会话覆盖','', '| 置顶序号 | 会话标题 | Session ID | 扫描记录数 | QDP相关性 |','|---|---|---|---:|---|']
for a in audit['sources']:
 role=next((k for k,v in SESS.items() if v==a['id']),None)
 text='存在实际QDP训练／warmup' if role in ['G1','Gi','Tree','TOPD','Co'] else ('设计、论文或结果汇总；无独立QDP训练run' if a['matching_records'] else '未找到QDP关键词记录')
 lines.append('| '+' | '.join(map(clean,[a['pinnedIndex'],a['title'],a['id'],f'{a["total_records"]:,}',text]))+' |')
lines += ['', '完整检索索引保留每条命中的原始文件字节偏移、行号、时间和SHA；被截断的展示摘录可按偏移回到原始记录核验。',
 '', '审计文件：'+ ' · '.join(link(D/n) for n in ['experiments.json','execution-attempts.json','inventory-summary.json','full-history-scan-audit.json','actual-metadata-evidence-index.json']),
 '', '边界：这是截至快照的历史清单和本地既有实际receipt核对；没有把当前会话活跃、Slurm RUNNING、提交成功或准备成功当作新增完整训练轮数。历史故障中仍未定位到具体根因的项已原样保留，不宣称所有故障均已修复。','']
(D/'REPORT.md').write_text('\n'.join(lines))

# Freeze only small cited receipt fields/hashes, not full histories or model artifacts.
manifest={}
for obj in experiments+attempts:
 for s in obj['sources']:
  if s['exists']:manifest[s['path']]=s
(D/'cited-source-manifest.json').write_text(json.dumps(list(manifest.values()),ensure_ascii=False,indent=2)+'\n')
frozen={'readback_scope':'Existing local actual receipt fields; no new all-node SSH audit',
 'Gi':{'source':source(G/'search-g1-actorhidden-period20-state.json'),'complete_outer':GI_COMPLETE,'probe_adam_per_rank':GI_STATE.get('probe_adam_per_rank'),'updated_utc':GI_STATE.get('updated_utc')},
 'TOPD':{'source':source(O/'STATE.json'),'complete_outer':TOP_COMPLETE,'status':TOP_STATE.get('status')},
 'Tree':{'source':source(T/'TRAINING_REWARD_HISTORY.json'),'complete_outer_reward_receipts':TREE_COMPLETE}}
(D/'frozen-local-key-facts.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
for n in ['REPORT.md','experiments.json','execution-attempts.json','inventory-summary.json']:
 assert not re.search(rb'hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',(D/n).read_bytes()),n
assert len({e['id'] for e in experiments})==len(experiments)
assert all(e['sources'] and any(s['exists'] for s in e['sources']) for e in experiments)
print(json.dumps(summary,ensure_ascii=False))
