# -*- coding: utf-8 -*-
"""Apply the recorded scientific/editorial review to the completed DPSK draft."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re

root=Path(__file__).resolve().parents[2]
source=root/'docs/V8_DPSK_RESEARCH_REPORT_DRAFT.md'
text=source.read_text(); edits=[]
def section(start,end,new,reason):
 global text
 a=text.index(start);b=text.index(end,a+len(start))
 text=text[:a]+new.rstrip()+'\n\n'+text[b:];edits.append(reason)
def replace(old,new,reason):
 global text
 if old not in text:raise ValueError('Missing review target: '+reason)
 text=text.replace(old,new);edits.append(reason)

section('**摘要**','**关键词**',
'''**摘要**　本文研究内容平台的未来曝光能从哪些历史信息中预测，以及这些信息对创作者竞争模型有什么启发。我们用真实曝光记录比较四种方法：不使用历史、只使用作品自身历史、只使用同账号其他作品的历史、同时使用两类历史。结果显示，作品自身历史带来主要预测改善；在此基础上加入账号共享信息，平均改善很小且不稳定。对没有自身历史的新作品，共享账号信息则有一定帮助。一个小型公开 YouTube 样本也显示作品历史有预测价值。根据这一信息区分，我们建立并求解了含旧作品存量与账号积累的两期生产博弈：旧作品可以挤出新增生产，而考虑未来状态会改变今天的选择。AI 降本和多账号是研究背景；真实拆号行为、成本冲击和平台最优规则仍需另外检验。''',
 'Lead with the actual research question and findings; remove a disclaimer-led abstract.')

section('## 1 研究问题、背景与实际贡献','## 2 概念、术语与一个贯穿例子',
'''## 1 研究问题、背景与实际贡献

设想一位创作者已经发布了很多作品，今天即使不更新，旧作品也可能继续获得展示。另一位创作者刚发布第一部作品，还没有这部作品的历史记录。平台如何分配曝光，不能只看两人今天各发了多少。已有作品的持续吸引力、创作者的稳定能力、账号积累的受众信息，都可能影响未来的分配。

这种区分也关系到 AI 背景下的策略问题。如果制作内容更便宜，创作者是否会增加生产？如果平台削弱同一账号多发作品的奖励，创作者是否会转向多账号？要分析这些行为，先要弄清楚曝光与哪些信息有关，避免把旧作品已经获得的流量，直接当成再制作一部新作品的回报。

本文回答两个相互衔接的问题：

1. **实验问题：**作品自己的过去表现、同账号其他作品的过去表现，各自能帮助预测多少未来曝光？新作品没有自身历史时，答案是否不同？
2. **博弈问题：**在明确给定的平台规则下，如果旧作品可以继续参与推荐、账号状态也会积累，两位创作者应怎样选择本期生产？允许开新号时，又多出什么权衡？

我们完成了两项工作。第一，用同一批候选作品和同一信息截止，比较不同历史来源，并把新作品单独检验，识别出这些信息各自适用的预测对象。第二，把旧作品存量、账号状态与新增生产分别写入一个可求解的两期博弈，分析当前选择为何会受未来影响。

两项工作的证据不同：真实数据检验预测信息，理论模型分析给定假设下的策略。数据中确实存在旧作品继续获得曝光的记录，但没有真实经济主体的多账号映射、AI 采用成本或平台规则变化。因此，本文完成的是曝光信息的区分和条件性的策略分析，尚未验证 AI 导致拆号的因果链。模型沿用已有的内容创作竞赛与存量分析工具，不主张创立新的基础博弈理论。''',
 'Replace version/history and repeated limitations with a coherent question, motivation and contribution.')

section('### 2.1 先分清五组对象','### 2.2 术语表',
'''### 2.1 阅读顺序

先把“作品”“账号”“创作者本人”分开，再看两类历史信息。下列术语表给出本报告的用法；随后用一个假设例子演示，如何从历史信息得到预测份额。''',
 'Remove five repeated definition paragraphs; retain a glossary and a worked example.')

# The two glossaries need definitions, not a third column of repeated caveats.
a=text.index('### 2.2 术语表');b=text.index('### 2.3 一个贯穿例子',a)
block=text[a:b]
lines=[]
for line in block.splitlines():
 if line.startswith('|'):
  cells=line.strip().strip('|').split('|')
  if len(cells)==3:line='|'+ '|'.join(cells[:2])+'|'
 lines.append(line)
text=text[:a]+'\n'.join(lines)+'\n\n'+text[b:]
edits.append('Reduce both glossaries to term and definition; preserve important distinctions in prose.')

replace('| 曝光权重 |', '| 历史曝光份额 | 过去某段时间，一个账号的曝光占当时同一候选池总曝光的比例。例如总曝光一万次、其中两千次属于该账号，历史份额就是20%。它不是下面两个历史权重 m、h 的别名。 |\n| 曝光权重 |',
        'Explicitly define the historical exposure-share term asked about by the reader.')
replace('训练期按年龄/时长分组的期望曝光；μ 是全局均值',
        '用训练期记录估计的同年龄、同时长作品的平均曝光；年龄指距发布日期的天数。用 $\\bar e$ 表示训练全局平均曝光',
        'Define age/duration baseline and separate its mean symbol from account maintenance cost.')
replace('目标作品过去曝光/年龄时长基准期望的残差比；样本少时把估计值拉向 1',
        '目标作品过去实际曝光相对于基准期望曝光的比值。记录少时向1平滑，避免少量曝光产生极端权重；旧记录每过7天权重减半，这个设定称为7日半衰期',
        'Explain memory, shrinkage and half-life without unexplained statistical jargon.')
replace('c 为单位生产成本；F 为一次性开户费；μ 为每期额外账号维护费',
        'c 为单位生产成本；F 为一次性开户费；μ 为每期额外账号维护费。它与测量部分的平均曝光 $\\bar e$ 无关',
        'Eliminate the overloaded mu symbol.')
replace('本期新生产 |','本期新增的标准化内容强度；连续模型可以取小数，不直接等于视频条数 |',
        'Define production units before fractional quantities are used.')
replace('S=θw(H)(B+x)^α；w(H)=(1+H)^b；α 是评分对内容量的弹性，1 为线性、小于 1 为边际递减',
        'S 是竞争积分；$\\alpha$ 控制内容增加带来多少积分：1为同比增加，小于1为新增奖励递减。$w(H)=(1+H)^b$ 中的 $b\\geq0$ 控制账号状态影响强弱。预测指数与理论规则没有被因果校准为同一参数',
        'Define score exponent and the previously undefined capital exponent.')
replace("B'=ρ_B B+x；H'=δ_H H+λ_H p",
        'ρ_B 是旧内容保留比例；δ_H 是账号状态保留比例；λ_H 控制曝光份额对下一期账号积累的影响，均是模型假设参数',
        'Define each state-transition coefficient rather than substituting an equation for a definition.')
replace('前瞻把今天行动对未来状态和对手反应计入；短视只优化当期；子博弈精炼要求每个子博弈都取最优',
        '前瞻考虑未来状态与对手反应，短视只优化当期。纳什均衡指对方策略不变时，任何一方单独改策略都不能提高收益；子博弈精炼还要求偏离后各局面中的继续策略也满足这一条件',
        'Correct equilibrium definition: individual best responses, not jointly optimal outcomes.')
replace('明天一元今天值多少','未来一单位效用折合今天多少',
        'Keep model utility distinct from cash income.')

text=text.replace('截止 t 前的信息预测 t 后的曝光','t 日前的信息预测 t 日曝光')
replace('截止 t 前信息预测 t 后曝光','t 日前信息预测 t 日曝光',
        'Correct the glossary time convention.')
text=text.replace('e_v/μ','e_v/\\bar e').replace('e/μ','e/\\bar e')
edits.append('Use bar-e for the empirical normalization mean, including the flow illustration.')

# Toy memory weights are assumed already smoothed; remove misleading raw counts.
text=text.replace('| 目标作品 | 账号 | 基准 e | 过去曝光 | 记忆 m（假定平滑后） |',
                  '| 目标作品 | 账号 | 基准分值 e | 记忆 m（假定已平滑） |')
text=text.replace('|---|---|---:|---:|---:|','|---|---|---:|---:|',1)
for old,new in [('| A | a | 400 | 800 | 2.0 |','| A | a | 400 | 2.0 |'),
                ('| B | a | 200 | 100 | 0.5 |','| B | a | 200 | 0.5 |'),
                ('| C | a | 200 | 无 | 1.0（先验） |','| C | a | 200 | 1.0（无自身历史） |'),
                ('| D | b | 300 | 300 | 1.0 |','| D | b | 300 | 1.0 |')]:
 text=text.replace(old,new)
text=text.replace('先只用作品记忆。取 α=1、γ=1，分数 z=e×m：',
 '基准分值可以统一缩放而不改变份额，因此例子省略共同的平均曝光除数。先只用作品记忆，取 α=1、γ=1，分数 z=e×m：')
text=text.replace('那需要 C 自己的历史，而新作没有',
 '那需要作品层特征或后续记录，公共 h 本身不能区分')
edits.append('Make the illustrative weights explicit assumptions; avoid a false smoothing calculation.')

replace('四个起点的测试窗有重叠，合并后有 28 个不同测试日。',
        '四个起点的测试窗互不重叠，共28个不同测试日；部分训练、校准期及后一起点可使用的过去记录会重用。',
        'Correct a factual error about overlapping test windows.')
text=text.replace('同一参数网格','共同的数量指数候选范围')
replace('份额的分母是这个池的总曝光。评分时每个作品日按实际曝光数加权，',
        '观测份额用池内实际曝光总和作分母；预测份额则用池内评分总和作分母。实际总曝光不作为预测输入，只在评价时给每个作品日加权。',
        'Separate observed-share and predicted-share denominators.')
text=text.replace('在截止 t 前可用的信息中，哪一种历史来源对 t 后曝光有预测增益',
                  '在 t 日前可用的信息中，哪一种历史来源能帮助预测 t 日曝光')

section('### 4.2 平均增量很小，而且集中','### 4.3 新作品：有限的正面结果，但不是双状态最好',
'''### 4.2 平均增量很小，而且集中

这个总体小增益对高曝光账号敏感。固定预测、仅去除曝光最高一个账号的评分贡献后，差值便从负转正；这不是把该账号从市场移走，也没有重新拟合模型。因此，不能把总体平均改善说成所有账号都受益。

不相交来源历史只覆盖26.04%的测试作品日、41.42%的曝光，限制了共享信息能用于哪些作品。无源历史时 h 回先验1，不新增个体信息，但其他账号的 h 仍会通过全池归一化改变其预测份额，不能把总体增量小仅归因于“稀释”。预定总体比较应当保留；新作品检验另回答没有自身历史时的信息价值。''',
 'Retain concentration and coverage without an unsupported dilution mechanism.')

# Preserve the new-work result and its adverse comparator; omit a tangential scoring branch.
start=text.index('### 4.3 新作品：有限的正面结果，但不是双状态最好')
end=text.index('### 4.4 两个负面结果',start)
new_section='''### 4.3 新作品：共享信息有用，但复杂模型未必最好

这一实验问：没有作品自身历史时，同账号其他作品的信息是否有帮助？输入是4,315部实际有上传自然日记录的新作品，作品记忆 m 回到先验1；共享信息 h 只来自不相交来源作品的过去记录。我们把四种方法在这些新作品上的预测概率重新归一化，比较新作之间的曝光分配，而不把新旧作品总份额混在同一个问题里。

| 方法 | 新作品子池内预测损失 |
|---|---:|
| 不使用历史 | 4.048581 |
| 只用作品自身历史 | 4.048581 |
| 只用账号共享历史 | 3.955176 |
| 两类历史一起用 | 4.017787 |

仅账号相对无状态改善−0.0934049，日期95%区间为[−0.1420583,−0.0420770]，不含零。它是四种方法中新作预测最好的；双状态反而比仅账号差。这说明没有自身历史的新作品可以利用同账号其他作品的信息，但增加更多特征并不保证测试表现更好。这里与上一表的候选池不同，不能直接横比两表损失的绝对大小。

这项改善会不会只是模型给“有来源历史”这一组多分了一些概率？结果后固定预测诊断将其拆开：710部有源历史新作的组内损失从2.526059降到2.437922，差−0.0881376，日期区间[−0.1657274,−0.0026242]；3,605部无源历史新作使用共同先验，组内差约零。总改善−0.0934049中，约−0.0494430来自两组之间的份额，约−0.0439620来自加权组内分配。

因此，改善不全是组间校正，有源历史组内部也有信息。去掉几个高曝光账号的评分贡献后仍有改善，但这项分解是看过主结果后的诊断，不是独立确认；它也不能区分稳定能力、题材匹配与不可转移的账号资本。

![新作品：共享信息及组内诊断](../src/v8/report/figures/experiment_new_works.svg){width=100%}

图2：左图比较同一新作子池的四种方法，越低越好。右图比较“仅账号减无状态”：全部新作与有源历史组内的区间在零左侧；无源历史组内差为零。星号表示结果后固定预测诊断，不是额外的独立实验。'''
text=text[:start]+new_section+'\n\n'+text[end:]
edits.append('Keep the primary new-work result, group diagnostic and adverse dual-state comparator; remove nonessential branches.')

start=text.index('另一个负结果来自一步预测。');end=text.index('### 4.5 外部小源',start)
text=text[:start]+'''我们还尝试预测下一日曝光的水平。加入更多观测状态与目录特征，没有超过直接沿用前一天表现的简单基准。这项失败说明，增加状态特征并未改善该短期预测任务；模型中的库存和资本转移方程仍是理论假设，不能用观测记忆的更新定义来证明它们。

'''+text[end:]
edits.append('Retain the failed state forecast as one relevant negative result, not a second technical chapter.')
start=text.index('| 频道 | 测试视频日/曝光 |');end=text.index('年龄×记忆相对年龄改善',start)
text=text[:start]+'''| 方法 | 四频道合并条件评分 |
|---|---:|
| 只用作品年龄 | 2.328489 |
| 年龄基准加作品历史 | 1.818669 |
| 直接用前日曝光加先验 | 1.718326 |

'''+text[end:]
edits.append('Replace the channel-name dump with the useful external comparison; retain concentration limits.')

text=text.replace('新号 B=H=0，w(0)=1', '模型假设新号 B=H=0，w(0)=1')
text=text.replace('固定状态、单个账号、对称内点（双方都生产正数量）的切片',
 '固定状态、禁止新增账号、两人条件相同的连续生产切片，假设 $R,c>0$、$0<\\alpha\\leq1$，且产能上限不约束')
text=text.replace('`n*=max(0, Rα/(4c)−B)`', '$$x^*=\\max\\{0,R\\alpha/(4c)-B\\}.$$')
text=text.replace('n*=0','x*=0')
text=text.replace('当 B 足够大时，x*=0','当旧存量 B>0 且足够大时，x*=0')

start=text.index('一个已核验的解析例可以说明差别。');end=text.index('### 5.4 开号',start)
text=text[:start]+'''一个固定单账号的两期例子说明这种差别：只优化当前收益时，第一期生产为0.5单位；考虑第二期状态和对手反应时，第一期约为1.83单位。这里的“单位”是连续的标准化内容强度，不是对实际视频条数的预测。这个例子只展示未来价值如何改变选择，不是“前瞻总会增产”的定理。

为使例子可复查，两位创作者的假设完全相同：能力系数1，内容评分采用线性形式α=1，初始旧存量2.5、账号状态0；收益池12、单位成本1；旧存量保留率0.25，账号状态保留率0.8，曝光积累系数1，贴现因子0.25，账号权重为1+H。第一期生产范围[0,2]、第二期[0,3]，账号数固定为一。完整逆推把每一种第一期偏离后的第二期最优反应计入收益；我们没有把逐期短视答案当成动态均衡。

'''+text[end:]
edits.append('Remove arbitrary grid counts and the incorrect cost-decrease-only opening claim; keep one fully specified forward-looking example.')

# Typeset the small set of load-bearing formulas rather than showing code fragments.
text=text.replace('`S_i=θ_i w(H_i)(B_i+x_i)^α`，`w(H)=(1+H)^b`，`b≥0`；',
 '$$S_i=\\theta_i w(H_i)(B_i+x_i)^\\alpha,\\qquad w(H)=(1+H)^b,\\ b\\geq0.$$')
text=text.replace('`p_i=S_i/(S_i+S_j)`，`u_i=R p_i−c x_i`；',
 '$$p_i=\\frac{S_i}{S_i+S_j},\\qquad u_i=Rp_i-cx_i.$$')
text=text.replace("`B_i'=ρ_B B_i+x_i`，`H_i'=δ_H H_i+λ_H p_i`。",
 "$$B_i'=\\rho_B B_i+x_i,\\qquad H_i'=\\delta_H H_i+\\lambda_H p_i.$$")
text=text.replace('`c=R p(1−p) α/(B+x)`。', '$$c=\\frac{Rp(1-p)\\alpha}{B+x}.$$')
text=text.replace('评分写成 `z_v=(e_v/\\bar e)m_v^γ`，`T_a=Σ z_v`，`s_v=h_a^β T_a^(α−1) z_v`；同账号汇总为 `Σ s_v=h_a^β T_a^α`。',
 '''评分先把基准和作品记忆结合，再用账号信息调整：

$$z_v=\\frac{e_v}{\\bar e}m_v^\\gamma,\\qquad T_a=\\sum_{v\\in a}z_v,\\qquad s_v=h_a^\\beta T_a^{\\alpha-1}z_v.$$

同账号汇总为 $\\sum_{v\\in a}s_v=h_a^\\beta T_a^\\alpha$；预测概率为目标作品评分除以候选池评分总和。''')
text=text.replace('每个方向有四个起点，训练、校准、测试三个窗口按时间顺序排列，分别为 14 天、7 天、7 天。',
 '每个方向有四个起点，训练、校准、测试三个窗口按时间顺序排列，分别为14天、7天、7天。训练期估计年龄/时长基准；校准期从预定候选中选参数；测试期只检验未来预测表现。')
text=text.replace('z=(e/\\bar e)m^γ','z=基准×作品记忆权重')
text=re.sub(r'把这条链画成文字流程：\s*```.*?```',
 '操作顺序是：先把目标作品的过去记录变成作品权重，再从不相交来源组构造账号权重；合成评分后，在当天候选池内归一化成预测份额。这个操作不需要预知当天的实际总曝光。',text,flags=re.S)
text=text.replace('### 5.4 开号：一个短政策情景，不是全文结论',
                  '### 5.4 允许新账号时，多出什么策略权衡')
text=text.replace('24 个线性补充情景确实没有开号，但这只是有限情景',
                  '所求解的线性基准情景没有出现开号，但这不是适用于所有动态环境的定理')

text=text.replace('32 次 α 全选 1；48 个主情景中 34 个前瞻改变、6 个首期开号，24 个线性补充 0 开号',
                  '当前网格均选线性形式；旧存量可以挤出新增生产，前瞻选择可以不同于短视')
text=text.replace('不是因为模型可以验证数据、数据可以校准模型','两项工作采用同一信息分层，但没有完成结构参数的因果校准')
text=text.replace('旧作品持续曝光与数据相容，但 B/H 转移没有被验证',
 '旧作品继续获得曝光在日志中可见，但 B/H 的结构转移没有被识别')
text=text.replace('一句话总结：本文没有证明 AI 导致拆号；它做的是把旧作品、账号信息与新生产分开，并说明在现有证据下哪些主张成立、哪些必须等待更好的数据。',
 '本文的实质结论是：作品自身历史提供了主要预测信息，账号共享信息对新作品有有限帮助；策略模型因此应把旧作品状态与新增生产分开。真实的AI成本效应、拆号收益与最优平台政策，仍需要相应的身份、成本和制度证据。')
text=text.replace('arXiv:2202.10842. 官方代码与数据：',
 'DOI: 10.1145/3511808.3557220. 官方代码与数据：')
text=text.replace('## 参考资料',
 '### 数据与写作说明\n\n实验使用公开元数据，行级原始记录不随本报告重新分发。报告由本地DeepSeek Harness通读项目档案后起草，再由Codex按保存的结果逐表核对、纠正表述并排版；这不等于外部人工同行评审。核心数据入口与相关研究列于下方。\n\n## 参考资料')
text=re.sub(r'\| 有效结果是什么 \|.*?\|',
 '| 有效结果是什么 | 自身历史提供主要预测改善；账号共享信息对新作有有限帮助；当前网格未支持预设数量压缩；状态模型能区分旧作收益和新增生产，并展示前瞻选择与短视的差别 |',text)

text=text.replace('模型只有两期、有限动作、给定参数','理论分析只覆盖两期、给定参数与可行范围')
text=text.replace('平台先承诺推荐规则，两位创作者同时行动；',
                  '平台先承诺推荐规则，假设双方知道规则、状态和成本；两位创作者同时行动；')
target=root/'docs/AI_CREATOR_RESEARCH_SUMMARY_ZH.md';target.write_text(text)
receipt=dict(edited_utc=datetime.now(timezone.utc).isoformat(),
             dpsk_draft_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             final_markdown_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
             draft_bytes=source.stat().st_size,final_bytes=target.stat().st_size,
             chinese_characters=len(re.findall(r'[\u4e00-\u9fff]',text)),changes=edits,
             originals_preserved=True)
(root/'docs/V8_AUDIT/editorial_changes.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='changes'},ensure_ascii=False))
