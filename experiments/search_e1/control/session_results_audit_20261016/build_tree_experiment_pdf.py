"""Auditable, vector-first Chinese report of the tree experiment family."""
from pathlib import Path
import json, math, hashlib, datetime, re
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

D=Path(__file__).resolve().parent
ROOT=D.parent.parent
OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
TMP=ROOT/'tmp/pdfs/tree-experiments-20261016';TMP.mkdir(parents=True,exist_ok=True)
TARGET=OUT/'tree-experiments-ablation-review-20261016.pdf'
R=json.loads((D/'all-seven-results.json').read_text())
F=json.loads((D/'final-seven-table-data.json').read_text())
NAMES=['Tree-GRPO original guarded200','Tree BRANPO fresh200 3360','Tree nested branching fresh200','Recursive A','Recursive B four GPU','C1 root-only ablation150','C2 random-final ablation150']
LABELS=['Tree-GRPO','Tree-BRANPO','Tree-Nested','递归A','递归B','C1 root-only','C2 random-final']
COLORS=['#4477AA','#B17D34','#CF5360','#168D7B','#E38927','#AC72A6','#506B95']
RUNS=[next(r for r in R['runs'] if r['name']==n) for n in NAMES]
for r in RUNS:
    for p in r['fixed512']+r.get('additional_fixed512',[]):
        if p.get('em') is not None:assert p['n']==512 and abs(p['em']*512-round(p['em']*512))<1e-7
pdfmetrics.registerFont(TTFont('CN','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CN',italic='CN',boldItalic='CN')
INK=HexColor('#263B49');MUTED=HexColor('#697F8C');TEAL=HexColor('#168D7B');BLUE=HexColor('#4477AA');GRID=HexColor('#DFE7EC');BG=HexColor('#F1F5F7')
W,H=A4;M=43;CW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='CN',fontSize=10.2,leading=15.5,textColor=INK,spaceAfter=7,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CN',fontSize=8.5,leading=12.6,textColor=MUTED,spaceAfter=5,wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='CN',fontSize=25,leading=34,textColor=INK,spaceAfter=12,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='CN',fontSize=19,leading=26,textColor=INK,spaceAfter=11,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='CN',fontSize=12.5,leading=18,textColor=TEAL,spaceBefore=9,spaceAfter=6,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=8.5,leading=12,textColor=INK,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='CN',fontSize=8.5,leading=12,textColor=white,wordWrap='CJK'),
 'formula':ParagraphStyle('formula',fontName='CN',fontSize=10,leading=17,textColor=INK,wordWrap='CJK'),
}
story=[]
def clean(s):return str(s).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
def P(s,kind='body'):return Paragraph(clean(s),styles[kind])
def add(s,kind='body'):story.append(P(s,kind))
def heading(title,subtitle=None):
    add(title,'h1')
    if subtitle:add(subtitle,'small')
def section(title):add(title,'h2')
def pct(v):return '未生成' if v is None else f'{100*v:.2f}%'
def table(rows,widths=None,small=False,compact=False):
    if widths is None:widths=[CW/len(rows[0])]*len(rows[0])
    cells=[[P(str(v),'head' if i==0 else 'cell') for v in row] for i,row in enumerate(rows)]
    if compact:
        hs=ParagraphStyle('compact-head',parent=styles['head'],fontSize=7.5,leading=11)
        cs=ParagraphStyle('compact-cell',parent=styles['cell'],fontSize=8,leading=12)
        cells=[[Paragraph(clean(str(v)),hs if i==0 else cs) for v in row] for i,row in enumerate(rows)]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('LEFTPADDING',(0,0),(-1,-1),4 if compact else 7),('RIGHTPADDING',(0,0),(-1,-1),4 if compact else 7),
      ('TOPPADDING',(0,0),(-1,-1),7 if not small else 5),('BOTTOMPADDING',(0,0),(-1,-1),7 if not small else 5),
      ('LINEBELOW',(0,0),(-1,0),.6,INK),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,BG]),
      ('LINEBELOW',(0,1),(-1,-1),.35,GRID)]))
    story.append(t);story.append(Spacer(1,9))
def box(text,kind='body',color='#EDF5F2'):
    t=Table([[P(text,kind)]],colWidths=[CW])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HexColor(color)),('BOX',(0,0),(-1,-1),.5,GRID),
      ('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(t);story.append(Spacer(1,8))
def new():story.append(PageBreak())
class Diagram(Flowable):
    def __init__(self,mode,height=105):Flowable.__init__(self);self.width=CW;self.height=height;self.mode=mode
    def draw(self):
        c=self.canv;c.setFont('CN',9)
        def node(x,y,label,fill='#EFF4F7',width=78):
            c.setFillColor(HexColor(fill));c.setStrokeColor(GRID);c.roundRect(x,y,width,29,6,fill=1,stroke=1)
            c.setFillColor(INK);c.drawCentredString(x+width/2,y+10,label)
        def edge(x1,y1,x2,y2):
            c.setStrokeColor(MUTED);c.setLineWidth(1);c.line(x1,y1,x2,y2)
            a=math.atan2(y2-y1,x2-x1);p=c.beginPath();p.moveTo(x2,y2);p.lineTo(x2-5*math.cos(a-.5),y2-5*math.sin(a-.5));p.lineTo(x2-5*math.cos(a+.5),y2-5*math.sin(a+.5));p.close();c.setFillColor(MUTED);c.drawPath(p,fill=1,stroke=0)
        if self.mode=='baseline':
            node(15,66,'根U');node(15,13,'根V')
            for x,label in [(150,'U 原轨迹'),(265,'U1 随机'),(380,'U2 随机')]:
                c.setStrokeColor(MUTED);c.line(93,80,111,80);c.line(111,80,111,102);c.line(111,102,x+39,102);edge(x+39,102,x+39,95);node(x,66,label)
            for x,label in [(150,'V 原轨迹'),(265,'V1 随机')]:
                c.setStrokeColor(MUTED);c.line(93,27,111,27);c.line(111,27,111,49);c.line(111,49,x+39,49);edge(x+39,49,x+39,42);node(x,13,label)
        elif self.mode=='nested':
            node(20,65,'U 原轨迹');node(180,65,'U1 随机');edge(98,79,180,79)
            node(20,10,'V 原轨迹');node(180,10,'V1 异奖');node(340,10,'V2 对V1异奖')
            edge(98,24,180,24);edge(258,24,340,24)
            c.setFillColor(MUTED);c.drawString(289,76,'最终保留2 + 3条')
        elif self.mode=='c1':
            node(10,65,'唯一原根')
            for x,label in [(120,'子1'),(220,'子2'),(320,'子3'),(420,'子4')]:
                c.setStrokeColor(MUTED);c.line(49,65,49,52);c.line(49,52,x+39,52);edge(x+39,52,x+39,39);node(x,10,label)
            c.setFillColor(MUTED);c.drawString(225,78,'四个子不能再作父')
        c.setFillColor(MUTED)
class LineChart(Flowable):
    def __init__(self,indices,kind='fixed',height=215):Flowable.__init__(self);self.width=CW;self.height=height;self.indices=indices;self.kind=kind
    def draw(self):
        c=self.canv;x0,y0,pw=38,27,CW-50
        legend_rows=math.ceil(len(self.indices)/3)
        ph=self.height-30-legend_rows*15-y0
        c.setFont('CN',7.5);c.setStrokeColor(GRID);c.setLineWidth(.45)
        for yy in range(0,61,10):
            y=y0+yy/60*ph;c.line(x0,y,x0+pw,y);c.setFillColor(MUTED);c.drawRightString(x0-6,y-3,str(yy))
        for xx in [0,50,100,150,200]:
            x=x0+xx/200*pw;c.drawCentredString(x,y0-13,str(xx))
        c.setFillColor(MUTED);c.drawString(0,y0+ph+7,'EM (%)');c.drawRightString(CW,y0-24,'outer iteration')
        for z,i in enumerate(self.indices):
            c.setFillColor(HexColor(COLORS[i]));c.setStrokeColor(HexColor(COLORS[i]))
            col=z%3;row=z//3;x=col*(CW/3);y=self.height-9-row*15
            c.setLineWidth(1.7);c.line(x,y,x+17,y);c.setFont('CN',8);c.drawString(x+22,y-3,LABELS[i])
            r=RUNS[i]
            if self.kind=='fixed':pts=[(p['iter'],p['em']) for p in r['fixed512'] if p.get('em') is not None and p['iter'] in [5,50,100,150,200]]
            else:
                raw=r.get('train_series',r.get('training_series',[]));pts=sorted((int(p.get('iter',p.get('iteration'))),p['em']) for p in raw)
                c.setStrokeColor(Color(*HexColor(COLORS[i]).rgb(),alpha=.22));c.setLineWidth(.4)
                for (a,b),(x,y) in zip(pts,pts[1:]):
                    if x==a+1:c.line(x0+a/200*pw,y0+100*b/60*ph,x0+x/200*pw,y0+100*y/60*ph)
                pts=[(pts[j][0],sum(v for _,v in pts[j-4:j+1])/5) for j in range(4,len(pts)) if pts[j][0]-pts[j-4][0]==4]
                c.setStrokeColor(HexColor(COLORS[i]))
            c.setLineWidth(1.65)
            for (a,b),(x,y) in zip(pts,pts[1:]):c.line(x0+a/200*pw,y0+100*b/60*ph,x0+x/200*pw,y0+100*y/60*ph)
            if self.kind=='fixed':
                for x,y in pts:c.circle(x0+x/200*pw,y0+100*y/60*ph,2.2,fill=1,stroke=0)

# 01 - scope and experiment map
add('树相关实验\n机制、消融与结果总览'.replace('\n','<br/>'),'title')
add('代码实现与实际评测记录核对版 | 2026-10-16','small')
box('当前可核验的主要结论：C1取消多代递归后，150轮固定512题EM与递归A同为48.05%；递归B在150轮比A多答对7题，但不能据此证明浅层加权更优；Tree-BRANPO在200轮固定512题仅比原Tree多1题，完整七集平均反而更低。')
section('本稿覆盖哪些独立setting')
table([['实验','改变／要检验的机制','完成进度'],
 ['Tree-GRPO','两根固定五叶，作为本组Tree参照','200/200'],
 ['Tree-BRANPO','尾部优先与一次有限回退，保留全部叶','200/200'],
 ['Tree-Nested','随机分支 + 奖励筛选的V → V1 → V2链','187/200'],
 ['递归A','单根、多代递归、异奖励接纳；均匀选择父','200/200'],
 ['递归B','在A式递归上，父概率按1/(generation+1)加权','169/200'],
 ['C1 root-only','相对A取消多代递归：只有原根能作父','150/150'],
 ['C2 random-final','相对A取消最终训练组的奖励差异优先选择','48/150']], [104,310,CW-414])
section('读数与证据边界')
add('这里的“效果”首先指固定dev512（NQ、HotpotQA各256题）的EM；训练EM属于各自实际选中轨迹，采样分布不同，不能当泛化准确率横排。完整终评为51,713题，七集平均与全题加权EM分开列。')
add('数据冻结：A/C1/C2核验至10月16日13:19；Tree-Nested使用13:17的实际远端输出快照；B为12:54。Tree本轮直读超时，未用聊天文字替代数据。本文没有重新启动、修改训练或追加评测。[S1-S3]','small')
add('七组之外：第9页登记后续参数消融与复用标签，并单独保留旧3154/3234历史结果，防止将恢复分支或旧setting接到当前曲线上。','small')
new()

# 02 - shared objective and baseline
heading('01 共同训练合同与Tree参照','先区分“改变采样器”与“改变优势／损失”。')
table([['项目','已核验的设置'],['初始化','官方Qwen2.5-3B-Instruct；revision aa8e72537993ba99e69dfaafa59ed015b17504d1；各新setting独立fresh训练。'],
 ['数据／批次','train169,028；dev512；目标每outer 512题 × 5条最终轨迹 = 2,560条。候选生成量可能更多。'],
 ['优化','mini-batch128，epochs1；actor全局micro16，old/ref全局32；名义每outer20次更新机会，真实Adam另查账本。'],
 ['学习率／正则','LR1e-6；57个有效outer warmup后常数；entropy0，reference KL系数0；PPO clip0.2，gradient clip1。'],
 ['生成／检索','temperature1、top_p1；max_turns4；top3；context4096，单次response500、observation500；不强制额外最后回答。'],
 ['防护的实际差异','Tree/BRANPO/Nested配置包含Dual-Clip3；A/B/C1/C2的document objective不额外加入Dual-Clip loss，阈值3仅保留监控；KL刹车、尾部刹车和指数防护仍在。']], [103,CW-103],True)
section('优势公式：所谓inter并非“两个根均值相减”')
box('Z<sub>G</sub>(i) = (r<sub>i</sub> - mean<sub>G</sub>(r)) / (sample_std<sub>G</sub>(r) + 10<super>-6</super>)<br/>A<sub>i</sub> = Z<sub>question</sub>(i) + Z<sub>(question, root)</sub>(i)','formula')
add('第一项对同题所有最终叶标准化，第二项对同题、同根的叶标准化。不是每个分叉点各算一个局部A，也没有递归边上的reward回传。单根A/B/C1/C2两分组相同，因此实际为2Z；全同奖组优势为0。[S4]')
section('原Tree-GRPO当前200轮参照')
add('每题两条完整根，最终根配额[3,2]。从根轨迹已有行动前snapshot中均匀抽取分叉点，生成三条续写；配置rounds=1，当前不是无限层递归。最终五叶全部训练，不因奖相同而拒绝候选。[S4]')
story.append(Diagram('baseline'))
add('有效模型token承担loss，检索观测仅作条件；完整轨迹共享前缀仍随每个叶计入。归约为题等权、题内轨迹等权、轨迹内token均值。两根[3,2]总质量为3/5与2/5，并非每根1/2。','small')
new()

# 03 - tail and nested
heading('02 尾部回退与嵌套链：两个不同实验')
section('Tree-BRANPO：有限尾部采样，非完整BranPO算法')
add('相对Tree参照，改的是分叉起点及有限回退。仍两根、配额[3,2]、每题仅三次新续写：第一条从原根最后一个模型动作之前分叉；配额3的那棵根还剩一个位置，若首次续写与原根同奖，第二条向前退一个原动作；若已异奖，则复用尾部位置。')
add('全部生成叶都保留，仍从原根snapshot续写。它不会一直回退到异奖，也不会让新生成子轨迹继续当父。奖励、优势与PPO目标沿用Tree参照。[S5]')
box('效果：iter200 dev512为250/512 = 48.83%，原Tree为249/512 = 48.63%，仅多1题；完整终评七集平均40.92%，低于原Tree的42.00%。现有结果不支持“尾部优先已带来整体改进”。')
section('Tree-Nested：随机一侧，另一侧进行两段奖励筛选')
story.append(Diagram('nested'))
add('每题先生成根U、V。U1在U的有效动作起点均匀选一个，只续写一次且无条件保留。V1从V的尾部向前逐有效动作尝试，首个与V终奖不同的候选被保留；V2再沿V1的真实完整路径重复同一过程，比较对象是V1。')
add('每阶段非初始有效位置最多一次；全部同奖时，从初始状态fresh续写一次并无条件保留。最终固定U/U1/V/V1/V2，根配额[2,3]。被拒候选不进入训练，但全部生成与检索成本记录。它没有A/B的统一“每题10次”预算；上界由父路径的合法位置数及fallback决定。[S6]')
table([['例子／边界','含义'],['V奖0，V1奖1，V2奖0','若两段都找到异奖，二元奖励会交替，不代表V2优于V1。'],['所有候选同奖','仍保留单次fallback，有限搜索不保证最终混奖。'],['优势与信用','仍题内+根内标准化；记录真实V1父亲，不新增节点局部优势。']], [139,CW-139],True)
add('现有结果：187轮已完成；dev150为48.44%，dev185为49.41%；已记录dev165峰值50.39%。尚无dev200和完整终评，峰值不替代最终成绩。','small')
new()

# 04 - recursive A/B
heading('03 递归A与B：父选择分布的干预')
section('共同状态机')
add('① 每题只生成一条完整根，代数g=0，放入接纳集L。② 在L中选择仍有合法分叉位置的父。③ 从合法行动前snapshot续写，继承原思考、检索观测、行为μ与已用turn预算。④ 子与父异奖则进L，g(child)=g(parent)+1；同奖则进候选池F，并在同父上逐动作向前回退。⑤ 达到5条接纳或耗尽10次续写预算；不足时从F无放回补齐，不追加生成、不复制轨迹。')
add('合法位置下界是父轨迹与所有较低代数历史记录的最大共同前缀长度；共同前缀按“模型动作+对应观测”比较。不能退到该边界之前，也不能给分支重新发放完整4轮交互预算。[S7-S8]')
section('唯一主要设计变量：下一次选择谁作为父')
box('A：P(parent=i) = 1 / |E|<br/>B：P(parent=i) = [1/(g<sub>i</sub>+1)] / Σ<sub>j∈E</sub>[1/(g<sub>j</sub>+1)]','formula')
table([['合格父的代数','A选择概率','B选择概率'],['0：原根','33.33%','54.55%'],['1：根的子','33.33%','27.27%'],['2：子的子','33.33%','18.18%']], [180,160,CW-340],True)
add('B偏向浅层单条父轨迹，意图减少预算集中在深层谱系；深层仍有机会。这个权重不乘到loss上，也不按reward高低决定父概率。同代多条父的概率会累加。奖励0→1和1→0都可接纳，不是只留成功回答。')
section('实现差异与效果解释')
add('B额外过滤空、终止及不可恢复snapshot；完成的A版本没有同样显式过滤。A运行于8张H100，B为4张RTX4090；全局micro规模相同，但工程pipeline不同。A/B固定dev和final题目ID一致，训练输入及顺序指纹不同、完整集合相同尚未证明，因此不能称严格单变量对照。')
table([['iter','A dev512','B dev512','B - A'],['50','38.87% (199)','40.04% (205)','+6题'],['100','46.09% (236)','45.51% (233)','-3题'],['150','48.05% (246)','49.41% (253)','+7题']], [66,160,160,CW-386],True)
add('当前不能判定B稳定优于A；B尚未到200轮。A200完整终评七集平均43.12%。','small')
new()

# 05 - mechanism ablations
heading('04 C1与C2：真正针对A机制的消融')
section('C1 root-only：多代递归是否必要？')
add('仅把可选父集合改成[原根]；子轨迹始终generation=1，不能继续当父。起点均匀、合法边界、同父回退、异奖接纳、候选补齐及10次预算都保持。不是取消分叉，也不是五条独立从头rollout。[S9]')
story.append(Diagram('c1',100))
add('实现证据：第150轮512题共产生5,017条真实非根候选，全部generation=1；最终保留512条根和2,048条一代子，shortfall=0。实际每rank Adam累计3,000、跳过0，训练终态exit0。[S10]')
table([['iter','A','C1','C1 - A'],['50','38.87%','41.21%','+12题'],['100','46.09%','47.46%','+7题'],['150','48.05%','48.05%','0题']], [65,145,145,CW-355],True)
box('在这个单seed、150轮终点，取消多代递归没有降低汇总EM。A为NQ108/256、Hotpot138/256；C1为110/256、136/256。总分相同不表示逐题预测相同，也不能据此证明递归永久无用。')
section('C2 random-final：最终奖励差异筛选是否必要？')
add('构树严格复用A：奖励仍决定接纳父与回退停止。构树结束后固定保留原根，再从所有真实完整非根候选（L与F的并集）均匀无放回选4条，替换原最终训练组。用seed/题ID/outer派生的独立RNG，不推进构树RNG。它只去掉最终选择，不是取消构树中的reward指导。[S9]')
add('举例：同一候选树中，A最终组可能是奖序列[0,1,0,1,0]；C2的随机组可能为[0,1,1,1,1]，也可能偏向0。随机抽样不看奖励，例子不表示它倾向正确回答。相同权重和构树RNG下，短对照应生成同一候选树；更新后策略自然分化。')
add('当前完成48/150；dev5为100/512 = 19.53%，比A5多4题；无dev50/100/150。早期一点评测不能判断最终筛选有益或有害。C1/C2本轮均未安排51,713题终评。','small')
new()

# 06 - fixed validation results
heading('05 固定512题：同轮次结果放在一起')
rows=[['实验','iter5','iter50','iter100','iter150','iter200']]
for lab,r in zip(LABELS,RUNS):
    values={p['iter']:p['em'] for p in r['fixed512'] if p.get('em') is not None}
    rows.append([lab]+['不适用' if s>r.get('target_iter',200) else pct(values.get(s)) for s in [5,50,100,150,200]])
table(rows,[112]+[(CW-112)/5]*5,True)
story.append(LineChart(list(range(7)),'fixed',250))
add('点均为准确checkpoint轮次；未发布不补零、不拿邻近轮次代替。Nested的额外dev185=253/512（49.41%）不接到200轮端点。C2当前只有iter5，因此只画单点。','small')
section('目前能支持的读数')
add('• C1相对A在50/100轮更高，150轮打平；优势主要表现为早中期，尚无终点增益。<br/>• B相对A有正有负，150轮领先7题；缺200轮与完整终评。<br/>• BRANPO200比原Tree多1题，Nested150比原Tree多10题；两者均不是只改一个变量的统一消融。<br/>• 同seed、不同工程历史和采样量下的观测分差，不等于跨seed稳定算法收益。')
add('“iter150”比较的是更新轮数，不是严格相同生成token、候选数或GPU小时。Tree/BRAN固定五叶，Nested与递归搜索会额外生成被拒候选；仅比较2,560条保留轨迹会低估采样成本。','small')
new()

# 07 - training curves
heading('06 训练EM：看动态，不能替代固定验证')
story.append(LineChart([0,1,2],'train',220))
story.append(Spacer(1,8))
story.append(LineChart([3,4,5,6],'train',220))
rows=[['实验','已完成outer','末轮训练EM','近10轮均值']]
for lab,r in zip(LABELS,RUNS):rows.append([lab,f"{r['completed_iter']}/{r.get('target_iter',200)}",pct(r.get('train_latest_em')),pct(r.get('train_recent10_mean'))])
table(rows,[155,100,125,CW-380],True)
add('淡线为已完成outer原值，粗线为最近5轮均值；缺失点留空。C1首轮没有相应原生标量，曲线从第2轮起。训练EM受异奖选择、候选补齐和当前题目cohort影响；不能把“选中轨迹更难”直接解释为模型退化。','small')
new()

# 08 - final test and cost
heading('07 完整终评与成本：另一个比较维度')
add('仅以下三组已有完整51,713题终评。七集平均为等权macro；全题EM为按题数加权micro。两者不可混列排序。[S2]')
labels=['NQ','Trivia','PopQA','Hotpot','2Wiki','MuSiQue','Bamboogle']
finals=[next(r for r in F['ours'] if r['experiment']==s) for s in ['递归Tree A','Tree-GRPO','Tree-BRANPO 3360']]
rows=[['iter200']+labels+['七集平均']]
for r in finals:rows.append([r['experiment'].replace('递归Tree A','递归A').replace(' 3360','')]+[f'{100*r["dataset_em"][k]:.2f}' for k in F['datasets']]+[f'{100*r["macro_em"]:.2f}'])
table(rows,[96]+[(CW-96)/8]*8,True,True)
add('上表单位均为%。各集题数：3,610 / 11,313 / 14,267 / 7,405 / 12,576 / 2,417 / 125。','small')
table([['iter200','正确题数／51,713','全题EM','全题F1'],*[[r['experiment'].replace('递归Tree A','递归A'),f"{r['correct']:,}",pct(r['micro_em']),pct(r['f1'])] for r in finals]], [145,145,105,CW-395],True)
box('递归A在当前三组完整终评中最高：七集平均43.12%，比原Tree高1.13个百分点；但它同时改变根数、采样／筛选与有效目标形态，不能把增益全部归因于多代递归。BRANPO在dev200略高、完整终评较低，说明单个小验证集的微小优势可能不保持。')
section('C1与A的150轮成本证据')
ratio=(113750940-100589702)/113750940*100
table([['记录范围','A','C1'],['累计emitted token','113,750,940','100,589,702'],['原生step累计秒','105,651.570','100,951.584'],['step外rolling保存秒','未形成同等完整口径','6,640.051']], [172,168,CW-340],True)
add(f'C1记录的emitted token少{ratio:.2f}%，但包含padding与丢弃尾部，不是有效训练token；保存和计时覆盖不同，不能称总计算量节省{ratio:.2f}%。这些记录支持进一步做EM对实际生成token／GPU小时比较，尚不支持严格成本优越性。[S10]')
section('保留的数学限制')
add('真实token行为μ只提供生成概率分母；没有纳入“首次异奖接受”、位置选择、终选与fallback的归一化概率。因此奖励筛选后的PPO是selected-data surrogate，不是已证明对原始未筛选策略无偏的估计器。共享前缀也没有去重，不能称已解决节点信用分配。[S6-S9]')
new()

# 09 - plan registry and history
heading('08 参数消融登记、复用标签与历史实验')
section('已存在标签，不等于新增独立训练')
table([['标签','实际复用谁','如何解释'],['budget-10 / position-uniform / generation-4','递归A','同一组5/50/100/150成绩的基准标签。'],['generation-1','C1 root-only','单根且最大代数1时，与C1机制相同。']], [175,100,CW-275],True)
section('后续参数方向：代码已有定义，尚无独立有效成绩')
table([['轴','具体实现','结果状态'],['续写预算5 / 20','只改每题continuation总次数；根不计入，回退每次计入；最终目标仍5。','已有计划；当前汇总无独立评测节点。'],
 ['分叉偏早 / 偏晚','N个合法位置：偏早权重N,N-1,...,1；偏晚1,2,...,N；后续回退规则不改。','已有代码选项；当前汇总无独立评测节点。'],
 ['最大递归代数2','只允许generation < 2的记录作父；子generation最多2；其余按A。','已有代码选项；当前汇总无独立评测节点。']], [116,267,CW-383],True)
add('“无独立评测节点”仅描述截至本稿证据快照的产物，不等于代码没写或任务永久不跑；不得给计划填入A的成绩冒充新实验。C2完整节点尚未齐，闭环汇总为21/44标签节点，23项仍缺。[S9-S10]','small')
section('更早Tree历史：只作背景，不混入当前曲线')
table([['旧run','当时可核验结果','为何单列'],['3154：旧Instruct300','最佳iter100为203/512 = 39.65%；历史150/200降至0。','旧配置与恢复前链；不是当前guarded200参照。'],['3234：旧恢复实验','最佳iter110为210/512 = 41.02%。','恢复链与训练预算不同，不接到fresh200曲线上。']], [142,211,CW-353],True)
add('旧非Instruct资产与此前其他算法不纳入这7组当前机制比较；本稿不据缺少同口径成绩断言其更差。[S11]','small')
section('结论应停在哪里')
add('目前最直接的机制证据来自A与C1：取消多代递归在150轮没有降低汇总EM。最终筛选的作用还在等C2后续节点；浅层父加权的作用还在等B200及更严谨对照；Nested需等200与完整终评。若要证明“树方法为什么有效”，应按同一训练输入、同一硬件／预算、多个seed逐项控制，而非仅比较各组观测峰值。')
new()

# 10 - compact audit appendix
heading('附录 证据索引与复现定位','仅列小型审计、代码和必要配置；原始轨迹、权重与完整预测仍在远端。')
sources=[
 ('S1','结果快照与来源链','outputs/session-refresh-20261016-additions/{all-seven-results,a-updated-runs,tree-updated-runs,b-updated-runs}.json。包含逐outer标量、固定512题数与来源、完成账本；时间详见第1页。'),
 ('S2','完整七集终评','同目录final-seven-table-data.json；逐集EM与整数正确数重算macro和micro，总题数51,713；原Tree、BRANPO、A保留各自远端summary身份。'),
 ('S3','采集限制','tree-updated-runs.json中的actual_remote_observations与remote_connection_note：13:16:45／13:17:52实际远端输出；本轮socket超时未伪装fresh直读。'),
 ('S4','原Tree与共同目标','库A：search_r1/baselines/rollout.py:459；advantages.py:57；objectives.py:36；配置ppo_tree_fresh200.yaml与生产overrides。'),
 ('S5','有限尾部BRANPO','库A：release/bran_tree_grpo/README.md、production_overrides.json；rollout.py:510的_tail_branches；生产源码8b6f99d1。'),
 ('S6','Nested实现','库Tree：work/tree-speed-repair-20261008/nested-branch-source/search_r1/baselines/tree_nested_branch.py:51；outputs/tree-nested-branch/{IMPLEMENTATION_PLAN,MATH_REVIEW,CODE_REVIEW}.md。'),
 ('S7','递归A','库A：recursive_branch.py:125、rollout.py:_recursive_branches；release/recursive_bran_a/README.md；H100实际配置入口h100_migration/prepare_train.py。'),
 ('S8','递归B','库B：recursive_branch.py:36及47；rollout.py:649；运行CP169绑定源码65e96b32，与核心文件逐字节比对。'),
 ('S9','C1/C2与参数实现','库A：recursive_ablation.py:87及143；docs/recursive-mechanism-ablation-plan-20261014.md。A恢复参照源码a1ec6385，目标／RNG／预算独立冻结。'),
 ('S10','闭环消融与成本','库A：reports/recursive-ablation-{closed-comparison,C1-outer150,C2-first5}-20261016/下aggregate、summary、findings与来源SHA。'),
 ('S11','历史成绩','outputs/all-eight-session-history-512-ranking-20261007.json；只使用有512题、模型与source checkpoint身份的记录。'),
]
for key,title,text in sources:add(f'<font color="#168D7B">[{key}] {title}</font><br/>'+escape(text),'small')
section('路径缩写与会话')
add('库A / 库B：本工作区work/recursive-bran-experiment-a-20261011与-b-20261011。库Tree：/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-2。','small')
table([['会话','负责的主要setting'],['01a0fcbc-0b46-7502-8b66-d8b21a0e032d','原Tree、BRANPO、Nested'],['01a1028b-b626-7f91-93db-0099ff1623c5','递归A、C1、C2与后续参数'],['01a12e27-764b-7432-a5f2-74d88ee4a8d4','递归B']], [290,CW-290],True)
add('数据与代码审计边界：固定dev IDs与完整final IDs在A/B一致；ordered train IDs、prompt／row指纹不一致，训练集合逐字节相同未获证明。B文档的“相同A数据”对的是早期HKUST A，不能套用于当前H100 A。','small')

def page(c,doc):
    c.saveState();c.setFillColor(INK);c.rect(0,H-8,W,8,fill=1,stroke=0)
    c.setFont('CN',8);c.setFillColor(MUTED);c.drawString(M,H-34,'TREE EXPERIMENTS / 机制与消融审计');c.drawRightString(W-M,H-34,'2026-10-16')
    c.setStrokeColor(GRID);c.line(M,38,W-M,38);c.drawString(M,25,'基于实际代码与已核验产物；不同setting、数据口径分别保留')
    c.drawRightString(W-M,25,f'{doc.page} / 10');c.restoreState()
doc=SimpleDocTemplate(str(TARGET),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=57,bottomMargin=49,title='树相关实验：机制、消融与结果',author='Experiment Audit',subject='Tree-GRPO / BRANPO / Nested / Recursive A,B / C1,C2')
doc.build(story,onFirstPage=page,onLaterPages=page)
reader=PdfReader(str(TARGET));assert len(reader.pages)==10, f'Unexpected page overflow: {len(reader.pages)}'
text='\n'.join(p.extract_text() or '' for p in reader.pages)
for token in ['Tree-GRPO','Tree-BRANPO','Tree-Nested','C1 root-only','C2 random-final','48.05%','49.41%','51,713','generation']:
    assert token in text,token
assert '\ufffd' not in text
receipt={'pdf':str(TARGET),'pages':len(reader.pages),'bytes':TARGET.stat().st_size,'sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),
 'source_report_sha256':hashlib.sha256((D/'all-seven-results.json').read_bytes()).hexdigest(),'source_final_sha256':hashlib.sha256((D/'final-seven-table-data.json').read_bytes()).hexdigest(),
 'settings':LABELS,'data_freeze_cst':'2026-10-16: A13:19, Nested13:17, B12:54','new_remote_training_actions':False,'output_format':'pdf_only'}
(D/'tree-pdf-build-verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
