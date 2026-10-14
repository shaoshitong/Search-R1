"""Editable three-chart overview from the user's latest drawing and exact aggregates.

Preserve the incoming drawing before replacing it. No data smoothing or invented
curves; icons and diagrams are native editable mxCells.
"""
from pathlib import Path
import json, math, hashlib, datetime, shutil, xml.etree.ElementTree as E

D=Path(__file__).resolve().parent
ns={'__file__':str(D/'make_drawio.py')}
exec(compile((D/'make_drawio.py').read_text().split('curve=json.load')[0],str(D/'make_drawio.py'),'exec'),ns)
Draw=ns['Draw']; OUT=ns['OUT']
P=OUT/'hard-conditioned-soft-reward-amplification.drawio'
BACK=D/'selection-storyboard-source-backup.json'
VERIFY=D/'selection-storyboard-verification.json'
incoming=P.read_bytes(); incoming_sha=hashlib.sha256(incoming).hexdigest()
if BACK.exists():
    previous=json.loads(VERIFY.read_text())['drawio_sha256'] if VERIFY.exists() else None
    recorded=json.loads(BACK.read_text())
    assert incoming_sha in [recorded['sha256'],previous], 'New user edits found; preserve them before regenerating.'
else:
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=OUT/'backups'; folder.mkdir(exist_ok=True)
    backup=folder/f'user-three-chart-storyboard-{stamp}.drawio'
    backup.write_bytes(incoming)
    BACK.write_text(json.dumps({'path':str(backup),'sha256':incoming_sha,'bytes':len(incoming),'preserved_at':stamp},indent=2))

data=json.loads((D/'hard-conditioned-soft-data-20261015.json').read_text())
row=next(r for r in data['steps'] if r['step']==20)
z=row['conditions']['all_correct']; mixed=row['conditions']['hard_std_positive']
curve=json.loads((D/'fixed512-curve-data.json').read_text())

INK='#293943'; MUTED='#73818B'; GRID='#E6EDF0'; PURPLE='#8C6BB1'; TEAL='#168D7B'
BLUE='#397CB5'; PERF_PURPLE='#9171B4'; PERF_TEAL='#128E7D'
LAV='#F3EFF8'; MINT='#EAF6F1'; CREAM='#FFF7E8'; BORDER='#CCD9DE'
f=Draw(); f.page('Observation → normalization → selection',1220,1245)
f.file.set('storyboardVersion','three-chart-selection-v1')
f.file.set('conditionDefinition','all-correct-vs-mixed; all-wrong-excluded')

def txt(s,x,y,w,h=24,size=18,color=INK,align='left',bold=False):
    return f.text(s,x,y,w,h,size,color,align,bold)
def box(x,y,w,h,fill='#FFFFFF',stroke=BORDER,r=18,sw=1.3):
    return f.cell('',f'rounded=1;arcSize={r};absoluteArcSize=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};',x,y,w,h)
def ellipse(x,y,w,h,fill,stroke='none',sw=1.2):
    return f.cell('',f'ellipse;fillColor={fill};strokeColor={stroke};strokeWidth={sw};',x,y,w,h)
def line(pts,color=INK,width=1.5,dash=False): f.line(pts,color,width,dash)
def arrow(pts,color=TEAL,width=3.5,curved=True):
    f.serial+=1
    c=E.SubElement(f.root,'mxCell',id='e'+str(f.serial),edge='1',parent='1',style=f'edgeStyle=none;curved={int(curved)};rounded=1;html=0;endArrow=block;endFill=1;endSize=9;startArrow=none;strokeColor={color};strokeWidth={width};')
    g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    E.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'})
    E.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
    if len(pts)>2:
        a=E.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in pts[1:-1]: E.SubElement(a,'mxPoint',x=str(x),y=str(y))
    return c
def tick(x,y,s=1,color=TEAL): line([(x,y+4*s),(x+4*s,y+8*s),(x+12*s,y-2*s)],color,2.2*s)
def cross(x,y,s=1,color='#B3756B'):
    line([(x,y),(x+9*s,y+9*s)],color,1.8*s); line([(x+9*s,y),(x,y+9*s)],color,1.8*s)
def sparkle(x,y,s=1,color='#D2AC67'):
    line([(x,y-6*s),(x,y+6*s)],color,1.4); line([(x-6*s,y),(x+6*s,y)],color,1.4)
def robot(x,y,s=1,face='happy',color=BLUE):
    line([(x+30*s,y),(x+30*s,y-10*s)],color,2*s)
    ellipse(x+26*s,y-15*s,8*s,8*s,CREAM,color,1.5*s)
    box(x,y,60*s,45*s,'#FFFFFF',color,12*s,1.8*s)
    box(x+8*s,y+9*s,44*s,25*s,'#EDF5F8','none',7*s)
    for ex in [17,38]: ellipse(x+ex*s,y+15*s,5*s,7*s,color)
    for ex in [12,43]: ellipse(x+ex*s,y+25*s,6*s,3*s,'#F4D1CE')
    if face=='happy':
        c=arrow([(x+25*s,y+27*s),(x+30*s,y+31*s),(x+35*s,y+27*s)],color,1.3*s)
        c.set('style',c.get('style').replace('endArrow=block','endArrow=none'))
    else:
        line([(x+26*s,y+28*s),(x+34*s,y+28*s)],color,1.3*s)
        line([(x+17*s,y+10*s),(x+24*s,y+12*s)],color,1.2*s)
    box(x+9*s,y+49*s,42*s,22*s,color,color,8*s)
    ellipse(x+26*s,y+55*s,7*s,7*s,'#FFFFFF')
    line([(x+9*s,y+54*s),(x-2*s,y+45*s)],color,2.2*s)
    line([(x+51*s,y+54*s),(x+65*s,y+45*s)],color,2.2*s)
    for ex in [16,38]: box(x+ex*s,y+73*s,12*s,5*s,CREAM,color,3*s,1.2*s)
def magnifier(x,y,s=1):
    ellipse(x,y,36*s,36*s,'#FFFFFF',PURPLE,2*s)
    line([(x+29*s,y+29*s),(x+44*s,y+44*s)],PURPLE,5*s)
    line([(x+10*s,y+10*s),(x+14*s,y+7*s)],'#B7A2CE',2*s)
def qpaper(x,y,label='Q',s=1,fill='#FFFFFF',color=BLUE):
    box(x+4*s,y-4*s,36*s,43*s,'#F1F5F6',BORDER,6*s)
    box(x,y,36*s,43*s,fill,color,6*s)
    txt(label,x+3*s,y+5*s,30*s,20*s,18*s,color,'center',True)
    for dy,w in [(29,19),(34,13)]: line([(x+7*s,y+dy*s),(x+(7+w)*s,y+dy*s)],color,.8*s)
def funnel(x,y,s=1):
    f.cell('',f'shape=mxgraph.flowchart.manual_operation;direction=north;fillColor={MINT};strokeColor={TEAL};strokeWidth=1.5;',x,y,47*s,32*s)
    box(x+19*s,y+30*s,9*s,18*s,'#FFFFFF',TEAL,2*s)
    sparkle(x+43*s,y-3*s,.5,TEAL)
def badge(n,label,x,y,color,fill):
    ellipse(x,y,28,28,fill,color,1.2)
    txt(str(n),x,y,28,28,16,color,'center',True)
    txt(label,x+39,y-1,500,29,23,INK,bold=True)
def axes(y,lo,hi,ymax,xt,yt,xlabel,ylabel):
    x=126; w=510; h=220
    X=lambda v:x+(v-lo)/(hi-lo)*w
    Y=lambda v:y+h-v/ymax*h
    for v in yt:
        line([(x,Y(v)),(x+w,Y(v))],GRID,.9)
        txt(f'{v:g}',x-43,Y(v)-10,32,20,15,MUTED,'right')
    for v in xt:
        line([(X(v),y+h),(X(v),y+h+5)],MUTED,.8)
        txt(f'{v:g}',X(v)-26,y+h+10,52,21,15,MUTED,'center')
    line([(x,y),(x,y+h),(x+w,y+h)],'#7A8991',1.3)
    txt(xlabel,x,y+h+41,w,23,17,INK,'center')
    c=txt(ylabel,5,y+h/2-12,90,24,17,INK,'center');c.set('style',c.get('style').replace('whiteSpace=wrap','whiteSpace=nowrap')+'rotation=270;')
    return X,Y
def histogram(h,X,Y,color,total):
    assert sum(h['counts'])==total
    pts=[(X(h['edges'][0]),Y(0))]
    for i,n in enumerate(h['counts']):
        x0,x1=X(h['edges'][i]),X(h['edges'][i+1]); yy=Y(100*n/total)
        if n: f.cell('',f'fillColor={color};fillOpacity=10;strokeColor=none;',x0,yy,x1-x0,Y(0)-yy)
        pts.extend([(x0,yy),(x1,yy)])
    pts.append((X(h['edges'][-1]),Y(0)))
    clean=[]
    for p in pts:
        if not clean or clean[-1]!=p:clean.append(p)
    line(clean,color,2.1)
def mini_legend(label,x,y,color,w=115,dash=False,marker=None):
    line([(x,y+10),(x+26,y+10)],color,2.4,dash)
    if marker: f.marker(x+13,y+10,color,marker,8)
    txt(label,x+35,y,w,21,15,color)

# One shared frame, three equal plotting areas, and a restrained visual story.
txt('From Equal Correctness to Better Question Selection',35,22,1150,37,28,INK,'center',True)
box(30,80,1160,1118,'#FFFFFF',BORDER,25,1.5)
for y,fill in [(102,'#F5F8FC'),(462,LAV),(822,MINT)]:
    box(731,y,431,302,fill,'none',19)
for y in [442,802]: line([(55,y),(1165,y)],'#E9EEF0',1.1)

# (a): true histogram at iter20; unchanged bins/counts, legible linear axes.
badge(1,'(a) Original Soft Reward',82,113,BLUE,'#EAF2FA')
txt('Same rollout snapshot · iteration 20',126,146,500,20,14,MUTED)
X,Y=axes(180,-.15,1.65,40,[0,.5,1,1.5],[0,10,20,30,40],'Soft reward r','Bin frequency (%)')
for s,col in [(mixed,TEAL),(z,PURPLE)]: histogram(s['soft_reward_hist'],X,Y,col,s['trajectories'])
mini_legend('All correct',365,183,PURPLE)
mini_legend('Mixed',515,183,TEAL,70)
txt('All correct: EM = [1, 1, 1, 1, 1]',755,118,380,25,20,PURPLE,bold=True)
for i in range(5):
    box(760+i*62,161,48,35,'#FFFFFF','#CBD9CC',10)
    tick(774+i*62,174,.9)
txt('Hard diversity = 0',759,212,295,25,19,TEAL)
robot(767,265,.75,'happy',BLUE)
magnifier(831,254,.65)
txt('Equal correctness,',890,264,252,26,21,INK,bold=True)
txt('different Soft scores.',890,292,250,26,21,INK,bold=True)
txt('The purple distribution still has spread.',758,365,390,23,16,MUTED)

# The transitions sit in the same frame and use curved, rounded native arrows.
arrow([(946,412),(978,432),(978,452),(946,472)],PURPLE,3.6)
box(748,420,165,39,'#FFFFFF','#D9CFE6',12,1)
txt('Normalize per question',752,424,157,31,15,PURPLE,'center')

# (b): all-correct Hard A is identically zero, whereas actual Soft A is broad.
badge(2,'(b) Standardized Advantage',82,473,PURPLE,LAV)
txt('Same groups · A = (r − mean) / (sample std + 10⁻⁶)',126,506,530,20,14,MUTED)
X,Y=axes(540,-1.85,1.85,20,[-1.5,-1,-.5,0,.5,1,1.5],[0,5,10,15,20],'Standardized Soft advantage A','Bin frequency (%)')
line([(X(0),540),(X(0),760)],'#BEC8CF',1,True)
for s,col in [(mixed,TEAL),(z,PURPLE)]: histogram(s['soft_advantage_hist'],X,Y,col,s['trajectories'])
mini_legend('All correct',365,543,PURPLE)
mini_legend('Mixed',515,543,TEAL,70)
txt('All correct ≠ zero Soft advantage',754,478,390,25,20,PURPLE,bold=True)
box(759,527,384,59,'#FFFFFF','#D9CFE6',13,1)
txt('Hard EM:  A = 0 for all five responses',775,540,360,28,19,TEAL)
robot(768,626,.75,'concerned',PURPLE)
ellipse(881,620,35,35,'#EAE2F1');txt('−',881,620,35,35,27,PURPLE,'center')
ellipse(934,620,35,35,'#EAE2F1');txt('+',934,620,35,35,26,PURPLE,'center')
txt('Soft A spans −1.79 to +1.77',881,666,267,26,18,PURPLE,bold=True)
txt('Relative preferences appear even when\nall answers have the same correctness.',758,727,390,44,17,MUTED)
sparkle(1092,616,.6,PURPLE)

arrow([(946,772),(978,792),(978,812),(946,832)],TEAL,3.6)
box(748,781,165,39,'#FFFFFF','#C7E1D7',12,1)
txt('Prefer diverse questions',752,785,157,31,15,TEAL,'center')

# (c): all three actual evaluation series, same width as (a) and (b).
badge(3,'(c) Easy Solution',82,833,TEAL,MINT)
mini_legend('Soft Reward',127,870,BLUE,104,marker='ellipse')
mini_legend('Soft → Hard',279,870,PERF_PURPLE,119,True,'rhombus')
mini_legend('Easy Selection + Soft Reward',448,870,PERF_TEAL,225,False,'triangle')
x=126;y=918;w=510;h=197
XX=lambda v:x+(v-45)/160*w; YY=lambda v:y+h-(v-35)/20*h
for v in [35,40,45,50,55]:
    line([(x,YY(v)),(x+w,YY(v))],GRID,.9);txt(str(v),x-43,YY(v)-10,32,20,15,MUTED,'right')
for v in [50,100,150,200]:
    line([(XX(v),y+h),(XX(v),y+h+5)],MUTED,.8);txt(str(v),XX(v)-26,y+h+10,52,21,15,MUTED,'center')
line([(x,y),(x,y+h),(x+w,y+h)],'#7A8991',1.3)
txt('Training iteration',x,y+h+41,w,23,17,INK,'center')
cc=txt('Fixed-512 EM (%)',5,y+h/2-12,90,24,17,INK,'center');cc.set('style',cc.get('style').replace('whiteSpace=wrap','whiteSpace=nowrap')+'rotation=270;')
for j,s in enumerate(curve['series']):
    color=[BLUE,PERF_PURPLE,PERF_TEAL][j]
    pts=[(XX(p['iteration']),YY(p['em']*100)) for p in s['points']]
    line(pts,color,2.7,j==1)
    for px,py in pts:f.marker(px,py,color,['ellipse','rhombus','triangle'][j],9)
    p=s['points'][-1];px,py=pts[-1]
    txt(f'{p["em"]*100:.2f}%',px-71,py+([8,-24,-27][j]),80,22,17,color,'right',True)

# Compact native icon schematic; selection depends on hard EM diversity, not Soft std.
txt('Roll out, select, then train',754,838,390,27,21,TEAL,bold=True)
qpaper(763,894,'Q₁',.75);qpaper(777,921,'Q₂',.75);qpaper(791,948,'Qₙ',.75)
txt('Question pool',751,982,120,23,15,MUTED,'center')
arrow([(840,923),(861,923),(881,923)],TEAL,2.6,False)
txt('5 rollouts / question',886,879,260,24,16,INK,'center')
for i,k in enumerate([5,2,3]):
    yy=909+i*34; col='#AAB6B1' if k==5 else TEAL
    box(888,yy,254,29,'#FFFFFF','#D9E4DF',8,1)
    txt('Q'+str(i+1),894,yy+2,28,25,14,col,'center')
    for j in range(5):
        if j<k:tick(937+37*j,yy+11,.65,col)
        else:cross(937+37*j,yy+10,.65,'#B8786D')
    if k!=5:box(887,yy-1,256,31,'none',TEAL,8,1.5)
arrow([(1015,1016),(1015,1036),(981,1051)],TEAL,2.7)
funnel(863,1041,.7)
box(917,1048,143,58,'#FFFFFF',TEAL,12,1.3)
txt('std(EM) > 0',923,1053,132,26,17,TEAL,'center',True)
txt('diverse groups',923,1079,132,20,15,MUTED,'center')
arrow([(1066,1075),(1081,1075),(1095,1075)],TEAL,2.4,False)
robot(1100,1039,.59,'happy',TEAL)
txt('Train',1096,1098,61,22,15,TEAL,'center')
txt('Mixed-first; all-correct fallback if needed.',756,1133,394,24,15,MUTED)
box(757,1161,387,27,'#F4FBF7','none',8)
txt('200 iter: 52.73% EM  ·  +11.33 pp vs Soft',766,1162,372,25,16,TEAL,'center',True)

txt('Distributions: same iter20 rollout groups; all-wrong excluded. No smoothing. 43 all-correct / 169 mixed groups.',43,1205,1140,21,13,MUTED,'center')
txt('Soft preferences among equal-EM answers do not by themselves prove incorrect gradients. Curves are observed evaluation results.',43,1226,1140,18,12,MUTED,'center')

f.write(P.name)
# Check native geometry, exact data, single page, and a bounded canvas.
root=E.parse(P).getroot(); pages=root.findall('diagram'); assert len(pages)==1
cells=pages[0].find('mxGraphModel/root').findall('mxCell'); ids=[c.get('id') for c in cells]; assert len(ids)==len(set(ids))
assert not any('shape=image' in c.get('style','') or 'image=' in c.get('style','') for c in cells)
for c in cells:
    if c.get('vertex')=='1':
        g=c.find('mxGeometry');xx=float(g.get('x',0));yy=float(g.get('y',0));ww=float(g.get('width',0));hh=float(g.get('height',0))
        assert xx>=0 and yy>=0 and xx+ww<=1220.1 and yy+hh<=1245.1,(c.get('id'),xx,yy,ww,hh)
for s in [z,mixed]:
    for key in ['soft_reward_hist','soft_advantage_hist']: assert sum(s[key]['counts'])==s['trajectories']
for s in curve['series']:
    assert [p['iteration'] for p in s['points']]==[50,100,150,200]
    for p in s['points']: assert p['n']==512 and p['em']==p['correct']/512
audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_backup':json.loads(BACK.read_text()),'drawio_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'pages':1,'native_editable_cells':len(cells),'images_embedded':False,'chart_widths':[510,510,510],'distribution_iteration':20,'histogram_counts_unchanged':True,'all_correct_groups':z['groups'],'mixed_groups':mixed['groups'],'all_wrong_excluded':True,'normalization':'per-question sample std, correction=1, epsilon=1e-6','selection':'prefer Hard EM std>0 (mixed-first), all-correct fallback if needed; not Soft std sorting','fixed512_points':[{ 'name':s['name'],'points':[{k:p[k] for k in ['iteration','correct','n','em']} for p in s['points']]} for s in curve['series']],'arrow_style':'native curved connectors with filled arrowheads','scope':'diagram only; no training changes'}
VERIFY.write_text(json.dumps(audit,indent=2))
print(P)
