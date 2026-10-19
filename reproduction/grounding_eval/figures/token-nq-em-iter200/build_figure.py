"""Audited normal-inference NQ cost / accuracy figure; all points iter200.

Editable vector primitives in diagrams.net. Export with draw.io desktop CLI.
The input files are small summaries; raw inference traces stay on their clusters.
"""
import json, hashlib, math
from pathlib import Path
import xml.etree.ElementTree as E

OUT = Path(__file__).resolve().parent
SRC = Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs')
SIX = SRC/'grounding-eval-20261019/six-method-grounding-final-20261019.json'
DYN = SRC/'dynrpo-nq-token-cost-20261019/dynrpo-normal-token-cost-final.json'
REF = OUT.parent/'time-em-iter50-100/time-em-iter50-100.pdf'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
six, dyn = json.loads(SIX.read_text()), json.loads(DYN.read_text())
labels = ['Search-G1', 'TOP-D (α=0.1)', 'OPD (α=1)', 'OPSD',
          'Soft->Hard', 'Easy Selection', 'DynRPO']
rows = []
for i, m in enumerate(six['methods']):
    n = m['normal']; c = n['inference_cost_tokens']
    assert n['questions'] == c['questions'] == 3610
    assert abs(n['EM'] - n['correct']/3610) < 1e-12
    assert c['measured_input_total'] + c['measured_raw_output_total'] == c['input_plus_raw_output_total']
    rows.append(dict(code=chr(65+i), method=m['method'], label=labels[i], iteration=200,
        questions=3610, correct=n['correct'], em_percent=100*n['EM'],
        input_tokens=c['measured_input_total'], raw_generated_tokens=c['measured_raw_output_total'],
        total_tokens=c['input_plus_raw_output_total'],
        mean_tokens=c['mean_input_plus_raw_output_per_question']))
d = dyn['comparison_with_original_six'][-1]
assert d['method'] == 'DynRPO' and dyn['checkpoint_iteration'] == 200
rows.append(dict(code='G', method='DynRPO', label=labels[-1], iteration=200,
    questions=d['questions'], correct=round(dyn['NQ_EM']*d['questions']),
    em_percent=100*dyn['NQ_EM'], input_tokens=d['cumulative_input_tokens'],
    raw_generated_tokens=d['raw_generated_tokens'], total_tokens=d['total'], mean_tokens=d['mean']))
for r in rows:
    assert math.isclose(r['mean_tokens'], r['total_tokens']/r['questions'], abs_tol=1e-10)
assert len(rows) == 7
audit = dict(axes=dict(x='Mean cumulative actual input + raw generated tokens per NQ question',
    y='Normal inference NQ exact match (%)'), checkpoint_iteration=200,
    sample_size=3610, points=rows,
    source_files=[dict(path=str(p), sha256=sha(p)) for p in [SIX,DYN,REF]],
    canonical_question_gold_order_sha256=dyn['canonical_question_gold_order_sha256'],
    limitations=[
        'Paired measured NQ EM and token cost from the same normal inference evaluation for each model; historical seven-dataset NQ scores are not substituted.',
        'All actual input history is counted again on each generation request. Raw completion tails discarded by legacy action parsers are included. CF, pilot, training, and probe costs are excluded.',
        'DynRPO aborts native generation at the first complete action; historical adapters may generate discarded tails. This is a native-implementation usage comparison, not a controlled estimate of policy-only token efficiency.',
        'Original six adapters retained native per-method loops. Retrieval fanout migration: OPSD 8->2 and Easy Selection 8->1; cropped document parity sampled, not full-run original query parity.',
        'Single measurement per model; no run-level variance or invented error bars. Points across algorithms are not connected.'],
    style='Latest user-edited time-em-iter50-100 PDF: italic Times, hollow markers, muted blue / brick red, horizontal dotted grid, blue hachure inset legend.')
(OUT/'data-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')

W,H=2010,600
mx=E.Element('mxfile',host='app.diagrams.net',version='28.2.5')
dia=E.SubElement(mx,'diagram',id='nq-token-em-200',name='NQ EM vs normal inference tokens - iter200')
model=E.SubElement(dia,'mxGraphModel',dx=str(W),dy=str(H),grid='0',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='0',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF')
root=E.SubElement(model,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
count=0
INK='#35444E';BLUE='#2E738F';RED='#A34B40';AXIS='#52616C'
def ident():
    global count
    count+=1;return 'cell-'+str(count)
def rect(value,x,y,w,h,style,id=None):
    c=E.SubElement(root,'mxCell',id=id or ident(),value=value,style=style,vertex='1',parent='1')
    E.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
def text(s,x,y,w,h,size=20,color=INK,bold=False,align='center',rotation=0,bg='none',id=None):
    return rect(s,x,y,w,h,f'text;html=0;fillColor={bg};strokeColor=none;fontFamily=Times New Roman;fontSize={size};fontStyle={3 if bold else 2};fontColor={color};align={align};verticalAlign=middle;spacing=0;whiteSpace=wrap;overflow=visible;rotation={rotation};',id)
def line(x1,y1,x2,y2,color=AXIS,width=1.5,dashed=False,sketch=False,id=None):
    style=f'edgeStyle=none;endArrow=none;startArrow=none;strokeColor={color};strokeWidth={width};'
    if dashed:style+='dashed=1;dashPattern=4 5;'
    if sketch:style+='sketch=1;jiggle=0.18;disableMultiStroke=1;'
    c=E.SubElement(root,'mxCell',id=id or ident(),edge='1',parent='1',style=style)
    g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    E.SubElement(g,'mxPoint',x=str(x1),y=str(y1),attrib={'as':'sourcePoint'})
    E.SubElement(g,'mxPoint',x=str(x2),y=str(y2),attrib={'as':'targetPoint'})
rect('',0,0,W,H,'fillColor=#FFFFFF;strokeColor=none;','artboard')
L,R,T,B=110,1900,35,495
XMIN,XMAX=800,2800;YMIN,YMAX=28,50
X=lambda x:L+(x-XMIN)/(XMAX-XMIN)*(R-L)
Y=lambda y:B-(y-YMIN)/(YMAX-YMIN)*(B-T)
for yy in [28,32,36,40,44,48,50]:
    y=Y(yy)
    if yy>28:line(L,y,R,y,'#E3E8EB',.75,True)
    line(L-5,y,L,y,AXIS,1)
    text(str(yy),L-52,y-15,38,30,20,align='right')
for xx in [800,1200,1600,2000,2400,2800]:
    x=X(xx);line(x,B,x,B+6,AXIS,1);text(f'{xx:,}',x-39,B+12,78,30,20)
line(L,T,L,B,AXIS,1.5,sketch=True);line(L,B,R,B,AXIS,1.5,sketch=True)
text('NQ EM (%)',-30,251,180,30,23,rotation=270,id='axis-y-title')
text('Mean tokens per question (cumulative input + raw generation)',525,550,1060,34,23,id='axis-x-title')
text('Iteration 200',930,44,200,32,23,color=RED,id='iteration-key')
# Label offsets separate the near-identical token coordinates of E and G.
offsets={'A':(31,10),'B':(25,-20),'C':(25,-4),'D':(-28,9),
         'E':(-28,8),'F':(26,-19),'G':(-28,-25)}
for r in rows:
    x,y=X(r['mean_tokens']),Y(r['em_percent']); col=BLUE if r['code']=='G' else RED
    shape='ellipse' if r['code']=='G' else 'rhombus';s=12
    rect('',x-s/2,y-s/2,s,s,f'{shape};fillColor=#FFFFFF;strokeColor={col};strokeWidth=1.9;sketch=1;jiggle=0.12;disableMultiStroke=1;',id='point-'+r['code'])
    dx,dy=offsets[r['code']];lx,ly=x+dx,y+dy
    line(x+dx*.23,y+dy*.23,lx,ly,col,.7)
    text(r['code'],lx-10,ly-12,20,24,17,col,True,bg='#FFFFFF',id='label-'+r['code'])
# Same latest-reference hachure legend, placed between high and low points.
lx,ly,lw,lh=530,280,875,132
rect('',lx,ly,lw,lh,'rounded=1;arcSize=9;absoluteArcSize=1;fillColor=#B9D6EC;fillOpacity=88;strokeColor=#7FA6C1;strokeOpacity=85;strokeWidth=1.15;sketch=1;jiggle=0.18;disableMultiStroke=1;fillStyle=hachure;hachureGap=5;fillWeight=0.8;',id='method-legend-background')
for i,r in enumerate(rows):
    x=lx+20+(i//4)*437;y=ly+8+(i%4)*29
    col=BLUE if r['code']=='G' else INK
    text(r['code'],x,y,23,27,18,col,True,id='legend-code-'+r['code'])
    text(r['label'],x+36,y,370,27,18,col,align='left',id='legend-name-'+r['code'])
E.indent(mx)
(OUT/'token-nq-em-iter200.drawio').write_bytes(E.tostring(mx,encoding='utf-8',xml_declaration=True))
assert len(root.findall("mxCell[@id='point-G']"))==1
print(json.dumps([dict(code=r['code'],method=r['label'],tokens=r['mean_tokens'],EM=r['em_percent']) for r in rows],indent=2))
