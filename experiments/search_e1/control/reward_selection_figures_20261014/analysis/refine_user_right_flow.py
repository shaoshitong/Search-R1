"""Re-layout only the user's right-side method flow in editable gray mxCells.

Every left-side cell is preserved, including the user's backgrounds, arrows,
axes, labels, curves and three robots. No explanatory copy is added.
"""
from pathlib import Path
import json, hashlib, datetime, xml.etree.ElementTree as E, copy, re, urllib.parse, zlib, base64

D=Path(__file__).resolve().parent
O=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison')
P=O/'hard-conditioned-soft-reward-amplification.drawio'
B=D/'right-flow-user-source-backup-20261015.json'
A=D/'right-flow-verification-20261015.json'
raw=P.read_bytes(); sha=hashlib.sha256(raw).hexdigest()
if not B.exists():
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=O/'backups';folder.mkdir(exist_ok=True)
    backup=folder/f'user-before-gray-right-flow-{stamp}.drawio';backup.write_bytes(raw)
    B.write_text(json.dumps({'path':str(backup),'sha256':sha,'bytes':len(raw),'preserved_at':stamp},indent=2))
backup=json.loads(B.read_text()); original=Path(backup['path']).read_bytes()
assert hashlib.sha256(original).hexdigest()==backup['sha256']
last=json.loads(A.read_text())['drawio_sha256'] if A.exists() else None
assert sha in [backup['sha256'],last], 'User has made subsequent changes; back up the new source rather than overwrite it.'

doc=E.fromstring(original); graph=doc.find('.//mxGraphModel/root'); cells=list(graph)
prefix='vV9Gi2tyWV5SowquxRN_-'
def right(c):
    cid=c.get('id',''); m=re.fullmatch(r'[ve](\d+)',cid)
    return bool(m and 370<=int(m.group(1))<=424) or cid in {prefix+str(n) for n in [8,9,10,11,12,13]}
right_ids=[c.get('id') for c in cells if right(c)]
protected=[c for c in cells if not right(c)]
def normalized(c):
    return {'tag':c.tag,'attrs':dict(sorted(c.attrib.items())),'text':c.text.strip() if c.text and c.text.strip() else None,'children':[normalized(k) for k in c]}
snap={c.get('id'):normalized(c) for c in protected}
old_text=[c.get('value') for c in cells if right(c) and c.get('value')]
for c in cells:
    if right(c):graph.remove(c)

serial=0
INK='#53585B'; EDGE='#72787C'; LIGHT='#B9BEC2'; LINE='#D6DADD'; FILL='#F5F6F6'; MUTED='#B8BEC2'
CX=787.5
def cell(value,style,x,y,w,h):
    global serial
    serial+=1
    c=E.Element('mxCell',id=f'gray-flow-v{serial}',value=value,style=style,vertex='1',parent='1')
    E.SubElement(c,'mxGeometry',x=f'{x:.4f}',y=f'{y:.4f}',width=f'{w:.4f}',height=f'{h:.4f}',attrib={'as':'geometry'})
    graph.append(c); return c
def box(x,y,w,h,fill='#FFFFFF',stroke=LINE,sw=1.2,r=12):
    return cell('',f'rounded=1;absoluteArcSize=1;arcSize={r};fillColor={fill};strokeColor={stroke};strokeWidth={sw};',x,y,w,h)
def text(value,x,y,w,h=26,size=17,color=INK,bold=False,italic=False):
    return cell(value,f'text;html=0;whiteSpace=nowrap;overflow=visible;fillColor=none;strokeColor=none;fontFamily=Times New Roman;fontSize={size};fontColor={color};fontStyle={int(bold)+2*int(italic)};align=center;verticalAlign=middle;spacing=0;',x,y,w,h)
def line(points,color=EDGE,width=1.6,arrow=False):
    global serial
    serial+=1
    c=E.SubElement(graph,'mxCell',id=f'gray-flow-e{serial}',edge='1',parent='1',style=f'edgeStyle=none;rounded=1;curved=0;html=0;strokeColor={color};strokeWidth={width};endArrow={"block" if arrow else "none"};endFill=1;endSize=8;startArrow=none;')
    g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
    E.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),attrib={'as':'sourcePoint'})
    E.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),attrib={'as':'targetPoint'})
    if len(points)>2:
        a=E.SubElement(g,'Array',attrib={'as':'points'})
        for x,y in points[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
    return c
def check(x,y,color=EDGE):line([(x-5,y),(x-1,y+4),(x+6,y-5)],color,1.65)
def cross(x,y,color=EDGE):
    line([(x-4,y-4),(x+4,y+4)],color,1.45);line([(x-4,y+4),(x+4,y-4)],color,1.45)

# A clean neutral column replaces only the user's right-hand green sketch frame.
container=box(671,118,233,1046,FILL,LIGHT,1.2,22)
# Keep this frame above the protected layers: the source contains a detached
# grid segment e337 entirely within the right column. This hides that stray
# segment without changing even one left-chart cell or its original geometry.
text('Easy\nSelection',697.5,156,180,52,22,INK,True,True)
line([(727.5,225),(847.5,225)],LINE,1)

# Three question papers have equal size and symmetric positions around the axis.
papers=[]
for i,(x,label) in enumerate([(722.5,'Q₁'),(769.5,'Q₂'),(816.5,'Qₙ')]):
    box(x+3,266,36,46,'#E8EBED',LINE,1,5)
    box(x,269,36,46,'#FFFFFF',EDGE,1.2,5)
    text(label,x+3,276,30,21,16,INK,True)
    line([(x+8,303),(x+28,303)],LIGHT,.9)
    line([(x+8,308),(x+22,308)],LIGHT,.9)
    papers.append({'x':x,'center':x+18,'label':label})
text('Question\nPool',707.5,339,160,42,17,INK,True)
line([(CX,391),(CX,435)],EDGE,1.8,True)

# Natural coordinates remove the old rotated parent/child group. The outcomes
# remain exactly the user's five checks / two checks / three checks.
outcomes=[]
for label,k,x in [('Q1',5,714.5),('Q2',2,766.5),('Q3',3,818.5)]:
    selected=k<5; col=EDGE if selected else MUTED
    box(x,463,42,236,'#FFFFFF' if selected else '#EFF1F2',EDGE if selected else '#D9DDE0',1.5 if selected else 1,9)
    text(label,x+4,473,34,24,15,col,True)
    line([(x+9,507),(x+33,507)],'#E1E4E6',.9)
    for i in range(5):
        if i<k:check(x+21,530+i*33,col)
        else:cross(x+21,530+i*33,col)
    outcomes.append({'label':label,'correct':k,'incorrect':5-k,'selected':selected,'center':x+21})
text('N rollouts',707.5,716,160,26,17,INK,True)
line([(CX,759),(CX,803)],EDGE,1.8,True)

# Symmetric funnel and selection stage, with the user's existing labels only.
stencil=E.Element('shape',name='Symmetric selection funnel',w='64',h='64',aspect='fixed',strokewidth='inherit')
bg=E.SubElement(stencil,'background');path=E.SubElement(bg,'path')
for tag,x,y in [('move',4,5),('line',60,5),('line',39,29),('line',39,57),('line',25,57),('line',25,29)]:E.SubElement(path,tag,x=str(x),y=str(y))
E.SubElement(path,'close');E.SubElement(bg,'fillstroke')
co=zlib.compressobj(wbits=-15);compressed=co.compress(urllib.parse.quote(E.tostring(stencil,encoding='unicode'),safe="~()*!.'").encode())+co.flush()
encoded=base64.b64encode(compressed).decode()
cell('',f'shape=stencil({encoded});fillColor=#E2E6E8;strokeColor={EDGE};strokeWidth=1.5;',CX-30,817,60,60)
box(CX-79,895,158,64,'#FFFFFF',EDGE,1.3,12)
text('std(EM) > 0',CX-74,901,148,26,18,INK,True)
text('diverse groups',CX-74,928,148,23,16,EDGE)
line([(CX,984),(CX,1028)],EDGE,1.8,True)
box(CX-85,1065,170,54,'#E7EAEC',LIGHT,1.2,12)
text('GRPO Training',CX-80,1078,160,27,18,INK,True)

# Check preservation, wording and equal spacing before saving the user's file.
after={c.get('id'):c for c in graph}
assert all(normalized(after[cid])==s for cid,s in snap.items())
new_text=[c.get('value') for c in graph if c.get('id','').startswith('gray-flow-') and c.get('value')]
assert sorted(old_text)==sorted(new_text),(old_text,new_text)
assert [r['correct'] for r in outcomes]==[5,2,3]
assert [r['selected'] for r in outcomes]==[False,True,True]
assert papers[0]['center']+papers[2]['center']==2*CX
assert outcomes[0]['center']+outcomes[2]['center']==2*CX
assert outcomes[1]['center']==CX
doc.set('rightFlowStyle','user-logic-gray-symmetric-v1')
E.indent(doc,space='  ');E.ElementTree(doc).write(P,encoding='utf-8',xml_declaration=True)
check_doc=E.parse(P);check_graph=check_doc.getroot().find('.//mxGraphModel/root')
final={c.get('id'):c for c in check_graph}
assert all(normalized(final[cid])==s for cid,s in snap.items())
protected_digest=hashlib.sha256(json.dumps(snap,sort_keys=True).encode()).hexdigest()
audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_backup':backup,'drawio_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'left_cells_preserved':len(snap),'protected_cells_sha256':protected_digest,'right_original_cells_replaced':len(right_ids),'right_new_cells':serial,'added_explanatory_text':False,'text_values_unchanged':sorted(old_text),'right_bounds':{'x':671,'y':118,'width':233,'height':1046},'centerline':CX,'arrows':'three identical 44-unit vertical arrows, centerline x787.5','question_centers':[q['center'] for q in papers],'rollout_groups':outcomes,'palette':[INK,EDGE,LIGHT,LINE,FILL,MUTED],'left_curves_axes_labels_frames_robots_unchanged':True,'native_editable':True}
A.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps({'file':str(P),'protected':len(snap),'right_original':len(right_ids),'right_new':serial,'added_text':False},ensure_ascii=False))
