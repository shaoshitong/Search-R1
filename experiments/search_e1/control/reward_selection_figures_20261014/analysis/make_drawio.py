# -*- coding: utf-8 -*-
from pathlib import Path
import json,xml.etree.ElementTree as E,math,shutil
D=Path(__file__).resolve().parent
OUT=Path('/Users/shaoshitong/Downloads/WWW_2026_PAPER/drawio_figure/reward-selection-comparison');OUT.mkdir(exist_ok=True)
COLORS=['#397CB5','#9171B4','#128E7D'];TEXT='#27323A';MUTED='#697780';GRID='#E7EDF1'
class Draw:
 def __init__(self):self.file=E.Element('mxfile',host='Electron',version='28.0.6');self.serial=0
 def page(self,name,w,h):
  p=E.SubElement(self.file,'diagram',name=name,id='page-'+str(len(self.file)));m=E.SubElement(p,'mxGraphModel',dx=str(w),dy=str(h),grid='0',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),background='#ffffff',math='0',shadow='0');self.root=E.SubElement(m,'root');E.SubElement(self.root,'mxCell',id='0');E.SubElement(self.root,'mxCell',id='1',parent='0')
 def cell(self,value,style,x,y,w,h):
  self.serial+=1;c=E.SubElement(self.root,'mxCell',id='v'+str(self.serial),value=str(value),style=style,vertex='1',parent='1');E.SubElement(c,'mxGeometry',x=f'{x:.4f}',y=f'{y:.4f}',width=f'{w:.4f}',height=f'{h:.4f}',as_='geometry').attrib['as']='geometry';del c[0].attrib['as_'];return c
 def text(self,s,x,y,w,h=28,size=20,color=TEXT,align='left',bold=False,rotate=0):
  return self.cell(s,f'text;html=0;whiteSpace=wrap;overflow=visible;fillColor=none;strokeColor=none;fontFamily=Times New Roman;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};align={align};verticalAlign=middle;spacing=0;rotation={rotate};',x,y,w,h)
 def line(self,points,color=TEXT,width=1,dash=False):
  self.serial+=1;c=E.SubElement(self.root,'mxCell',id='e'+str(self.serial),style=f'edgeStyle=none;rounded=0;html=0;endArrow=none;startArrow=none;strokeColor={color};strokeWidth={width};dashed={int(dash)};dashPattern=8 5;',edge='1',parent='1');g=E.SubElement(c,'mxGeometry',relative='1');g.set('as','geometry');E.SubElement(g,'mxPoint',x=f'{points[0][0]:.4f}',y=f'{points[0][1]:.4f}',as_='sourcePoint').attrib['as']='sourcePoint';del g[0].attrib['as_'];end=E.SubElement(g,'mxPoint',x=f'{points[-1][0]:.4f}',y=f'{points[-1][1]:.4f}');end.set('as','targetPoint')
  if len(points)>2:
   a=E.SubElement(g,'Array');a.set('as','points')
   for x,y in points[1:-1]:E.SubElement(a,'mxPoint',x=f'{x:.4f}',y=f'{y:.4f}')
 def marker(self,x,y,color,shape='ellipse',size=11):self.cell('',f'shape={shape};rotation={270 if shape=="triangle" else 0};aspect=fixed;fillColor=#ffffff;strokeColor={color};strokeWidth=2.5;',x-size/2,y-size/2,size,size)
 def bar(self,x,y,w,h,color,opacity=80):self.cell('',f'rounded=0;fillColor={color};fillOpacity={opacity};strokeColor={color};strokeWidth=.8;',x,y,max(w,.2),max(h,.2))
 def axes(self,x,y,w,h,xmin,xmax,ymin,ymax,xt,yt,xlabel,ylabel):
  X=lambda v:x+(v-xmin)/(xmax-xmin)*w;Y=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
  for v in yt:
   self.line([(x,Y(v)),(x+w,Y(v))],GRID,1);self.text(f'{v:g}',x-52,Y(v)-13,40,26,18,MUTED,'right')
  for v in xt:self.line([(X(v),y+h),(X(v),y+h+6)],MUTED,1);self.text(f'{v:g}',X(v)-38,y+h+14,76,24,18,MUTED,'center')
  self.line([(x,y),(x,y+h),(x+w,y+h)],'#465660',1.4);self.text(xlabel,x,y+h+51,w,28,21,TEXT,'center');self.text(ylabel,x-160,y+h/2-15,170,28,21,TEXT,'center',rotate=270)
  return X,Y
 def hist(self,hist,x,y,w,h,color,label=None,xlim=None,ymax=None):
  total=sum(hist['counts']);e=hist['edges'];counts=hist['counts'];lo,hi=xlim or (e[0],e[-1]);top=ymax or max(counts)/max(total,1)*100*1.12
  return {'edges':e,'counts':counts,'total':total,'lo':lo,'hi':hi,'top':top}
 def write(self,name):
  p=OUT/name;E.indent(self.file,space='  ');E.ElementTree(self.file).write(p,encoding='utf-8',xml_declaration=True);return p

curve=json.load((D/'fixed512-curve-data.json').open());f=Draw();f.page('Fixed512 EM',1080,740)
f.text('Fixed 512-question evaluation',90,28,890,32,25,bold=True)
# Stacked legend avoids squeezing the long user-specified labels.
for j,row in enumerate(curve['series']):
 y=95+j*35;f.line([(92,y),(134,y)],COLORS[j],3.2,dash=j==1);f.marker(113,y,COLORS[j],['ellipse','rhombus','triangle'][j],10);f.text(row['name'],149,y-15,670,30,22)
X,Y=f.axes(115,228,845,375,45,205,35,55,[50,100,150,200],[35,40,45,50,55],'Training iteration','EM (%)')
for j,row in enumerate(curve['series']):
 pts=[(X(p['iteration']),Y(p['em']*100)) for p in row['points']];f.line(pts,COLORS[j],3.2,j==1)
 for i,(p,(x,y)) in enumerate(zip(row['points'],pts)):
  f.marker(x,y,COLORS[j],['ellipse','rhombus','triangle'][j],11)
  if i<3:continue
  # Labels use deliberate offsets at iter50 and iter200 to avoid collisions.
  if j==0:dy=-33 if i==0 else (8 if i<3 else 10)
  elif j==1:dy=10 if i==0 else -30
  else:dy=-31
  dx=9 if i==0 else (-28 if i<3 else -66);f.text(f'{p["em"]*100:.2f}',x+dx,y+dy,70,25,17,COLORS[j],'center')
f.write('fixed512-em-curves.drawio')
# Native editable histogram panels. No image cells are embedded.
data=json.load(open(D.parent/'g1-diversity-20261014/distributions-and-evidence.json'));old=data['old_soft'];stats=Draw();notes=[]
def panel_axes(draw,x,y,title,xmin,xmax,ymax,xt,yt,xlabel):
 draw.text(title,x-4,y-53,530,35,23,bold=True);return draw.axes(x,y,450,260,xmin,xmax,0,ymax,xt,yt,xlabel,'Frequency (%)')
def histogram(draw,hist,X,Y,color,total=None):
 total=total or sum(hist['counts'])
 for i,n in enumerate(hist['counts']):
  if n:draw.bar(X(hist['edges'][i]),Y(n/total*100),X(hist['edges'][i+1])-X(hist['edges'][i]),Y(0)-Y(n/total*100),color,42)
def step_line(draw,hist,X,Y,color,total=None,dash=False):
 total=total or sum(hist['counts']);pts=[]
 for i,n in enumerate(hist['counts']):pts.extend([(X(hist['edges'][i]),Y(n/total*100)),(X(hist['edges'][i+1]),Y(n/total*100))])
 draw.line(pts,color,2.3,dash)
for step in [20,100,140]:
 r=next(v for v in old['rows'] if v['step']==step);stats.page(f'Soft vs hard, iter{step}',1240,1090)
 stats.text(f'Reward distributions · iteration {step}',105,30,1030,36,27,bold=True)
 stats.text('Same 512 questions × 5 responses from the Soft Reward experiment',105,74,1030,28,20,MUTED)
 x,y=135,178;X,Y=panel_axes(stats,x,y,'(a) Raw reward values',-.15,1.65,90,[0,.5,1,1.5],[0,30,60,90],'Reward r')
 # Hard reward is an exact binary distribution, not inferred from a trajectory mean.
 for v,n in [(0,r['trajectories']-round(r['em']*r['trajectories'])),(1,round(r['em']*r['trajectories']))]:stats.bar(X(v)-4,Y(n/r['trajectories']*100),8,Y(0)-Y(n/r['trajectories']*100),'#99A2AA',75)
 step_line(stats,r['R_hist'],X,Y,COLORS[0]);stats.text('Hard Reward',x+25,y+12,160,25,18,'#7B8791');stats.text('Soft Reward',x+265,y+12,160,25,18,COLORS[0])
 x,y=755,178;X,Y=panel_axes(stats,x,y,'(b) Within-question reward std',0,1.01,100,[0,.25,.5,.75,1],[0,25,50,75,100],'Sample std over 5 rewards')
 hardstd_hist={'edges':r['R_std_hist']['edges'],'counts':[0]*len(r['R_std_hist']['counts'])}
 import bisect
 for k,n in enumerate(r['correct_count_hist']):
  sd=math.sqrt(k*(5-k)/20);idx=min(len(hardstd_hist['counts'])-1,bisect.bisect_right(hardstd_hist['edges'],sd)-1);hardstd_hist['counts'][idx]+=n
 step_line(stats,hardstd_hist,X,Y,'#89959F',dash=True);step_line(stats,r['R_std_hist'],X,Y,COLORS[0]);stats.text(f'Hard mean std: {r["C_std"]["mean"]:.4f}',x+14,y+32,290,25,18,'#7B8791');stats.text(f'Soft mean std: {r["R_std"]["mean"]:.4f}',x+14,y+62,290,25,18,COLORS[0])
 x,y=135,638;X,Y=panel_axes(stats,x,y,'(c) Normalized reward / advantage',-1.85,1.85,100,[-1.5,-1,-.5,0,.5,1,1.5],[0,25,50,75,100],'A = (r − group mean) / (group std + ε)')
 step_line(stats,r['C_A_hist'],X,Y,'#89959F',dash=True);step_line(stats,r['R_A_hist'],X,Y,COLORS[0]);stats.text('All trajectories, including zero advantages',x+15,y+18,420,25,18,MUTED)
 x,y=755,638;X,Y=panel_axes(stats,x,y,'(d) Advantages in nonconstant groups',-1.85,1.85,40,[-1.5,-1,-.5,0,.5,1,1.5],[0,10,20,30,40],'Normalized advantage A')
 step_line(stats,r['C_active_A_hist'],X,Y,'#89959F',dash=True);step_line(stats,r['R_active_A_hist'],X,Y,COLORS[0]);stats.text('Hard Reward',x+20,y+15,170,25,18,'#7B8791');stats.text('Soft Reward',x+230,y+15,170,25,18,COLORS[0])
 stats.text('std uses correction = 1; ε = 10⁻⁶. Soft rewards are historical frozen-probe reconstructions.',105,1020,1100,27,18,MUTED)
 notes.append({'iteration':step,'hard_mean_group_std':r['C_std']['mean'],'soft_mean_group_std':r['R_std']['mean'],'groups':512,'trajectories':2560,'raw_reward_std_global':r['R']['std'],'soft_reward_coverage':'historical reconstruction, not online bitwise values'})
stats.write('reward-std-and-advantage-distributions.drawio')
# Paired per-question std, including and excluding exactly constant groups.
q=Draw();q.page('Question std, iter100',1240,690);r=next(v for v in old['rows'] if v['step']==100)
q.text('Per-question reward diversity = sample std',105,30,1060,36,27,bold=True)
q.text('Same 512 questions × 5 original responses · iteration 100 · Soft Reward experiment',105,76,1060,28,20,MUTED)
import bisect
hard={'edges':r['R_std_hist']['edges'],'counts':[0]*len(r['R_std_hist']['counts'])}
for k,n in enumerate(r['correct_count_hist']):
 sd=math.sqrt(k*(5-k)/20);hard['counts'][min(len(hard['counts'])-1,bisect.bisect_right(hard['edges'],sd)-1)]+=n
for j,active in enumerate([False,True]):
 x=135+j*620;y=186;X,Y=panel_axes(q,x,y,'(a) All question groups' if not active else '(b) Nonconstant question groups',0,1.01,100 if not active else 65,[0,.25,.5,.75,1],[0,25,50,75,100] if not active else [0,20,40,60],'Within-question sample std')
 for h,color,label,zero,mean in [(hard,'#89959F','Hard Reward',round(r['C_zero_group_fraction']*512),r['C_std']['mean']),(r['R_std_hist'],COLORS[0],'Soft Reward',round(r['R_zero_group_fraction']*512),r['R_std']['mean'])]:
  hh={'edges':h['edges'],'counts':list(h['counts'])}
  if active:hh['counts'][0]-=zero
  total=sum(hh['counts']);assert total==512-zero if active else total==512
  step_line(q,hh,X,Y,color,dash=label=='Hard Reward');yy=y+22 if label=='Hard Reward' else y+57
  q.text(f'{label}: mean std = {mean/(1-zero/512) if active else mean:.4f}',x+22,yy,410,24,18,color)
 q.text('Hard: zero-std 81.45%  ·  Soft: zero-std 48.63%' if not active else 'Hard: 95 groups  ·  Soft: 263 groups',x-15,y+361,500,27,18,MUTED)
q.text('Hard and Soft rewards use identical original question groups; histograms share the same bins.',105,633,1100,26,18,MUTED)
q.write('question-level-reward-std.drawio')

# Per-question std comparison across the three policies, using the same Hard Reward definition.
g=Draw();g.page('Hard Reward std comparison',1080,780);g.text('Within-question Hard Reward std',105,30,860,35,27,bold=True)
for j,r in enumerate(data['runs']):
 label={'Search-G1':'Soft Reward','G1→R1':'Soft Reward -> Hard Reward','mixed-first':'Easy Selection+Soft Reward'}[r['name']];y=99+j*34;g.line([(105,y),(144,y)],COLORS[j],3,j==1);g.text(label,160,y-14,790,28,21)
X,Y=g.axes(120,250,835,370,0,200,0,.55,[0,50,100,150,200],[0,.1,.2,.3,.4,.5],'Training iteration','Mean question std')
for j,r in enumerate(data['runs']):
 pts=[(X(v['step']),Y(v['C_std']['mean'])) for v in r['rows']];g.line(pts,COLORS[j],2.6,j==1)
 if j<2:
  for x,y in pts:g.marker(x,y,COLORS[j],['ellipse','rhombus'][j],7)
g.text('Hard Reward = strict EM ∈ {0,1}; std is computed within each question, then averaged.',105,722,880,30,18,MUTED)
g.write('hard-reward-group-std-curves.drawio')
(OUT/'fixed512-curve-data.json').write_text(json.dumps(curve,ensure_ascii=False,indent=2));(OUT/'reward-std-summary.json').write_text(json.dumps({'definition':'sample std within each original 5-response question group, correction=1','paired_soft_hard':notes,'missing':'Per-trajectory Soft Reward for Easy Selection+Soft Reward and counterfactual Soft Reward after the switch were not persisted; requires original historical probe replay','source':'outputs/g1-diversity-20261014/distributions-and-evidence.json'},ensure_ascii=False,indent=2))
print('Native editable drawio documents written to',OUT)
# Actual outer-update reward means, read from the native update ledger.
means=json.load((D/'native-reward-means.json').open());m=Draw();m.page('Actual mean reward',1080,790);m.text('Actual training reward',105,30,890,35,27,bold=True)
for j,s in enumerate(means['series']):
 y=99+j*34;m.line([(105,y),(145,y)],COLORS[j],3,j==1);m.text(s['name'],160,y-14,800,28,21)
X,Y=m.axes(120,250,835,370,0,200,0,1.3,[0,50,100,150,200],[0,.25,.5,.75,1,1.25],'Training iteration','Mean reward')
for j,s in enumerate(means['series']):m.line([(X(v['step']),Y(v['critic/rewards/mean'])) for v in s['rows'] if v['critic/rewards/mean'] is not None],COLORS[j],2.6,j==1)
m.line([(X(50),250),(X(50),620)],'#9CA8B1',1,True);m.text('Reward switch after iter50',X(50)+10,282,305,26,18,MUTED)
m.text('Native critic/rewards/mean. Different reward definitions have different scales.',105,722,905,30,18,MUTED);m.write('actual-mean-reward-curves.drawio')
# Comparable strict-EM distributions for all policies. This is an EM-component
# normalization, not the actual soft-reward advantage of the two soft policies.
tri=Draw()
for iteration in [100,200]:
 rs=[next(v for v in run['rows'] if v['step']==iteration) for run in data['runs']]
 tri.page(f'EM components, iter{iteration}',1240,1160);tri.text(f'Hard Reward (EM) distributions · iteration {iteration}',105,28,1060,36,27,bold=True)
 for j,run in enumerate(data['runs']):
  name={'Search-G1':'Soft Reward','G1→R1':'Soft Reward -> Hard Reward','mixed-first':'Easy Selection+Soft Reward'}[run['name']];y=90+31*j;tri.line([(110,y),(147,y)],COLORS[j],2.8,j==1);tri.text(name,165,y-14,1000,28,20)
 x,y=135,260;X,Y=panel_axes(tri,x,y,'(a) Hard Reward values',-.2,1.25,100,[0,1],[0,25,50,75,100],'Strict EM reward ∈ {0,1}')
 for j,r in enumerate(rs):
  for val,n in [(0,2560-round(r['em']*2560)),(1,round(r['em']*2560))]:tri.bar(X(val)+(j-1)*12-5,Y(n/2560*100),10,Y(0)-Y(n/2560*100),COLORS[j],65)
 x,y=755,260;X,Y=panel_axes(tri,x,y,'(b) Per-question Hard Reward std',0,.6,100,[0,.2,.4,.6],[0,25,50,75,100],'Sample std of 5 EM rewards')
 for j,r in enumerate(rs):
  for sd,n in [(0,r['correct_count_hist'][0]+r['correct_count_hist'][5]),(math.sqrt(.2),r['correct_count_hist'][1]+r['correct_count_hist'][4]),(math.sqrt(.3),r['correct_count_hist'][2]+r['correct_count_hist'][3])]:tri.bar(X(sd)+(j-1)*8-3,Y(n/512*100),6,Y(0)-Y(n/512*100),COLORS[j],65)
  tri.text(f'Mean std: {r["C_std"]["mean"]:.4f}',x+22,y+25+j*27,330,25,18,COLORS[j])
 x,y=135,720;X,Y=panel_axes(tri,x,y,'(c) Normalized EM rewards · all groups',-1.85,1.85,100,[-1.5,-1,-.5,0,.5,1,1.5],[0,25,50,75,100],'A_EM = (EM − group mean) / (group std + ε)')
 for j,r in enumerate(rs):step_line(tri,r['C_A_hist'],X,Y,COLORS[j],dash=j==1)
 x,y=755,720;X,Y=panel_axes(tri,x,y,'(d) Normalized EM · nonconstant groups',-1.85,1.85,40,[-1.5,-1,-.5,0,.5,1,1.5],[0,10,20,30,40],'Normalized EM advantage A_EM')
 for j,r in enumerate(rs):step_line(tri,r['C_active_A_hist'],X,Y,COLORS[j],dash=j==1)
 tri.text('EM component only: soft-policy actual advantages require per-trajectory soft rewards. std correction = 1; ε = 10⁻⁶.',105,1100,1100,32,17,MUTED)
tri.write('hard-reward-distributions-three-methods.drawio')
(OUT/'native-reward-means.json').write_text(json.dumps(means,indent=2));print('Native reward means and three-method EM distributions written')
