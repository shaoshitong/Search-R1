"""Native editable reward histograms: solid closed outlines and explicit denominators.
This is a geometry-only repair using unchanged aggregate counts; no remote reads.
"""
from pathlib import Path
import json,math,bisect,hashlib,xml.etree.ElementTree as E
D=Path(__file__).resolve().parent
# Reuse only the drawing factory, never execute the other figure generators.
ns={'__file__':str(D/'make_drawio.py')}
exec(compile((D/'make_drawio.py').read_text().split('curve=json.load')[0],str(D/'make_drawio.py'),'exec'),ns)
Draw=ns['Draw'];OUT=ns['OUT'];TEXT=ns['TEXT'];MUTED=ns['MUTED']
HARD='#B46F26';SOFT='#397CB5'

def paired_hist(draw,hh,sh,X,Y):
 assert hh['edges']==sh['edges'];totals=[sum(h['counts']) for h in [hh,sh]];assert min(totals)>0
 geometry=[]
 # Native vector fills first; outlines remain visible above both fills.
 for h,color,total in zip([hh,sh],[HARD,SOFT],totals):
  for i,n in enumerate(h['counts']):
   if n:
    x=X(h['edges'][i]);y=Y(n/total*100);w=X(h['edges'][i+1])-x;high=Y(0)-y
    draw.cell('',f'rounded=0;fillColor={color};fillOpacity=12;strokeColor=none;',x,y,w,high)
 for h,color,total in zip([hh,sh],[HARD,SOFT],totals):
  # Both boundary bins return to y=0. Exact bin counts stay unchanged.
  points=[(X(h['edges'][0]),Y(0))]
  for i,n in enumerate(h['counts']):points.extend([(X(h['edges'][i]),Y(n/total*100)),(X(h['edges'][i+1]),Y(n/total*100))])
  points.append((X(h['edges'][-1]),Y(0)))
  compressed=[]
  for p in points:
   if not compressed or p!=compressed[-1]:compressed.append(p)
  draw.line(compressed,color,2.5,False)
  # Hard-reward peak markers distinguish it even where outlines overlap.
  if color==HARD:
   for i,n in enumerate(h['counts']):
    if n:draw.marker(X((h['edges'][i]+h['edges'][i+1])/2),Y(n/total*100),color,'ellipse',4.5)
  geometry.append({'counts':h['counts'],'edges':h['edges'],'denominator':total,'closed_boundary':True,'solid_outline':True,'points':compressed})
 return geometry

def axes(d,x,y,title,lo,hi,top,xt,yt,xlabel):
 d.text(title,x-4,y-61,550,36,25,bold=True)
 return d.axes(x,y,500,280,lo,hi,0,top,xt,yt,xlabel,'Bin frequency (%)')

def hard_std_hist(r):
 e=r['R_std_hist']['edges'];ct=[0]*(len(e)-1)
 for k,n in enumerate(r['correct_count_hist']):
  value=math.sqrt(k*(5-k)/20);i=min(len(ct)-1,bisect.bisect_right(e,value)-1);ct[i]+=n
 return {'edges':e,'counts':ct}

def build(old):
 d=Draw();notes=[];audits=[]
 for step in [20,100,140]:
  r=next(v for v in old['rows'] if v['step']==step);assert r['groups']==512 and r['trajectories']==2560
  hzero=round(r['C_zero_group_fraction']*512);szero=round(r['R_zero_group_fraction']*512);hn=5*(512-hzero);sn=5*(512-szero)
  assert sum(r['C_active_A_hist']['counts'])==hn and sum(r['R_active_A_hist']['counts'])==sn
  d.page(f'Soft vs hard, iter{step}',1420,1450)
  d.text(f'Reward distributions · iteration {step}',105,30,1190,38,29,bold=True)
  d.text('Identical original rollouts: 512 questions × 5 responses · Soft Reward experiment',105,80,1230,30,21,MUTED)
  d.line([(111,141),(159,141)],HARD,2.6);d.marker(135,141,HARD,'ellipse',6);d.text('Hard Reward (strict EM)',179,126,330,30,22,HARD)
  d.line([(586,141),(634,141)],SOFT,2.6);d.text('Soft Reward (G1 form)',654,126,440,30,22,SOFT)
  panels=[]
  # (a) Both rewards share the same bins. Binary hard reward is exactly 0/1.
  hh={'edges':r['R_hist']['edges'],'counts':[0]*len(r['R_hist']['counts'])}
  for value,count in [(0,2560-round(r['em']*2560)),(1,round(r['em']*2560))]:hh['counts'][bisect.bisect_right(hh['edges'],value)-1]+=count
  x,y=135,255;X,Y=axes(d,x,y,'(a) Original reward values',-.15,1.65,90,[0,.5,1,1.5],[0,30,60,90],'Reward r');panels.append(paired_hist(d,hh,r['R_hist'],X,Y))
  d.text('2,560 trajectories per reward; shared bin edges',x,y+378,550,27,19,MUTED)
  d.text(f'Hard mean = {r["em"]:.4f}   ·   Soft mean = {r["R"]["mean"]:.4f}',x,y+414,550,27,19,TEXT)
  # (b) Padding places sigma=0 away from the y-axis to show both vertical edges.
  x,y=835,255;X,Y=axes(d,x,y,'(b) Per-question reward std',-.025,1.035,100,[0,.25,.5,.75,1],[0,25,50,75,100],'Sample std across 5 rewards');panels.append(paired_hist(d,hard_std_hist(r),r['R_std_hist'],X,Y))
  d.text(f'Mean std: Hard {r["C_std"]["mean"]:.4f}   ·   Soft {r["R_std"]["mean"]:.4f}',x,y+378,550,27,19,TEXT)
  d.text(f'Exactly std = 0: Hard {hzero}/512   ·   Soft {szero}/512',x,y+414,550,27,19,MUTED)
  d.text('First bin also includes small positive std values',x,y+450,550,27,18,MUTED)
  # (c) All original groups, constant-group zero advantages included.
  x,y=135,875;X,Y=axes(d,x,y,'(c) Advantages · all trajectories',-1.85,1.85,100,[-1.5,-1,-.5,0,.5,1,1.5],[0,25,50,75,100],'Standardized advantage A');panels.append(paired_hist(d,r['C_A_hist'],r['R_A_hist'],X,Y))
  d.text('All 512 groups · n = 2,560 per reward',x,y+378,550,27,19,TEXT)
  d.text('Constant groups retained; their advantages are zero',x,y+414,550,27,18,MUTED)
  # (d) Reward-specific group selection and reward-specific denominators.
  x,y=835,875;X,Y=axes(d,x,y,'(d) Advantages · question std > 0',-1.85,1.85,40,[-1.5,-1,-.5,0,.5,1,1.5],[0,10,20,30,40],'Standardized advantage A');panels.append(paired_hist(d,r['C_active_A_hist'],r['R_active_A_hist'],X,Y))
  d.text(f'Hard: {512-hzero} groups / n = {hn:,}   ·   Soft: {512-szero} groups / n = {sn:,}',x,y+378,550,27,19,TEXT)
  d.text('Filter each reward separately; renormalize retained n',x,y+414,550,27,18,MUTED)
  d.text('A = (r − question mean) / (question sample std + 10⁻⁶); sample std uses correction = 1.',105,1373,1230,29,19,TEXT)
  d.text('Soft rewards use historical frozen-probe reconstruction; raw scalar normalization, not PPO clipping.',105,1408,1230,27,18,MUTED)
  audits.append({'iteration':step,'panels':panels,'c_denominators':[2560,2560],'d_denominators':[hn,sn],'std_zero_group_counts':[hzero,szero]})
  notes.append({'iteration':step,'hard_mean_group_std':r['C_std']['mean'],'soft_mean_group_std':r['R_std']['mean'],'groups':512,'trajectories':2560,'raw_reward_std_global':r['R']['std'],'soft_reward_coverage':'historical reconstruction, not online bitwise values'})
 d.write('reward-std-and-advantage-distributions.drawio')
 audit={'repair':'closed solid histogram outlines; fills; peak markers; explicit c/d selection and denominators','data_counts_unchanged':True,'pages':audits};(D/'histogram-repair-geometry.json').write_text(json.dumps(audit,indent=2))
 return d,notes

if __name__=='__main__':
 data=json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open());build(data['old_soft']);print('Repaired all 3 native draw.io histogram pages')
