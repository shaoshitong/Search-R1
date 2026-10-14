"""Preserve the user's native drawing edits while replacing reward-only filtering
with conditions defined exclusively by Hard Reward's within-question std.
"""
from pathlib import Path
import json,copy,math,hashlib,bisect,xml.etree.ElementTree as E
from repair_reward_histograms import Draw,OUT,TEXT,MUTED,paired_hist
D=Path(__file__).resolve().parent
ZERO='#8C6BB1';POS='#168D7B'
BACK=json.load((D/'hard-conditioned-source-backup.json').open());p=OUT/'reward-std-and-advantage-distributions.drawio';original=Path(BACK['path']).read_bytes()
assert hashlib.sha256(original).hexdigest()==BACK['sha256']
condition=json.load((D/'hard-conditioned-soft-data-20261015.json').open());by={v['step']:v for v in condition['steps']}

def outline(draw,h,X,Y,color):
 n=sum(h['counts']);assert n>0;points=[(X(h['edges'][0]),Y(0))]
 for i,count in enumerate(h['counts']):
  x0,x1=X(h['edges'][i]),X(h['edges'][i+1]);yy=Y(count/n*100)
  if count:draw.cell('',f'rounded=0;fillColor={color};fillOpacity=11;strokeColor=none;',x0,yy,x1-x0,Y(0)-yy)
  points.extend([(x0,yy),(x1,yy)])
 points.append((X(h['edges'][-1]),Y(0)));clean=[]
 for q in points:
  if not clean or q!=clean[-1]:clean.append(q)
 draw.line(clean,color,2.7,False)
 return {'denominator':n,'counts':h['counts'],'edges':h['edges']}

def chart(draw,x,title,lo,hi,xt,xlabel,zero,positive,key,top=100):
 y=255;draw.text(title,x-4,y-61,550,36,25,bold=True);X,Y=draw.axes(x,y,500,280,lo,hi,0,top,xt,[0,10,20,30,40] if top==40 else [0,25,50,75,100],xlabel,'Bin frequency (%)')
 return [outline(draw,v[key],X,Y,color) for v,color in [(zero,ZERO),(positive,POS)]]

def move(c,dx,dy):
 if c.get('vertex')=='1':
  g=c.find('mxGeometry')
  if g is not None:
   g.set('x',str(float(g.get('x',0))+dx));g.set('y',str(float(g.get('y',0))+dy))
 for q in c.findall('.//mxPoint'):
  if q.get('as')!='offset':
   q.set('x',str(float(q.get('x',0))+dx));q.set('y',str(float(q.get('y',0))+dy))

def normalized(c):
 def val(k,v):
  if k in ['x','y','width','height']:
   try:return round(float(v),4)
   except ValueError:pass
  return v
 return {'tag':c.tag,'attrs':{k:val(k,v) for k,v in c.attrib.items()},'children':[normalized(q) for q in c]}

def filter_left_panels(graph,A,C,drawing,step,z,pos):
 old=json.load((D.parent/'g1-diversity-20261014/distributions-and-evidence.json').open());base=next(r for r in old['old_soft']['rows'] if r['step']==step)
 allwrong=base['correct_count_hist'][0];groups=z['groups']+pos['groups'];n=groups*5
 correct=round(z['groups']*5+pos['hard_em']*pos['trajectories'])
 def isdata(c):return not c.get('value') and any(color in c.get('style','') for color in ['#B46F26','#397CB5'])
 for c in A+C:
  if isdata(c):graph.remove(c)
 softR={'edges':z['soft_reward_hist']['edges'],'counts':[a+b for a,b in zip(z['soft_reward_hist']['counts'],pos['soft_reward_hist']['counts'])]}
 hardR={'edges':softR['edges'],'counts':[0]*len(softR['counts'])}
 for value,count in [(0,n-correct),(1,correct)]:hardR['counts'][bisect.bisect_right(hardR['edges'],value)-1]+=count
 X=lambda v:135+(v+.15)/1.8*500;Y=lambda v:535-v/90*280;paired_hist(drawing,hardR,softR,X,Y)
 hardA={'edges':base['C_A_hist']['edges'],'counts':list(base['C_A_hist']['counts'])};hardA['counts'][bisect.bisect_right(hardA['edges'],0)-1]-=allwrong*5
 softA={'edges':z['soft_advantage_hist']['edges'],'counts':[a+b for a,b in zip(z['soft_advantage_hist']['counts'],pos['soft_advantage_hist']['counts'])]}
 X=lambda v:835+(v+1.85)/3.7*500;Y=lambda v:535-v/100*280;paired_hist(drawing,hardA,softA,X,Y)
 assert sum(hardR['counts'])==sum(softR['counts'])==sum(hardA['counts'])==sum(softA['counts'])==n
 for c in A:
  value=c.get('value','')
  if value.startswith('(a) Original'):c.set('value','(a) Rewards · all-wrong excluded')
  elif value.startswith('2,560 trajectories'):c.set('value',f'{groups} question groups / {n:,} trajectories per reward')
  elif value.startswith('Hard mean'):c.set('value',f'Hard mean = {correct/n:.4f}   ·   Soft mean = {(z["soft_reward"]["mean"]*z["trajectories"]+pos["soft_reward"]["mean"]*pos["trajectories"])/n:.4f}')
 for c in C:
  value=c.get('value','')
  if value.startswith('(b) Advantages'):c.set('value','(b) Advantages · all-wrong excluded')
  elif value.startswith('All 512 groups'):c.set('value',f'{groups} question groups · n = {n:,} per reward')
  elif value.startswith('Constant groups retained'):c.set('value','All-correct groups: Hard A = 0; Soft A may be nonzero')
 for c in graph:
  if c.get('value','').startswith('Identical original rollouts:'):c.set('value','Question groups with at least one correct answer; all-wrong groups excluded')
 return {'groups':groups,'trajectories':n,'excluded_all_wrong_groups':allwrong}

def main():
 history=D/'hard-conditioned-drawio-preservation.json'
 previous=json.load(history.open()).get('new_file_sha256') if history.exists() else None
 assert hashlib.sha256(p.read_bytes()).hexdigest() in [BACK['sha256'],previous],'File changed since recorded version; refuse to overwrite newer edits'
 root=E.fromstring(original);audit=[]
 for page in root.findall('diagram'):
  step=int(page.get('name').split('iter')[-1]);r=by[step]['conditions'];z=r['all_correct'];pos=r['hard_std_positive'];model=page.find('mxGraphModel');graph=model.find('root');cells=list(graph)
  def index(prefix):return next(i for i,c in enumerate(cells) if c.get('value','').startswith(prefix))
  ia,ib,ic,id_=map(index,['(a) Original','(b) Per-question','(c) Advantages','(d) Advantages'])
  footer=index('A = (r');A=cells[ia:ib];C=cells[ic:id_];removed=cells[ib:ic]+cells[id_:footer]
  # Existing left panels remain native cells with the user's styles intact.
  a_before=[normalized(c) for c in A];c_before=[normalized(c) for c in C]
  for c in removed:graph.remove(c)
  for c in C:move(c,700,-620)
  C[0].set('value','(b) Advantages · all trajectories')
  for c in cells[footer:]:
   g=c.find('mxGeometry')
   if g is not None:
    y=float(g.get('y'));g.set('y',str(y-581));g.set('width','1930')
  model.set('pageWidth','2120');model.set('pageHeight','890');model.set('dx','2120');model.set('dy','890')
  drawing=Draw();drawing.root=graph;drawing.serial=10000+step*1000
  panels=chart(drawing,1535,'(c) Soft advantage · all-correct vs mixed',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'Standardized Soft Reward advantage A',z,pos,'soft_advantage_hist',top=40)
  for yy,v,color,label in [(633,z,ZERO,'All correct'),(669,pos,POS,'Mixed outcomes')]:
   drawing.line([(1535,yy+14),(1575,yy+14)],color,2.7);drawing.text(f'{label}: {v["groups"]} groups / {v["trajectories"]:,} trajectories',1592,yy,450,27,19,color)
  drawing.text(f'All-correct groups: Soft std > 0 in {z["active_soft_groups"]}/{z["groups"]}',1535,713,550,28,18,MUTED)
  drawing.text('All wrong excluded; purple = Hard std 0, green = Hard std > 0',1535,747,550,27,17,MUTED)
  assert [normalized(c) for c in A]==a_before
  for c in C:move(c,-700,620)
  C[0].set('value',c_before[0]['attrs']['value']);assert [normalized(c) for c in C]==c_before
  for c in C:move(c,700,-620)
  C[0].set('value','(b) Advantages · all trajectories')
  selection=filter_left_panels(graph,A,C,drawing,step,z,pos)
  audit.append({'step':step,'a_axis_styles_preserved':True,'a_data_filtered_by_user_request':True,'b_axis_styles_preserved':True,'b_data_filtered_by_user_request':True,'removed_old_std_panel':True,'removed_old_reward_specific_filter':True,'new_conditions':'All-correct Hard std=0 versus mixed Hard std>0; all-wrong groups excluded in all panels','conditional_histograms':panels,'common_selection':selection})
 # Explicit source metadata prevents future legacy regeneration from replacing user edits.
 root.set('conditionDefinition','hard-all-correct-vs-mixed');E.indent(root,space='  ');E.ElementTree(root).write(p,encoding='utf-8',xml_declaration=True)
 # Detailed comparison: raw Soft reward, centered reward difference, normalized A.
 detail=Draw()
 for step in [20,100,140]:
  r=by[step]['conditions'];z=r['all_correct'];pos=r['hard_std_positive'];detail.page(f'Hard-conditioned Soft, iter{step}',2120,980)
  detail.text(f'Soft Reward distributions · all-correct vs mixed question groups · iteration {step}',105,30,1930,39,29,bold=True)
  detail.text('All-wrong question groups excluded; same original five-response groups, selected by Hard EM outcomes',105,82,1930,30,21,MUTED)
  for x,v,color,label in [(111,z,ZERO,'All correct (Hard std = 0)'),(880,pos,POS,'Mixed outcomes (Hard std > 0)')]:
   detail.line([(x,141),(x+48,141)],color,2.7);detail.text(f'{label}: {v["groups"]} groups / {v["trajectories"]:,} trajectories',x+68,126,650,30,22,color)
  configs=[(135,'(a) Original Soft Reward',-.15,1.65,[0,.5,1,1.5],'Soft reward r','soft_reward_hist'),(835,'(b) Within-question Soft difference',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'δr = r − question mean','soft_centered_delta_hist'),(1535,'(c) Standardized Soft advantage',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'A = δr / (question std + 10⁻⁶)','soft_advantage_hist')]
  for x,title,lo,hi,xt,xlabel,key in configs:
   chart(detail,x,title,lo,hi,xt,xlabel,z,pos,key,top=40 if key=='soft_advantage_hist' else 100)
   for yy,v,color,label in [(633,z,ZERO,'All correct'),(669,pos,POS,'Mixed outcomes')]:
    field={'soft_reward_hist':'soft_reward','soft_centered_delta_hist':'soft_centered_delta','soft_advantage_hist':'soft_advantage'}[key];q=v[field]['quantiles'];detail.text(f'{label}: 10–90% [{q[2]:.4f}, {q[6]:.4f}]',x,yy,550,28,19,color)
  detail.text('A = (r − question mean) / (question sample std + 10⁻⁶); std correction = 1. Histograms use equal trajectory weighting.',105,766,1930,29,20,TEXT)
  detail.text(f'All-correct groups with nonzero Soft std: {z["active_soft_groups"]}/{z["groups"]}. Frequencies normalized separately within all-correct / mixed subsets.',105,806,1930,29,20,MUTED)
  if step==100:
   example=condition['examples'][0];detail.text('Actual all-correct group: Hard rewards [1,1,1,1,1] give A_Hard = [0,0,0,0,0].',105,851,1930,29,20,TEXT)
   detail.text('Soft std = 2.38035×10⁻⁵; δr_min = −4.12941×10⁻⁵ → A_min = −1.66485. Nonzero Soft ordering, despite identical EM.',105,888,1930,29,20,TEXT)
  else:
   count=z['small_raw_delta_nontrivial_advantage']['count'];n=z['trajectories'];detail.text(f'All-correct subset: {count}/{n} trajectories have |δr| < 10⁻⁴ but |A| > 0.1.',105,867,1930,30,20,TEXT)
  detail.text('Historical frozen-probe reconstruction; not guaranteed bitwise recovery of online rewards.',105,935,1930,28,18,MUTED)
 detail.write('hard-conditioned-soft-reward-amplification.drawio');(OUT/'hard-conditioned-soft-data.json').write_text(json.dumps(condition,indent=2));(D/'hard-conditioned-drawio-preservation.json').write_text(json.dumps({'source_backup':BACK,'new_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':audit},indent=2));print('Updated user-edited three-panel figure; wrote conditional raw/delta/A companion')
if __name__=='__main__':main()
