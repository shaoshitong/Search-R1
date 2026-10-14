"""Preserve the user's native drawing edits while replacing reward-only filtering
with conditions defined exclusively by Hard Reward's within-question std.
"""
from pathlib import Path
import json,copy,math,hashlib,xml.etree.ElementTree as E
from repair_reward_histograms import Draw,OUT,TEXT,MUTED
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

def chart(draw,x,title,lo,hi,xt,xlabel,zero,positive,key):
 y=255;draw.text(title,x-4,y-61,550,36,25,bold=True);X,Y=draw.axes(x,y,500,280,lo,hi,0,100,xt,[0,25,50,75,100],xlabel,'Bin frequency (%)')
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

def main():
 history=D/'hard-conditioned-drawio-preservation.json'
 previous=json.load(history.open()).get('new_file_sha256') if history.exists() else None
 assert hashlib.sha256(p.read_bytes()).hexdigest() in [BACK['sha256'],previous],'File changed since recorded version; refuse to overwrite newer edits'
 root=E.fromstring(original);audit=[]
 for page in root.findall('diagram'):
  step=int(page.get('name').split('iter')[-1]);r=by[step]['conditions'];z=r['hard_std_zero'];pos=r['hard_std_positive'];model=page.find('mxGraphModel');graph=model.find('root');cells=list(graph)
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
  panels=chart(drawing,1535,'(c) Soft advantage by Hard std',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'Standardized Soft Reward advantage A',z,pos,'soft_advantage_hist')
  for yy,v,color,label in [(633,z,ZERO,'Hard std = 0'),(669,pos,POS,'Hard std > 0')]:
   drawing.line([(1535,yy+14),(1575,yy+14)],color,2.7);drawing.text(f'{label}: {v["groups"]} groups / {v["trajectories"]:,} trajectories',1592,yy,450,27,19,color)
  drawing.text(f'Within Hard-zero groups: {z["active_soft_groups"]}/{z["groups"]} have Soft std > 0',1535,713,550,28,18,MUTED)
  drawing.text('Each curve uses all trajectories in its Hard-defined subset',1535,747,550,27,17,MUTED)
  assert [normalized(c) for c in A]==a_before
  for c in C:move(c,-700,620)
  C[0].set('value',c_before[0]['attrs']['value']);assert [normalized(c) for c in C]==c_before
  for c in C:move(c,700,-620)
  C[0].set('value','(b) Advantages · all trajectories')
  audit.append({'step':step,'original_a_cells_preserved_exactly':True,'original_c_cells_style_and_data_preserved':True,'removed_old_std_panel':True,'removed_old_reward_specific_filter':True,'new_conditions':'Hard std=0 versus Hard std>0; never filter by Soft std','conditional_histograms':panels})
 # Explicit source metadata prevents future legacy regeneration from replacing user edits.
 root.set('conditionDefinition','hard-std-zero-vs-positive');E.indent(root,space='  ');E.ElementTree(root).write(p,encoding='utf-8',xml_declaration=True)
 # Detailed comparison: raw Soft reward, centered reward difference, normalized A.
 detail=Draw()
 for step in [20,100,140]:
  r=by[step]['conditions'];z=r['hard_std_zero'];pos=r['hard_std_positive'];detail.page(f'Hard-conditioned Soft, iter{step}',2120,980)
  detail.text(f'Soft Reward distributions conditioned on Hard Reward std · iteration {step}',105,30,1930,39,29,bold=True)
  detail.text('Same original five-response question groups; all groups included within each Hard-defined subset',105,82,1930,30,21,MUTED)
  for x,v,color,label in [(111,z,ZERO,'Hard std = 0'),(880,pos,POS,'Hard std > 0')]:
   detail.line([(x,141),(x+48,141)],color,2.7);detail.text(f'{label}: {v["groups"]} groups / {v["trajectories"]:,} trajectories',x+68,126,650,30,22,color)
  configs=[(135,'(a) Original Soft Reward',-.15,1.65,[0,.5,1,1.5],'Soft reward r','soft_reward_hist'),(835,'(b) Within-question Soft difference',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'δr = r − question mean','soft_centered_delta_hist'),(1535,'(c) Standardized Soft advantage',-1.85,1.85,[-1.5,-1,-.5,0,.5,1,1.5],'A = δr / (question std + 10⁻⁶)','soft_advantage_hist')]
  for x,title,lo,hi,xt,xlabel,key in configs:
   chart(detail,x,title,lo,hi,xt,xlabel,z,pos,key)
   for yy,v,color,label in [(633,z,ZERO,'Hard std = 0'),(669,pos,POS,'Hard std > 0')]:
    field={'soft_reward_hist':'soft_reward','soft_centered_delta_hist':'soft_centered_delta','soft_advantage_hist':'soft_advantage'}[key];q=v[field]['quantiles'];detail.text(f'{label}: 10–90% [{q[2]:.4f}, {q[6]:.4f}]',x,yy,550,28,19,color)
  detail.text('A = (r − question mean) / (question sample std + 10⁻⁶); std correction = 1. Histograms use equal trajectory weighting.',105,766,1930,29,20,TEXT)
  detail.text(f'Hard-zero groups with nonzero Soft std: {z["active_soft_groups"]}/{z["groups"]}. Each curve is separately normalized within its Hard-defined subset.',105,806,1930,29,20,MUTED)
  if step==100:
   example=condition['examples'][0];detail.text('Actual all-correct group: Hard rewards [1,1,1,1,1] give A_Hard = [0,0,0,0,0].',105,851,1930,29,20,TEXT)
   detail.text('Soft std = 2.38035×10⁻⁵; δr_min = −4.12941×10⁻⁵ → A_min = −1.66485. Nonzero Soft ordering, despite identical EM.',105,888,1930,29,20,TEXT)
  else:
   count=z['small_raw_delta_nontrivial_advantage']['count'];n=z['trajectories'];detail.text(f'Hard-zero subset: {count}/{n} trajectories have |δr| < 10⁻⁴ but |A| > 0.1.',105,867,1930,30,20,TEXT)
  detail.text('Historical frozen-probe reconstruction; not guaranteed bitwise recovery of online rewards.',105,935,1930,28,18,MUTED)
 detail.write('hard-conditioned-soft-reward-amplification.drawio');(OUT/'hard-conditioned-soft-data.json').write_text(json.dumps(condition,indent=2));(D/'hard-conditioned-drawio-preservation.json').write_text(json.dumps({'source_backup':BACK,'new_file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pages':audit},indent=2));print('Updated user-edited three-panel figure; wrote conditional raw/delta/A companion')
if __name__=='__main__':main()
