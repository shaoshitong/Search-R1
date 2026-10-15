"""Publication-style plots and an offline Canvas from the seven-session audit."""
from pathlib import Path
import json,hashlib,math,copy,datetime
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

D=Path(__file__).resolve().parent
OUT=D/'figures';OUT.mkdir(exist_ok=True)
report=json.loads((D/'all-seven-results.json').read_text())
chain=json.loads((D/'plot-chain-audit.json').read_text())
histpath=D.parent/'training-and-fixed512-em-20261014-2030/metrics-and-evidence.json'
history=json.loads(histpath.read_text())
spec=[
('Soft Reward (Search-G1)','Soft Reward','#397CB5','Reward & selection'),
('Soft Reward -> Hard Reward','Soft → Hard','#9368B7','Reward & selection'),
('Easy Selection+Soft Reward','Easy Selection + Soft','#009E73','Reward & selection'),
('QProbe sigma3 fresh200 (latest setting)','Q-probe σ3 · G1','#444C57','Q-probe'),
('Tree-GRPO original guarded200','Tree-GRPO','#4477AA','Tree & recursion'),
('Tree BRANPO fresh200 3360','Tree-BRANPO','#AA7733','Tree & recursion'),
('Tree nested branching fresh200','Tree-Nested','#D04A53','Tree & recursion'),
('Recursive A','Recursive A','#168D7B','Tree & recursion'),
('C1 root-only ablation150','C1 root-only','#AC72A6','Tree & recursion'),
('Recursive B four GPU','Recursive B','#E38927','Tree & recursion'),
('GiGPO safety fresh200 3347','GiGPO','#A6577F','Other baselines'),
('Three-block Q-probe stdloss 3399 → CP5 recovery3400','Q-probe 3-block','#9A69AF','Q-probe'),
('CoSearch common-em-v4 dual-model 713379','CoSearch dual-model','#DF8E32','Reward & selection'),
('HPC3 Q-probe MLP old loss 719225','Q-probe MLP · stopped117','#738194','Q-probe'),
('HPC3 Q-probe sigma3 new loss 721485','Q-probe σ3 · Co','#168D7B','Q-probe'),
('TOP-D alpha0.1','TOP-D α=0.1','#E1764B','Other baselines'),
('TOP-D alpha1','TOP-D α=1','#66578C','Other baselines')]
models=[]
for idx,(name,label,color,family) in enumerate(spec):
 r=next(r for r in report['runs'] if r['name']==name)
 raw=r.get('train_series',r.get('training_series',[]))
 if name.startswith('CoSearch common'):raw=chain['cosearch']['chosen']
 pts={int(p.get('iter',p.get('iteration'))):float(p['em']) for p in raw if p.get('em') is not None}
 if name.startswith('GiGPO'):
  old=next(m for m in history['models'] if m['name']=='GiGPO')
  supplement={i:v for i,v in old['train'] if 51<=i<=200 and str(i) in old['train_source']}
  assert len(supplement)==150
  for i,v in pts.items():assert abs(v-supplement[i])<2e-8
  supplement.update(pts);pts=supplement
 fixed=copy.deepcopy(r['fixed512'])
 if name.startswith('GiGPO'):
  for e in r.get('historical_ancestor_fixed512_unrechecked',[]):fixed.append(dict(e,historical=True,historical_unrechecked_this_turn=True))
 models.append({'id':str(idx),'name':name,'label':label,'family':family,'color':color,'train':[[i,pts[i]] for i in sorted(pts)],'fixed512':fixed,'completed':r['completed_iter'],'target':r.get('target_iter',200),'status':r.get('status'),'note':r.get('note'),'fresh':True})
# Keep already displayed historical baselines; their dates remain explicit.
for key,label,color in [('BAPO','BAPO 3359 · historical','#BE6A47'),('OPSD','OPSD 3304 · historical','#82934E'),('Search-E1 Archive4','E1 Archive4 / R1 · historical','#7B8A93')]:
 old=next(m for m in history['models'] if m['name']==key)
 fixed=[dict(e,iter=e['iteration'],n=512,historical=True) for e in old['fixed512']]
 models.append({'id':key,'name':key,'label':label,'family':'Other baselines','color':color,'train':old['train'],'fixed512':fixed,'completed':old['latest_iter'],'target':200,'status':'historical_snapshot_20261014','note':'Historical scalar snapshot from2026-10-14; not refreshed during the current seven-session audit.','fresh':False})
for m in models:
 assert len({p[0] for p in m['train']})==len(m['train'])
 assert all(0<=p[1]<=1 for p in m['train'])
 for e in m['fixed512']:
  if e.get('em') is not None:assert e['n']==512

font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B7C2CB','axes.labelcolor':'#40515D','xtick.color':'#667987','ytick.color':'#667987','axes.unicode_minus':False,'savefig.facecolor':'white'})
INK='#263B49';MUTED='#72838E';GRID='#E7EDF0'
STAMP='2026-10-15 · 16:58–17:11 CST'
def seg(pts):
 out=[]
 for p in pts:
  if not out or p[0]!=out[-1][-1][0]+1:out.append([])
  out[-1].append(p)
 return out
def avg(pts):
 return [[pts[j][0],sum(p[1] for p in pts[j-4:j+1])/5] for j in range(4,len(pts)) if pts[j][0]-pts[j-4][0]==4]
def series(ax,m,limit=200,inset=False):
 pts=[p for p in m['train'] if p[0]<=limit]
 ls='--' if m['name']=='Soft Reward -> Hard Reward' or not m['fresh'] else '-'
 for s in seg(pts):ax.plot([p[0] for p in s],[100*p[1] for p in s],color=m['color'],alpha=.23,lw=.9,ls=ls)
 sm=avg(pts)
 if not sm:sm=pts
 for k,s in enumerate(seg(sm)):
  ax.plot([p[0] for p in s],[100*p[1] for p in s],color=m['color'],lw=2.2 if not inset else 1.4,ls=ls,label=m['label'] if k==0 else None,marker='o' if len(s)<=5 else None,markersize=4)
 if pts:
  p=pts[-1];ax.scatter(p[0],p[1]*100,s=24,color=m['color'],edgecolor='white',lw=.5,zorder=5)
 else:ax.plot([],[],color=m['color'],label=m['label']+' · no completed outer')
def style(ax,xmax=200,ymax=85):
 ax.set_xlim(0,xmax);ax.set_ylim(0,ymax);ax.set_xlabel('Outer iteration');ax.set_ylabel('Rollout EM (%)');ax.grid(axis='y',color=GRID,lw=.8);ax.set_axisbelow(True)

fig,axes=plt.subplots(2,2,figsize=(19.8,11.5));fig.subplots_adjust(left=.065,right=.975,bottom=.10,top=.81,wspace=.21,hspace=.56)
families=['Reward & selection','Tree & recursion','Other baselines','Q-probe']
titles=['(a) Reward & question selection','(b) Tree branching & recursive sampling','(c) Other baselines · historical lines marked','(d) New Q-probe settings · first12 iterations']
for j,(ax,family,title) in enumerate(zip(axes.flat,families,titles)):
 mm=[m for m in models if m['family']==family]
 for m in mm:series(ax,m,12 if j==3 else 200)
 style(ax,12 if j==3 else 200,70 if j in [1,2] else (65 if j==3 else 85))
 ax.set_title(title,loc='left',fontsize=15,color=INK,pad=58)
 ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.015),ncol=2 if j!=1 else 3,frameon=False,fontsize=10.6,handlelength=2.1,columnspacing=1.3)
 if j==3:
  ax.text(.02,.96,'G1 σ3: outer1 rollout28.98%;\nAdam update unfinished.\nNo completed-iteration point.',transform=ax.transAxes,va='top',color=MUTED,fontsize=10.5)
  ia=ax.inset_axes([.58,.62,.39,.32]);old=next(m for m in mm if m['name'].startswith('HPC3 Q-probe MLP'));series(ia,old,117,True);ia.set_xlim(0,120);ia.set_ylim(15,65);ia.grid(axis='y',color=GRID);ia.tick_params(labelsize=8);ia.set_title('Stopped MLP run · full117',fontsize=9,color=MUTED)
fig.suptitle('Training rollout EM',x=.065,y=.975,ha='left',fontsize=26,color=INK,fontweight='bold')
fig.text(.065,.931,STAMP+'  |  Native rollout EM; settings and recovery lineages kept separate.',color=MUTED,fontsize=12)
fig.text(.065,.052,'Faint = each completed outer; bold = trailing5-outer mean. Missing data are gaps. Selected-question / tree-rollout EMs use different sampling distributions.',color=MUTED,fontsize=10.5)
fig.text(.065,.027,'GiGPO51–200 includes source-bound historical scalars (12 points rechecked). BAPO / OPSD / E1 Archive use the labeled10/14 historical snapshot.',color=MUTED,fontsize=10.5)
fig.savefig(OUT/'training-em-curves.png',dpi=165);fig.savefig(OUT/'training-em-curves.svg');plt.close(fig)

panels=[]
for step in [5,50,100,150,200]:
 rows=[];missing=[]
 for m in models:
  e=next((e for e in m['fixed512'] if e['iter']==step and e.get('em') is not None),None)
  if e is None:missing.append(m['label']);continue
  historical=not m['fresh'] or bool(e.get('historical'))
  label=m['label']+(' · historical50' if e.get('historical_unrechecked_this_turn') else '')
  rows.append({'label':label,'em':e['em'],'correct':int(round(e['em']*512)),'color':m['color'],'historical':historical,'source':e,'id':m['id']})
 rows.sort(key=lambda r:(-r['em'],r['label']));panels.append({'iteration':step,'rows':rows,'missing':missing})
fig,axes=plt.subplots(3,2,figsize=(20,18.5));fig.subplots_adjust(left=.185,right=.965,bottom=.075,top=.905,wspace=.82,hspace=.36)
for ax,p in zip(axes.flat,panels):
 rows=p['rows'];n=len(rows);ax.barh(range(n),[r['em']*100 for r in rows],color=[r['color'] for r in rows],height=.66,alpha=.9)
 ax.set_yticks(range(n),[r['label'] for r in rows],fontsize=10.8);ax.invert_yaxis();ax.set_xlim(0,60);ax.set_xlabel('Fixed-512 EM (%)');ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=9)
 for j,r in enumerate(rows):ax.text(r['em']*100+.7,j,f'{r["em"]*100:.2f}%  ({r["correct"]})',va='center',fontsize=10.8,color=INK)
 ax.set_title(f'iter{p["iteration"]}  ·  {n} evaluated settings',loc='left',fontsize=17,color=INK,pad=16,fontweight='bold')
ax=axes.flat[-1];ax.axis('off')
ax.text(-.25,.94,'Evaluation contract',transform=ax.transAxes,fontsize=21,color=INK,fontweight='bold')
notes=['Fixed512 = NQ256 + HotpotQA256.','Bars use the exact checkpoint iteration.','Numbers in parentheses are correct /512.','Unavailable checkpoints are omitted, never zero-filled.','','Mixed iter5 is64-question stage evaluation;','it is excluded from the512-question bars.','','TOP-D α=0.1 iter200 uses native47.85%;','the separate48.44% re-evaluation is retained','in the evidence, not substituted silently.','','Historical rows are explicitly labeled.','Training rollout EM is shown in a separate figure.']
ax.text(-.25,.83,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=14,color=MUTED,linespacing=1.65)
fig.suptitle('Fixed512 evaluation EM · checkpoint rankings',x=.055,y=.972,ha='left',fontsize=27,color=INK,fontweight='bold')
fig.text(.055,.942,STAMP+'  |  iter5 /50 /100 /150 /200 · all available audited settings + labeled historical baselines',color=MUTED,fontsize=12)
fig.text(.055,.03,'Per-iteration ranking of observed validation scores. Different settings, sample selection and inference execution are not a controlled algorithm-only comparison.',color=MUTED,fontsize=11)
fig.savefig(OUT/'fixed512-em-bars.png',dpi=150);fig.savefig(OUT/'fixed512-em-bars.svg');plt.close(fig)
for p in panels:
 rows=p['rows'];fig,ax=plt.subplots(figsize=(13,7.7));fig.subplots_adjust(left=.31,right=.94,bottom=.13,top=.83)
 ax.barh(range(len(rows)),[r['em']*100 for r in rows],color=[r['color'] for r in rows],height=.67,alpha=.9)
 ax.set_yticks(range(len(rows)),[r['label'] for r in rows],fontsize=12);ax.invert_yaxis();ax.set_xlim(0,60);ax.grid(axis='x',color=GRID);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=10);ax.set_xlabel('Fixed-512 EM (%)')
 for j,r in enumerate(rows):ax.text(r['em']*100+.65,j,f'{r["em"]*100:.2f}%  ({r["correct"]}/512)',va='center',fontsize=11,color=INK)
 fig.suptitle(f'iter{p["iteration"]} · 固定512题EM',x=.055,y=.962,ha='left',fontsize=24,color=INK,fontweight='bold')
 fig.text(.055,.90,STAMP+' · 仅本轮已有的512题成绩；未发布不补零；历史结果已标注。',fontsize=11,color=MUTED)
 fig.savefig(OUT/f'fixed512-iter{p["iteration"]}.png',dpi=165);plt.close(fig)

final_file=D/'final-seven-table-data.json'
final=json.loads(final_file.read_text()) if final_file.exists() else None
if final:
 rows=final['rows'];table_rows=[]
 for r in rows:
  label=r['experiment']+(f" · iter{r['checkpoint_iter']}" if r.get('checkpoint_iter') is not None else '')
  if label.startswith('Easy Selection'):label='Easy Selection\n+ Soft Reward · iter200'
  vals=['—' if r['dataset_em'][s] is None else f"{r['dataset_em'][s]*100:.2f}%" for s in final['datasets']]
  table_rows.append([label]+vals+['未报告' if r['macro_em'] is None else f"{r['macro_em']*100:.2f}%"])
 table_rows.append(['CoSearch双模型 · iter200']+['—']*7+['评测中'])
 fig,ax=plt.subplots(figsize=(19,10.5));ax.axis('off');fig.subplots_adjust(left=.035,right=.975,top=.81,bottom=.145)
 t=ax.table(cellText=table_rows,colLabels=['实验／checkpoint']+final['dataset_labels']+['七集平均'],colWidths=[.26]+[.088]*7+[.124],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,2.35)
 for (i,j),c in t.get_celld().items():
  c.set_edgecolor('#E0E7EB');c.set_linewidth(.7);c.PAD=.05
  if i==0:c.set_facecolor('#304958');c.get_text().set_color('white');c.get_text().set_weight('bold')
  else:
   ref=i<=len(rows) and rows[i-1]['kind'].startswith('paper')
   c.set_facecolor('#EFF4F7' if ref else ('#FFFFFF' if i%2 else '#F7F9FA'));c.get_text().set_color(INK)
   if j==0:c.get_text().set_ha('left')
   if j==8:c.get_text().set_weight('bold')
   if i<=len(rows) and rows[i-1]['experiment']=='Easy Selection+Soft Reward':c.set_facecolor('#E2F2EC');c.get_text().set_weight('bold')
 # Extra height for the deliberately wrapped Easy Selection label.
 easy=next(i+1 for i,r in enumerate(rows) if r['experiment']=='Easy Selection+Soft Reward')
 for j in range(9):t[(easy,j)].set_height(t[(easy,j)].get_height()*1.5)
 fig.suptitle('51,713题最终测试 · 七数据集EM',x=.035,y=.97,ha='left',fontsize=26,color=INK,fontweight='bold')
 fig.text(.035,.917,STAMP+'  ·  按七集等权平均排序；BAPO为iter150，其余已完成本地行为iter200。',fontsize=13,color=MUTED)
 fig.text(.035,.085,'七集平均 = 七个数据集EM的算术平均；与按51,713题加权的整体EM不同。论文行沿用用户给出的原文参考值。',fontsize=11.5,color=MUTED)
 fig.text(.035,.051,'BAPO / OPSD沿用10月14日已核验摘要；Easy Selection评分已完整，收尾曾异常，正式终态验收待核。CoSearch尚无完整终评。',fontsize=11.5,color=MUTED)
 fig.savefig(OUT/'final51713-seven-dataset-table.png',dpi=165);fig.savefig(OUT/'final51713-seven-dataset-table.svg');plt.close(fig)
payload={'stamp':STAMP,'models':models,'ranking':panels,'final':final,'notes':{'training':'Per-outer rollout EM; sample selection differs. No completed point is assigned to partial outer1.','testing':'Fixed NQ256+Hotpot256; no64-question stage points.','final':'Seven-dataset macro is unweighted; micro is pooled51713 EM.'}}
(D/'visual-payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
(D/'visual-build-verification.json').write_text(json.dumps({'created_at_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_sha256':hashlib.sha256((D/'all-seven-results.json').read_bytes()).hexdigest(),'co_canonical_outers':len(chain['cosearch']['chosen']),'co_superseded_records_excluded':6,'models':len(models),'gap_preserving':True,'new_partial_outer_plotted':False,'fixed512_only':True,'ranking_counts':{p['iteration']:len(p['rows']) for p in panels},'artifacts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in OUT.glob('*') if p.is_file()]},indent=2))
print(json.dumps({'figures':str(OUT),'models':len(models),'ranking_counts':{p['iteration']:len(p['rows']) for p in panels},'final_table_ready':bool(final)}))
