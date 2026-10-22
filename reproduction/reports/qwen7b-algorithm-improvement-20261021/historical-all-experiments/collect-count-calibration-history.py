from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];rows=[];inventory=[]
def load(rel):return json.loads((R/rel).read_text())
def add(exp,cp,state,test,top,k,n,source,train=None,pool='HF固定G5独立2048（旧产品/count支线）',note='',**more):
 rows.append(dict(experiment=exp,checkpoint=cp,state=state,train_BA=train,test_BA=test,top_mixed=top,top_k=k,pool_questions=n,pool_identity=pool,source=str((R/source).resolve()),comparability_note=note,status='MEASURED',**more))
for cp in [256,512,768,938]:
 rel='direct-hidden-count/actual-cursor-%06d-fixed-evaluation.json'%cp
 if (R/rel).exists():x=load(rel)['metrics']['selection2048FreshG5HF']
 else:rel='direct-hidden-count/actual-final938-summary.json';x=load(rel)['fixed_validation_metrics']['selection2048FreshG5HF']
 p=x['pool'];s=x['selected512'];add('直接hidden六类count（block27＋head）',f'CP{cp}','已发布评分模型；未拆raw/EMA',p['mixed_balanced_accuracy'],s['mixed_questions'],512,2048,rel,note='30016题/938更新；固定训练BA未留在该评测回执，不能用在线batch替代')
for cp in [256,512,768,938]:
 rel='direct-count-headlr1e4/actual-cursor-%06d-accepted.json'%cp;mx=load(rel)['metrics'];x=mx if 'pool'in mx else mx.get('selection2048FreshG5HF',mx.get('selection2048'));p=x['pool'];s=x['selected512'];add('直接count：仅head LR升至1e-4',f'CP{cp}','已发布评分模型；未拆raw/EMA',p['mixed_balanced_accuracy'],s['mixed_questions'],512,2048,rel,note='与原count单次938更新计划匹配；不是head-only小样本诊断')
rel='direct-count-capacity/FINAL.json';x=load(rel)
for lr,epochs in x['metrics'].items():
 for ep,states in epochs.items():
  for st,m in states.items():
   fit=m['fit_primaryG5'];test=m['internal_holdout_primaryG5'];top=test['Top256_of1024'];add('冻结E1 count-head拟合容量 LR='+lr,'E'+ep,st,test['Mixed_BA'],top['Mixed_count'],256,1024,rel,train=fit['Mixed_BA'],pool='head-only内部：fit512/holdout1024',note='原native已见题目；Top256不能充当2048→512独立结果',train_pool_questions=512)
rel='direct-count-large-capacity/actual-complete-fit-curve.json';x=load(rel)
for node in x['nodes']:
 for domain in ['NQ','HotpotQA','pooled']:
  test=node['held_'+domain]if domain!='pooled'else node['held_pooled'];top=test['Top25percent'];fit=node['fit_primaryG5'];add('冻结E1大样本count-head /'+domain,'E'+str(node['epoch']),node['mode'],test['Mixed_BA'],top['Mixed_count'],top['questions'],test['questions'],rel,train=fit['Mixed_BA'],pool='原warmup题目内部留出/'+domain,note='fit30016，holdout5120(NQ2560+Hotpot2560)，head-only20ep；不是native未见问题',train_pool_questions=30016)
for domain in ['NQ','HotpotQA','pooled']:
 m=x['baseline'][domain];top=m['Top25percent'];add('大样本count同cache原E1产品基线/'+domain,'E1冻结','published baseline',m['Mixed_BA'],top['Mixed_count'],top['questions'],m['questions'],rel,pool='原warmup题目内部留出/'+domain,note='不是新增训练；和该内部count实验逐题匹配')
# Same-backend old product models, provenance is the already published aggregate.
root=R.parents[2].parents[1];relp=root/'outputs/qwen7b-warmup-20261019/reports/peer8192-calibration-and-E2-final-summary.json';x=json.loads(relp.read_text())
for name,m in x['same_HF_fair_grid'].items():
 add('旧product-q同HF复评/'+name,name,'源报告公布模型',m['BA'],m['Top512_mixed'],512,2048,'platt-g10-calibration/RESULT.md',note='旧轨迹BCE+产品Mixed路线/同HF公平复评；与原vLLM指标分开',extra_source=str(relp))
rel='platt-g10-calibration/independent-final-audit.json';x=load(rel);m=x['external'];top=m['Top512'];add('Platt两参数标定peer256（12内部候选冻结1）','冻结winner','frozen calibrated model',m['balanced_accuracy'],top['Mixed_count']if'Mixed_count'in top else top.get('Mixed',top.get('mixed_questions')),512,2048,rel,note='训练6144、参数holdout2048，外部2048只评冻结winner；未用外部调阈值')
# Decision rules change no learned weights/ranking: record as diagnostics.
rel='full938-prior-decision-diagnostic/FINAL.json';x=load(rel)
for v in x['results']:
 for rule in ['primary','train_prior_decision']:
  m=v[rule];add('训练先验decision诊断/'+v['name'],rule,'same saved weights',m['balanced_accuracy'],v['unchanged_ranking']['Top512_Mixed'],512,2048,rel,note='只改训练侧先验decision规则，非新训练；Top排序保持不变')
# Append-only production and the exact-all252/random4 matched sampling experiment.
for cp in [512,768,938]:
 rel='append-only-682/actual-cursor-%06d-independent-audit.json'%cp;d=load(rel)['evaluations'];key=next(k for k in d if k.startswith('selection'));m=d[key];m=m.get('metrics',m);p=m['pool'];top=m['selected512'];add('Append-only peer256继续扩数据',f'CP{cp}','published evaluator',p['mixed_balanced_accuracy'],top['mixed_questions'],512,2048,rel,note='保留原256更新，增加682至938；无Adam重置、未重复原问题前缀')
rel='exact-uniform-all-G5/actual-final-independent-audit.json';d=load(rel);key=next(k for k in d['evaluations']if k.startswith('selection'));m=d['evaluations'][key]['metrics'];p=m['pool'];top=m['selected512'];add('G10 exact全部252个G5子集','CP136','published evaluator',p['mixed_balanced_accuracy'],top['mixed_questions'],512,2048,rel,pool='旧native固定G5独立2048',note='6528题/136更新，与random4同预算；后端口径与HF复评分开')
for name,body in d['baselines'].items():
 m=body['metrics'];p=m['pool'];top=m['selected512'];add('Exact-G5匹配参考/'+name,'frozen reference','published evaluator',p['mixed_balanced_accuracy'],top['mixed_questions'],512,2048,rel,pool='旧native固定G5独立2048',note='同exact252协议逐题配对，非额外新增训练')
for folder in ['frozenE1-head-objective-family','count53312-data-scaling','count53312-scaling','coarse-event-cached-forks','coarse-fixture-stack-diagnostic','prior-logit-event-loss']:
 inventory.append(dict(experiment=folder,status='PREPARED_OR_QUALIFICATION_ONLY_NO_TRAINED_OUTCOME_PROVEN_IN_LOCAL_RESULTS',note='不把准备、合成CPU测试或数据采集当成已训练模型',source=str((R/folder).resolve())))
out=dict(rows=rows,inventory=inventory,no_new_evaluation=True,all_scalars_from_existing_reports=True,full_weights_or_tensor_loaded=False)
(R/'historical-all-experiments/count-capacity-and-calibration.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'rows':len(rows),'inventory':len(inventory)}))
