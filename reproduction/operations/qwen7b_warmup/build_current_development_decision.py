"""Summarize existing small fixed-pool metrics only; never score or read final."""
import hashlib,json,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[3];R=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021'
def load(name):return json.loads((R/name).read_bytes())
def proof(name):
 p=R/name;r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
P80='viewed2048-two-seed-P80-paired-comparison/actual-latest.json';WIN='viewed2048-current64-window2018-sameE10-comparison/actual-latest.json';FULL='packed-variance-full169028/agent-observations-mae/attempt-749576/actual-same-epoch-three-arm-10.json';BASE='mixed-direct-e1-unique-g5/E1-matched-training-independent-baseline.json';p80=load(P80)['result'];win=load(WIN)['result'];full=load(FULL);base=load(BASE);rows=[]

def common(metrics,family):
 cw=metrics['classwise'];aw=cw['all_wrong'];ac=cw['all_correct'];top=metrics['Top512'];loss=None;kind=None;missing=None
 if family=='direct':kind='Mixed BCE';loss=metrics.get('Mixed_BCE',metrics.get('BCE'))
 elif family=='topk':kind='fixed-pool TopK-ramp loss';missing='Not computed/saved on fixed diagnostic pools; online optimizer-batch loss is deliberately not substituted.'
 else:kind='variance MSE(v)';loss=metrics.get('variance_MSE')
 return dict(questions=metrics['questions'],true_Mixed_questions=metrics.get('Mixed_questions'),fixed_diagnostic_loss=dict(kind=kind,value=loss,missing_reason=missing if loss is None else None),Mixed_BA=metrics['Mixed_balanced_accuracy'],Mixed_recall=metrics['Mixed_recall'],all_wrong_FPR=aw['predicted_Mixed_fraction'],all_correct_FPR=ac['predicted_Mixed_fraction'],Top512_Mixed=top['Mixed'],Top512_fraction=top['fraction'],Mixed_AUC=metrics.get('Mixed_ROC_AUC'),threshold=metrics['threshold'],variance_MSE=metrics.get('variance_MSE'),variance_MAE=metrics.get('variance_MAE'))
def add_model(name,seed,epoch,state,objective,train,dev,source,budget):
 rows.append(dict(experiment=name,seed=seed,epoch=epoch,state=state,actual_training_objective=objective,budget=budget,trainA=None,trainB=common(train,'topk'if objective=='TopK-ramp'else'variance'),development=common(dev,'topk'if objective=='TopK-ramp'else'variance'),source=source))

# E1 training-A must never masquerade as the newer, consistent training-B.
t=base['fixed_training']['metrics']['EMA']['fixed_train2048'];pool=t['pool'];a=dict(questions=2048,true_Mixed_questions=1082,fixed_diagnostic_loss=dict(kind='Mixed BCE',value=t['binaryMixedBCE'],missing_reason=None),Mixed_BA=pool['mixed_balanced_accuracy'],Mixed_recall=pool['mixed_recall'],all_wrong_FPR=pool['allwrong_false_positive_rate'],all_correct_FPR=pool['allcorrect_false_positive_rate'],Top512_Mixed=t['selected512']['mixed_questions'],Top512_fraction=t['selected512']['mixed_fraction'],Mixed_AUC=pool['mixed_roc_auc'],threshold=.5,variance_MSE=None,variance_MAE=None)
rows.append(dict(experiment='E1 baseline',seed=None,reference_only=True,epoch=1,state='EMA',actual_training_objective='direct Mixed BCE',budget=dict(training_questions=169028,cursor=2577,earlier_micro_batch_updates_and_hard_replay=True),trainA=a,trainB=None,trainB_missing_reason='Original E1 was not scored on fixed trainB in these saved fixed diagnostics.',development=common(p80['independent_summaries']['E1_EMA'],'direct'),source=dict(trainA=BASE+'#/fixed_training/metrics/EMA/fixed_train2048',development=P80+'#/result/independent_summaries/E1_EMA')))
for family,epoch,prefix,objective,attempts in [('CURRENT64',5,'CURRENT64','TopK-ramp',640),('MSE-M5',7,'MSE_M5','MSE(4v)',448)]:
 for seed in [1,2]:
  key=prefix+'_seed'+str(seed);add_model(family,seed,epoch,'raw',objective,p80['fixed_train_B_separate'][key]['metrics'],p80['independent_summaries'][key],dict(summary=P80,key=key,remote=p80['source_proofs'][key]),dict(training_questions=8192,checkpoint_batch_attempts=attempts,global_batch=64 if family=='CURRENT64'else 128))
for family in ['CURRENT64','WINDOW2018']:
 for seed in [1,2]:
  key=family+'_seed'+str(seed);add_model(family,seed,10,'raw','TopK-ramp',win['fixed_train_B_separate'][key]['metrics'],win['independent_summaries'][key],dict(summary=WIN,key=key,remote=win['source_proofs'][key]),dict(training_questions=8192,checkpoint_batch_attempts=1280,global_batch=64,actual_metadata=win['actual_training_metadata'][key]))
for arm in ['mse','mae','z_mse_anchor']:
 for state in ['raw','EMA']:
  metrics=full['arms'][arm]['producer_metrics'];converted={}
  for role in ['fixed_train2048_G5','fixed_validation2048_G5']:
   m=metrics[role][state];d=m['predeclared_G5_classification_diagnostic'];cw=m['classwise'];losskey='variance_MAE'if arm=='mae'else'variance_MSE';losskind='variance MAE(v)'if arm=='mae'else'variance MSE(v)';converted[role]=dict(questions=2048,true_Mixed_questions=888 if role.startswith('fixed_train')else 663,fixed_diagnostic_loss=dict(kind=losskind,value=m[losskey],missing_reason=None,note='Original-unit fixed-pool diagnostic, not normalized Z-MSE+anchor total or online batch loss.'if arm=='z_mse_anchor'else'Fixed diagnostic in variance units; actual model trains on4v.'),Mixed_BA=d['balanced_accuracy'],Mixed_recall=d['Mixed_recall'],all_wrong_FPR=cw['all_wrong']['diagnostic_false_positive_rate'],all_correct_FPR=cw['all_correct']['diagnostic_false_positive_rate'],Top512_Mixed=m['Top512']['Mixed'],Top512_fraction=m['Top512']['Mixed_fraction'],Mixed_AUC=None,Mixed_AUC_missing_reason='This saved E10 producer metric does not contain ROC-AUC.',threshold=.08,variance_MSE=m['variance_MSE'],variance_MAE=m['variance_MAE'])
  rows.append(dict(experiment='fullP20 '+arm,seed=1,epoch=10,state=state,actual_training_objective=dict(mse='MSE(4v)',mae='MAE(4v)',z_mse_anchor='global-batch standardized MSE + original4v MSE anchor')[arm],budget=dict(training_questions=169028,cursor=13210,layer=21,prefix=20,M='uniform3..10'),trainA=None,trainB=converted['fixed_train2048_G5'],development=converted['fixed_validation2048_G5'],source=dict(summary=FULL,arm=arm,state=state,job_id=full['arms'][arm]['job_id'])))
assert len(rows)==15 and all(z['development']['true_Mixed_questions']==663 for z in rows)
q=load('full169028-L15P80-MSE-ownG5/qualification-watch-latest.json');z=load('seed2-agent-observer/agent-seed2-watch-latest.json');out=dict(status='CURRENT_VIEWED_NQ_DEVELOPMENT_DECISION_FROM_EXISTING_FIXED_METRICS',generated_at=time.time(),rows=rows,pools=dict(trainA=dict(questions=2048,Mixed=1082,all_wrong=720,all_correct=246,note='Old original E1 fixed training-A; never directly compared to trainB.'),trainB=dict(questions=2048,Mixed=888,all_wrong=950,all_correct=210,authoritative_metadata_SHA256='e6e74d92160efb0740ba5c0316c4ee7686f06a499c90cafd5a545f325b62209b'),development=dict(questions=2048,Mixed=663,all_wrong=1055,all_correct=330,NQ_only=True,own_G5_same_question_trajectory_order_and_EM_verified=True)),comparisons=dict(selected_P80_two_seed=p80['family_two_seed_average'],MSE_minus_current64_selected=p80['MSE_minus_current64'],window_minus_current_sameE10=win['paired_comparisons']),fullP80_input_status=q,full_Z_seed2_actual_status=dict(time=z['time'],job_id=z['cells']['Z']['job_id'],rank_epochs=z['cells']['Z']['actual_rank_epochs'],rank_cursors=z['cells']['Z']['actual_rank_cursors'],completed_eval_epochs=z['cells']['Z']['evaluation_epochs'],failures=z['cells']['Z']['failures']),limits=['Metrics are viewed NQ development, not untouched final4096 or seven-benchmark generalization.','TrainA and trainB are distinct question/G5 diagnostic pools and cannot establish train-set improvement against each other.','TopK sigmoid scores are ranking outputs; threshold0.5 BA is diagnostic, not calibrated Mixed probability.','Variance threshold0.08 is predeclared; MSE/MAE losses are variance-unit diagnostics, not BCE.','Missing fixed-pool TopK loss and fullP20 E10 AUC remain null; no online batch loss substituted.','Bootstrap intervals condition on selected models and paired questions, not training-seed uncertainty.'],source_proofs=[proof(n)for n in [P80,WIN,FULL,BASE,'full169028-L15P80-MSE-ownG5/qualification-watch-latest.json','seed2-agent-observer/agent-seed2-watch-latest.json']],new_inference=False,checkpoint_or_tensor_opened=False,final4096_read=False,Slurm_mutated=False)
snapshots=R/'current-development-decision-sources';snapshots.mkdir(exist_ok=True)
for index,ref in enumerate(out['source_proofs']):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];snapshot=snapshots/('%02d-'%index+Path(ref['path']).parent.name+'--'+Path(ref['path']).name);snapshot.write_bytes(raw);ref['snapshot']=str(snapshot.relative_to(REPO))
(R/'CURRENT-DEVELOPMENT-DECISION.json').write_text(json.dumps(out,indent=2)+'\n')
def pct(v):return'—'if v is None else'%.2f%%'%(100*v)
def loss(m):
 if m is None:return'—'
 d=m['fixed_diagnostic_loss'];return'缺失（未计算）'if d['value']is None else d['kind']+' %.6f'%d['value']
def table(role):
 a=['| checkpoint | 固定诊断loss | Mixed BA | Mixed召回 | 全错误报 | 全对误报 | Top512 Mixed |','|---|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  m=row[role]
  if role=='trainA'and m is None:continue
  label='%s / %s / E%d %s'%(row['experiment'],'reference'if row['seed']is None else'seed'+str(row['seed']),row['epoch'],row['state'])
  if m is None:a.append('| '+label+' | — | — | — | — | — | — |');continue
  a.append('| %s | %s | %s | %s | %s | %s | %d/512 (%s) |'%(label,loss(m),pct(m['Mixed_BA']),pct(m['Mixed_recall']),pct(m['all_wrong_FPR']),pct(m['all_correct_FPR']),m['Top512_Mixed'],pct(m['Top512_fraction'])))
 return'\n'.join(a)
text='''# Current development decision

现有信号仍在**已浏览的 NQ 开发集**：MSE-M5 E7/raw 两seed为278/512与277/512，均值54.20%；原E1 EMA为205/512（40.04%）。这尚非最终4096确认，也不是七集结论。

CURRENT64与WINDOW2018在完全相同E10/raw、每seed相同采样计划及1280次global64批次尝试下，窗口的两seed平均Top512仅提高2.2461个百分点，配对题目95%区间[0.0000,+4.6875]个百分点触及零；单seed区间均跨零，不能称稳健确认提升。BA诊断反而平均下降7.40个百分点。

## 固定训练A：原E1历史诊断

A为2048题、1082 Mixed；仅原E1有这份指标。后续模型未在A评估。不能把A的训练结果与B直接作前后改进对照。

'''+table('trainA')+'''

## 固定训练B：后续模型一致诊断

B及所有8192题pilot均为NQ。B为2048题、888 Mixed、950全错、210全对，固定ownG5；原E1没有这份已保存评估，空缺保持为空。TopK固定池loss没有计算/保存，不能拿在线batch loss补上。MSE/MAE数值使用原方差v单位，实际训练输出回归4v；Z行的固定诊断MSE不等于其标准化训练总loss。

'''+table('trainB')+'''

## 已浏览开发集：同一2048题及固定ownG5

开发池为663 Mixed、1055全错、330全对，全部NQ；所选轨迹标签已独立通过actual ownEM/seal核验。Top512按预测分数排序，不按strictEM或分类阈值选题。TopK的0.5阈值只作BA诊断；方差模型使用预设v≥0.08。

'''+table('development')+'''

## 当前行动与边界

- 领先探索信号为MSE-M5 E7/raw两seed均约54%，但MSE与CURRENT64的已选checkpoint比较还混合epoch/global batch/更新预算差异，不是单独损失因果比较。
- WINDOW2018应按同E10比较，不能用CURRENT64的E5替代对照，也不能把触零区间说成已确认收益。
- fullP20三臂E10的raw/EMA全部保留。full Z第二seed仍按单独状态记录，不拿进行中的epoch替代第一seedE10。
- fullP80物理源输入仍以实际生产验收状态为准；本报告生成时的qualifier状态见下，不把source-stage当训练ready。训练只由root取得actual ready后唯一提交。
- 最终4096仍未读取，没有新推理、模型加载、CP读取、阈值拟合或Slurm操作。完整JSON保留全部原指标、AUC缺项、来源与配对区间；未保存的指标明确为null。

Qualifier snapshot: `'''+q['status']+'`, timestamp `'+str(q['time'])+'`, fixture complete=`'+str(q['fixture_complete_exists'])+'`, accepted=`'+str(q['fixture_accepted_exists'])+'''`。

[完整指标、来源及比较区间](CURRENT-DEVELOPMENT-DECISION.json)
'''
(R/'CURRENT-DEVELOPMENT-DECISION.md').write_text(text);print(json.dumps(dict(status=out['status'],rows=len(rows),fullP80=q['status'],final4096_read=False)))
