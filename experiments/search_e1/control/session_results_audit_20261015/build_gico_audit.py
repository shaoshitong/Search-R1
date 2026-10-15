from pathlib import Path
import json,datetime,statistics,hashlib
D=Path(__file__).resolve().parent
read=lambda p:json.loads((D/p).read_text())
gi=read('gigpo-actual-2.json');gt=read('gigpo-targeted-1.json');cx=read('co-extra-2.json');ct=read('co-targeted-1.json');ca=read('co-actual-2.json');cm=read('co-milestones-1.json')
GI='01a0fcbb-863f-7450-926d-db537f648265';CO='01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef'
def base(name,sid,r,complete,status,note):
 return dict(name=name,session=sid,output_dir=r,completed_iter=complete,train_latest_em=None,train_recent10_mean=None,train_recent_mean_n=0,fixed512=[],final51713=dict(complete=False,em=None,f1=None,n=None,path=None),status=status,note=note)
def attach_train(o,t):
 t=sorted([x for x in t if x.get('em') is not None],key=lambda x:x.get('iter',x.get('iteration',0)));o['training_series']=t
 if t:o.update(train_latest_em=t[-1]['em'],train_recent10_mean=statistics.mean(x['em'] for x in t[-10:]),train_recent_mean_n=min(10,len(t)))
def attach_eval(o,v):
 uniq={}
 for x in v:
  if x.get('iter') is not None:uniq[x['iter']]=x
 o['fixed512']=sorted(uniq.values(),key=lambda x:x['iter'])
runs=[]
o=base('GiGPO safety fresh200 3347',GI,gi['runs'][0]['root'],200,'TRAINING_AND_FULL_FINAL_SCORED','本回合直接读 calibration.log 的全51713指标；训练/评测完成与全部归档验收分开。')
final=next(x for x in gi['runs'][0]['metrics'][0]['rows'] if x.get('val/qa_count/all')==51713)
o['final51713']={'complete':True,'em':final['val/em/all'],'f1':final['val/f1/all'],'n':51713,'path':gi['runs'][0]['metrics'][0]['path'],'sha256':gt['roots'][0].get('final_log_sha256'),'datasets':{k[len('val/em/'):]:v for k,v in final.items() if k.startswith('val/em/')}}
p=D/'oldgigpo-metrics-final.json'
if p.exists():
 m=read(p.name);rec=m.get('records',[]);attach_train(o,[{'iter':x['iter'],'em':x['metrics'].get('critic/score/mean'),'path':x['path'],'sha256':x['sha256']} for x in rec if x.get('complete')]);attach_eval(o,[{'iter':x['iter'],'em':x['metrics']['val/em/all'],'f1':x['metrics'].get('val/f1/all'),'n':512,'path':x['path'],'sha256':x['sha256']} for x in rec if x['metrics'].get('val/qa_count/all')==512])
runs.append(o)
o=base('Three-block Q-probe stdloss 3399 → CP5 recovery3400',GI,gt['roots'][1]['root'],5,'3399_FAILED_AFTER_VALIDATION_RECOVERY3400_RUNNING','3399完成5轮/每rank100Adam后monitor tensor hash审计退出1；CP5配对CPU状态+全SHA校验完成，恢复3400已实际RUNNING并进入检索stage，尚未读到新完整outer。初始dev25.39%的推理流程与旧MLP初始17.77%有差异，不能归为probe收益。')
attach_train(o,gt['roots'][1]['train']);attach_eval(o,gt['roots'][1]['fixed512']);o['recovery_output_dir']=gt['roots'][2]['root'];o['gpu']='h100-8, 8 H100';o['optimizer_per_rank']=100;o['final51713']['reason']='current fresh run has only5 completed iterations';runs.append(o)
o=base('CoSearch common-em-v4 dual-model 713379',CO,ca['runs'][0]['root'],200,'TRAINING_COMPLETE_FINAL_EVAL_RECOVERY_RUNNING','原训练controller FAILED仅因为final_eval退出1；200个completed-outer common EM receipts实有，不将历史final失败误报训练未完成。双模型当前终评由TOPD会话代评，job721542 ACD1-19。')
attach_train(o,ca['runs'][0]['training']);vals=[]
for x in ca['runs'][0]['json']:
 if '/common-validation/' in x['path']:
  d=x['data'];vals.append({'iter':int(Path(x['path']).name.split('.')[0]),'em':d['em'],'f1':d['f1'],'n':d['questions'],'path':x['path'],'sha256':x['sha256']})
attach_eval(o,vals);progress=next(x['data'] for x in ca['runs'][1]['json'] if x['path'].endswith('/progress.json'));o['final51713'].update(n=progress['rows'],target_n=51713,path=ca['runs'][1]['root']+'/progress.json',observed_epoch=progress['epoch'],reason='generation in progress, no final EM/F1 summary');o['gpu']='final inference ACD1-19, 4 H100';runs.append(o)
o=base('HPC3 Q-probe MLP old loss 719225',CO,ct['roots'][1]['root'],117,'HUMAN_CANCELLED_SUPERSEDED','sacct实测CANCELLED by14090；真实actual-updates1..117，每rank2340Adam。原session独立科学审计cutoff108，109..117日志存在但未冒称同级验收。50/100EM按输出日志每分源256题三位小数反推唯一整数正确数；F1仅日志三位近似。')
attach_train(o,ct['roots'][1]['train']);attach_eval(o,ct['roots'][1]['fixed512']);o['scientific_audit_cutoff_iter']=108;o['optimizer_per_rank']=2340;o['gpu']='historical ACD1-56,4 H100';o['final51713']['reason']='explicitly superseded before200'
for x in cm['roots'][1]['log_eval']:
 m=x['metrics'];n1=m.get('val/evaluated_count/nq');n2=m.get('val/evaluated_count/hotpotqa')
 if n1==n2==256:
  c=round(m['val/test_score/nq']*256)+round(m['val/test_score/hotpotqa']*256);o['fixed512'].append(dict(iter=x['iter'],em=c/512,f1=m.get('val/weighted_f1'),n=512,correct=c,path=ct['roots'][1]['root']+'/logs/search-g1-common8-training.log',em_method='unique integer count reconstructed from separately printed 256-question source EM values',f1_precision='rounded3decimal'))
o['fixed512']=sorted(o['fixed512'],key=lambda x:x['iter']);runs.append(o)
o=base('HPC3 Q-probe sigma3 new loss 721485',CO,ct['roots'][0]['root'],ct['roots'][0]['completed_iter'],'RUNNING_ACTUAL_FORWARD_PROGRESS','两次实读8→9完整outer、各rank160→180Adam；sigma3三分类损失新setting，与旧719225和CoSearch算法分开。CPU归档5恢复进行中不代表训练失败；先前旧CPU归档失败已保留。')
attach_train(o,ct['roots'][0]['train']);attach_eval(o,ct['roots'][0]['fixed512']);o['gpu']='ACD1-21,4 H100';o['optimizer_per_rank']=180;o['final51713']['reason']='current run9of200, final not due';runs.append(o)
report={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'training_agent_messages_sent':False,'remote_training_modified':False,'source_files':[],'runs':runs,'limits':['training EM across 512 questions aggregates five sampled responses/question; fixed512 uses one greedily inferred response/question','Full final51713 uses all7datasets; incomplete generation has no publishable final EM','Local only scalar summaries and source/audit scripts; raw rollout/tensor/checkpoint retained remote']}
for p in D.glob('*actual*.json'):
 report['source_files'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for n in ['gigpo-targeted-1.json','co-targeted-1.json','co-extra-2.json','co-milestones-1.json','gigpo-milestones-1.json','oldgigpo-metrics-final.json']:
 p=D/n
 if p.exists():report['source_files'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(D/'gigpo-co-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for r in runs:print(r['name'],r['completed_iter'],r['train_latest_em'],r['train_recent10_mean'],[(x['iter'],x['em']) for x in r['fixed512']],r['final51713'].get('em'),r['status'])
