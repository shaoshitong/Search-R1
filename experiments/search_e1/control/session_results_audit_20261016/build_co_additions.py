from pathlib import Path
import json,copy,statistics,datetime,hashlib
D=Path(__file__).resolve().parent
old=json.loads((D.parent/'session-refresh-20261015-current/all-seven-results.json').read_text())
observations=[json.loads((D/f'co-observation-{i}.json').read_text()) for i in (1,2,3)]
obs=observations[-1];sigma,fin,co=obs['runs']
def file(run,name):return next(x for x in run['files'] if Path(x['path']).name==name)
name='CoSearch common-em-v4 dual-model 713379';r=copy.deepcopy(next(x for x in old['runs'] if x['name']==name))
plan=file(co,'recovery-plan.json');p=plan['data'];assert p['step']==100 and str(p['source_job'])=='709923' and str(p['prior_history']['prior_job'])=='709265' and p['prior_history']['prior_through']==5
canonical=sorted([x for x in co['train'] if x['job']==('709265' if x['iter']<=5 else '709923' if x['iter']<=100 else '713379')],key=lambda x:x['iter'])
assert [x['iter'] for x in canonical]==list(range(1,201))
r['training_series']=canonical;r['train_latest_em']=canonical[-1]['em'];r['train_recent10_mean']=statistics.mean(x['em'] for x in canonical[-10:]);r['train_recent_mean_n']=10;r['target_iter']=200
r['fixed512']=sorted(co['fixed512'],key=lambda x:x['iter'])
c=file(fin,'complete.json');a=file(fin,'acceptance.json');binding=file(fin,'binding.json');f=file(fin,'final-completion-report.json')
assert c['data']['status']=='COSEARCH_DUAL_MODEL_FINAL51713_COMPLETE' and c['data']['rows']==51713
assert a['data']['completion_sha256']==c['sha256'] and c['data']['binding_sha256']==binding['sha256']
assert a['data']['status']=='COSEARCH_DUAL_MODEL_FINAL51713_INDEPENDENT_ACCEPTANCE_PASS'
assert a['data']['full_question_identity_and_independent_EM_F1_pass'] and a['data']['both_exported_models_readback_and_SHA_pass'] and a['data']['owned_cleanup_pass']
assert f['data']['evaluation_and_audit_completed_zero_exit'] and f['data']['completion_receipt_sha256']==c['sha256']
m=c['data']['metrics'];assert m==a['data']['metrics']
ds=m['datasets'];assert sum(x['questions'] for x in ds.values())==51713 and sum(x['correct'] for x in ds.values())==m['correct']==25234
assert all(x['em']==x['correct']/x['questions'] for x in ds.values())
macro=statistics.mean(x['em'] for x in ds.values());micro=m['correct']/51713
assert abs(macro-m['macro_dataset_em'])<1e-12 and micro==m['em']
r['final51713']={'complete':True,'iter':200,'n':51713,'correct':25234,'em':micro,'f1':m['f1'],'macro_em':macro,'macro_f1':statistics.mean(x['f1'] for x in ds.values()),'source_metrics':{s:{'count':x['questions'],'correct':x['correct'],'em':x['em'],'f1':x['f1']} for s,x in ds.items()},'datasets':ds,'path':c['path'],'sha256':c['sha256'],'acceptance_path':a['path'],'acceptance_sha256':a['sha256'],'scorer_version':m['scorer_version'],'completed_epoch':c['data']['epoch'],'independent_acceptance':True}
r['status']='TRAINING_AND_FINAL51713_INDEPENDENTLY_ACCEPTED';r['note']='Native双模型CoSearch：主actor和单独训练reranker均来自iter200；检索top50再rerank top5，greedy。与单模型方法推理计算预算不同。200outer canonical按7092651–5/7099236–100/713379101–200，当前245收据中45条旧失败后缀不入训练曲线。完整终评51713、模型导出SHA、独立重评分及作业退出0已核验。';r['gpu']='final evaluation721542 ACD1-19, 4 H100, COMPLETED; allocation released';r['audit_file']=str(D/'co-updated-runs.json')
name='HPC3 Q-probe sigma3 new loss 721485';s=copy.deepcopy(next(x for x in old['runs'] if x['name']==name))
us=sigma['updates'];assert [u['step'] for u in us]==list(range(1,sigma['completed_iter']+1))
assert len({u['attempt_id'] for u in us})==len(us)
train=sorted(sigma['train'],key=lambda x:x['iter']);assert [x['iter'] for x in train]==list(range(1,len(us)+1)) and all(x['n']==512 and x['em'] is not None for x in train)
s.update(completed_iter=len(us),target_iter=200,training_series=train,train_latest_em=train[-1]['em'],train_recent10_mean=statistics.mean(x['em'] for x in train[-10:]),train_recent_mean_n=min(10,len(train)),optimizer_per_rank=us[-1]['optimizer_writes_after'])
vals=[x for x in sigma['validation_ledger'] if x['n']==512];assert len({x['iter'] for x in vals})==len(vals) and all(x['scalar_readback_correct'] and x['unique_question_ids']==512 and x['correct']/512==x['em'] for x in vals)
s['fixed512']=sorted(vals,key=lambda x:x['iter'])
for row in s['fixed512']:row['sha256']=sigma['validation_ledger_proof']['sha256']
s['final51713']={'complete':False,'n':None,'em':None,'f1':None,'path':None,'reason':f'current run{len(us)}of200, final not due'}
s['note']='Q-probe sigma3三分类损失新setting，与旧MLP及CoSearch dual分开。完整训练outer1..177以actual-updates正式attempt绑定rollout-audit；178候选采集不计已完成。fixed512直接从validation-identities.jsonl在远端逐题标量重算正确数/EM/F1，读取0/5/50/100/150；未下载预测文本或逐题数组。当前训练日志未见新致命错误；training721449错误属于历史预启动，不误计当前721485失败。';s['audit_file']=str(D/'co-updated-runs.json')
obs_summary=[]
for d in observations:
 q=d['runs'][0];obs_summary.append({'utc':d['utc'],'completed_iter':q['completed_iter'],'actor_optimizer_after':q['updates'][-1]['optimizer_writes_after'],'trainer_bytes':next(x['bytes'] for x in q['logs'] if x['path'].endswith('/trainer.log')),'rank_last':[x['latest'] for x in q['rank_optimizer']]})
out={'observed_at_cst':datetime.datetime.fromisoformat(obs['utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'read_only':True,'remote_state_changed':False,'training_agent_messages_sent':False,'runs':[r,s],'observations':obs_summary,'canonical_co_lineage_proof':{'path':plan['path'],'sha256':plan['sha256'],'data':p,'total_raw_rows':len(co['train']),'canonical_rows':len(canonical),'superseded_rows_excluded':len(co['train'])-len(canonical)},'sources':[{'path':str(D/f'co-observation-{i}.json'),'sha256':hashlib.sha256((D/f'co-observation-{i}.json').read_bytes()).hexdigest()} for i in (1,2,3)]}
(D/'co-updated-runs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'observed_at':out['observed_at_cst'],'co':{'macro':macro,'micro':micro,'fixed':[(x['iter'],x['em']) for x in r['fixed512']]},'sigma':{'complete':len(us),'latest_em':s['train_latest_em'],'near10':s['train_recent10_mean'],'fixed':[(x['iter'],x['em'],x['f1']) for x in s['fixed512']]},'path':str(D/'co-updated-runs.json')},ensure_ascii=False))
