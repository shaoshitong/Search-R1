"""Read-only pinned-session result merge. Preserve settings and source lineage."""
from pathlib import Path
import json,copy,hashlib,datetime,statistics
D=Path(__file__).resolve().parent
TZ=datetime.timezone(datetime.timedelta(hours=8));now=datetime.datetime.now(TZ)
def read(n):return json.loads((D/n).read_text())
def stamp(t):return datetime.datetime.fromtimestamp(t,TZ).isoformat()
def ev(n):p=D/n;return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
rpt=read('base-all-seven-results.json');by={r['name']:r for r in rpt['runs']};updated=[];added=[];before={r['name']:r.get('completed_iter') for r in rpt['runs']}
def update(r,series,fixed,completed,observed,source,note,status='training_running'):
 old={x['iter']:x['em'] for x in r.get('train_series',[])}
 for x in series:
  assert 0<=x['em']<=1
  if x['iter'] in old:assert abs(x['em']-old[x['iter']])<2e-8,(r['name'],x['iter'],old[x['iter']],x['em'])
 assert len({x['iter'] for x in series})==len(series)
 assert all(1<=x['iter']<=completed for x in series)
 for e in fixed:
  assert e['n']==512 and abs(e['em']*512-round(e['em']*512))<1e-7
  e['correct']=round(e['em']*512)
 r.update(completed_iter=completed,train_series=series,fixed512=fixed,train_latest_em=series[-1]['em'] if series else None,train_recent10_mean=statistics.mean(x['em'] for x in series[-10:]) if series else None,train_recent10_n=min(10,len(series)),status=status,observed_at_cst=stamp(observed),audit_file=str(D/source),note=note)
 updated.append(r['name'])
 return r
# Recursive B: inherited1..8, current native log9 onward, completed checkpoint only.
x=read('b-current.json');r=by['Recursive B four GPU'];n=x['checkpoint']['completed'];pts={p['iter']:p for p in r['train_series'] if p['iter']<9};fixed={p['iter']:p for p in r['fixed512'] if p.get('em') is not None}
for lg in x['logs']:
 for p in lg['rows']:
  if not 9<=p['iter']<=n:continue
  m=p['metrics'];proof=dict(iter=p['iter'],path=lg['path'],line=p['line'],sha256=p['sha256'])
  if 'critic/score/mean' in m:pts[p['iter']]=dict(proof,em=m['critic/score/mean'])
  if m.get('val/qa_count/all')==512:fixed[p['iter']]=dict(proof,n=512,em=m['val/em/all'],f1=m.get('val/f1/all'),correct=round(m['val/em/all']*512))
assert sorted(pts)==list(range(1,n+1));assert all(w['adam']==20*n and w['finite'] for w in x['checkpoint']['workers'])
update(r,[pts[i] for i in sorted(pts)],[fixed[i] for i in sorted(fixed)],n,x['epoch'],'b-current.json','Canonical same training chain1–3 /4–8 /9 onward; fresh native manifest and full emitted scalar log checked. Complete200 checkpoint; final51713 is generating, no partial EM substituted.')
r['status']='training_complete_final_generating'
r['checkpoint_manifest']=x['checkpoint'];r['final51713']=dict(complete=False,n=0,em=None,f1=None,progress=x['final_progress'],reason='No completed final51713 metric marker. Trajectory event count is progress only.')
# C2 independent mechanism ablation.
x=read('a-current.json');r=by['C2 random-final ablation150'];n=max(z['data']['completed_outer'] for z in x['receipts']);pts={};fixed={}
for lg in x['logs']:
 for p in lg['rows']:
  m=p['metrics'];i=p['iter']
  if i>n or m.get('training/actual_outer_updates')!=i:continue
  proof=dict(iter=i,path=lg['path'],line=p['line'],sha256=p['sha256'])
  if 'critic/score/mean' in m:pts[i]=dict(proof,em=m['critic/score/mean'])
  if m.get('val/qa_count/all')==512:fixed[i]=dict(proof,n=512,em=m['val/em/all'],f1=m.get('val/f1/all'),correct=round(m['val/em/all']*512))
update(r,[pts[i] for i in sorted(pts)],[fixed[i] for i in sorted(fixed)],n,x['epoch'],'a-current.json','Independent random-final ablation, target150; missing first scalar is a gap. New fixed50/100/150 taken from actual complete native scalar lines.')
by['C2 random-final ablation150']['status']='training_complete150' if n==150 else 'training_running'
# HPC sigma3: strict training EM from full actual G5 labels, not soft reward.
hpc=read('jxu957-current.json');sigma=hpc['runs'][0];r=by[sigma['name']]
pts={p['iter']:p for p in sigma['train']};assert sorted(pts)==list(range(1,201))
fixed=[dict(z,correct=round(z['em']*512)) for z in sigma['fixed'] if z['n']==512 and z['iter'] in [5,50,100,150,200]]
update(r,[pts[i] for i in sorted(pts)],fixed,200,hpc['epoch'],'jxu957-current.json','Strict generated selected512x5 training EM; classifier accuracy/soft reward excluded. Full final scoring exists; wrapper later failed cleanup with trainer left owned subprocesses. Final acceptance incomplete.','scoring_complete_cleanup_failed')
f=next(z for z in sigma['fixed'] if z['n']==51713);first=next(z for z in read('jxu957-current-first.json')['runs'][0]['fixed'] if z['n']==51713);identity=read('sigma-final-identity-readback.json')['final'][0]
assert f['row_sha256']==first['row_sha256']==identity['row_sha256'];assert identity['rows']==51713 and identity['final'] and identity['split']=='final'
canonical=read('sigma-canonical-data-readback.json');assert canonical['ordered_ids_and_sources_equal'] and canonical['canonical_rows']==51713
manifest=next(z['data'] for z in sigma['files'] if z['path'].endswith('/run-manifest.json'))
assert f['run_id']==manifest['run_id'] and f['validation_sha256']==manifest['input_sha256']['final.parquet']
r['final51713']=dict(complete=True,scoring_complete=True,final_acceptance_complete=False,n=51713,correct=int(f['correct']),em=f['em'],f1=f['f1'],path=f['path'],sha256=f['row_sha256'],sha256_scope='exact final identity ledger row',source_metrics=f['source_metrics'],dataset_sha256=f['validation_sha256'],run_id=f['run_id'],cleanup_error='trainer left owned subprocesses',identity_readback=ev('sigma-final-identity-readback.json'),canonical_dataset_readback=ev('sigma-canonical-data-readback.json'))
# Prefix variants from the two H100 source nodes: deduplicate inherited complete receipts.
for filename,session in [('h100-8-current.json','01a0fcbb-863f-7450-926d-db537f648265'),('h100-10-current.json','01a0fcbc-0b46-7502-8b66-d8b21a0e032d')]:
 x=read(filename)
 for q in x['runs']:
  pts={}
  for p in q['points']:
   if p['iter'] in pts:assert pts[p['iter']]['em']==p['em']
   assert p['n']==2560 and p['actor_adam']==[20*p['iter']]*8
   pts[p['iter']]=p
  n=max(pts) if pts else 0;assert sorted(pts)==list(range(1,n+1))
  fixed=[]
  for v in q['qa']:
   z=v['data'];i=int(z['actor_version'].rsplit('actor',1)[1]);assert z['qa_count']==512
   fixed.append(dict(iter=i,n=512,em=z['em'],f1=z['f1'],correct=round(z['em']*512),path=v['path'],sha256=v['sha256'],evaluation_contract=z['evaluation_contract'],question_source_sha256=z.get('question_source_sha256')))
  r=dict(name=q['name'],session=session,output_dir=q['roots'][-1] if q['roots'] else None,target_iter=200,additional_fixed512=[],final51713=dict(complete=False,n=0,em=None,f1=None,reason='Fresh Prefix variant has not completed200.'),training_metric='strict EM on actual selected512x5 actor trajectories')
  stop=q['name'] in ['Prefix20 frozen encoder every-outer','Prefix20 actor-hidden period20']
  update(r,[pts[i] for i in sorted(pts)],fixed,n,x['epoch'],filename,'Independent Prefix variant. Actual complete joint receipts only, own strict EM. Legacy async FP32 QA evaluation; inference contract differs from some historical settings. '+('Historical setting; not spliced to the LR1e-5 recovery branch.' if stop else 'Current same-run chain includes authorized probe-LR change; not appended to historical GiGPO or Tree-Nested.'),'historical_stopped' if stop else 'training_running')
  rpt['runs']=[z for z in rpt['runs'] if z['name']!=r['name']];rpt['runs'].append(r);by[r['name']]=r;added.append(r['name'])
# Periodic25 Q-probe: ancestor1–5 +6–10 + current11 onward, actual full-budget outers.
x=read('mou176-current.json');q=x['runs'][0];anc=read('topd-ancestors.json');n=max(p['iter'] for p in q['native_updates']);pts={}
for p in anc['train']+q['train']:
 if p['iter']>n:continue
 if p['iter'] in pts:assert pts[p['iter']]['em']==p['em']
 pts[p['iter']]=p
assert sorted(pts)==list(range(1,n+1));fixed={}
for p in anc['fixed']+q['fixed']:
 if p['n']==512 and p['iter'] in [5,50,100,150,200]:
  if p['iter'] in fixed:assert fixed[p['iter']]['em']==p['em']
  fixed[p['iter']]=dict(p,correct=round(p['em']*512))
r=dict(name='Periodic25 Q-probe 725286',session='01a0fcbe-15f8-7600-ae84-9487d924560a',output_dir=q['root'],target_iter=200,additional_fixed512=[],final51713=dict(complete=False,n=0,em=None,f1=None))
update(r,[pts[i] for i in sorted(pts)],[fixed[i] for i in sorted(fixed)],n,x['epoch'],'mou176-current.json','Periodic25 question probe setting in TOP-D session, not TOP-D distillation. Strict selected G5 trajectory EM, ancestors1–5/r13,6–10/r15 and11 onward/r16; partial next rollout excluded.')
rpt['runs']=[z for z in rpt['runs'] if z['name']!=r['name']];rpt['runs'].append(r);by[r['name']]=r;added.append(r['name'])
# Current Co Prefix20: completed head positive_fraction = actual strict label mean.
q=hpc['runs'][1];n=max(set(p['iter'] for p in q['native_updates']) & set(p['outer'] for p in q['probe_commits']))
pts=[p for p in q['train'] if p['iter']<=n];assert sorted(p['iter'] for p in pts)==list(range(1,n+1))
fixed=[p for p in q['fixed'] if p['n']==512 and p['iter'] in [5,50,100,150,200]]
r=dict(name=q['name'],session='01a0fcbe-65a4-7fb1-9cc8-8a74046bf8ef',output_dir=q['root'],target_iter=200,additional_fixed512=[],final51713=dict(complete=False,n=0,em=None,f1=None))
update(r,pts,fixed,n,hpc['epoch'],'jxu957-current.json','Fresh Prefix-only20 fourGPU; epoch68 warmup initialization; trajectoryBCE, all-correct virtual0. TrainingEM is before_positive_fraction of TRUE own strict labels, NOT probe binary_accuracy or softreward.')
rpt['runs']=[z for z in rpt['runs'] if z['name'] not in ['Prefix20 joint4 727414',q['name']]];rpt['runs'].append(r);by[r['name']]=r;added.append(r['name'])
# New A800 Prefix5 P10 is a separate setting.
x=read('g1-current.json')['result'];columns=x['history_columns'];pts=[]
for row in x['complete_outer_scalar_history']:
 z=dict(zip(columns,row));assert z['actor_adam_per_rank']==20*z['outer'] and 0<=z['selected_strict_correct']<=2560
 pts.append(dict(iter=z['outer'],em=z['selected_strict_correct']/2560,n=2560,correct=z['selected_strict_correct'],sha256=z['joint_sha256'],actual_em_selection=z['actual_em_selection']))
n=x['complete_outer'];assert len(pts)==n
fixed=[dict(iter=p['outer'],n=512,em=p['em'],f1=p['f1'],sha256=p['metadata_sha256'],question_source_sha256=p['question_source_sha256']) for p in x['qa_milestones'] if p.get('qa_count')==512 and p.get('em') is not None]
r=dict(name='Prefix5 actor-hidden P10 candidate2560',session='01a0fb1b-d3ff-74b3-aa13-49ca57871f92',target_iter=200,additional_fixed512=[],final51713=dict(complete=False,n=0,em=None,f1=None))
update(r,pts,fixed,n,x['observed_unix'],'g1-current.json','A800 fresh official actor; question+5 actorhidden; candidate2560 period10; refresh rounds actual EM mixed-first, ordinary predictedTop512; all-correct virtual0. Same-run recovery27->28; old model peaks not mixed.')
rpt['runs'].append(r);by[r['name']]=r;added.append(r['name'])
# New budget5 ablation, independent of C2 and Recursive A.
x=read('budget5-current.json');n=max(z['data']['completed_outer'] for z in x['receipts']);pts={};fixed={}
for log in x['logs']:
 for p in log['rows']:
  i=p['iter'];m=p['metrics']
  if i>n or m.get('training/actual_outer_updates')!=i:continue
  proof=dict(iter=i,path=log['path'],line=p['line'],sha256=p['sha256'])
  if 'critic/score/mean' in m:pts[i]=dict(proof,em=m['critic/score/mean'])
  if m.get('val/qa_count/all')==512:fixed[i]=dict(proof,n=512,em=m['val/em/all'],f1=m.get('val/f1/all'))
r=dict(name='Recursive budget5 ablation150',session='01a1028b-b626-7f91-93db-0099ff1623c5',target_iter=150,additional_fixed512=[],final51713=dict(complete=False,n=0,em=None,f1=None))
update(r,list(pts.values()),list(fixed.values()),n,x['epoch'],'budget5-current.json','Independent budget5 ablation, actual own complete outer and fixed512 scalar logs. Not attached to C2 or A.')
rpt['runs'].append(r);by[r['name']]=r;added.append(r['name'])
rpt.update(observed_report_at_cst=now.isoformat(),plot_stamp='Pinned-session refresh · '+now.strftime('%Y-%m-%d %H:%M')+' CST',viewer_subtitle='10月17日午间更新：递归B200、C2 150、budget5、Prefix20/1024、Periodic25与A800 Prefix5 P10；未完成评分不补零。',updated_names=updated,new_setting_names=added,training_agent_messages_sent=False,remote_jobs_changed=False,source_snapshot_files=[ev(n) for n in ['a-current.json','b-current.json','jxu957-current.json','mou176-current.json','h100-8-current.json','h100-10-current.json','topd-ancestors.json','g1-current.json','budget5-current.json']])
(D/'all-seven-results.json').write_text(json.dumps(rpt,ensure_ascii=False,indent=2)+'\n')
# Full seven-source table: one new scored final row, acceptance caveat kept explicit.
final=read('base-final-seven-table-data.json');ds=final['datasets'];f=by['HPC3 Q-probe sigma3 new loss 721485']['final51713'];src=f['source_metrics'];counts={k:src[k]['count'] for k in ds};correct={k:round(src[k]['em']*counts[k]) for k in ds}
assert sum(counts.values())==51713 and sum(correct.values())==f['correct'];assert abs(sum(correct.values())/51713-f['em'])<1e-12
row=dict(experiment='Q-probe σ3 · Co会话†',checkpoint_iter=200,dataset_em={k:src[k]['em'] for k in ds},dataset_correct=correct,source_counts=counts,count=51713,correct=f['correct'],micro_em=f['em'],macro_em=sum(src[k]['em'] for k in ds)/7,f1=f['f1'],kind='ours',evidence=ev('jxu957-current.json'),final_identity_current=f,final_acceptance_complete=False)
final['ours']=[r for r in final['ours'] if r['experiment']!=row['experiment']]+[row];final['ours'].sort(key=lambda r:-r['macro_em']);final['rows']=sorted(final['ours']+final['paper_references'],key=lambda r:r['macro_em'] if r['macro_em'] is not None else -1,reverse=True)
final.update(created_cst=now.isoformat(),pending_display=[dict(label='递归Tree B · '+str(by['Recursive B four GPU']['completed_iter'])+'/200',status='评测中')],plot_footer='† σ3完整评分已发布；包装清理报错，最终验收未完成。递归B训练200完成，最终评分仍在生成；其余终评保留原日期。',comparison_note='七集平均为等权macro。不同选题/检索/推理设置需分别比较；新Prefix/Periodic25尚无完整终评。')
final['pending']=copy.deepcopy(final['pending_display'])
final['checks']=[dict(experiment=r['experiment'],counts_sum=sum(r['source_counts'].values()),integer_correct_counts_verified=all(abs(r['dataset_em'][k]*r['source_counts'][k]-r['dataset_correct'][k])<1e-7 for k in ds),macro_recomputed_from_unrounded_dataset_em=abs(sum(r['dataset_em'].values())/7-r['macro_em'])<1e-12) for r in final['ours']]
assert all(c['counts_sum']==51713 and c['integer_correct_counts_verified'] and c['macro_recomputed_from_unrounded_dataset_em'] for c in final['checks'])
(D/'final-seven-table-data.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
def pct(x):return '—' if x is None else f'{x*100:.2f}%'
md=['51,713题终评；七集平均为等权macro。本次没有新增完整终评；递归B的200轮训练已完成，终评生成中。','','| 实验／checkpoint | '+' | '.join(final['dataset_labels'])+' | 七集平均 |','|---|'+'---:|'*8]
for r in final['rows']:
 md.append('| '+r['experiment']+(f"，iter{r['checkpoint_iter']}" if r.get('checkpoint_iter') is not None else '')+' | '+' | '.join(pct(r['dataset_em'][k]) for k in ds)+' | '+pct(r['macro_em'])+' |')
md+=['',final['plot_footer'],'',final['comparison_note']];(D/'final-seven-table.md').write_text('\n'.join(md)+'\n')
summary=dict(created_cst=now.isoformat(),updated_runs=[dict(name=r['name'],session=r['session'],complete=r['completed_iter'],train_em=r['train_latest_em'],recent10=r['train_recent10_mean'],before=before.get(r['name']),fixed512=[dict(iter=p['iter'],em=p['em'],correct=p['correct']) for p in r['fixed512'] if p.get('em') is not None],status=r['status'],observed_at_cst=r.get('observed_at_cst')) for r in rpt['runs'] if r['name'] in updated],new_final=None,new_final_completed=False,B_final_progress=read('b-current.json')['final_progress'],read_only=True,probe_classifier_not_QA=True)
(D/'updated-session-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False)[:6500])
