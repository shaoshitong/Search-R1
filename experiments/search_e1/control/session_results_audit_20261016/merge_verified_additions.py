"""Merge two read-only session audits into the previous checked figure data.

Only scalar summaries live locally. Prior settings keep their source dates.
"""
from pathlib import Path
import copy, datetime, hashlib, json

D = Path(__file__).resolve().parent
BASE = D.parent / 'session-refresh-20261015-current'
DATASETS = ['nq','triviaqa','popqa','hotpotqa','2wikimultihopqa','musique','bamboogle']
COUNTS = dict(zip(DATASETS,[3610,11313,14267,7405,12576,2417,125]))
def read(p):return json.loads(p.read_text())
def evidence(p):return {'local_audit':str(p),'local_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
base = read(D/'base-all-seven-results.json')
report = copy.deepcopy(base)
updates = []
sources = []
for name in ['b-updated-runs.json','co-updated-runs.json']:
    p = D/name
    doc = read(p)
    sources.append(evidence(p))
    updates.extend(doc['runs'])
by_name = {r['name']:i for i,r in enumerate(report['runs'])}
updated_names = set()
for r in updates:
    name = r['name']
    assert name in by_name, 'Unknown setting requires an explicit new plot specification: '+name
    old = report['runs'][by_name[name]]
    assert r['session'] == old['session']
    assert r['completed_iter'] >= old['completed_iter']
    points = r.get('train_series',r.get('training_series',[]))
    steps = [int(v.get('iter',v.get('iteration'))) for v in points]
    assert len(steps) == len(set(steps)), 'Duplicate completed outer in '+name
    assert all(1 <= i <= r['completed_iter'] for i in steps)
    assert all(0 <= v['em'] <= 1 for v in points)
    for e in r['fixed512']:
        if e.get('em') is not None:
            assert e['n'] == 512
            assert abs(e['em']*512-round(e['em']*512)) < 1e-7
    r['audit_refreshed_20261016'] = True
    r['audit_file'] = str(D/('b-updated-runs.json' if name.startswith('Recursive B') else 'co-updated-runs.json'))
    report['runs'][by_name[name]] = r
    updated_names.add(name)
for r in base['runs']:
    if r['name'] not in updated_names:
        assert report['runs'][by_name[r['name']]] == r
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
report['observed_report_at_cst'] = now.isoformat()
report['plot_stamp'] = 'B / Co update · 2026-10-16 '+now.strftime('%H:%M')+' CST'
report['viewer_subtitle'] = 'B与CoSearch于10月16日更新；其余setting沿用10月15日审计，BAPO/OPSD/E1历史线沿用10月14日。训练rollout、固定512验证和51,713题终评分开。'
report['base_audit'] = evidence(D/'base-all-seven-results.json')
report['updated_names'] = sorted(updated_names)
report['addition_sources'] = sources
report['training_agent_messages_sent'] = False
report['remote_jobs_changed'] = False
(D/'all-seven-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

# Keep the already validated final rows; add only complete, seven-source scores.
final = read(D/'base-final-seven-table-data.json')
final['created_cst'] = now.isoformat()
final['observed_window_cst'] = 'B / CoSearch: '+report['observed_report_at_cst']+'; other rows retain prior source dates'
added_finals = []
for r in updates:
    f = r.get('final51713',{})
    if not f.get('complete'):continue
    assert f['n'] == 51713
    if r['name'].startswith('CoSearch common'):
        label = 'CoSearch 双模型'
    elif r['name'].startswith('Recursive B'):
        label = '递归Tree B'
    elif r['name'].startswith('HPC3 Q-probe sigma3'):
        label = 'Q-probe σ3（CoSearch会话）'
    else:continue
    metrics = f['source_metrics']
    assert set(metrics) == set(DATASETS)
    assert {k:int(metrics[k]['count']) for k in DATASETS} == COUNTS
    em = {k:float(metrics[k]['em']) for k in DATASETS}
    correct = {k:round(em[k]*COUNTS[k]) for k in DATASETS}
    assert all(abs(em[k]*COUNTS[k]-correct[k]) < 1e-7 for k in DATASETS)
    assert abs(sum(correct.values())/51713-f['em']) < 1e-10
    row = {'experiment':label,'checkpoint_iter':200,'dataset_em':em,
           'dataset_correct':correct,'correct':sum(correct.values()),
           'source_counts':COUNTS,'count':51713,'micro_em':f['em'],
           'macro_em':sum(em.values())/7,'f1':f.get('f1'),'kind':'ours',
           'scope':'full_test_dataset_breakdown_rechecked_2026-10-16',
           'evidence':{'path':f.get('path'),'sha256':f.get('sha256')},
           'sources':[evidence(Path(r['audit_file']))],'final_identity_current':f}
    final['ours'] = [x for x in final['ours'] if x['experiment'] != label]+[row]
    added_finals.append(label)
final['ours'].sort(key=lambda r:-r['macro_em'])
final['rows'] = sorted(final['ours']+final['paper_references'],key=lambda r:r['macro_em'] if r['macro_em'] is not None else -1,reverse=True)
final['pending_display'] = []
co = next(r for r in report['runs'] if r['name'].startswith('CoSearch common'))
if not co['final51713'].get('complete'):
    final['pending_display'].append({'label':'CoSearch双模型 · iter200','status':'评测中'})
    co_note = 'CoSearch完整终评尚未发布。'
else:
    co_note = 'CoSearch双模型完整终评已加入。'
for r in updates:
    if r['name'].startswith('Recursive B') and not r.get('final51713',{}).get('complete'):
        final['pending_display'].append({'label':f"递归Tree B · {r['completed_iter']}/200",'status':'待终评'})
    if r['name'].startswith('HPC3 Q-probe sigma3') and not r.get('final51713',{}).get('complete'):
        final['pending_display'].append({'label':f"Q-probe σ3 · Co会话 · {r['completed_iter']}/200",'status':'待终评'})
final['plot_footer'] = '本次更新B与CoSearch；其余终评保留原核查日期。BAPO/OPSD为10月14日摘要。'+co_note
final['comparison_note'] = 'CoSearch双模型：检索top50后重排取5；检索计算预算与其他setting不同，排名不等于同预算算法优劣。'
final['pending'] = [x for x in final.get('pending',[]) if not (co['final51713'].get('complete') and x['experiment'].startswith('CoSearch'))]
for r in final['ours']:
    assert r['source_counts'] == COUNTS
    assert abs(sum(r['dataset_em'].values())/7-r['macro_em']) < 1e-14
    assert abs(sum(r['dataset_correct'].values())/51713-r['micro_em']) < 1e-12
(D/'final-seven-table-data.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
def pct(v):return '—' if v is None else f'{v*100:.2f}%'
lines = ['完整51,713题终评；七集平均为等权macro。B/CoSearch于10月16日更新，其他行保留原核查日期。','',
         '| 实验／checkpoint | '+' | '.join(final['dataset_labels'])+' | 七集平均 |',
         '|---|'+'---:|'*8]
for r in final['rows']:
    name = r['experiment']+(f"，iter{r['checkpoint_iter']}" if r.get('checkpoint_iter') is not None else '')
    lines.append('| '+name+' | '+' | '.join(pct(r['dataset_em'][k]) for k in DATASETS)+' | **'+(pct(r['macro_em']) if r['macro_em'] is not None else '未报告')+'** |')
for p in final['pending_display']:
    lines.append('| '+p['label']+' | '+' | '.join(['—']*7)+' | '+p['status']+' |')
lines += ['',final['plot_footer'],'',final['comparison_note'],'','论文行沿用用户提供的参考数值，本次未重新核对原文。B尚未完成200轮及51,713题终评。']
(D/'final-seven-table.md').write_text('\n'.join(lines)+'\n')
receipt = {'updated_names':sorted(updated_names),'new_final_rows':added_finals,
           'unchanged_settings_preserved':True,'source_files':sources,
           'created_cst':now.isoformat(),'read_only':True}
(D/'merge-verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
