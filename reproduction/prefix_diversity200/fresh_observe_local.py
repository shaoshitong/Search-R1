"""Read actual fresh-run counters and small evidence; never controls training."""
import argparse
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

from reproduction.h1008_qprobe.management_transport import remote, connection_options
from .poll_local import SCRIPT
from .watch_actual import EXTRA
from .resource_sample import combine_resource_summaries


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def observe(root, output, connection):
    root, output = Path(root), Path(output)
    config_path = output / 'CURRENT_CONFIG.json'
    c = json.loads(config_path.read_text())
    registry_path=output/'QA_STANDALONE_REGISTRY.json'
    qa_registry=json.loads(registry_path.read_text()) if registry_path.exists() else []
    connection_info = json.loads(Path(connection).read_text())
    series_path = output / 'COMPLETED_ITERATION_SCALAR_EVIDENCE.json'
    series = json.loads(series_path.read_text()) if series_path.exists() else {}
    verified = max(map(int, series), default=0)
    assert sorted(map(int, series)) == list(range(1, verified + 1))
    extra = 'import hashlib\nif value.get("latest_joint"):\n'
    extra += '\n'.join(' ' + line for line in EXTRA.replace('VERIFIED', str(verified)).splitlines())
    extra += '\nelse:\n value["new_complete_facts"]=[]\n'
    extra += '\n' + inspect.getsource(combine_resource_summaries) + '\n'
    extra += r'''
value['config_sha256']=hashlib.sha256((base/'preparation/launch-config.json').read_bytes()).hexdigest()
resource_segments=[]
for segment in ('resource5s-v1','resource5s-v2'):
 p=base/'preparation'/segment/'summary.json'
 if p.exists():
  raw=p.read_bytes();summary=json.loads(raw)
  assert summary['base']==str(base) and summary['run_id']==value['latest_joint']['run_id'] and summary['job']==value['native']['JobId']
  summary['segment_source']=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  resource_segments.append(summary)
value['resource_summary']=combine_resource_summaries(resource_segments)
value['qa_completed']=[]
ancestors=[base];seen={str(base)};current_config=json.loads((base/'preparation/launch-config.json').read_text())
for _ in range(16):
 parent=current_config.get('resume_joint',{}).get('source_base')
 if not parent:break
 ancestor=pathlib.Path(parent)
 assert ancestor.parent==base.parent and ancestor.name.startswith('search-g1-prefix-diversity200-') and str(ancestor) not in seen
 current_config=json.loads((ancestor/'preparation/launch-config.json').read_text());assert current_config['run_id']==value['latest_joint']['run_id']
 ancestors.append(ancestor);seen.add(str(ancestor))
if value.get('qa_pending') is None:
 for ancestor in ancestors:
  pending_path=ancestor/'logs/qa-pending.json'
  if pending_path.exists():
   value['qa_pending']=json.loads(pending_path.read_text());break
for ordinal in (5,50,100,150,200):
 for ancestor in ancestors:
  p=ancestor/'outers'/('outer-%06d'%ordinal)/'validation/complete.json'
  if p.exists():
   qa=json.loads(p.read_text());assert qa['qa_count']==512 and qa['run_id']==value['latest_joint']['run_id']
   assert qa['actor_version']==qa['run_id']+'-actor%04d'%ordinal
   value['qa_completed'].append(dict(outer=ordinal,result=qa,source_base=str(ancestor),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));break
p=base/'final-evaluation/complete.json'
value['final_qa']=json.loads(p.read_text()) if p.exists() else None
'''
    extra += '\nstandalone_specs='+repr(qa_registry)+'\n'
    extra += r'''
for spec in standalone_specs:
 p=pathlib.Path(spec['canonical_retained18_base'])
 ordinal=spec['outer'];count=spec.get('expected_count',512)
 assert ordinal in (5,50,100,150,200) and count in (512,51713) and (count==512 or ordinal==200)
 assert p.parent==pathlib.Path('/mnt/data/shaoshitong') and p.name.startswith('search-g1-qa-i1i6-4a1cc2d5-outer%04d-'%ordinal)
 cfg=p/'preparation/config.json';receipt=p/'validation/complete.json'
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==spec['config_sha256']
 conf=json.loads(cfg.read_text());qa=json.loads(receipt.read_text())
 assert hashlib.sha256(receipt.read_bytes()).hexdigest()==spec['qa_receipt_sha256']
 assert conf['qa_source_commit']==spec['source_commit'] and conf['node']==spec['node']
 assert qa['run_id']==conf['run_id']==value['latest_joint']['run_id']
 assert conf['qa_standalone']['outer']==ordinal and conf['qa_standalone']['expected_count']==count
 assert qa['actor_version']==qa['run_id']+'-actor%04d'%ordinal and qa['qa_count']==count
 assert qa['question_source_sha256']==current_config['validation_questions_sha256' if count==512 else 'final_questions_sha256']
 assert qa['evaluation_contract']==spec['evaluation_contract']
 if count==51713:
  assert len(qa['per_source'])==7 and sum(r['qa_count']for r in qa['per_source'].values())==51713
  assert spec['native_unit_cleanup_verified'] and spec['independent_saved_raw_cpu_recompute']
  value['final_qa']=dict(qa,standalone_source_base=str(p),source_commit=spec['source_commit'])
 else:value['qa_completed'].append(dict(outer=spec['outer'],result=qa,source_base=str(p),node=spec['node'],
  receipt_sha256=spec['qa_receipt_sha256'],source_commit=spec['source_commit'],independent_saved_raw_cpu_recompute=True,
  native_unit_cleanup_verified=spec['native_unit_cleanup_verified']))
value['qa_completed']=sorted({r['outer']:r for r in value['qa_completed']}.values(),key=lambda r:r['outer'])
value['native_qa_pending']=value.get('qa_pending')
resolved={str(r['outer'])for r in value['qa_completed']}
value['qa_pending']={k:v for k,v in (value.get('qa_pending')or{}).items()if k not in resolved}
'''
    code = SCRIPT.replace('print(json.dumps(value,allow_nan=False))',
        extra + '\nprint(json.dumps(value,allow_nan=False))').replace('BASE', repr(c['base']))
    transport = output / ('read-' + str(time.time_ns()) + '-transport.json')
    value = json.loads(remote(code, transport, target=c['node'], **connection_options(connection_info)))
    assert value['base'] == c['base']
    assert value['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    evidence = transport.with_name(transport.name.replace('-transport', '-actual'))
    save(evidence, value)
    save(output / 'LATEST_ACTUAL.json', value)
    for fact in value['new_complete_facts']:
        assert len(fact['actor']) == len(fact['probe']) == 8
        assert all(r['writes'] == 20 for r in fact['actor'] + fact['probe'])
        series[str(fact['outer'])] = fact
    joint, stage, native = value.get('latest_joint') or {}, value.get('stage') or {}, value.get('native') or {}
    n = joint.get('outer', 0)
    assert sorted(map(int, series)) == list(range(1, n + 1))
    save(series_path, series)
    from .fresh_metrics_local import summarize
    summarize(root)
    state = json.loads((output / 'STATUS.json').read_text())
    state.update(base=c['base'],initial_official_fresh_run_source=c.get('initial_official_fresh_run_source',c['base']),
        stage=stage.get('stage'), outer_in_progress=stage.get('outer'),
        native_state=native.get('JobState', 'UNKNOWN'), job_id=native.get('JobId'),
        active_training_job=native.get('JobId'),training_job=native.get('JobId'),current_remote_state=native.get('JobState','UNKNOWN'),
        actual_complete_outer=n,
        actor_adam_per_rank=(value.get('actor') or {}).get('actual_adam_steps_per_rank', [0]*8),
        probe_adam_per_rank=(value.get('probe') or {}).get('actual_adam_steps_per_rank', [0]*8),
        latest_actual_observed_unix=value['observed_unix'], latest_actual_evidence=str(evidence),current_status_evidence=str(evidence),
        config_sha256=value['config_sha256'], source_commit=c['source_commit'],
        latest_joint_source_commit=joint.get('source_commit'),resume_joint=c.get('resume_joint'),
        qa_pending=value.get('qa_pending'),native_qa_pending=value.get('native_qa_pending'),
        final_qa=value.get('final_qa'),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), goal_complete=False)
    if n:
        state['latest_actor_selection'] = joint['actor_selection']
        state['latest_probe_metrics'] = joint['probe_metrics']
        state['last_complete_phase_seconds'] = sum(x['elapsed_seconds'] for x in joint['phase_timings'].values())
    if value['qa_completed']:
        state['latest_qa'] = value['qa_completed'][-1]
    if value['resource_summary']:
        save(output / 'RESOURCE5S_SUMMARY_LATEST.json', value['resource_summary'])
    save(output / 'STATUS.json', state)
    save(root / 'outputs/STATUS.json', state)
    lines = [
        'Search-G1 I1-I6 fresh200 actual state.', '',
        'Job%s %s on %s; complete %d/200; current stage %s.' %
        (state['job_id'], state['native_state'], c['node'], n, state['stage']),
        'Saved actor/probe Adam per rank: %s / %s.' %
        (state['actor_adam_per_rank'], state['probe_adam_per_rank']),
        'Latest QA: ' + json.dumps({k:v for k,v in state.get('latest_qa',{}).items()if k!='result'} | {'result':{k:v for k,v in state.get('latest_qa',{}).get('result',{}).items()if k in ('actor_version','qa_count','em','f1','per_source','question_source_sha256','completed_utc')}}),
        'Unresolved milestone QA: '+json.dumps(state.get('qa_pending') or {}),
        'Final51713 QA: '+json.dumps({k:v for k,v in (state.get('final_qa')or{}).items()if k in ('actor_version','qa_count','em','f1','per_source','question_source_sha256','completed_utc','standalone_source_base','source_commit')}),
        'Frozen trainer pending ledger (standalone accepted results are separate): '+json.dumps(state.get('native_qa_pending') or {}),
        'Checkpoint5 CPU acceptance: '+json.dumps(state.get('checkpoint5_cpu_acceptance','not yet available')),
        'Frozen source: ' + c['source_commit'] + '; actual config SHA: ' + value['config_sha256'],
        '5120G5 prefix20 -> Top1024 same-prefix continuation -> actual EM variance Top512 actor; probe ALL1024/5120 one natural epoch20Adam atLR1e-6.',
        'Shared BF16 train/QA interaction4096/action500/reserve500, early final and protected observation wrapper; strict scorer unchanged. Original logical two-row objective and fixed before/after monitor restored; finite thresholds diagnostic-only. Actual reward-grounding refits50/100/150.',
        'This logical run originated from official actor0 and best retained warmup probe model-only/online Adam0. No repeated warmup or initial QA. First actor outer uses scheduledLR0; nonzero parameter change requires a later positive-LR update.',
        'Current continuation boundary: '+json.dumps(c.get('resume_joint','original fresh launch; no continuation')),
        'Old3427 ended at131 before the protocol repair and was explicitly invalidated by latest human authorization. Its131 are historical and do not count in this fresh run. Old owned wrong outputs were removed; protected shared inputs remain.',
        'Whole-card resource peaks cover the actual five-second sample window only; observed owned CPU is an incomplete lower bound. All large assets stay remote on retained18, exported as same-cluster NFS.200/full51713/checkpoint validation/cost and cleanup remain unfinished.',
        'Evidence: ' + str(evidence),
    ]
    if n:
        lines += ['Latest selection: ' + json.dumps(joint['actor_selection']),
                  'Last complete normal phase seconds: ' + str(state['last_complete_phase_seconds'])]
    report = '\n\n'.join(lines) + '\n'
    (output / 'REPORT.md').write_text(report)
    (root / 'outputs/REPORT.md').write_text(report)
    for target in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/output.name,
                   Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/output.name):
        target.mkdir(parents=True, exist_ok=True)
        for name in ('STATUS.json', 'REPORT.md', 'COMPLETED_ITERATION_SCALAR_EVIDENCE.json', 'RESOURCE5S_SUMMARY_LATEST.json'):
            if (output/name).exists(): shutil.copyfile(output/name, target/name)
    return dict(evidence=str(evidence), native=state['native_state'], job=state['job_id'],
        complete_outer=n, stage=state['stage'], outer_in_progress=stage.get('outer'),
        actor_adam=state['actor_adam_per_rank'], probe_adam=state['probe_adam_per_rank'],
        actual_rank_update_ledgers=value.get('actual_rank_update_ledgers'),
        progress=value.get('current_prefix'), log_tails=value['log_tails'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--connection', required=True)
    a = p.parse_args()
    print(json.dumps(observe(a.root, a.output, a.connection), allow_nan=False))
