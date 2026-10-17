"""Read actual fresh-run counters and small evidence; never controls training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

from reproduction.h1008_qprobe.management_transport import remote
from .poll_local import SCRIPT
from .watch_actual import EXTRA


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def observe(root, output, connection):
    root, output = Path(root), Path(output)
    config_path = output / 'CURRENT_CONFIG.json'
    c = json.loads(config_path.read_text())
    cp = json.loads(Path(connection).read_text())['control_path']
    series_path = output / 'COMPLETED_ITERATION_SCALAR_EVIDENCE.json'
    series = json.loads(series_path.read_text()) if series_path.exists() else {}
    verified = max(map(int, series), default=0)
    assert sorted(map(int, series)) == list(range(1, verified + 1))
    extra = 'import hashlib\nif value.get("latest_joint"):\n'
    extra += '\n'.join(' ' + line for line in EXTRA.replace('VERIFIED', str(verified)).splitlines())
    extra += '\nelse:\n value["new_complete_facts"]=[]\n'
    extra += r'''
value['config_sha256']=hashlib.sha256((base/'preparation/launch-config.json').read_bytes()).hexdigest()
p=base/'preparation/resource5s-v1/summary.json'
value['resource_summary']=json.loads(p.read_text()) if p.exists() else None
value['qa_completed']=[]
for ordinal in (5,50,100,150,200):
 p=base/'outers'/('outer-%06d'%ordinal)/'validation/complete.json'
 if p.exists():
  qa=json.loads(p.read_text());assert qa['qa_count']==512
  value['qa_completed'].append(dict(outer=ordinal,result=qa))
p=base/'final-evaluation/complete.json'
value['final_qa']=json.loads(p.read_text()) if p.exists() else None
'''
    code = SCRIPT.replace('print(json.dumps(value,allow_nan=False))',
        extra + '\nprint(json.dumps(value,allow_nan=False))').replace('BASE', repr(c['base']))
    transport = output / ('read-' + str(time.time_ns()) + '-transport.json')
    value = json.loads(remote(code, transport, control_path=cp, target=c['node']))
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
    state = json.loads((output / 'STATUS.json').read_text())
    state.update(stage=stage.get('stage'), outer_in_progress=stage.get('outer'),
        native_state=native.get('JobState', 'UNKNOWN'), job_id=native.get('JobId'),
        active_training_job=native.get('JobId'), actual_complete_outer=n,
        actor_adam_per_rank=(value.get('actor') or {}).get('actual_adam_steps_per_rank', [0]*8),
        probe_adam_per_rank=(value.get('probe') or {}).get('actual_adam_steps_per_rank', [0]*8),
        latest_actual_observed_unix=value['observed_unix'], latest_actual_evidence=str(evidence),
        config_sha256=value['config_sha256'], source_commit=c['source_commit'],
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
        'Latest QA: ' + json.dumps(state.get('latest_qa', 'not yet produced under the new protocol')),
        'Frozen source: ' + c['source_commit'] + '; actual config SHA: ' + value['config_sha256'],
        '5120G5 prefix20 -> Top1024 same-prefix continuation -> actual EM variance Top512 actor; probe ALL1024/5120 one natural epoch20Adam atLR1e-6.',
        'Shared BF16 train/QA interaction4096/action500/reserve500, early final and protected observation wrapper; strict scorer unchanged. Original logical two-row objective and fixed before/after monitor restored; finite thresholds diagnostic-only. Actual reward-grounding refits50/100/150.',
        'Official actor fresh0; best retained warmup probe model-only with online Adam0. No repeated warmup or initial QA. First actor outer uses scheduledLR0; nonzero parameter change requires a later positive-LR update.',
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
