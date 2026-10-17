"""Derive current progress from existing local JSON; never observes or controls a remote job."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

RUN_ID = '4a1cc2d500fe40979ef8ca4792959438'
RUNTIME = '33ef232e605f7864086d1685d9a4d8ea0d67f4ab'
NAME = 'search-g1-i1i6-fresh200-20261017'
COMPONENTS = ('trajectory_bce', 'group_bce', 'total_loss')


def need(condition, message):
    if not condition: raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def contiguous(outers):
    n = 0
    while n + 1 in outers: n += 1
    return n or None


def phase_exits_zero(fact):
    timings = fact.get('phase_timings') or {}
    codes = [phase.get('exit_code') for phase in timings.values()]
    for code in codes:
        need(code is None or type(code) is int, 'Phase exit code is invalid')
    return all(code == 0 for code in codes) if codes and None not in codes else None


class Inputs:
    def __init__(self):
        self.files, self.missing = {}, set()

    def read(self, path, required=False):
        path = Path(path).absolute()
        need(path != Path('/mnt') and not str(path).startswith('/mnt/'), 'Only local files may be read')
        for cursor in reversed((path, *path.parents)):
            need(not cursor.is_symlink(), 'Local evidence paths must not redirect through symlinks')
        if not path.exists():
            need(not required, 'Missing required local input: '+str(path))
            self.missing.add(path); return None
        need(path.is_file() and path.stat().st_size <= 64*1024**2, 'Input is not a small JSON evidence file')
        before = path.stat(); raw = path.read_bytes(); after = path.stat()
        signature = lambda s:(s.st_ino,s.st_size,s.st_mtime_ns)
        need(signature(before) == signature(after), 'Input changed during local read')
        self.files[path] = dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)

    def stable(self):
        for path, proof in self.files.items():
            need(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == proof['sha256'],
                 'Input changed while summarizing; rerun after collectors finish')
        need(all(not p.exists() for p in self.missing), 'An input appeared while summarizing')


def validate_facts(facts, n, joint):
    verified = {'actor': {}, 'probe': {}}
    for key, fact in facts.items():
        outer = int(key)
        need(str(outer) == key and 1 <= outer <= (n if n is not None else 200)
             and fact['outer'] == outer and sha(fact['joint_sha256']), 'Fact outer/SHA is outside the current joint boundary')
        selection = fact['actor_selection']
        need(selection['run_id'] == RUN_ID and selection['outer'] == outer, 'Fact selection belongs to another run/outer')
        version = RUN_ID + ('-official0' if outer == 1 else '-actor%04d' % (outer-1))
        need(selection['actor_version'] == version, 'Fact sampling actor identity differs')
        metrics = fact.get('probe_metrics')
        if metrics:
            need(metrics['run_id'] == RUN_ID and metrics['outer'] == outer
                 and metrics['actor_version'] == version, 'Probe metrics identity differs')
        phase_exits_zero(fact)  # Timings retain failed attempts before a successful joint retry.
        for role in verified:
            rows = fact.get(role)
            if rows is None: continue
            need(len(rows) == 8 and {r['rank'] for r in rows} == set(range(8)), 'Direct facts must contain eight unique ranks')
            for row in rows:
                last = row['last']
                need(row['writes'] == 20 and last['rank'] == row['rank'] and last['outer'] == outer
                     and last['actual_adam_steps'] == 20*outer and sha(row['sha256']), 'Direct rank write counter/identity differs')
                if role == 'actor':
                    restore = row['restore']
                    need(last['event'] == 'ACTUAL_ADAM_WRITE' and last['writes_this_attempt'] == 20
                         and restore['actor_version'] == version and restore['actual_adam_before'] == 20*(outer-1)
                         and restore['outer'] == outer and restore['rank'] == row['rank'],
                         'Actor direct restore/write boundary differs')
                else:
                    need(last['step'] == 20 and row['first']['step'] == 1
                         and row['first']['actual_adam_steps'] == 20*(outer-1)+1
                         and row['first']['outer'] == outer and row['first']['rank'] == row['rank'], 'Probe direct write boundary differs')
            verified[role][outer] = rows
    if n is not None and str(n) in facts:
        need(facts[str(n)]['joint_sha256'] == joint['sha256'], 'Latest fact does not bind the latest actual joint SHA')
    return verified


def paired_summary(data, facts, n):
    rows = {}; seen = set(); unbound = {}
    for row in (data or {}).get('rows', []):
        outer = row['outer']
        need(type(outer) is int and 1 <= outer <= (n if n is not None else 200) and outer not in seen,
             'Paired loss is duplicated or outside the joint boundary')
        seen.add(outer)
        need(row['run_id'] == RUN_ID, 'Paired loss belongs to another run')
        if n is None or str(outer) not in facts: continue
        fact = facts[str(outer)]; metrics = fact.get('probe_metrics') or {}
        need(row['joint_sha256'] == fact['joint_sha256']
             and row['actor_version'] == fact['actor_selection']['actor_version']
             and row['questions'] == 1024 and row['trajectories'] == 5120
             and row['same_batch_comparison'] is True and row['actual_probe_updates_per_rank'] == 20,
             'Paired loss does not bind the same committed1024G5')
        bindings = {'after_probe_checkpoint_sha256':metrics.get('checkpoint', {}).get('sha256'),
                    'selection_file_sha256':fact['actor_selection'].get('prefix_selection_sha256'),
                    'own_em_sha256':metrics.get('own_em_sha256')}
        missing = [key for key,expected in bindings.items() if expected is None]
        for row_key, expected in bindings.items():
            if expected is not None:
                need(sha(expected) and row.get(row_key) == expected, 'Paired loss checkpoint/data SHA differs')
        if missing:
            unbound[outer] = missing; continue
        for key in COMPONENTS:
            before, after, delta = row['before'][key], row['after'][key], row['delta_after_minus_before'][key]
            need(all(finite(v) for v in (before, after, delta))
                 and math.isclose(after-before, delta, rel_tol=1e-12, abs_tol=1e-12), 'Paired loss delta is not finite/actual')
        for side in ('before','after'):
            need(math.isclose(row[side]['total_loss'], row[side]['trajectory_bce']+row[side]['group_bce'],
                              rel_tol=1e-12, abs_tol=1e-12), 'Paired total loss components differ')
        rows[outer] = row
    pending = [i for i in range(1, (n or 0)+1) if i not in rows]
    directions = {key:dict(improved=sum(r['delta_after_minus_before'][key] < 0 for r in rows.values()) if rows else None,
        worsened=sum(r['delta_after_minus_before'][key] > 0 for r in rows.values()) if rows else None,
        unchanged=sum(r['delta_after_minus_before'][key] == 0 for r in rows.values()) if rows else None,
        mean_delta=math.fsum(r['delta_after_minus_before'][key] for r in rows.values())/len(rows) if rows else None)
        for key in COMPONENTS}
    return dict(status='covered' if n is not None and not pending else 'pending', joint_boundary=n,
        through_outer=contiguous(rows), covered_outers=sorted(rows), count=len(rows), pending_outers=pending,
        boundary_pending=n is None, missing_binding_sha_by_outer=unbound,
        component_directions=directions, latest=rows[max(rows)] if rows else None,
        scope='Same-batch1024G5 components only; changing outer batches and QA/calibration benefit are separate.')


def cost_summary(coverage, data, facts, n):
    result = {}
    for role, metric, declared in (('generation','combined_actual_sampled_tokens','sampled_policy_tokens_committed'),
                                   ('retrieval','total_distinct_logical_retrieval_calls','logical_retrieval_calls_committed')):
        source = (data or {}).get(role); rows = {}; total = None
        if source and 'error' not in source:
            need(source['run_id'] == RUN_ID, 'Cost evidence belongs to another run')
            for key, row in source.get('complete_outer_rows', {}).items():
                outer = int(key)
                need(str(outer) == key and row['outer'] == outer and 1 <= outer <= (n if n is not None else 200),
                     'Cost evidence is outside the actual joint boundary')
                if 'run_id' in row: need(row['run_id'] == RUN_ID, 'Cost row belongs to another run')
                if n is None or key not in facts: continue
                hashes = [value for path,value in row.get('receipt_sha256', {}).items()
                          if path.endswith('/outers/outer-%06d/complete.json' % outer)]
                need(hashes == [facts[key]['joint_sha256']], 'Cost row joint identity differs')
                need(type(row[metric]) is int and row[metric] >= 0, 'Cost count must be an actual nonnegative integer')
                rows[outer] = row
            if rows: total = sum(row[metric] for row in rows.values())
            if rows and coverage and declared in coverage and len(rows) == len(source.get('complete_outer_rows', {})):
                need(coverage[declared] == total, 'Cost coverage total differs from bound original rows')
        pending = [i for i in range(1, (n or 0)+1) if i not in rows]
        result[role] = dict(status='covered' if n is not None and not pending else 'pending',
            through_outer=contiguous(rows), covered_outers=sorted(rows), pending_outers=pending, total=total,
            boundary_pending=n is None, error=source.get('error') if source else None)
    result['scope'] = 'Committed generation tokens/logical retrieval only; reservation, retries, QA and unknown CPU remain separate.'
    return result


def trend_summary(facts, n):
    rows = {}
    for key, fact in facts.items():
        if n is None: continue
        outer = int(key); selection = fact['actor_selection']; row = {'outer':outer}
        for role, count in (('continued1024',1024),('actor512',512)):
            item = selection.get(role)
            if item is None: row[role+'_mixed_mean'] = row[role+'_prediction_mean'] = None; continue
            hist = item['correct_count_histogram']
            need(set(hist) == set(map(str,range(6))) and all(type(v) is int and v >= 0 for v in hist.values())
                 and sum(hist.values()) == item['questions'] == count, 'Selection histogram/question count differs')
            mixed = sum(hist[str(k)] for k in range(1,5))
            need(item['mixed_questions'] == mixed and item['mixed_fraction'] == mixed/count, 'Selection mixed fraction is inconsistent')
            prediction = item.get('mean_predicted_mixed_probability')
            need(prediction is None or finite(prediction) and 0 <= prediction <= 1, 'Selection prediction is not a finite probability')
            row[role+'_mixed_mean'], row[role+'_prediction_mean'] = mixed/count, prediction
        if selection.get('actor512') and selection.get('continued1024'):
            need(selection['actor512']['mixed_questions'] == min(512,selection['continued1024']['mixed_questions']),
                 'Actor subset mixed count is inconsistent with strict-EM Top512')
        value = (fact.get('probe_metrics') or {}).get('test', {}).get('balanced_accuracy')
        need(value is None or finite(value) and 0 <= value <= 1, 'Fixed512 balanced accuracy is invalid')
        row['fixed512_post_update_balanced_accuracy_mean'] = value; rows[outer] = row
    fields = ('continued1024_mixed_mean','actor512_mixed_mean','actor512_prediction_mean',
              'fixed512_post_update_balanced_accuracy_mean')
    def group(first, last):
        wanted = list(range(first,last+1)); available = [rows[i] for i in wanted if i in rows]
        values = {field:math.fsum(r[field] for r in available)/len(wanted)
                  if len(available) == len(wanted) and all(r[field] is not None for r in available) else None for field in fields}
        return dict(first_outer=first,last_outer=last,outer_count=len(available),expected_outer_count=len(wanted),
            pending_outers=[i for i in wanted if i not in rows],metrics_pending=any(v is None for v in values.values()),**values)
    groups = [group(i,min(i+4,n)) for i in range(1,(n or 0)+1,5)]
    return dict(joint_boundary=n,through_outer=contiguous(rows),five_outer_means=groups,
        recent5=group(max(1,n-4),n) if n else None, latest=({
            'outer':n,'selected512_mixed_fraction':rows[n]['actor512_mixed_mean'],
            'selected512_mean_prediction':rows[n]['actor512_prediction_mean'],
            'continued1024_mixed_fraction':rows[n]['continued1024_mixed_mean'],
            'post_update_fixed512_probe_balanced_accuracy':rows[n]['fixed512_post_update_balanced_accuracy_mean']} if n in rows else None),
        pending_outers=[i for i in range(1,(n or 0)+1) if i not in rows],boundary_pending=n is None,
        scope='Descriptive changing-question scalar trends; a partial final five-round bin is distinct from the latest five actual outers. Not ECE, raw-data recomputation or QA benefit; nonmixed strict-EM does not imply zero soft-reward advantage.')


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def summarize(root, classification, note='', *, mirror_roots=None):
    need(classification in ('progress','verified_wait') and len(note) <= 1000, 'Classification or short note is invalid')
    root = Path(root).resolve(); need(root != Path('/mnt') and not str(root).startswith('/mnt/'), 'Summary root must be local')
    out = root/'outputs'/NAME; inputs = Inputs()
    load = lambda name, required=False:inputs.read(out/name, required)
    config = load('CURRENT_CONFIG.json', True); actual = load('LATEST_ACTUAL.json', True)
    status = load('STATUS.json')
    need(config['run_id'] == RUN_ID and config['source_commit'] == RUNTIME, 'Current runtime identity differs')
    config_sha = inputs.files[out/'CURRENT_CONFIG.json']['sha256']
    need(actual['base'] == config['base'] and actual['config_sha256'] == config_sha, 'Actual/config identity differs')
    if status:
        need(status['run_id'] == RUN_ID and status['source_commit'] == RUNTIME
             and status.get('base',config['base']) == config['base']
             and status.get('config_sha256',config_sha) == config_sha, 'Status identity differs')
    joint = actual.get('latest_joint'); n = joint.get('outer') if joint else None
    need(n is None or type(n) is int and 1 <= n <= 200, 'Latest joint boundary is invalid')
    clocks = {}
    if joint:
        need(joint['run_id'] == RUN_ID and joint['source_commit'] == RUNTIME and sha(joint['sha256']), 'Latest joint identity differs')
        for role in ('actor','probe'):
            clocks[role] = joint.get(role+'_adam')
            need(clocks[role] is None or clocks[role] == [20*n]*8, 'Joint actual Adam clock differs')
    live = {}
    for role in ('actor','probe'):
        receipt = actual.get(role)
        if receipt:
            need(receipt['run_id'] == RUN_ID and type(receipt['completed_outer']) is int
                 and 0 <= receipt['completed_outer'] <= 200 and (n is None or n <= receipt['completed_outer'] <= n+1)
                 and receipt.get('source_commit',RUNTIME) == RUNTIME
                 and receipt['actual_adam_steps_per_rank'] == [20*receipt['completed_outer']]*8, 'Live role identity/clock differs')
        live[role] = dict(saved_outer=receipt.get('completed_outer') if receipt else None,
            saved_adam_per_rank=receipt.get('actual_adam_steps_per_rank') if receipt else None,
            status_adam_per_rank=(status or {}).get(role+'_adam_per_rank'), pending=receipt is None)
    facts = load('COMPLETED_ITERATION_SCALAR_EVIDENCE.json') or {}
    verified = validate_facts(facts,n,joint)
    pairs = paired_summary(load('PROBE_SAME_BATCH_LOSS_PAIRS.json'),facts,n)
    coverage = load('COST_COVERAGE_CURRENT.json'); costs = cost_summary(coverage,load('COST_CURRENT_PROGRESS.json'),facts,n)
    trend = trend_summary(facts,n)
    resources = {}
    for name in ('RESOURCE_COST_CURRENT.json','RESOURCE5S_SUMMARY_LATEST.json'):
        data = load(name)
        if data:
            need(data['run_id'] == RUN_ID, 'Resource evidence belongs to another run')
            resources[name] = dict(source=inputs.files[out/name],actual_complete_outer=data.get('actual_complete_outer'),
                snapshot_created_unix=data.get('created_unix'),last_observed_unix=data.get('last_observed_unix'),
                joint_boundary=n,boundary_matches=data.get('actual_complete_outer') == n if data.get('actual_complete_outer') is not None else None,
                values={k:data[k] for k in ('total_native_gpu_reservation_hours','total_native_cpu_reserved_capacity_hours',
                    'observed_native_owned_cpu_core_seconds_lower_bound','cpu_accounting_complete','scope','coverage') if k in data})
        else: resources[name] = dict(status='pending')
    optimization = dict(run_id=RUN_ID,complete_outer=(min((contiguous(v) or 0) for v in verified.values()) or None) if n is not None else None,
        joint_boundary=n,roles={},resource_snapshots=resources,
        scope='Verified successful committed rank ledgers only. Eight rank calls are one synchronized update; missing evidence is pending, not extrapolated.')
    for role, rows in verified.items():
        known = sorted(rows) if n is not None else []
        optimization['roles'][role] = dict(actual_adam_per_rank=[20*len(known)]*8 if known else None,
            synchronized_adam_updates=20*len(known) if known else None,total_distributed_rank_adam_calls=160*len(known) if known else None,
            direct_rank_ledgers=8*len(known) if known else None,covered_outers=known,through_outer=contiguous(known),
            pending_outers=[i for i in range(1,(n or 0)+1) if i not in known],boundary_pending=n is None,
            scope='Counts cover only covered_outers, not the latest saved-state clock.')
    stage = actual.get('stage') or {}
    if stage.get('run_id') is not None: need(stage['run_id'] == RUN_ID, 'Live stage belongs to another run')
    need(actual.get('observed_unix') is None or finite(actual['observed_unix']), 'Actual observation timestamp is invalid')
    summary = dict(schema_version=1,run_id=RUN_ID,runtime_source=RUNTIME,config_sha256=config_sha,
        observed_unix=actual.get('observed_unix'),summarized_unix=time.time(),classification=classification,note=note,
        actual_complete_outer=n,joint_sha256=joint['sha256'] if joint else None,joint_adam_per_rank=clocks,
        in_flight=dict(stage=stage,roles=live,rank_write_evidence=actual.get('actual_rank_update_ledgers')),
        active_job=(actual.get('native') or {}).get('JobId'),cost_coverage=costs,
        paired_loss_coverage={k:pairs[k] for k in ('status','through_outer','pending_outers','count')},
        direct_fact_coverage={role:optimization['roles'][role]['pending_outers'] for role in verified},
        resource_snapshots=resources,latest_five=trend['recent5'],
        all_latest_phase_exits_zero=phase_exits_zero(facts.get(str(n)) or {}),
        all8_actor_probe20_actual_writes=(True if n is not None and all(n in v for v in verified.values()) else None),
        full_goal_completion_assessed=False,scope='Local derived summary only; raw observations, goal state, final200/51713 acceptance and cleanup are not adjudicated.')
    goal = load('GOAL_TURN_PROGRESS_CURRENT.json') or {}; requirement = load('REQUIREMENT_EVIDENCE_CURRENT.json') or {}
    for old in (goal,requirement):
        need(old.get('run_id',RUN_ID) == RUN_ID, 'Existing derived document belongs to another run')
    legacy_fields = ('actual_complete_outer','latest_joint_sha256','current_turn_verified_increment')
    previous = {key:copy.deepcopy(goal[key]) for key in legacy_fields if key in goal}
    current = dict(actual_complete_outer=n,latest_joint_sha256=summary['joint_sha256'],
                   current_turn_verified_increment=note)
    if previous and any(previous.get(key) != value for key,value in current.items()):
        goal['previous_turn_progress_annotation'] = dict(previous,
            scope='Previous top-level boundary/SHA and caller-supplied annotation; retained before this local update.')
    for document in (goal,requirement):
        document.update(run_id=RUN_ID,actual_complete_outer=n,derived_progress_summary=summary)
    goal.update(classification=classification,note=note,actual_evidence=str(out/'LATEST_ACTUAL.json'),observed_unix=actual.get('observed_unix'),
        latest_joint_sha256=summary['joint_sha256'],current_turn_verified_increment=note,
        current_turn_verified_increment_kind='caller_supplied_annotation_from_cli_note',
        actor_adam_per_rank=clocks.get('actor'),probe_adam_per_rank=clocks.get('probe'),
        outer_in_progress=stage.get('outer'),active_job=summary['active_job'],
        all_latest_phase_exits_zero=summary['all_latest_phase_exits_zero'],
        all8_actor_probe20_actual_writes=summary['all8_actor_probe20_actual_writes'],
        loss_pairs_through_outer=pairs['through_outer'],loss_pair_pending_outers=pairs['pending_outers'],
        cost_pending_outers=sorted(set(costs['generation']['pending_outers'])|set(costs['retrieval']['pending_outers'])),
        sampled_policy_tokens=costs['generation']['total'],logical_retrieval_calls=costs['retrieval']['total'])
    documents = {'GOAL_TURN_PROGRESS_CURRENT.json':goal,'REQUIREMENT_EVIDENCE_CURRENT.json':requirement,
        'OPTIMIZATION_COST_CURRENT.json':optimization,'PROBE_PAIRED_LOSS_COMPONENT_SUMMARY.json':pairs,
        'MIXED_SELECTION_TREND_CURRENT.json':trend,'CURRENT_PROGRESS_SUMMARY.json':summary}
    if mirror_roots is None:
        mirror_roots = (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/NAME,
            Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/NAME)
    need(len(mirror_roots) == 2, 'Keep the two established local mirrors')
    mirrors = [Path(path).resolve() for path in mirror_roots]
    status_targets = [out/'STATUS.json',root/'outputs/STATUS.json']+[p/'STATUS.json' for p in mirrors]
    status_values = {}
    for path in status_targets:
        value = status if path == out/'STATUS.json' else inputs.read(path)
        if value is None: value = copy.deepcopy(status)
        if value is not None:
            need(value.get('run_id',RUN_ID) == RUN_ID, 'Status mirror belongs to another run')
            status_values[path] = dict(value,derived_progress_summary=summary)
    summary['sources'] = [v for p,v in inputs.files.items() if p.name not in ('STATUS.json','GOAL_TURN_PROGRESS_CURRENT.json','REQUIREMENT_EVIDENCE_CURRENT.json')]; summary['missing_inputs'] = [str(p) for p in sorted(inputs.missing)]
    text = (f"Joint complete outer: {n if n is not None else 'pending'} / 200. "
        f"Classification: {classification}. Live saved actor/probe states remain separately reported.\n\n"
        f"Cost coverage generation/retrieval: {costs['generation']['through_outer']} / {costs['retrieval']['through_outer']}; "
        f"paired loss coverage: {pairs['through_outer']}. Missing coverage remains pending.\n\n"
        f"Note: {note}\n\nSource: {out/'LATEST_ACTUAL.json'}; source observation Unix: {actual.get('observed_unix')}. "
        "Final200/full51713/cleanup completion is not assessed by this local tool.\n")
    writes = {}
    encode = lambda value: (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    for directory in [out]+mirrors:
        for name,value in documents.items(): writes[directory/name] = encode(value)
        writes[directory/'CURRENT_PROGRESS_SUMMARY.md'] = text.encode()
    writes[root/'outputs/CURRENT_PROGRESS_SUMMARY.json'] = encode(summary)
    writes[root/'outputs/CURRENT_PROGRESS_SUMMARY.md'] = text.encode()
    for path,value in status_values.items(): writes[path] = encode(value)
    inputs.stable()  # Validate everything before creating/replacing any output.
    for path,raw in writes.items(): atomic_write(path,raw)
    return dict(actual_complete_outer=n,classification=classification,summary=str(out/'CURRENT_PROGRESS_SUMMARY.json'),
        writes=len(writes),goal_state_changed=False,remote_operations=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--classification',choices=('progress','verified_wait'),required=True)
    parser.add_argument('--note',default='');args=parser.parse_args()
    print(json.dumps(summarize(args.root,args.classification,args.note),allow_nan=False))


if __name__=='__main__': main()
