#!/usr/bin/env python3
"""Pure local FULL107..118 publication from actual post119 source readback.

Never manufactures a poll, rewrites a live progress counter, contacts a remote,
or writes STATUS/REPORT/CURRENT. NativeN guard supplies historical counts;
the unchanged live119 poll is only the later observation/identity/audit-stat pin.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

MODE = 'RETROSPECTIVE_SOURCE_READBACK'
GAP_SCHEMA = 'tree_nested_historical_gap_scalar_readback_v1'
SOURCE = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + SOURCE
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INVOCATION = 'a313bda919834d24b4c5d491698d39ce'
SCOPE = ('Historical source scalars read after outer119. Original live119 poll stays byte-identical; '
         'it is not an observation at historical outerN. Metric/rank/controller/dev come from '
         'SHA-bound original files in the gap readback; cumulative counts/attempts come from nativeN '
         'persisted guard. Later actual next120 entry is an as-of-readback fact, not a historical marker. '
         'Original writer/archive JSON metadata only; current PT availability unknown; no cold reload.')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def held(base, name, expected=None):
    name = Path(name).name
    if not name.endswith('.json') or name.startswith(('STATUS', 'REPORT', 'CURRENT')):
        raise ValueError('Only immutable task scalar JSON is allowed')
    raw = (base / name).read_bytes()
    if len(raw) > 4_000_000 or (expected and digest(raw) != expected):
        raise ValueError('Immutable source SHA/size differs: ' + name)
    return name, json.loads(raw), dict(path=name, bytes=len(raw), sha256=digest(raw))


def bound_file(value):
    if not isinstance(value, dict) or not isinstance(value.get('bytes'), int) or value['bytes'] <= 0:
        raise ValueError('Original remote file size/path binding missing')
    if not value.get('path', '').startswith(RUN + '/') or not re.fullmatch('[0-9a-f]{64}', value.get('sha256', '')):
        raise ValueError('Original remote file identity missing')
    return value


def components(base, outer, gap_name, gap_sha, anchor_name, anchor_sha):
    if not 107 <= outer <= 118:
        raise ValueError('Only actual historical107..118 bridge is supported')
    gn, gap, gb = held(base, gap_name, gap_sha)
    an, anchor, ab = held(base, anchor_name, anchor_sha)
    pn, poll, pb = held(base, anchor['source_poll'], anchor['source_poll_sha256'])
    nn, native, nb = held(base, f'NATIVE_HISTORY_{outer}_METADATA_READBACK.json')
    if (gap.get('summary_schema') != GAP_SCHEMA or gap.get('requested_outers') != list(range(107, 120))
            or gap.get('source_commit') != SOURCE or gap.get('config_sha256') != CONFIG
            or gap.get('run') != RUN or gap.get('unit') != UNIT or gap.get('read_only') is not True
            or gap.get('remote_files_written') is not False or gap.get('tensor_contents_read') is not False):
        raise ValueError('Actual gap source/config/read-only scope differs')
    if (anchor.get('completed_outer') != 119 or (anchor.get('actual_next_outer_entered') or 0) < 120
            or anchor.get('source_commit') != SOURCE or anchor.get('config_sha256') != CONFIG
            or anchor.get('run') != RUN or anchor.get('unit') != UNIT
            or anchor.get('actual_training_pid') != 503255 or anchor.get('actual_training_invocation_id') != INVOCATION
            or poll.get('source_commit') != SOURCE or poll.get('run') != RUN or poll.get('unit') != UNIT
            or poll.get('progress', {}).get('actual_completed_outer_from_attempt_results') != 119
            or poll.get('progress', {}).get('errors') or gap['observed_epoch'] < poll['epoch']):
        raise ValueError('Unchanged actual119/live120 retrospective anchor differs')
    sync_path = Path(__file__).with_name('sync_nested_native_progress.py')
    sync = module(sync_path, 'retrospective_pure_sync_validators')
    sync.validate_restore(poll)
    metric = sync.validate_metric({'iteration_metrics': gap['iteration_metrics']}, outer)
    original_ranks = gap['direct_rank_summaries_by_outer'][str(outer)]
    ranks = sync.validate_ranks({'last_rank_summaries': original_ranks}, metric, outer)
    guard = native['identity']['guard_continue_accounting']
    # This in-memory validator argument is explicitly native-derived, never a saved poll.
    native_progress = dict(training_attempts_seen=guard['training_attempts'],
        completed_actual_optimizer_calls_per_rank=guard['completed_optimizer_calls_per_rank'],
        all_skipped_attempts=guard['refused_attempts'])
    record, guard, completed, totals = sync.validate_native(native, native_progress, outer)
    archived = [x for x in gap['native_metadata_readbacks'] if x.get('outer') == outer]
    if len(archived) != 1 or any(native.get(k) != archived[0].get(k) for k in
            ('source_commit', 'outer', 'receipt_sha256', 'manifest_sha256', 'identity', 'receipt')):
        raise ValueError('Individual native readback differs from actual gap archive')
    lines = [x for x in gap['metric_line_sources'] if x.get('outer') == outer]
    if (len(lines) != 1 or lines[0].get('path') != RUN + '/stage.log'
            or not re.fullmatch('[0-9a-f]{64}', lines[0].get('line_sha256', ''))
            or not isinstance(lines[0].get('byte_start'), int) or lines[0]['byte_end'] <= lines[0]['byte_start']):
        raise ValueError('Actual complete metric source line binding absent')
    rank_files = gap['direct_rank_source_files_by_outer'][str(outer)]
    if len(rank_files) != 8:
        raise ValueError('Eight original rank file bindings are required')
    for rank, meta in enumerate(rank_files):
        bound_file(meta)
        if not meta['path'].endswith('/step-%06d-attempt-%06d-rank-%02d.json' % (outer, outer, rank)):
            raise ValueError('Original rank file path differs')
    controller = gap['rollout_cost_by_outer'][str(outer)]
    controller_file = bound_file(gap['rollout_cost_source_files_by_outer'][str(outer)])
    if (controller.get('status') != 'TRAINING_ROLLOUT_COST_COMPLETE' or controller.get('candidate_outer') != outer
            or controller.get('attempt_id') != outer or controller.get('completed_outer_before') != outer - 1
            or controller.get('code_commit') != SOURCE or controller.get('config_sha256') != CONFIG
            or not controller_file['path'].endswith('/attempt-%06d.json' % outer)):
        raise ValueError('Actual historical COMPLETE controller identity differs')
    audit = [x for x in poll['output_metadata'] if re.fullmatch(
        'output/tree-audit/step_' + str(outer) + r'_rollout_\d+\.json', x.get('path', ''))]
    if len(audit) != 1 or not isinstance(audit[0].get('bytes'), int) or not sync.number(audit[0].get('mtime')):
        raise ValueError('Fixed completed historical audit stat pin absent from actual live poll')
    devs = [x for x in gap['development'] if x.get('outer') == outer]
    dev = devs[0] if len(devs) == 1 else None
    if len(devs) > 1 or (outer in (110, 115)) != bool(dev):
        raise ValueError('Actual historical periodic dev presence differs')
    dev_file = None
    if dev is not None:
        if (not sync.valid_dev(dev) or dev['metrics'].get('val/qa_count/nq') != 256
                or dev['metrics'].get('val/qa_count/hotpotqa') != 256):
            raise ValueError('Actual historical dev512 identity incomplete')
        dev_files = [x for x in gap['development_source_files'] if x.get('path', '').endswith('/' + dev['path'])]
        if len(dev_files) != 1:
            raise ValueError('Actual dev cost file binding missing')
        dev_file = bound_file(dev_files[0])
    _, prior, _ = held(base, 'FULL_OUTER106_PROGRESS_RECEIPT.json')
    latest = max([x for x in [prior.get('latest_actual_dev512')] + gap['development']
                  if x and x['outer'] <= outer], key=lambda x: x['outer'])
    pins = {gn: gb, an: ab, pn: pb, nn: nb}
    resource = anchor.get('input_bindings', {}).get('resource')
    if resource:
        rn, _, rb = held(base, resource['path'], resource['sha256'])
        if rb['bytes'] != resource['bytes']:
            raise ValueError('Actual119 observation resource binding differs')
        pins[rn] = rb
    return dict(gap_name=gn, gap=gap, gap_binding=gb, anchor_name=an, anchor=anchor, anchor_binding=ab,
        poll_name=pn, poll=poll, poll_binding=pb, native_name=nn, native=native, native_binding=nb,
        metric=metric, ranks=ranks, controller=controller, native_record=record, guard=guard,
        completed=completed, totals=totals, dev=dev, latest_dev=latest, bindings=pins,
        remote_bindings=dict(metric_line=lines[0], rank_files=rank_files, controller_file=controller_file,
                             dev_file=dev_file, resolved_config=gap['resolved_config']),
        validator_sha256=digest(sync_path.read_bytes()))


def cost_evidence(base, full, poll, native):
    x = components(base, full['completed_outer'], full['source_readback'], full['source_readback_sha256'],
                   full['source_anchor_receipt'], full['source_anchor_receipt_sha256'])
    if (poll != x['poll'] or native != x['native'] or full.get('iteration_metrics') != x['metric']
            or full.get('last_rank_summaries') != x['ranks'] or full.get('actual_dev512') != x['dev']
            or full.get('per_rank_successful_adam') != x['completed']
            or full.get('retrospective_original_file_bindings') != x['remote_bindings']):
        raise ValueError('Retrospective FULL differs from its actual immutable source readback')
    return x


def receipt(outer, x):
    metric, dev, guard = x['metric'], x['dev'], x['guard']
    dev_wall = metric.get('timing_s/testing') if dev is not None else None
    if dev is not None and dev_wall is None:
        raise ValueError('Actual periodic dev lacks its own fullmetric testing wall')
    total = metric['timing_s/step']
    return dict(status=MODE + '_FULL_OUTER' + str(outer) + '_NATIVE_COMPLETE', evidence_mode=MODE,
        observed_epoch=x['gap']['observed_epoch'], source_commit=SOURCE, config_sha256=CONFIG, run=RUN, unit=UNIT,
        target_outer=200, restored_outer=5, completed_outer=outer,
        actual_next_outer_entered=x['anchor']['actual_next_outer_entered'],
        actual_next_outer_entry_scope='Actual later live119 anchor next120 entry; no historical contemporaneous marker claimed.',
        retrospective_observation_completed_outer=x['poll']['progress']['actual_completed_outer_from_attempt_results'],
        retrospective_evidence_scope=SCOPE, retrospective_original_file_bindings=x['remote_bindings'],
        historical_counter_source='nativeN.identity.guard_continue_accounting; never relabel live119 counters as N',
        historical_training_attempts=guard['training_attempts'], all_skipped_attempts=guard['refused_attempts'],
        per_rank_successful_adam=x['completed'], per_rank_actual_total_adam=x['totals'],
        partial_optimizer_calls_per_rank=guard['partial_optimizer_calls_per_rank'],
        actual_outer_successful_writes_per_rank={str(r['rank']): r['actual_successful_writes'] for r in x['ranks']},
        actual_outer_processed_opportunities_per_rank={str(r['rank']): r['actual_opportunities_processed'] for r in x['ranks']},
        actual_outer_skipped_opportunities_per_rank={str(r['rank']): r['actual_opportunities_skipped'] for r in x['ranks']},
        full_metric_processed_opportunities=metric['actor/optimizer_opportunities_processed'],
        full_metric_skipped_opportunities=metric['actor/optimizer_opportunities_skipped'],
        no_all20_writes_or_zero_skip_assumption=True, last_rank_summaries=x['ranks'],
        native_manifest_sha256=x['native']['manifest_sha256'], native_history_receipt_sha256=x['native']['receipt_sha256'],
        native_driver_next_step=x['native_record']['driver']['next_step'], native_workers=x['native_record']['workers'],
        native_evidence_scope=SCOPE, independent_cold_reload_performed=False, tensor_contents_read=False,
        PT_current_availability_unknown=True, exact_executor_resume=False,
        iteration_metrics=metric, full_iteration_metric_pending=False, full_outer_wall_seconds=total,
        actual_dev512=dev, latest_actual_dev512=x['latest_dev'], latest_dev512_outer=x['latest_dev']['outer'],
        normal_dev512_wall_seconds=dev_wall, training_archive_wall_seconds_excluding_current_dev=total - (dev_wall or 0),
        dev_score_updated_this_turn=False,
        source_poll=x['poll_name'], source_poll_sha256=x['poll_binding']['sha256'],
        source_readback=x['gap_name'], source_readback_sha256=x['gap_binding']['sha256'],
        source_metric_file=x['gap_name'], source_metric_sha256=x['gap_binding']['sha256'],
        source_metric_selector='iteration_metrics[step=' + str(outer) + ']; metric_line_sources[outer=' + str(outer) + ']',
        source_rank_file=x['gap_name'], source_rank_sha256=x['gap_binding']['sha256'],
        source_rank_selector='direct_rank_summaries_by_outer.' + str(outer),
        source_original_dev_summary=x['gap_name'] if dev else None,
        source_original_dev_summary_sha256=x['gap_binding']['sha256'] if dev else None,
        source_original_dev_selector='development[outer=' + str(outer) + ']' if dev else None,
        source_anchor_receipt=x['anchor_name'], source_anchor_receipt_sha256=x['anchor_binding']['sha256'],
        native_metadata_readback=x['native_name'], native_metadata_readback_sha256=x['native_binding']['sha256'],
        input_bindings={name: value for name, value in x['bindings'].items()},
        actual_training_pid=503255, actual_training_invocation_id=INVOCATION,
        local_control_script_sha256=digest(Path(__file__).read_bytes()), pure_validator_sha256=x['validator_sha256'],
        remote_operations=False, live_poll_manufactured_or_modified=False, current_status_or_report_written=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--gap', default='HISTORICAL_GAP107_119_SCALAR_READBACK.json')
    p.add_argument('--anchor-receipt', default='FULL_OUTER119_PROGRESS_RECEIPT.json')
    args = p.parse_args();base = args.base.resolve(strict=True)
    _, _, gb = held(base, args.gap);_, _, ab = held(base, args.anchor_receipt)
    book = json.loads((base / 'PER_OUTER_METRICS_COMPACT.json').read_bytes())
    mirrors = [Path(v) for v in book['mirrors']]
    # All dependent evidence is validated before the first immutable output.
    items = []
    for outer in range(107, 119):
        x = components(base, outer, args.gap, gb['sha256'], args.anchor_receipt, ab['sha256'])
        name = f'FULL_OUTER{outer}_PROGRESS_RECEIPT.json'
        raw = (json.dumps(receipt(outer, x), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
        if len(raw) > 100000:
            raise ValueError('Retrospective FULL exceeds small metadata budget')
        items.append((name, raw))
    for name, raw in items:
        for root in [base] + mirrors:
            path = root / name
            if path.exists() and path.read_bytes() != raw:
                raise ValueError('Existing immutable FULL retained; no overwrite: ' + name)
    for name, raw in items:
        for root in [base] + mirrors:
            path = root / name
            if not path.exists():
                tmp = path.with_name(path.name + '.tmp');tmp.write_bytes(raw);os.replace(tmp, path)
            if path.read_bytes() != raw:
                raise ValueError('Small immutable publication readback differs')
        print(json.dumps(dict(path=name, bytes=len(raw), sha256=digest(raw), evidence_mode=MODE,
                              both_mirrors_byte_equal=True)), flush=True)


if __name__ == '__main__':
    main()
