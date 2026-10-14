"""Read closed native sparse trees; never construct a new tree or touch training.

The existing report.tree_events supplies the tree counts. This companion binds
them to actual closed dev512/post-evaluation boundaries and adds defined token
and recorded cost coverage. Large weights, tensor files, models and GPU/HTTP
interfaces are never opened.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from release.recursive_bran_a.ablation150_aggregate_watch_v1 import watch
from release.recursive_bran_a.ablation150_reports_v1 import report
from release.recursive_bran_a.ablation150_v1.contracts import planned_conditions, read_receipt, sha

MILESTONES = (5, 50, 100, 150)


def integer(value, label, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError('Invalid recorded integer: ' + label)
    return int(value)


def stable_bytes(path):
    """Only small JSON evidence; compare metadata before/after the actual read."""
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    signature = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns)
    if signature(before) != signature(after) or len(raw) != before.st_size:
        raise ValueError('Closed source changed during read')
    return raw, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns)


def bounded(path, parent):
    return watch.bounded_path(path, parent)


def tree_rows(path, receipt, outer):
    raw, origin = stable_bytes(path)
    if not raw.endswith(b'\n'):
        raise ValueError('Closed tree contains an unfinished JSONL line')
    rows = [watch.finite(json.loads(line)) for line in raw.splitlines()]
    if not rows or rows[0].get('event') != 'question_start' or rows[-1].get('event') != 'question_end':
        raise ValueError('A closed tree needs its actual start and terminal event')
    start, end = rows[0], rows[-1]
    if sum(row['event'] == 'question_end' for row in rows) != 1:
        raise ValueError('Tree has multiple terminal events')
    if start.get('target') != 5 or start.get('budget') != receipt['condition']['budget']:
        raise ValueError('Tree target/budget differs from its immutable condition')
    for index, row in enumerate(rows, 1):
        if (row.get('event_seq') != index or row.get('outer') != outer
                or row.get('question_id') != start['question_id']
                or any(row.get(key) != receipt[key] for key in ('run_id', 'config_sha256', 'source_commit'))):
            raise ValueError('Tree event sequence or actual source/run/config/question identity differs')
    final_ids = end['final_ids']
    if len(set(final_ids)) != len(final_ids) or end['complete_group'] != (len(final_ids) == 5):
        raise ValueError('Final actual tree selection has duplicate IDs or inconsistent completeness')
    children = {}
    roots = set()
    branch_starts, branch_ends = {}, {}
    for row in rows:
        if row['event'] == 'search_start':
            roots.update(value['id'] for value in row['eligible'] if value['generation'] == 0)
        elif row['event'] == 'branch_start':
            attempt = integer(row['attempt'], 'branch attempt', 1)
            if attempt in branch_starts:
                raise ValueError('Repeated branch attempt')
            branch_starts[attempt] = row
        elif row['event'] == 'branch_end':
            attempt = integer(row['attempt'], 'branch end attempt', 1)
            if attempt in branch_ends or row['status'] != 'complete':
                raise ValueError('A native closed training tree retains an errored/repeated branch')
            branch_ends[attempt] = row
            if row['id'] in children or row['reward'] not in (0, 1) or row['parent_reward'] not in (0, 1):
                raise ValueError('Candidate identity or binary reward differs')
            integer(row['generation'], 'candidate generation', 1)
            children[row['id']] = row
    attempts = integer(end['attempts'], 'question attempts')
    expected = set(range(1, attempts + 1))
    if set(branch_starts) != expected or set(branch_ends) != expected or attempts > start['budget']:
        raise ValueError('Closed tree omits actual branch attempt coverage')
    if not roots:
        roots = set(final_ids) - set(children)
    if len(roots) != 1 or not roots.issubset(final_ids) or not set(final_ids).issubset(roots | set(children)):
        raise ValueError('Selected tree root or candidate identity cannot be reconstructed')
    if integer(end['accepted_branches'], 'accepted branches') != sum(
            row['source'] == 'contrast_accept' for row in children.values()):
        raise ValueError('Actual contrast acceptance count differs from candidate events')
    return rows, origin, next(iter(roots)), children


def root_policy_tokens(path, root_id, tree_id):
    """Read only an already saved root JSON to obtain its policy-mask count.

    The raw JSON stays on the node. No token IDs, text, probabilities or private
    question identities are copied into the aggregate report.
    """
    if not path.is_file():
        return None, None
    raw, proof = stable_bytes(path)
    state = json.loads(raw)['state']
    mask = state['model_mask']
    if (state['trajectory'] != root_id or state['tree'] != tree_id
            or len(mask) != len(state['response']) or any(type(value) is not int or value not in (0, 1) for value in mask)):
        raise ValueError('Saved root policy-mask identity/alignment differs')
    return sum(mask), proof


def tree_summary(roots, receipt, outer, scratch_parent):
    """Only one already closed keypoint; source tree coverage must be all 512."""
    sources, question_ids = [], set()
    policy_tokens = 0
    mask_missing = 0
    branch_cost = collections.Counter()
    selected_branch_cost = collections.Counter()
    with tempfile.TemporaryDirectory(prefix='tree-read-', dir=scratch_parent) as directory:
        view = Path(directory)
        for root in roots:
            for path in sorted((root/'recursive-events').glob('*/*/events.jsonl')):
                bounded(path, root/'recursive-events')
                with path.open() as stream:
                    first = stream.readline()
                if not first.endswith('\n'):
                    continue  # An unrelated live tree does not establish coverage.
                first_row = json.loads(first)
                if first_row.get('outer') != outer:
                    continue
                rows, origin, root_id, children = tree_rows(path, receipt, outer)
                question_id = rows[0]['question_id']
                if question_id in question_ids:
                    raise ValueError('Duplicate closed question tree at the same keypoint')
                question_ids.add(question_id)
                link = view/'recursive-events'/str(len(sources))/'0/events.jsonl'
                link.parent.mkdir(parents=True)
                link.symlink_to(path.resolve())
                root_path = bounded(path.parent/(root_id+'.json'), path.parent)
                root_count, root_proof = root_policy_tokens(root_path, root_id, rows[0]['root_id'])
                selected = rows[-1]['final_ids']
                counts = [root_count]
                for candidate_id, candidate in children.items():
                    cost = candidate.get('cost', {})
                    for key in ('new_generated_tokens', 'retrieval_calls', 'model_calls', 'branch_wall_seconds'):
                        if key in cost:
                            value = report.number(cost[key], 'branch cost ' + key)
                            if value < 0:
                                raise ValueError('Negative recorded branch cost')
                            branch_cost[key] += value
                            if candidate_id in selected:
                                selected_branch_cost[key] += value
                    if candidate_id in selected:
                        counts.append(integer(cost['full_training_model_tokens'], 'selected full policy tokens')
                                      if 'full_training_model_tokens' in cost else None)
                if any(count is None for count in counts):
                    mask_missing += 1
                else:
                    policy_tokens += sum(counts)
                # This manifest is remote/private; it retains source paths, never question IDs.
                sources.append(dict(events_path=str(path), events=origin, root_path=str(root_path), root=root_proof))
        if len(question_ids) != 512:
            return dict(status='INCOMPLETE_TREE_COVERAGE', closed_questions=len(question_ids), expected_questions=512), sources
        summary = report.tree_events([view], receipt)[outer]
        if summary['questions'] != 512 or summary['selected_reward_count_unavailable'] != 0:
            raise ValueError('Existing tree collector disagrees with actual 512 tree/reward coverage')
        # Read-back only the accepted small events; no complete raw candidate JSON read.
        for source in sources:
            if sha(source['events_path']) != source['events']['sha256']:
                raise ValueError('Closed source tree changed while existing report collector read it')
        summary.update(status='COMPLETE_CLOSED_TREE512', selected_policy_token_occurrences=(
            policy_tokens if mask_missing == 0 else None), policy_denominator_missing_questions=mask_missing,
            recorded_all_candidate_branch_cost_sum=dict(branch_cost),
            recorded_selected_nonroot_branch_cost_sum=dict(selected_branch_cost),
            branch_cost_scope='Child suffix generated tokens, retrieval calls and shared-batch elapsed times from '
                'already saved branch_end.cost. Summed branch elapsed times overlap across questions; not GPU wall.')
        return summary, sources


def cost_summary(lines, outer, summary, recovered):
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch)/'native.log'
        path.write_text(''.join(lines[index] for index in sorted(lines) if index <= outer))
        metrics = report.native_log([path])
    prefix = {index: row for index, row in metrics.items() if 1 <= index <= outer}
    keys = sorted({key for row in prefix.values() for key in row if key.startswith(('rollout/', 'timing_s/'))})
    totals = {key: sum(row[key] for row in prefix.values() if key in row) for key in keys}
    coverage = {key: sum(key in row for row in prefix.values()) for key in keys}
    boundary = metrics[outer]
    inherited = boundary.get('rollout/raw_inherited_leaf_tokens')
    denominator = summary.get('selected_policy_token_occurrences')
    if inherited is not None:
        inherited = integer(inherited, 'native inherited leaf token occurrences')
    if denominator is not None and inherited is not None and inherited > denominator:
        raise ValueError('Native repeated policy occurrences exceed the exact selected policy denominator')
    complete_prefix = sorted(prefix) == list(range(1, outer+1))
    return dict(recorded_boundary_scalars={key: value for key, value in boundary.items()
            if key.startswith(('rollout/', 'timing_s/', 'training/', 'actor/lr', 'actor/pg_loss'))},
        recorded_prefix_sums=totals, per_scalar_outer_coverage=coverage, complete_scalar_prefix=complete_prefix,
        allocated_gpu_hours_from_recorded_outer_wall=(totals['timing_s/step']*8/3600
            if 'timing_s/step' in totals else None), recovered_original_sampling_cost_missing=bool(recovered),
        complete_recorded_emitted_token_prefix=bool(complete_prefix and not recovered and all(
            'rollout/raw_emitted_tokens' in value for value in prefix.values())),
        complete_end_to_end_cost=False,
        inherited_policy_occurrence_fraction=dict(numerator=inherited, denominator=denominator,
            value=(inherited/denominator if inherited is not None and denominator else None),
            definition='Duplicate retained sampled policy-token occurrences across selected leaves divided '
                'by all selected leaf policy-token occurrences. Numerator is native raw_inherited_leaf_tokens '
                '(total raw action-token occurrences minus unique(sample_id,offset)); denominator sums saved '
                'root.model_mask plus selected branch_end.cost.full_training_model_tokens. '
                'This is not the average child inherited-prefix fraction.'),
        cost_scope='Existing native scalars only. Raw emitted/prefill may include padding and discarded tails '
            'by native convention. Stage timers overlap. GPU hours = eight allocated GPUs × recorded step wall, '
            'not GPU activity/lease billing. A zero-write replay excludes original sampling cost.')


def collect(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    planned = planned_conditions()
    if output == base or any(output.is_relative_to(base/row['group']) for row in planned):
        raise ValueError('Separate analysis output is required; never write inside training groups')
    output.mkdir(parents=True, exist_ok=True)
    study = watch.read(base/'study-progress.json')
    if study['conditions'] != planned or study['requested_labels'] != 11:
        raise ValueError('Frozen full study conditions differ')
    state = dict(protocol='recursive_ablation150_tree_summary_v1', time=time.time(), goal_complete=False,
        scientific_source_commit=study['source_commit'], required_unique_reports=28, reports={}, pending=[], errors=[])
    for row in planned:
        if row.get('reuse'):
            continue
        label = row['group']
        group_base = base/label
        if not (group_base/'receipts/configuration.json').is_file():
            continue
        try:
            receipt = read_receipt(group_base/'receipts/configuration.json')
            if (receipt['group'] != label or receipt['condition'] != row['options']
                    or receipt['source_commit'] != study['source_commit']):
                raise ValueError('Actual configuration/source differs from frozen condition')
            roots = [bounded(path, group_base/'output') for path in sorted((group_base/'output').glob('attempt-*')) if path.is_dir()]
            closed, origins = watch.closed_records(roots, receipt)
            for outer in MILESTONES:
                if outer not in closed:
                    continue
                destination = output/label/f'outer-{outer:03d}'
                complete = destination/'complete.json'
                if complete.is_file():
                    accepted = watch.read(complete)
                    if set(accepted['files']) != {'summary.json', 'provenance-private.json'}:
                        raise ValueError('Persisted tree report contains unexpected file pointers')
                    for name, proof in accepted['files'].items():
                        if sha(destination/name) != proof['sha256']:
                            raise ValueError('Persisted tree summary changed')
                    if (accepted['run_id'] != receipt['run_id'] or accepted['config_sha256'] != receipt['config_sha256']
                            or accepted['boundary_sha256'] != closed[outer]['checkpoint']['sha256']):
                        raise ValueError('Existing report belongs to a different actual boundary')
                    state['reports'].setdefault(label, {})[str(outer)] = dict(directory=str(destination), complete_sha256=sha(complete))
                    continue
                boundary, dev = watch.boundary(closed[outer], receipt, group_base)
                lines, log_origins = watch.native_lines(group_base, closed)
                if not set(range(1, outer+1)).issubset(lines):
                    state['pending'].append(dict(group=label, outer=outer, reason='Native scalar prefix still unavailable'))
                    continue
                trees, tree_sources = tree_summary(roots, receipt, outer, output)
                if trees['status'] != 'COMPLETE_CLOSED_TREE512':
                    state['pending'].append(dict(group=label, outer=outer, **trees))
                    continue
                recovered = any(watch.read(root/'zero-write-recovered.json', optional=True) is not None for root in roots)
                entry = watch.read(group_base/'receipts/current-entry.json', optional=True)
                recovered = recovered or bool(entry and entry.get('zero_write_manifest'))
                costs = cost_summary(lines, outer, trees, recovered)
                destination.mkdir(parents=True, exist_ok=True)
                summary = dict(protocol=state['protocol'], group=label, outer=outer, run_id=receipt['run_id'],
                    config_sha256=receipt['config_sha256'], source_commit=receipt['source_commit'],
                    dev512=dev, native_boundary=boundary, tree=trees, costs=costs, goal_complete=False,
                    scope='One actual closed sparse milestone only; no A replay, new model generation, tensor reload '
                          'or per-question output. Training/generation implementation is unchanged.')
                provenance = dict(configuration_receipt_sha256=sha(group_base/'receipts/configuration.json'),
                    boundary=boundary, closed_native_record=origins[outer],
                    native_log_lines={str(index): log_origins[index] for index in range(1, outer+1)},
                    sparse_tree_sources=tree_sources,
                    collector_source_sha256=sha(__file__), original_tree_report_source_sha256=sha(report.__file__),
                    original_mask_definition='search_r1/baselines/rollout.py SearchRollout._compose',
                    original_mask_definition_source_sha256=sha(Path(__file__).resolve().parents[3]/
                                                               'search_r1/baselines/rollout.py'),
                    full_tensor_or_logit_reload_performed=False)
                watch.atomic_json(destination/'summary.json', summary)
                watch.atomic_json(destination/'provenance-private.json', provenance)
                files = {name: dict(bytes=(destination/name).stat().st_size, sha256=sha(destination/name))
                         for name in ('summary.json', 'provenance-private.json')}
                watch.atomic_json(complete, dict(protocol=state['protocol'], group=label, outer=outer,
                    run_id=receipt['run_id'], config_sha256=receipt['config_sha256'],
                    boundary_sha256=boundary['sha256'], files=files, goal_complete=False))
                state['reports'].setdefault(label, {})[str(outer)] = dict(directory=str(destination), complete_sha256=sha(complete))
        except (ValueError, KeyError, FileNotFoundError, TypeError, OSError) as error:
            state['errors'].append(dict(group=label, error=str(error)))
    state['collected_unique_reports'] = sum(len(value) for value in state['reports'].values())
    state['all_required_unique_tree_reports_present'] = state['collected_unique_reports'] == 28 and not state['errors']
    # Matched labels point to the same actual C1 trees; A tree coverage is not fabricated from EM-only packs.
    state['matched_reuse'] = {row['group']: dict(reuse_of=row['reuse'], tree_summary_same_actual_condition=(
        state['reports'].get(row['reuse']) if row['reuse'] != 'A-original' else None),
        new_training_or_generation=False) for row in planned if row.get('reuse')}
    state['status'] = 'ATTENTION_REQUIRED' if state['errors'] else 'SPARSE_TREE_REPORTS_COMPLETE' if (
        state['all_required_unique_tree_reports_present']) else 'ACTIVE'
    watch.atomic_json(output/'tree-summary-state.json', state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=60.)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('interval must be finite and at least one second')
    while True:
        value = collect(args.base, args.output)
        print(json.dumps({key: value[key] for key in ('time', 'status', 'collected_unique_reports', 'pending', 'errors')},
              allow_nan=False), flush=True)
        if value['errors']:
            return 2
        if args.once or value['all_required_unique_tree_reports_present']:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
