"""Outcome-based whole-question selection after Top1024 prefixes finish.

Binary population EM variance is K(5-K)/25. Ties use the already frozen
prefix prediction, then original candidate order; there is no reward probing
or additional sampling. Keep every G5 intact, including all-equal groups when
fewer than512 mixed groups exist. The probe still consumes all1024 groups.
"""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path

from .actor import canonical, read, sha, atomic_json, validate_record


def select_groups(selection, labels, k=512):
    groups = selection['selected']
    if len(groups) != 1024 or len(labels) != 5120 or k != 512:
        raise ValueError('Require1024 full G5 outcomes and512 actor questions')
    tids, qids, ordinals = set(), set(), set()
    rows = []
    for group in groups:
        ids = group['trajectory_ids']
        if (len(ids) != 5 or len(set(ids)) != 5 or tids.intersection(ids)
                or group['qid'] in qids or group['candidate_ordinal'] in ordinals
                or not math.isfinite(group['q']) or not 0 <= group['q'] <= 1):
            raise ValueError('Incomplete, duplicate, or invalid scored group')
        outcomes = []
        for tid in ids:
            value = labels[tid]
            if (value['qid'] != group['qid'] or type(value['strict_em']) is not int
                    or value['strict_em'] not in (0, 1)):
                raise ValueError('Outcome label belongs to a different own trajectory')
            outcomes.append(value['strict_em'])
        count = sum(outcomes)
        rows.append(dict(group, own_strict_em=outcomes, correct_count=count,
                         em_variance=count * (5 - count) / 25, mixed=0 < count < 5))
        tids.update(ids); qids.add(group['qid']); ordinals.add(group['candidate_ordinal'])
    if tids != set(labels):
        raise ValueError('Extra or missing actually continued labels')
    ranked = sorted(rows, key=lambda r: (-r['em_variance'], -r['q'], r['candidate_ordinal']))
    return ranked[:k], rows


def summarize(rows):
    n = len(rows)
    return dict(questions=n, trajectories=n * 5,
        mixed_questions=sum(r['mixed'] for r in rows),
        mixed_fraction=sum(r['mixed'] for r in rows) / n,
        mean_predicted_mixed_probability=math.fsum(r['q'] for r in rows) / n,
        mean_population_em_variance=math.fsum(r['em_variance'] for r in rows) / n,
        correct_count_histogram=dict(sorted(Counter(str(r['correct_count']) for r in rows).items())))


def select(config_path, outer):
    c = read(config_path); root = Path(c['base']) / 'outers' / ('outer-%06d' % outer)
    source_path, selection_path = root / 'full/complete.json', root / 'probe-score/selection.json'
    source, selection = read(source_path), read(selection_path)
    if (source['run_id'] != c['run_id'] or source['outer'] != outer
            or source['questions'] != 1024 or source['trajectories'] != 5120
            or selection['run_id'] != c['run_id'] or selection['outer'] != outer
            or source['actor_version'] != selection['actor_version']
            or source['selection_sha256'] != sha(selection_path)):
        raise ValueError('Completed Top1024 rollout/selection identity differs')
    own = source['own_em']; path = Path(own['path'])
    if path.stat().st_size != own['bytes'] or sha(path) != own['sha256']:
        raise ValueError('Actual own EM file differs from completed continuation')
    labels = {}
    for line in path.read_text().splitlines():
        row = json.loads(line); tid = row['trajectory_id']
        if (tid in labels or row['actor_version'] != source['actor_version']
                or row['selection_sha256'] != sha(selection_path)):
            raise ValueError('Duplicate or foreign own EM label')
        labels[tid] = row
    selected, all_groups = select_groups(selection, labels)
    wanted = {tid for group in selected for tid in group['trajectory_ids']}
    full_rows = {}
    for item in source['shards']:
        path = Path(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Actual completed full rollout shard changed')
        opener = gzip.open if path.suffix == '.gz' else open
        count = 0
        with opener(path, 'rt') as handle:
            for line in handle:
                count += 1; row = json.loads(line); tid = row['trajectory_id']
                if tid in wanted:
                    validate_record(row, source['actor_version'])
                    if (tid in full_rows or row['qid'] != labels[tid]['qid']
                            or row['prefix_sha256'] != labels[tid]['prefix_sha256']
                            or row['selection_sha256'] != sha(selection_path)):
                        raise ValueError('Selected full trajectory lost its original prefix identity')
                    full_rows[tid] = row
        if count != item['count']:
            raise ValueError('Full source shard record count differs')
    if set(full_rows) != wanted:
        raise ValueError('A selected own full trajectory is missing')
    out = root / 'actor-selection'; out.mkdir(exist_ok=False)
    selected_manifest = dict(run_id=c['run_id'], outer=outer, actor_version=source['actor_version'],
        selected=selected, outcome_selection='population_strict_EM_variance_K_times_5minusK_over25',
        tie_break='frozen_prefix_q_descending_then_original_candidate_ordinal',
        prefix_selection_sha256=sha(selection_path), full_rollout_sha256=sha(source_path),
        continued1024=summarize(all_groups), actor512=summarize(selected),
        probe_uses_all_continued1024=True, uncontinued_candidate_questions=4096,
        all_candidate_accuracy_available=False)
    atomic_json(out / 'selection.json', selected_manifest)
    shards = []
    for rank in range(8):
        path = out / ('rank-%d-full.jsonl.gz' % rank)
        ids = [tid for group in selected[rank::8] for tid in group['trajectory_ids']]
        with gzip.open(path, 'xt') as handle:
            for tid in ids:
                handle.write(canonical(full_rows[tid]).decode() + '\n')
        with path.open('rb') as handle: os.fsync(handle.fileno())
        shards.append(dict(path=str(path), count=len(ids), bytes=path.stat().st_size, sha256=sha(path)))
    complete = dict(source, questions=512, trajectories=2560, shards=shards,
        actor_selection_sha256=sha(out / 'selection.json'),
        source_full_rollout_path=str(source_path), source_full_rollout_sha256=sha(source_path),
        actual_selected_mixed_questions=selected_manifest['actor512']['mixed_questions'],
        actual_selected_mixed_fraction=selected_manifest['actor512']['mixed_fraction'],
        actor_selection_metrics=selected_manifest, raw_native_records_preserved=True)
    # Raw generation costs belong to all1024 continuations, not just actor512.
    complete['generation_cost_scope'] = 'all1024_source_full_rollout_not_additive_with_selected512'
    atomic_json(out / 'complete.json', complete)
    return dict(path=str(out / 'complete.json'), metrics=selected_manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--outer', type=int, required=True)
    args = parser.parse_args(); print(json.dumps(select(args.config, args.outer), allow_nan=False))
