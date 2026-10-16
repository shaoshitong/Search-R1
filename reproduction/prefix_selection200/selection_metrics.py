"""Read-only selection summaries from completed native-actor rounds.

Run on the source node, outside the frozen training repository. This reads only
small selection/strict-EM records; it never loads tensors or changes training.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_outer(joint_path):
    joint = read(joint_path)
    rollout = joint['rollout']
    labels_path = Path(rollout['own_em']['path'])
    # Imported complete rounds preserve their original immutable source paths.
    source_outer = labels_path.parent.parent
    selection_path = source_outer / 'probe-score/selection.json'
    selection = read(selection_path)
    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels = {row['trajectory_id']: row for row in rows}
    groups = selection['selected']
    assert len(groups) == 512 and len(rows) == len(labels) == 2560
    assert len({row['qid'] for row in groups}) == 512
    assert selection['actor_version'] == rollout['actor_version']
    assert digest(selection_path) == joint['selection_sha256'] == rollout['selection_sha256']
    assert digest(labels_path) == rollout['own_em']['sha256']
    seen, histogram = set(), Counter()
    for group in groups:
        ids = group['trajectory_ids']
        assert len(ids) == 5 and not seen.intersection(ids)
        seen.update(ids)
        own = [labels[tid] for tid in ids]
        assert all(row['qid'] == group['qid'] and row['strict_em'] in (0, 1)
                   and row['actor_version'] == rollout['actor_version'] for row in own)
        histogram[sum(row['strict_em'] for row in own)] += 1
    assert seen == set(labels)
    mixed = sum(histogram[k] for k in (1, 2, 3, 4))
    assert mixed == rollout['actual_selected_mixed_questions']
    assert sum(k * histogram[k] for k in range(6)) == rollout['selected_strict_correct']
    prefix = read(source_outer / 'prefix/complete.json')
    return dict(
        outer=joint['outer'], completed_unix=joint['observed_unix'],
        sampling_actor_version=rollout['actor_version'],
        probe_used_to_select_sha256=selection['probe_before_checkpoint_sha256'],
        selected_questions=512, selected_trajectories=2560,
        correct_count_histogram={str(k): histogram[k] for k in range(6)},
        all_wrong_questions=histogram[0], mixed_questions=mixed,
        all_correct_questions=histogram[5], actual_mixed_fraction=mixed / 512,
        mean_predicted_mixed_probability=math.fsum(row['q'] for row in groups) / 512,
        unselected_outcome_labels_available=bool(joint.get('periodic_probe_update')),
        actor_actual_adam_per_rank=joint['actor']['actual_adam_steps_per_rank'],
        probe_actual_adam_per_rank=joint['probe']['actual_adam_steps_per_rank'],
        phase_timings=joint['phase_timings'],
        native_capture_seconds_per_rank=[row['actor_hidden_forward_and_capture_seconds']
                                         for row in prefix['ranks']],
        periodic_probe_metrics=joint['probe_metrics'] if joint.get('periodic_probe_update') else None,
        selection_path=str(selection_path), selection_sha256=digest(selection_path),
        strict_em_path=str(labels_path), strict_em_sha256=digest(labels_path),
        joint_path=str(joint_path), joint_sha256=digest(joint_path),
    )


def collect(base):
    base = Path(base)
    config = read(base / 'preparation/launch-config.json')
    assert config['actor_hidden_native'] and config['probe_update_period'] == 20
    latest = read(base / 'latest-joint.json')
    rows, missing = [], []
    for outer in range(1, latest['outer'] + 1):
        path = base / 'outers' / ('outer-%06d' % outer) / 'complete.json'
        try:
            row = summarize_outer(path)
            assert row['outer'] == outer and read(path)['run_id'] == config['run_id']
            rows.append(row)
        except (OSError, ValueError, KeyError, AssertionError) as error:
            missing.append(dict(outer=outer, error=repr(error)))
    return dict(run_id=config['run_id'], base=str(base), observed_unix=time.time(),
                actual_complete_outer=latest['outer'], rows=rows, missing=missing,
                status='COMPLETE_SELECTION_METRICS' if not missing else 'PARTIAL_SELECTION_METRICS',
                training_modified=False, tensor_or_rollout_payload_loaded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base, output = Path(args.base).resolve(), Path(args.output).resolve()
    assert Path('/mnt/data') in base.parents and base in output.parents
    result = collect(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    with temporary.open('w') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps(dict(status=result['status'], actual_complete_outer=result['actual_complete_outer'],
                          rows=len(result['rows']), missing=result['missing'],
                          output=str(output), sha256=digest(output),
                          training_modified=False)))


if __name__ == '__main__':
    main()
