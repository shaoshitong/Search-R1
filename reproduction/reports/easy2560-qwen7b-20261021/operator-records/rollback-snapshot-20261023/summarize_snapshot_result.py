"""Summarize small diagnostic JSON; never read checkpoint/rollout tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(path):
    raw = json.loads(path.read_text())
    actual = json.loads(raw['stdout'])
    ranks = actual['reports']
    result = actual['benchmark_result']
    assert raw['returncode'] == 0
    assert 'COMPLETED|0:0|' in actual['accounting']['stdout']
    assert set(ranks) == {'0', '1', '2', '3'}
    assert all(row['measurement_complete'] and row['completed_constructors'] == 20 for row in ranks.values())
    assert all(row['actual_optimizer_writes'] == 20 and row['actual_training_forwards'] == 80 for row in result['ranks'])
    fields = ('constructor_body_wall_seconds_total', 'constructor_body_wall_seconds_median',
              'constructor_body_wall_seconds_min', 'constructor_body_wall_seconds_max',
              'parameter_snapshot_bytes_copied_total', 'adam_snapshot_bytes_copied_total',
              'pre_constructor_cuda_sync_seconds_total', 'snapshot_tensor_copies_all_cpu')
    return dict(schema='native4_cpu_rollback_cost_summary_v1', job_id=actual['job'],
        measured_unix=actual['epoch'], slurm=actual['accounting']['stdout'].strip(),
        operator_commit='d87d3b54c505b581250b152bccef0f01e9a5f071',
        checkpoint=result['checkpoint'], selection_step=51,
        actual_world_size=4, constructors_per_rank=20, actual_adam_writes_per_rank=20,
        training_forwards_per_rank=80, production_modified=False,
        ranks={rank: {key: row[key] for key in fields} for rank, row in ranks.items()},
        measured_fixed_work_wall_seconds=max(row['elapsed_seconds'] for row in result['ranks']),
        bytes_copied_per_rank_per_outer=ranks['0']['parameter_snapshot_bytes_copied_total'] + ranks['0']['adam_snapshot_bytes_copied_total'],
        four_rank_sum_is_wall_time=False, full_outer_contribution_proven=False,
        source_evidence=path.name, source_evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        actual_error_lines=actual['actual_error_lines'],
        limitations=['Four ranks run concurrently; never add their seconds and call that parallel wall time.',
                    'The legacy repeat includes instrumentation and a different node; do not subtract its wall from earlier modes as a pure component ablation.',
                    'Pre-constructor CUDA synchronization is excluded; exception rollback and snapshot release are not measured.',
                    'The production path is unchanged; this diagnostic executes old legacy work independently.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence), indent=2) + '\n')
