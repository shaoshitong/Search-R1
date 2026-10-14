"""Combine closed small scalar evidence without models or production changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PROTOCOL = 'recursive_bran_ablation150_cost_supplement_v1'
RUN = '260d1debb6214d75b3d1a5954d88f524'
CONFIG = 'b2a47bdbd8d721579e983062eb3ee942cc1e4b6aa021e7c49861e3763cdb10de'
SCIENCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
MILESTONES = (5, 50, 100, 150)
TOKEN_FIELDS = ('rollout/raw_emitted_tokens', 'rollout/generated_tokens')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scalar(value, name, integer=False):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid scalar: ' + name)
    require(value >= 0 and (not integer or value == int(value)), 'Invalid count/duration: ' + name)
    return int(value) if integer else float(value)


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'Missing SHA provenance: ' + name)
    return value


def bound_identity(value):
    require(value.get('run_id') == RUN and value.get('config_sha256') == CONFIG
            and value.get('scientific_source_commit') == SCIENCE, 'Original C1 identity differs')


def chosen_tokens(metrics, outer):
    # This is exactly the b95 collector's priority, not a cross-field equality.
    for key in TOKEN_FIELDS:
        if key in metrics:
            return scalar(metrics[key], 'outer%d/%s' % (outer, key), integer=True), key
    raise ValueError('Native emitted token field missing at outer ' + str(outer))


def combine(aggregate, original, capture):
    """Return a separate supplement. Failure never produces a guessed total."""
    result = dict(protocol=PROTOCOL, status='INCOMPLETE', training_action='NONE',
                  accepted_report_modified=False, supplemental_emitted_prefix=None,
                  end_to_end_cost_complete=False, exact_native_gen_prefix_seconds=None,
                  measured_GPU_active_seconds=None, allocator_billed_GPU_seconds=None)
    try:
        require(aggregate.get('protocol') == 'recursive_bran_ablation150_closed_milestone_v1',
                'Input must be an original b95 closed milestone aggregate')
        require(aggregate.get('milestone_report_complete') is True, 'Milestone report is not closed')
        label = aggregate.get('requested_label')
        result['requested_label'] = label
        if label not in ('C1-root-only', 'generation-1'):
            result.update(status='NO_FIX_REQUIRED', scope='Only the C1 original sampling gap is eligible')
            return result
        require(not aggregate.get('sampling_cost_supplement_applied'), 'Addition already applied to input')
        if label == 'generation-1':
            require(aggregate.get('reuse_of') == 'C1-root-only'
                    and aggregate.get('generated_new_training') is False, 'generation1 is not matched C1 reuse')
        else:
            require(aggregate.get('reuse_of') is None and aggregate.get('generated_new_training') is True,
                    'C1 native report provenance differs')
        milestone = aggregate.get('milestone')
        require(type(milestone) is int and milestone in MILESTONES, 'Invalid formal milestone')
        condition = aggregate['conditions'][label]
        require(condition.get('run_id') == RUN and condition.get('config_sha256') == CONFIG
                and condition.get('world_size') == 8, 'Aggregate C1 run/config/world differs')
        row = condition['milestones'][str(milestone)]
        require(row.get('outer') == milestone and row.get('complete_scalar_prefix') is True
                and row.get('scalar_coverage_count') == milestone
                and row.get('receipts_complete_through_outer') is True
                and row.get('complete_generated_token_prefix') is False,
                'Closed scalar/Adam prefix or the missing original token marker differs')
        coverage = row.get('cost_coverage', {})
        require(coverage.get('recovered_outer1_original_sampling_excluded') is True
                and coverage.get('end_to_end_coverage_complete') is False,
                'Aggregate does not explicitly exclude original recovered sampling')
        require(row.get('dev_status') == 'COMPLETE_NATIVE_AGGREGATE', 'Formal dev512 is not closed')
        bound_identity(original)
        require(original.get('protocol') == 'ablation150_original_zero_write_sampling_cost_v1'
                and original.get('status') == 'COMPLETE_RECORDED_ACTION_COUNTS_WITH_TIMING_LIMITS'
                and original.get('group') == 'C1-root-only'
                and original.get('original_outer') == 1
                and original.get('original_optimizer_calls_per_rank') == 0
                and original.get('reused_for_committed_outer1') is True
                and original.get('additional_sampling_or_training') is False,
                'Original action evidence is not the accepted single-reused zero-write sampling')
        action_coverage = original['coverage']
        require(action_coverage.get('initial_trajectories') == 512
                and action_coverage.get('selected_training_rows') == 2560
                and action_coverage.get('successful_unique_action_batches') == 32
                and action_coverage.get('generation_rpc_records') == 32
                and action_coverage.get('full_log_read') is True, 'Original physical action coverage differs')
        action = original['recorded_action_cost_sum']
        added = scalar(action['raw_emitted_tokens'], 'original emitted', integer=True)
        require(added == 834605, 'Accepted original physical emitted count differs')
        retained = scalar(action['raw_retained_tokens'], 'original retained', integer=True)
        discarded = scalar(action['raw_discarded_tail_tokens'], 'original discarded', integer=True)
        require(retained + discarded == added, 'Original retained/discarded conservation differs')
        require(action.get('context_mismatches') == 0, 'Original recorded context mismatch')
        bound_identity(capture)
        require(capture.get('protocol') == 'ablation150_closed_prefix_native_scalar_v1'
                and capture.get('recovered_outer1') is True, 'Closed recovered scalar capture differs')
        maximum = capture.get('completed_prefix_outer')
        require(type(maximum) is int and maximum in MILESTONES and maximum >= milestone,
                'Capture does not reach the requested closed milestone')
        require(capture.get('configuration_receipt_sha256') == aggregate.get('configuration_receipt_sha256'),
                'Frozen configuration receipt SHA differs')
        digest(capture['configuration_receipt_sha256'], 'configuration receipt')
        metrics = capture['prefix_metrics']
        require(set(metrics) == {str(v) for v in range(1, maximum + 1)}, 'Capture lacks exact closed scalar prefix')
        first = capture['outer1_metrics']
        require(first == metrics['1'] and all(key not in first for key in TOKEN_FIELDS),
                'First native generated/emitted fields are present or first scalar differs; no addition allowed')
        require(first.get('training/reused_saved_prepared_actor_batch') == 1
                and first.get('training/replayed_round_timing_excludes_original_sampling') == 1,
                'First native scalar lacks explicit replay sampling exclusion markers')
        for outer in range(1, maximum + 1):
            native = metrics[str(outer)]
            require(native.get('training/actual_outer_updates') == outer
                    and native.get('training/consumed_sampling_rounds') == outer,
                    'Native scalar closure/cursor differs at outer ' + str(outer))
            for key, value in native.items():
                require(type(value) in (int, float) and math.isfinite(value),
                        'Native scalar contains a nonfinite/non-numeric field')
            scalar(native.get('timing_s/step'), 'outer step wall')
        source_lines = capture['native_line_SHA_by_outer']
        require(set(source_lines) == set(metrics), 'Capture omits native line SHA coverage')
        for outer, value in source_lines.items():
            digest(value, 'native line outer' + outer)
        require(source_lines[str(milestone)] == aggregate['native_evidence']['native_log']['captured_line_sha256'],
                'Milestone native line does not match accepted b95 report')
        chosen = [chosen_tokens(metrics[str(outer)], outer) for outer in range(2, milestone + 1)]
        recorded = sum(value for value, _ in chosen)
        require(recorded == scalar(row['generated_tokens_recorded_sum'], 'accepted emitted prefix', integer=True),
                'Native prefix sum differs from accepted report; potential double addition or another gap')
        native_steps = sum(scalar(metrics[str(outer)]['timing_s/step'], 'native step wall')
                           for outer in range(1, milestone + 1))
        stages = row['recorded_stage_seconds_sum']
        require(math.isclose(native_steps, scalar(stages['timing_s/step'], 'accepted recorded step wall'),
                             rel_tol=1e-12, abs_tol=1e-7), 'Native wall prefix differs from accepted report')
        times = original['time_scopes']
        interval = scalar(times['first_RPC_to_compose_end_wall_seconds'], 'first RPC to compose wall')
        require(times.get('exact_native_timing_s_gen') is None
                and times.get('measured_GPU_active_seconds') is None
                and times.get('allocator_billed_GPU_seconds') is None, 'Original timing limits changed')
        result.update(status='COMPLETE_RECORDED_EMITTED_PREFIX_SUPPLEMENT', milestone=milestone,
            run_id=RUN, config_sha256=CONFIG, scientific_source_commit=SCIENCE,
            recovered_original_outer=1, added_once=True, generated_new_training=False,
            existing_recorded_emitted_prefix=recorded, original_physical_emitted_added_once=added,
            supplemental_emitted_prefix=recorded + added, complete_physical_emitted_prefix=True,
            native_field_priority=list(TOKEN_FIELDS),
            compatibility_generated_fallback_outers=[outer for outer, (_, key) in zip(range(2, milestone + 1), chosen)
                                                     if key == TOKEN_FIELDS[1]],
            original_action_coverage=action_coverage,
            original_retained_tokens=retained, original_discarded_tokens=discarded,
            recorded_native_outer_wall_seconds_unchanged=native_steps,
            recorded_native_gen_seconds_unchanged=stages.get('timing_s/gen'),
            recorded_outer_wall_gpu_hours_proxy_unchanged=row.get('allocated_gpu_hours_from_recorded_outer_wall'),
            separate_original_first_RPC_to_compose_seconds=interval,
            original_compose_seconds=scalar(times['compose_seconds'], 'original compose'),
            original_RPC_duration_sum_seconds=scalar(action['generation_rpc_wall_seconds'], 'original RPC sum'),
            original_prefill_tokens=scalar(action['prefill_tokens'], 'original prefill', integer=True),
            original_closed_log_sha256=digest(original['evidence']['full_closed_log']['sha256'], 'original log'),
            scope='Supplement only the one omitted original sampling token count; immutable b95 report and EM are unchanged.',
            limits=[
                'Physical emitted/prefill tokens include padding and discarded tails; not selected-leaf effective tokens.',
                'Native raw_emitted_tokens has priority; generated_tokens is only the compatibility fallback. Both fields need not have the same meaning.',
                'The native recorded outer wall/gen prefix excludes original pre-fault sampling and remains unchanged.',
                'The separate first-RPC-to-compose interval omits pre-first-RPC setup and is not exact native gen duration.',
                'Action/RPC/HTTP/compose components contain or overlap one another; they are not additive allocation billing.',
                'Original zero-Adam sampling was reused once as committed outer1 input; no extra training outer or sampling is credited.',
                'GPU active time and allocator billing remain unknown; complete emitted coverage does not mean complete end-to-end timing.',
            ])
    except (ValueError, KeyError, TypeError) as error:
        result.update(status='INCOMPLETE', reason=type(error).__name__ + ': ' + str(error))
    return result


def load(path):
    path = Path(path)
    require(path.stat().st_size <= 4 * 1024 * 1024, 'Only small aggregate/scalar JSON inputs are allowed')
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write(result, destination):
    """An independent namespace is required; idempotency never adds a supplement again."""
    destination = Path(destination)
    require(destination.name.startswith('cost-supplement-'), 'Use an independent cost-supplement-* directory')
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / 'summary.json'
    if summary.exists():
        previous = json.loads(summary.read_text())
        require(previous == result, 'Existing supplement differs; preserve it and use a new namespace')
        return
    if result['status'] == 'COMPLETE_RECORDED_EMITTED_PREFIX_SUPPLEMENT':
        note = ('# C1 首轮采样成本补充\n\n'
            f"第 {result['milestone']} 轮已闭合报告原记录物理 emitted token "
            f"{result['existing_recorded_emitted_prefix']:,}，只补回原首轮缺失的 "
            f"{result['original_physical_emitted_added_once']:,}，累计为 **{result['supplemental_emitted_prefix']:,}**。"
            '该原采样在零 Adam 写入时发生目录故障，2,560 行批次后来只复用一次；不增加训练轮数或采样次数。\n\n'
            f"已有 native outer wall 累计 {result['recorded_native_outer_wall_seconds_unchanged']:.3f} 秒保持原值；"
            f"原采样另有首 RPC 至 compose 结束窗口 {result['separate_original_first_RPC_to_compose_seconds']:.3f} 秒，"
            f"其中 RPC duration 累计 {result['original_RPC_duration_sum_seconds']:.3f} 秒、compose "
            f"{result['original_compose_seconds']:.3f} 秒。这些包含或重叠的时间不直接相加。"
            '窗口遗漏首 RPC 前准备，精确 native gen、GPU 活跃和资源计费时间仍未知。\n\n'
            '只补物理 token 计数；原报告、EM、已接受源码和训练均未修改。物理计数包括补齐与丢弃尾部，'
            '不等同训练叶子的有效 token。\n')
    else:
        note = '# 采样成本补充\n\n' + result['status'] + '：' + result.get('reason', result.get('scope', '输入不足')) + '\n\n未制造补充总量，未修改训练或已有报告。\n'
    (destination / 'findings-zh.md').write_text(note)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aggregate', required=True)
    parser.add_argument('--original-cost', required=True)
    parser.add_argument('--prefix-scalars', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    try:
        aggregate, a_sha = load(args.aggregate)
        original, o_sha = load(args.original_cost)
        capture, c_sha = load(args.prefix_scalars)
        result = combine(aggregate, original, capture)
        result['input_sha256'] = dict(closed_milestone_aggregate=a_sha, original_sampling_cost=o_sha,
                                     closed_native_prefix_scalars=c_sha)
        write(result, args.destination)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(protocol=PROTOCOL, status='INCOMPLETE', reason=type(error).__name__ + ': ' + str(error),
                      supplemental_emitted_prefix=None, training_action='NONE', accepted_report_modified=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result['status'] in ('COMPLETE_RECORDED_EMITTED_PREFIX_SUPPLEMENT', 'NO_FIX_REQUIRED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
