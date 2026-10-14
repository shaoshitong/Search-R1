"""Build a small public C1 report from existing accepted CPU summaries only."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aggregate', 'prefix-scalars', 'cost-supplement', 'saved-cost',
                 'tree-summary', 'a-retained-cost', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--a-reference-pack', type=Path, default=
        Path(__file__).resolve().parents[1] / 'ablation150_aggregate_watch_v1/a-reference-pack.json')
    args = parser.parse_args()
    inputs = {name: getattr(args, name.replace('-', '_')) for name in
              ('aggregate', 'prefix-scalars', 'cost-supplement', 'saved-cost',
               'tree-summary', 'a-retained-cost', 'a-reference-pack')}
    data = {name: load(path) for name, path in inputs.items()}
    aggregate, prefix, cost = (data[name] for name in
                               ('aggregate', 'prefix-scalars', 'cost-supplement'))
    saved, tree, retained, reference = (data[name] for name in
        ('saved-cost', 'tree-summary', 'a-retained-cost', 'a-reference-pack'))
    outer = aggregate['milestone']
    assert outer in (5, 50, 100, 150) and aggregate['milestone_report_complete']
    assert aggregate['requested_label'] == 'C1-root-only'
    assert prefix['status'] == 'COMPLETE_EXISTING_NATIVE_SCALAR_PREFIX'
    assert prefix['completed_prefix_outer'] == outer
    assert cost['status'] == 'COMPLETE_RECORDED_EMITTED_PREFIX_SUPPLEMENT'
    assert cost['input_sha256']['closed_milestone_aggregate'] == sha(inputs['aggregate'])
    assert cost['input_sha256']['closed_native_prefix_scalars'] == sha(inputs['prefix-scalars'])
    assert saved['status'] == 'COMPLETE_RECORDED_ROLLING_PREFIX_WITH_KEYPOINT_TIMING_LIMITS'
    assert saved['closed_prefix_outer'] == outer
    assert saved['rolling_save_coverage_outers'] == saved['expected_rolling_outers']
    assert tree['outer'] == outer and tree['tree']['status'] == 'COMPLETE_CLOSED_TREE512'
    assert reference['protocol'] == 'retained_A_native_aggregate_pack_v1'
    assert reference['new_inference'] is False
    assert retained['status'] == 'COMPLETE_EXISTING_A_ANCESTRY_SCALAR150_REPORT'
    condition = aggregate['conditions']['C1-root-only']
    assert condition['world_size'] == 8
    point = condition['milestones'][str(outer)]
    for record in (prefix, cost, saved, tree):
        assert record['run_id'] == condition['run_id']
        assert record['config_sha256'] == condition['config_sha256']
    assert prefix['scientific_source_commit'] == cost['scientific_source_commit'] == saved['scientific_source_commit'] == tree['source_commit']
    assert tree['tree']['questions'] == 512
    assert all(int(generation) == 1 for generation, count in
               tree['tree']['candidate_generations'].items() if count)
    metrics = prefix['prefix_metrics']
    assert set(metrics) == {str(i) for i in range(1, outer + 1)}
    assert all(math.isfinite(value) for row in metrics.values()
               for value in row.values() if isinstance(value, (int, float)))
    curve = []
    for step in (5, 50, 100, 150):
        if step > outer:
            continue
        native, baseline = metrics[str(step)], reference['milestones'][str(step)]['dev512']
        assert native['val/qa_count/all'] == baseline['count'] == 512
        hit_values = [native['val/em/all'] * 512, native['val/em/nq'] * 256,
                      native['val/em/hotpotqa'] * 256]
        assert all(abs(v - round(v)) < 1e-8 for v in hit_values)
        c_hits, nq, hotpot = map(lambda v: int(round(v)), hit_values)
        assert c_hits == nq + hotpot
        for label, hits, n, h in (
                ('A-original', baseline['hits'], baseline['nq']['hits'], baseline['hotpot']['hits']),
                ('C1-root-only', c_hits, nq, hotpot)):
            curve.append(dict(condition=label, outer=step, hits=hits, count=512,
                em=hits / 512, nq_hits=n, hotpot_hits=h,
                aggregate_difference_pp=(hits - baseline['hits']) / 512 * 100))
    current, baseline = point['dev512'], aggregate['conditions']['A-original']['milestones'][str(outer)]['dev512']
    assert curve[-1]['hits'] == current['all']['hits']
    assert curve[-2]['hits'] == baseline['all']['hits']
    assert curve[-1]['nq_hits'] == current['nq']['hits']
    assert curve[-1]['hotpot_hits'] == current['hotpotqa']['hits']
    assert curve[-2]['nq_hits'] == baseline['nq']['hits']
    assert curve[-2]['hotpot_hits'] == baseline['hotpotqa']['hits']
    a_cost = retained['prefixes'][str(outer)]
    assert a_cost['physical_emitted_token_prefix_complete']
    history = []
    fields = {'actual_adam_each_rank': 'training/actual_adam_calls_per_rank',
        'updates_this_outer': 'actor/optimizer_steps',
        'skips_this_outer': 'actor/optimizer_opportunities_skipped',
        'pg_loss': 'actor/pg_loss', 'selected_train_reward': 'critic/score/mean',
        'lr_before': 'actor/lr_before_outer', 'lr_after': 'actor/lr',
        'native_gen_seconds': 'timing_s/gen', 'native_update_seconds': 'timing_s/update_actor',
        'native_step_seconds': 'timing_s/step', 'raw_emitted_tokens_recorded': 'rollout/raw_emitted_tokens',
        'dev_em_if_milestone': 'val/em/all'}
    for step in range(1, outer + 1):
        history.append(dict(outer=step, **{label: metrics[str(step)].get(key)
                                          for label, key in fields.items()}))
    loss = [row['pg_loss'] for row in history if row['pg_loss'] is not None]
    assert all(math.isfinite(value) for value in loss)
    summary = dict(protocol='recursive_C1_closed_milestone_public_v1', milestone=outer,
        goal_complete=False, comparison_scope='Aggregate EM only; no paired/bootstrap/new A evaluation',
        scientific_source_commit=prefix['scientific_source_commit'], run_id=condition['run_id'],
        config_sha256=condition['config_sha256'], dev512={'A-original': baseline, 'C1-root-only': current},
        aggregate_difference_pp=(current['all']['hits'] - baseline['all']['hits']) / 512 * 100,
        actual_adam_each_rank=point['receipt_adam_each_rank'], skips_each_rank=point['receipt_skips_each_rank'],
        native_scalar_prefix_outers=outer, pg_loss={'first': history[0]['pg_loss'],
            'last': history[-1]['pg_loss'], 'observed_outers': len(loss),
            'minimum': min(loss) if loss else None, 'maximum': max(loss) if loss else None,
            'all_recorded_finite': True},
        cost={'A-original': {'retained_emitted_tokens': a_cost['recorded_prefix_sums']['rollout/raw_emitted_tokens'],
                'native_step_seconds': a_cost['recorded_prefix_sums']['timing_s/step'],
                'world_times_native_wall_GPU_hours_proxy': a_cost['allocated_gpu_hours_from_recorded_outer_wall'],
                'actual_adam_each_rank_recorded': a_cost['actual_adam_each_rank'],
                'rolling_save_prefix_seconds': None, 'historical_rework_covered': False},
            'C1-root-only': {'retained_emitted_tokens': cost['supplemental_emitted_prefix'],
                'original_emitted_added_once': cost['original_physical_emitted_added_once'],
                'native_step_seconds': cost['recorded_native_outer_wall_seconds_unchanged'],
                'world_times_native_wall_GPU_hours_proxy': cost['recorded_outer_wall_gpu_hours_proxy_unchanged'],
                'rolling_save_prefix_seconds_outside_native_step': saved['rolling_save_seconds_sum_outside_native_step'],
                'rolling_save_covered_outers': saved['rolling_save_coverage_outers'],
                'separate_original_first_RPC_to_compose_seconds': cost['separate_original_first_RPC_to_compose_seconds']},
            'end_to_end_time_complete': False, 'GPU_active_or_billed_time': None,
            'scope': 'Retained ancestry; padding/discarded tails included. Native clocks and separate rolling clocks differ. Historical infra differs; no causal speedup ratio.'},
        tree=tree['tree'], inherited_policy_occurrence_fraction=tree['costs']['inherited_policy_occurrence_fraction'],
        post_evaluation_manifest_sha256=aggregate['native_evidence']['boundary']['sha256'],
        limitations=['Single seed and sparse formal milestones do not establish stable causal superiority.',
            'Selected TRAIN reward and PPO loss are not fixed heldout EM; cohorts differ.',
            'Recorded Adam/finite scalars are not an independent tensor/logit/gradient reload.',
            'C1 original sampling token count is supplemented once; full original timing and GPU billing remain unknown.',
            'Branch elapsed times overlap; inherited fraction is over all selected policy occurrences, not mean child prefix length.'])
    source = Path(__file__).resolve()
    repo = source.parents[3]
    relative = source.relative_to(repo).as_posix()
    commit = subprocess.run(['git', 'log', '-1', '--format=%H', '--', relative],
                            cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    committed = subprocess.run(['git', 'show', f'{commit}:{relative}'], cwd=repo,
                               capture_output=True, check=True).stdout
    assert committed == source.read_bytes(), 'Report writer contains uncommitted edits'
    provenance = dict(writer_source_commit=commit, writer_source_sha256=sha(source),
        input_sha256={key: sha(value) for key, value in inputs.items()},
        additional_training=False, additional_evaluation=False, model_or_tensor_reads=False,
        remote_actions=False, accepted_input_reports_modified=False)
    table = '\n'.join(f"| {row['condition']} | {row['outer']} | {row['hits']}/512 ({row['em']*100:.4f}%) | {row['nq_hits']}/256 | {row['hotpot_hits']}/256 |"
                      for row in curve)
    delta = summary['aggregate_difference_pp']
    direction = '高于' if delta > 0 else '低于' if delta < 0 else '等于'
    remaining = [step for step in (5, 50, 100, 150) if step > outer]
    next_text = ('后续仅按' + '、'.join(map(str, remaining)) + '正常节点继续评测'
                 if remaining else '本组计划节点已齐，继续后续C2及唯一参数组')
    findings = f'''# C1取消多代递归：截至第{outer}轮

| 条件 | 轮次 | 总体EM | NQ | Hotpot |
| --- | --- | --- | --- | --- |
{table}

同一固定512题（NQ/Hotpot各256）、保留的原生评测协议：第{outer}轮C1汇总EM{direction}A，差值{delta:+.5f}个百分点。这是汇总差值，不是逐题新增/丢失统计；A使用已有结果，没有重评。当前单seed与稀疏节点不能确定最终或跨seed优劣。总目标仍为全部计划实验各150轮；整个goal尚未完成。

本节点树记录包含512题、{tree['tree']['candidates']}个真实非根候选；候选代数分布为{tree['tree']['candidate_generations']}，选中代数分布为{tree['tree']['selected_generations']}，shortfall={tree['tree']['shortfalls']}。这支持实际执行了root-only干预；奖励差异接纳仍按原规则执行，C2随后单独测试最终筛选。

累计实际Adam/rank={point['receipt_adam_each_rank']}，跳过/rank={point['receipt_skips_each_rank']}；完整{outer}轮原生标量均有限。逐轮PPO loss、所选TRAIN奖励和阶段计时见native-scalar-history.csv；这些不是固定验证曲线，也不是独立重算梯度或Adam张量。

保留训练链的实际emitted token：A={summary['cost']['A-original']['retained_emitted_tokens']:,}；C1={cost['supplemental_emitted_prefix']:,}，其中原首轮834,605仅补一次。包含padding和丢弃尾部，不能当作有效训练token。原生step累计：A={summary['cost']['A-original']['native_step_seconds']:.3f}秒；C1={cost['recorded_native_outer_wall_seconds_unchanged']:.3f}秒。C1另有{saved['rolling_save_seconds_sum_outside_native_step']:.3f}秒滚动保存（{saved['rolling_save_coverage_outers']}轮）在step之外；关键保存/评测已计入原生step，不能重复相加。原首轮401.640秒RPC至compose窗口单列，缺少其前准备。A滚动前缀、精确GPU活跃/计费和完整端到端成本仍未知；历史infra不同，不据这些数值宣称因果提速。

模型、optimizer、原始树/轨迹和完整预测留远端；此目录只有小型聚合与来源SHA。{next_text}，不追加A重测或逐题补测。
'''
    outputs = {'summary.json': json_bytes(summary), 'provenance.json': json_bytes(provenance),
               'curve.csv': csv_bytes(curve), 'native-scalar-history.csv': csv_bytes(history),
               'findings-zh.md': findings.encode()}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        path = args.output / name
        if path.exists():
            assert path.read_bytes() == raw, f'Existing report differs: {name}'
        else:
            path.write_bytes(raw)
    print(json.dumps(dict(status='COMPLETE_PUBLIC_C1_MILESTONE_REPORT', milestone=outer,
        files=len(outputs), bytes=sum(map(len, outputs.values())), goal_complete=False,
        training_action='NONE', evaluation_action='NONE')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
