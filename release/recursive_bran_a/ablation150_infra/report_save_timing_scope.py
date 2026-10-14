"""Project accepted small save-timing receipts into a separate public report."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path, status):
    raw = Path(path).read_bytes()
    assert len(raw) < 1024 * 1024
    value = json.loads(raw)
    assert value['protocol'] == 'ablation150_closed_save_cost_readonly_v1'
    assert value['status'] == status
    assert value['Node_writes'] is False and value['training_action'] == 'NONE'
    return value, hashlib.sha256(raw).hexdigest()


def project(value):
    points = {}
    for outer, point in value['keypoint_worker_save_receipts'].items():
        phases = {}
        for phase in ('pre', 'post'):
            row = point[phase]
            assert set(row['worker_checkpoint_and_reload_seconds_by_rank']) == {str(i) for i in range(8)}
            phases[phase] = {key: row[key] for key in (
                'phase', 'worker_checkpoint_and_reload_seconds_by_rank',
                'worker_reload_verified_by_rank', 'worker_seconds_min',
                'worker_seconds_max', 'worker_seconds_sum_overlapping', 'scope')}
            phases[phase]['manifest_sha256'] = row['manifest']['sha256']
        points[outer] = phases
    identity = {key: value[key] for key in ('run_id', 'config_sha256', 'scientific_source_commit') if key in value}
    if 'segments' in value:
        identity['segments'] = [row['identity'] for row in value['segments'].values()]
    return dict(status=value['status'], group=value['group'], milestone=value['milestone'],
        identity=identity, keypoint_worker_save_receipts=points,
        rolling_coordinator_seconds_by_outer=value.get('rolling_coordinator_save_seconds_by_outer'),
        rolling_save_seconds_sum_outside_native_step=value.get('rolling_save_seconds_sum_outside_native_step'),
        exact_HF_export_wall_seconds=None, exact_keypoint_coordinator_save_seconds=None,
        measured_GPU_active_seconds=None, allocator_billed_GPU_seconds=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c1', required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    c1, c_sha = load(args.c1, 'COMPLETE_RECORDED_ROLLING_PREFIX_WITH_KEYPOINT_TIMING_LIMITS')
    a, a_sha = load(args.a, 'COMPLETE_RETAINED_A_KEYPOINT_WORKER_TIMINGS_WITH_WALL_LIMITS')
    assert c1['group'] == 'C1-root-only' and c1['milestone'] == 5
    assert a['group'] == 'A-original' and a['milestone'] == 150
    result = dict(protocol='recursive_ablation_recorded_save_timing_public_v1',
        C1_first5=project(c1), A_keypoints_through150=project(a),
        input_sha256=dict(C1=c_sha, A=a_sha), accepted_native_reports_modified=False,
        training_action='NONE', scopes=[
            'C1 rolling timer includes parallel worker RPC, driver/manifest and rolling reclamation; outside native step.',
            'Native save_checkpoint timing at protected evaluation is a no-op hook, not actual saving cost.',
            'Protected testing includes HF export, pre native save, evaluation and post native save; already inside step.',
            'Worker checkpoint/reload timers run concurrently; their sum is not coordinator wall.',
            'First5 pre-evaluation worker timers include reload verification; post timers and later points do not.',
            'Exact HF export, keypoint coordinator wall and A rolling-save prefix are unrecorded.',
            'No component sums reconstruct total end-to-end time, GPU activity or allocation billing.'])
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    path = destination / 'summary.json'
    if path.exists():
        assert path.read_text() == raw, 'Keep earlier reports immutable; use a new directory'
    else:
        path.write_text(raw)
    note = ('# 实际保存耗时与计时范围\n\n'
        f"C1 前5轮的四次非评测 rolling 保存累计 **{c1['rolling_save_seconds_sum_outside_native_step']:.3f} 秒**，"
        '各次约33.35–43.34秒。该计时包含八rank保存RPC、driver/manifest及旧rolling回收，发生在原生step计时之外。\n\n'
        '原生 `timing_s/save_checkpoint` 的微秒数是受保护评测路径里的空钩子，不能解释为真实保存只需微秒。'
        '第5轮的testing已包含HF导出、评测前native保存、512评测和评测后native保存；不能再把这些分量重复加到step。\n\n'
        'C1第5轮评测前各rank保存加reload为52.38–55.84秒，评测后保存为33.96–35.82秒。'
        'A的5/50/100/150节点逐rank计时见summary.json。八rank并行，不能将其和当成wall。\n\n'
        'HF导出和关键节点coordinator保存没有独立完整wall记录；A的rolling保存prefix也未恢复。'
        '这些项保持未知，不用减法或相加制造端到端耗时、GPU活跃时间或计费。此报告只投影已有小收据，没有改变训练、评测或旧报告。\n')
    (destination / 'findings-zh.md').write_text(note)
    print(json.dumps(dict(status='SAVED_PUBLIC_TIMING_SCOPE', files=2, bytes=len(raw.encode()))))


if __name__ == '__main__':
    main()
