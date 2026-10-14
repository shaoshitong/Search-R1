"""Produce a small report from the existing local A scalar capture, not Node logs."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

CAPTURE_SHA = '9629119492d24e50c8119cc67b57e9253a0b0f82eeeb21e76a5715948b8a1989'
CAPTURE_NAME = 'current-user-effects.json'
LOCAL_PROTOCOL = 'retained_A_local_native_scalar_capture100_cost_v1'


def reader_module():
    path = Path(__file__).with_name('read_retained_a_scalar_cost.py')
    spec = importlib.util.spec_from_file_location('retained_a_cost_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read_stable(path, expected, limit):
    path = Path(path).resolve()
    before = path.stat()
    if not path.is_file() or before.st_size > limit:
        raise ValueError('Only the bounded existing local scalar capture is allowed')
    raw = path.read_bytes()
    after = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    sha = hashlib.sha256(raw).hexdigest()
    if signature(before) != signature(after) or len(raw) != before.st_size or sha != expected:
        raise ValueError('Pinned local retained scalar evidence differs')
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=sha)


def summarize(capture_path, pack_path):
    reader, reader_proof = reader_module()
    if Path(capture_path).name != CAPTURE_NAME:
        raise ValueError('This mode only accepts the retained current-user-effects scalar capture')
    capture, capture_proof = read_stable(capture_path, CAPTURE_SHA, 1024 * 1024)
    pack, pack_proof = read_stable(pack_path, reader.PACK_SHA, 128 * 1024)
    if capture['entry']['run_id'] != reader.ORIGINAL_RUN:
        raise ValueError('Local retained entry belongs to another A run')
    if pack['protocol'] != 'retained_A_native_aggregate_pack_v1' or pack['new_inference'] is not False:
        raise ValueError('Local boundary pack is not the existing retained A reference')
    rows = {}
    for index in range(1, 101):
        recorded = capture['emitted_outer_metrics'][str(index)]
        values = {key: value for key, value in recorded.items()
                  if key in reader.COUNT_FIELDS or key in reader.ROLLOUT_SECONDS
                  or key in reader.ACCOUNTING or key.startswith('timing_s/')}
        for key, value in values.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid retained scalar: ' + key)
            if key in reader.COUNT_FIELDS or key in reader.ACCOUNTING:
                if value != int(value):
                    raise ValueError('Noninteger retained native count: ' + key)
                values[key] = int(value)
        if (any(key not in values for key in reader.REQUIRED)
                or values['training/actual_outer_updates'] != index
                or values['training/consumed_sampling_rounds'] != index
                or values['actor/scheduler_step_after_outer'] != index):
            raise ValueError('Retained local scalar prefix lacks complete outer/accounting coverage')
        rows[index] = dict(outer=index, metrics=values)
    boundary = {}
    for index in (5, 50, 100):
        accepted = pack['milestones'][str(index)]
        if any(capture['emitted_outer_metrics'][str(index)].get(key) != value
               for key, value in accepted['recorded_boundary_scalars'].items()):
            raise ValueError('Local retained boundary scalars differ from the existing A pack')
        boundary[str(index)] = dict(retained_boundary_scalars_equal=True,
            retained_native_log=accepted['source']['native_log'],
            retained_captured_line_sha256=accepted['source']['captured_line_sha256'],
            native_log_line_hash_freshly_reverified=False)
    prefixes = {str(index): reader.prefix(rows, index, True) for index in (5, 50, 100)}
    for value in prefixes.values():
        value['status'] = 'COMPLETE_RETAINED_LOCAL_NATIVE_SCALAR_CAPTURE_PREFIX'
        value['ancestry_identity_verified_by_fresh_node_reads'] = False
    source = Path(__file__).resolve()
    return dict(protocol=LOCAL_PROTOCOL, time=time.time(),
        status='COMPLETE_RETAINED_LOCAL_NATIVE_SCALAR_CAPTURE100', goal_complete=False,
        completed_local_scalar_coverage=100, captured_entry={key: capture['entry'][key]
            for key in ('time', 'pid', 'run_id', 'job')},
        capture_time=capture['time'], provenance=dict(capture=capture_proof,
            reference_pack=pack_proof, scalar_reader=reader_proof,
            summarizer=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
            fresh_local_file_sha=True, fresh_node_log_sha=False,
            original_complete_transcript_present=False),
        boundary_pack_matches=boundary, prefixes=prefixes,
        limits=['Only the already retained local native scalar capture1..100 is summarized. '
                'Captured original101..105 are excluded because the accepted final ancestry later resumes from100.',
                'No150 coverage claim, Node access, new A inference, question capture or native log-line '
                'hash reconstruction is performed in this local mode.',
                'Phase timers may overlap. Do not add them to total step. Scheduled dev/save may be inside step.',
                'Eight times recorded step wall is a configured-GPU-hour proxy, not active GPU utilization '
                'or total allocation/lease billing.',
                'Recorded raw_emitted_tokens follow native padding/discarded-tail conventions; '
                'they are not independent token reconstruction or effective training-token counts.',
                'Failed launches, discarded101..118/partial119 and all historical rework/startup/downtime '
                'costs are NOT covered. Actual Node two-segment identity/log evidence belongs to the separate reader.'])


def markdown(summary):
    lines = ['# A 前100轮：已有本地原生标量成本', '',
             '使用已保留的 `current-user-effects.json`，原始A运行 `1c806d…`、job3389。'
             '1–100轮标量连续、有限，5/50/100轮边界与已有A参考包一致。', '',
             '| prefix | 已记录 emitted token | step wall（小时） | 卡数×step（GPU小时代理） | 平均step（分钟/轮） |',
             '| --- | ---: | ---: | ---: | ---: |']
    for outer, prefix in summary['prefixes'].items():
        sums = prefix['recorded_prefix_sums']
        wall = sums['timing_s/step']
        lines.append(f"| {outer} | {sums['rollout/raw_emitted_tokens']:,} | {wall/3600:.4f} | "
                     f"{prefix['allocated_gpu_hours_from_recorded_outer_wall']:.4f} | {wall/int(outer)/60:.4f} |")
    lines += ['', '| prefix | gen（小时） | ref（小时） | update actor（小时） |',
              '| --- | ---: | ---: | ---: |']
    for outer, prefix in summary['prefixes'].items():
        sums = prefix['recorded_prefix_sums']
        lines.append(f"| {outer} | {sums['timing_s/gen']/3600:.4f} | "
                     f"{sums['timing_s/ref']/3600:.4f} | {sums['timing_s/update_actor']/3600:.4f} |")
    lines += ['', '阶段时间可能重叠，不能相加成为总耗时；step可能包括规定的评测/保存。'
              'GPU小时仅按8卡×已记录step wall计算。token采用原生字段口径，包含padding/丢弃尾部的记录约定。', '',
              '**范围仅为已有本地标量capture100。** 没有重新读Node完整日志、没有新A评测、没有150轮成本结论。'
              '本地保存的是标量摘要，不能称完整原始transcript；失败、重算、启动、停机和总租用成本均未覆盖。', '',
              '本地capture SHA：`' + summary['provenance']['capture']['sha256'] + '`。'
              '逐字段覆盖与完整来源见summary.json。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-pack', type=Path, default=Path(__file__).resolve().parents[1] /
                        'ablation150_aggregate_watch_v1/a-reference-pack.json')
    args = parser.parse_args()
    result = summarize(args.capture, args.reference_pack)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.output / 'findings-zh.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'], output=str(args.output.resolve()),
        files={name: dict(bytes=(args.output / name).stat().st_size,
                         sha256=hashlib.sha256((args.output / name).read_bytes()).hexdigest())
               for name in ('summary.json', 'findings-zh.md')})), flush=True)


if __name__ == '__main__':
    main()
