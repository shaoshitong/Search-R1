"""Write a small public report from the accepted existing A scalar-reader JSON."""
import argparse
import hashlib
import json
from pathlib import Path

INPUT_SHA = 'bef0e5b58514a238a9b1114fa7becc78fc155cb2b3fd2ab3064271affbb2c04b'
MILESTONES = ('5', '50', '100', '150')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def proof(value):
    return {key: value[key] for key in ('bytes', 'sha256')}


def build(input_path):
    before = input_path.stat()
    if not input_path.is_file() or before.st_size > 512 * 1024:
        raise ValueError('Only the accepted bounded small A-reader result is allowed')
    raw = input_path.read_bytes()
    after = input_path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or sha(raw) != INPUT_SHA):
        raise ValueError('Accepted A-reader input differs')
    value = json.loads(raw)
    if (value['protocol'] != 'retained_A_final_ancestry_scalar_cost_v1'
            or value['status'] != 'COMPLETE_RETAINED_ANCESTRY_SCALAR150'
            or value['errors'] or set(value['prefixes']) != set(MILESTONES)
            or value['boundary_pack_line_and_scalar_match'] != dict.fromkeys(MILESTONES, True)):
        raise ValueError('A-reader result lacks complete accepted ancestry scalar150 coverage')
    identity = value['identities']
    if (identity['native_worker_driver_restore_receipt_accepted'] is not True
            or identity['independent_tensor_reload_performed'] is not False):
        raise ValueError('A-reader native receipt scope differs')
    prefixes = {}
    for outer in MILESTONES:
        original = value['prefixes'][outer]
        if (original['status'] != 'COMPLETE_RETAINED_ANCESTRY_SCALARS'
                or original['observed_closed_outers'] != int(outer) or original['missing_outers']
                or original['historical_rework_cost_covered'] is not False):
            raise ValueError('A-reader required prefix is incomplete')
        prefixes[outer] = {key: original[key] for key in (
            'outer', 'status', 'observed_closed_outers', 'missing_outers',
            'per_scalar_outer_coverage', 'recorded_prefix_sums',
            'physical_emitted_token_prefix_complete',
            'allocated_gpu_hours_from_recorded_outer_wall', 'actual_adam_each_rank',
            'historical_rework_cost_covered')}
    segments = {}
    for name in ('original', 'recovery'):
        row = identity[name]
        segments[name] = {key: row[key] for key in (
            'run_id', 'config_sha256', 'declared_source_commit', 'data_identity')}
        segments[name]['small_receipt_proofs'] = {key: proof(row[key]) for key in (
            'configuration_receipt', 'resolved_config', 'entry_receipt')}
        log = value['logs'][name]
        segments[name]['retained_log'] = {key: log[key] for key in (
            'bytes', 'sha256', 'retained_outer_range', 'identical_repeated_scalar_lines',
            'full_source_log_read', 'line_hash_convention')}
    writer = Path(__file__).read_bytes()
    return dict(protocol='retained_A_native_ancestry_scalar150_public_report_v1',
        status='COMPLETE_EXISTING_A_ANCESTRY_SCALAR150_REPORT', goal_complete=False,
        observed_reader_time=value['time'], prefix_outer_count=150,
        configured_gpu_count=8, segments=segments, prefixes=prefixes,
        boundary_pack_line_and_scalar_match=value['boundary_pack_line_and_scalar_match'],
        provenance=dict(input_reader_result=dict(bytes=len(raw), sha256=sha(raw)),
            writer_source=dict(bytes=len(writer), sha256=sha(writer)),
            retained_boundary_pack=proof(value['retained_boundary_pack']),
            native_chain_receipts={key: proof(identity[key]) for key in (
                'parent_post100', 'continuation_binding', 'migration', 'final_post150')},
            native_worker_driver_restore_receipt_accepted=True,
            independent_tensor_reload_performed=False),
        scope=value['scopes'],
        runtime_timing_warning='历史两段运行的infra/runtime及资源状态存在差异。耗时差是观察结果，'
            '不能仅据此证明算法造成速度变化。',
        public_projection='Prefix aggregates and bounded source SHA provenance only; '
            'Node/local absolute paths, inodes, mtimes, PIDs and all150 raw scalar rows are omitted.')


def markdown(summary):
    lines = ['# A 前150轮：已有原生训练链的标量成本', '',
        '已读取既有日志和小型状态票据，没有新增A训练或评测。前1–100轮来自原始运行，'
        '101–150轮来自实际接受的saved100恢复运行；两段run/config/source分别保存。'
        '5/50/100/150边界的原始记录行SHA与参考包一致。', '',
        '| prefix | emitted token | 真实Adam/rank | step wall（小时） | 8×step（GPU小时代理） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for outer in MILESTONES:
        prefix = summary['prefixes'][outer]
        sums = prefix['recorded_prefix_sums']
        lines.append(f"| {outer} | {sums['rollout/raw_emitted_tokens']:,} | {prefix['actual_adam_each_rank']} | "
                     f"{sums['timing_s/step']/3600:.6f} | "
                     f"{prefix['allocated_gpu_hours_from_recorded_outer_wall']:.6f} |")
    lines += ['', '| prefix | gen（小时） | ref（小时） | update actor（小时） |',
              '| --- | ---: | ---: | ---: |']
    for outer in MILESTONES:
        sums = summary['prefixes'][outer]['recorded_prefix_sums']
        lines.append(f"| {outer} | {sums['timing_s/gen']/3600:.6f} | "
                     f"{sums['timing_s/ref']/3600:.6f} | {sums['timing_s/update_actor']/3600:.6f} |")
    lines += ['', '阶段时间可能重叠，不能相加成为总耗时；step采用原生日志口径，可能包含规定的评测与保存。'
              'emitted token采用原生padding/丢弃尾部记录约定，不等于有效训练token。', '',
              '**8×step仅为记录的配置GPU时间代理**，不是GPU活跃时间、Slurm租用时长或计费结论。'
              '原始101–118与partial119以及更早失败启动的重算、停机、初始化成本均未覆盖，不能把这里的累计值称为全部资源消耗。', '',
              summary['runtime_timing_warning'], '',
              '公开JSON仅保留prefix汇总、逐字段覆盖与来源SHA；没有绝对路径、PID或150轮逐行原始内容。'
              '独立Adam/tensor反序列化并未在此次reader中进行。', '',
              '接受的reader结果SHA：`' + summary['provenance']['input_reader_result']['sha256'] + '`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.output / 'findings-zh.md').write_text(markdown(summary))
    print(json.dumps(dict(status=summary['status'], files={name: dict(
        bytes=(args.output / name).stat().st_size,
        sha256=sha((args.output / name).read_bytes())) for name in ('summary.json', 'findings-zh.md')})), flush=True)


if __name__ == '__main__':
    main()
