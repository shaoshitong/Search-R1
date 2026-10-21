"""Format saved HPC3 observations into local plain text and a small JSON summary.

No SSH, subprocess, deletion, permission change, or training action is available
in this module. Two UIDs observe the SAME root; their totals are never added.
Default publication waits for both du result.json records and root-total rows.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path, PurePosixPath
import re
import time


DEFAULT_OUT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/'
    'users-shaoshitong-documents-remote-codex-control-6/outputs/easy2560-qwen7b-20261021')
REPORT_NAME = 'HPC3-storage-distribution-20261021.txt'
SUMMARY_NAME = 'HPC3-storage-distribution-20261021-summary.json'
OWN_7B = 'easy2560-corollout-joint4-qwen7b-20261021'
LARGE_SOURCE = 'dynrpo-qwen25-7b-probe-warmup-20261019'
ACCOUNT_NAMES = ('jxu', 'mou')


def timestamp(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return '未记录'
    return datetime.fromtimestamp(value, timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def _latest(out, pattern):
    files = list(out.glob(pattern))
    if not files:
        return None
    def order(path):
        match = re.search(r'-(\d+)\.json$', path.name)
        return (int(match.group(1)) if match else 0, path.stat().st_mtime_ns)
    return max(files, key=order)


def read_observation(path):
    outer = json.loads(path.read_text())
    if 'stdout' not in outer:
        return outer
    if outer.get('returncode', 0) != 0:
        raise ValueError('Saved observation transport failed: ' + str(path))
    documents = []
    for line in outer['stdout'].splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            documents.append(value)
    if not documents:
        raise ValueError('No JSON observation in ' + str(path))
    return documents[-1]


def parse_snapshot(document, filename):
    claim = document.get('claim.json') or {}
    command = claim.get('command') or []
    if not command or not isinstance(command[-1], str) or not command[-1].startswith('/'):
        raise ValueError('Missing actual du root in ' + filename)
    root = PurePosixPath(command[-1])
    rows = {}
    for row in document.get('rows', []):
        value, path = row['bytes'], PurePosixPath(row['path'])
        if type(value) is not int or value < 0 or not path.is_relative_to(root):
            raise ValueError('Invalid allocated-size row in ' + filename)
        relative = path.relative_to(root)
        if len(relative.parts) > 2:
            raise ValueError('Unexpected deeper row in depth2 observation: ' + str(path))
        if str(path) in rows:
            raise ValueError('Repeated du path in ' + filename)
        rows[str(path)] = value
    direct = {path: size for path, size in rows.items() if PurePosixPath(path).parent == root}
    result = document.get('result.json')
    ended = isinstance(result, dict) and bool(result) and str(root) in rows
    errors = document.get('errors') or {}
    lines = errors.get('lines') or []
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise ValueError('Invalid supplied du errors in ' + filename)
    error_bytes = errors.get('bytes', 0)
    supplied_bytes = len(('\n'.join(lines) + ('\n' if lines else '')).encode())
    affected = Counter()
    for line in lines:
        for item in re.findall(r"'(/[^']*)'", line):
            path = PurePosixPath(item)
            if path.is_relative_to(root) and path != root:
                affected[str(root / path.relative_to(root).parts[0])] += 1
    total = rows.get(str(root))
    subtotal = sum(direct.values())
    delta = None if total is None else total - subtotal
    completed_unix = next((result.get(key) for key in
        ('epoch', 'completed_unix', 'finished_unix', 'end_unix')
        if isinstance(result, dict) and isinstance(result.get(key), (int, float))), None)
    return dict(source_file=filename, uid=document.get('uid', claim.get('uid')),
        root=str(root), rows=rows, row_count=len(rows), direct=direct, direct_directory_count=len(direct),
        root_total_bytes=total, direct_subtotal_bytes=subtotal,
        root_minus_direct_subtotal_bytes=delta,
        small_root_metadata_difference=delta is not None and 0 <= delta <= 1024**2,
        scan_complete=ended, result=result, started_unix=claim.get('epoch'),
        completed_unix=completed_unix, observed_unix=document.get('epoch'),
        error_lines=lines, error_line_count=len(lines), error_bytes=error_bytes, supplied_error_bytes=supplied_bytes,
        all_error_bytes_present=error_bytes == supplied_bytes,
        permission_incomplete=any('Permission denied' in line for line in lines),
        affected_top_directories=dict(affected), scan_command=command)


def training_summary(document, filename):
    if document is None:
        return dict(source_file=None, observed_unix=None, unavailable=True,
                    note='Saved training observation unavailable; no current training claim')
    health = document.get('service_health') or {}
    receipt = document.get('preparation/cp1-full-state-source-receipt.json') or document.get('source_cp1_cpu_receipt') or {}
    stop = document.get('old/preparation/human-stop-after-cp1-for-optimized-external2-20261021.json') or {}
    completed = stop.get('actual_complete_outer') or {}
    return dict(source_file=filename, observed_unix=document.get('epoch'),
        new_job=document.get('newjob'), queue=document.get('queue'), accounting=document.get('accounting'),
        current_base=document.get('base'), current_base_allocated_bytes=document.get('current_base_allocated_bytes'),
        service_health={key: health[key] for key in ('ok', 'service_job_id', 'training_run_id',
            'hostname', 'endpoint', 'physical_gpu_count', 'logical_shards', 'observed_unix', 'statistics')
            if key in health},
        cp1_cpu_receipt={key: receipt[key] for key in ('status', 'step', 'world_size',
            'hf_saved_tensor_count', 'actual_optimizer_writes_per_rank', 'actual_cuda_restore_accepted')
            if key in receipt},
        completed_old_outer={key: completed[key] for key in ('step', 'update_status',
            'full_budget_complete', 'actual_optimizer_writes') if key in completed},
        restore_receipts=document.get('restore_receipts') or {},
        diagnostic_failures=document.get('diagnostic_failures') or {},
        actual_cuda_restore_accepted=(receipt.get('actual_cuda_restore_accepted') is True),
        note='Saved observation only; PENDING/RUNNING and CPU-state receipt do not prove resumed training progress')


def _size(value):
    return '未观测到' if value is None else f'{value / 10**9:,.6f} GB ({value:,} B)'


def _table(paths, views, *, relative_to=None):
    lines = ['  jxu957 (UID14090) GB       mou176 (UID21930) GB       路径']
    for path in paths:
        sizes = [views[name]['rows'].get(path) for name in ACCOUNT_NAMES]
        columns = [f'{size / 10**9:>20,.6f}' if size is not None else f'{"未观测到":>20}' for size in sizes]
        label = str(PurePosixPath(path).relative_to(relative_to)) if relative_to else path
        lines.append('  ' + '   '.join(columns) + '   ' + label)
    return lines


def render(views, training, generated_unix):
    roots = {view['root'] for view in views.values()}
    if len(roots) != 1:
        raise ValueError('Cannot compare UIDs observing different roots')
    root = next(iter(roots))
    complete = all(view['scan_complete'] for view in views.values())
    permission_gaps = any(view['permission_incomplete'] for view in views.values())
    all_errors_present = all(view['all_error_bytes_present'] for view in views.values())
    state = ('两份du进程均结束，根目录总计均已收到' if complete else
             '扫描未完成：至少一份缺少result.json或根目录总计；禁止将部分结果当作当前全量')
    lines = ['HPC3 实验目录存储分布（本地保存观测生成，2026-10-21）',
        '生成时间：' + timestamp(generated_unix), '统计根目录：' + root, '扫描状态：' + state,
        '删除操作：无。本报告生成器不执行远端命令、文件删除、chmod或训练操作。',
        '范围：仅上述实验根目录；不是整个HPC3文件系统、账号硬配额或Mac本地磁盘。',
        '单位：十进制GB=10^9 B，TB=10^12 B；du -B1为实际分配块占用，不是apparent-size。',
        '两个UID扫描的是同一根目录，属于不同权限视图；绝不将两个账号的数值相加。',
        '同一次du默认只统计一次硬链接；归属受遍历顺序/权限影响。符号链接不跟随，',
        '因此目录显示大小不等于单独删除该目录可释放的字节数，跨UID逐目录取最大值也不能冒充去重总量。',
        '扫描会跨越多个时刻，活动作业可能同时写入；这是扫描区间观测，不是原子文件系统快照。',
        '权限漏读：' + ('存在；所有根总量均为权限可见下限。' if permission_gaps else '已提供错误中未发现Permission denied。'),
        '错误清单完整性：' + ('已提供行的UTF-8字节数与错误文件长度一致。' if all_errors_present else
                              '至少一份保存观测没有提供全部错误字节；不能声称列出了远端全部错误。'), '']
    for name in ACCOUNT_NAMES:
        view = views[name]
        lines += [name + ' 权限视图：UID=' + str(view['uid']),
            '  输入：' + view['source_file'],
            '  开始：' + timestamp(view['started_unix']) + '；结束：' + timestamp(view['completed_unix']) +
            '；最后观测：' + timestamp(view['observed_unix']),
            '  du命令：' + ' '.join(view['scan_command']),
            '  结束记录：' + json.dumps(view['result'], ensure_ascii=False, sort_keys=True),
            '  根目录总计：' + _size(view['root_total_bytes']),
            '  已输出直接子目录数：' + str(view['direct_directory_count']),
            '  直接子目录小计：' + _size(view['direct_subtotal_bytes']),
            '  根总计减直接子目录小计：' + str(view['root_minus_direct_subtotal_bytes']) + ' B',
            '  小计复核：' + ('差额为非负且<=1MiB，符合根目录自身/根直属小文件开销。'
                if view['small_root_metadata_difference'] else
                '尚不能归为根元数据小差额；可能扫描未结束、根直属文件、权限差异或扫描期间写入，未强行归零。'), '']
    paths = set().union(*(set(view['direct']) for view in views.values()))
    def ordering(path):
        return (-views['jxu']['rows'].get(path, views['mou']['rows'].get(path, 0)), path)
    direct = sorted(paths, key=ordering)
    lines += ['全部直接子目录（jxu主视图降序；两列不可相加）'] + _table(direct, views, relative_to=root) + ['']
    lines += ['每个顶层目录的完整第二层清单（深层内容已递归计入这些du数值；本表显示至第二层）']
    for parent in direct:
        children = sorted({path for view in views.values() for path in view['rows']
                           if PurePosixPath(path).parent == PurePosixPath(parent)}, key=ordering)
        lines += ['', parent, '  父目录：jxu=' + _size(views['jxu']['rows'].get(parent)) +
                  '；mou=' + _size(views['mou']['rows'].get(parent))]
        lines += _table(children, views, relative_to=parent) if children else ['  未提供第二层目录行。']
    own = str(PurePosixPath(root) / OWN_7B)
    lines += ['', '当前Easy7B原作业目录专列：' + own]
    own_rows = sorted({path for view in views.values() for path in view['rows']
                       if path == own or PurePosixPath(path).parent == PurePosixPath(own)}, key=ordering)
    lines += _table(own_rows, views)
    large = str(PurePosixPath(root) / LARGE_SOURCE)
    large_children = sorted({path for view in views.values() for path in view['rows']
        if PurePosixPath(path).parent == PurePosixPath(large)}, key=ordering)
    feature_rows = sorted({path for view in views.values() for path in view['rows']
        if PurePosixPath(path).parent == PurePosixPath(large) and
        'feature' in PurePosixPath(path).name.lower()}, key=ordering)
    lines += ['', '大型共享来源目录：' + large,
        '  下列为真实第二层占用；不把algorithm-improvement等父目录的全部内容推断为features。']
    lines += _table(large_children, views, relative_to=large) if large_children else ['  当前观测未提供第二层条目。']
    lines += ['  第二层路径名含feature的条目数：' + str(len(feature_rows)) +
              '；更深层features的归属不能仅由depth2清单进一步细分。']
    lines += ['', '训练与独立检索服务（时间为此份保存观测的时间）',
        '状态输入：' + str(training.get('source_file')), '观测时间：' + timestamp(training.get('observed_unix')),
        '新训练作业：' + str(training.get('new_job'))]
    current_base = training.get('current_base')
    if current_base and current_base not in views['jxu']['rows'] and training.get('current_base_allocated_bytes') is not None:
        lines += ['全根扫描完成后新增的恢复目录（独立单目录du，未混入原扫描总计）：',
                  '  ' + current_base + '：' + _size(training['current_base_allocated_bytes']),
                  '  此目录未出现在上方63项19:13扫描中；上方总量仍保留原始扫描口径。']
    for key in ('queue', 'accounting'):
        value = training.get(key) or {}
        lines += [key + ' 原始输出（rc=' + str(value.get('rc')) + '）：', value.get('stdout', '未提供').rstrip()]
        if value.get('stderr'):
            lines += ['stderr：' + value['stderr'].rstrip()]
    for key in ('service_health', 'cp1_cpu_receipt', 'completed_old_outer', 'restore_receipts', 'diagnostic_failures'):
        lines += [key + '：', json.dumps(training.get(key), ensure_ascii=False, indent=2, sort_keys=True)]
    lines += ['训练解释：' + training['note'], '', '权限错误及其影响的顶层目录（保留所有已提供错误行）']
    for name in ACCOUNT_NAMES:
        view = views[name]
        lines += ['', name + '：错误文件=' + str(view['error_bytes']) + ' B，已提供=' +
                  str(len(view['error_lines'])) + '行；受影响顶层目录=' + str(len(view['affected_top_directories']))]
        for path, count in sorted(view['affected_top_directories'].items()):
            lines.append('  ' + str(count) + ' 行  ' + path)
        lines += view['error_lines'] or ['  无已提供错误行。']
    summary = dict(format='hpc3_saved_du_depth2_cross_uid_report_v1', generated_unix=generated_unix,
        root=root, scans_complete=complete, permission_incomplete=permission_gaps,
        full_namespace_without_permission_gaps=(complete and not permission_gaps and
            not any(view['error_lines'] for view in views.values())),
        account_views_never_added=True, primary_account='jxu', no_deletion_performed=True,
        report_scope='saved observations of one experiments root; no remote actions',
        account_views={name: {key: value for key, value in view.items() if key not in
            ('rows', 'direct', 'error_lines', 'scan_command')} for name, view in views.items()},
        own_7b={name: {path: size for path, size in view['rows'].items()
            if path == own or PurePosixPath(path).parent == PurePosixPath(own)} for name, view in views.items()},
        large_source={name: {path: size for path, size in view['rows'].items()
            if path == large or path in large_children} for name, view in views.items()}, training=training)
    return '\n'.join(lines) + '\n', summary


def build(out, *, allow_incomplete=False):
    out = Path(out).resolve(strict=True)
    views = {}
    for name in ACCOUNT_NAMES:
        path = _latest(out, 'space-user-observation-' + name + '-*.json')
        if path is None:
            raise ValueError('Missing saved ' + name + ' observation')
        views[name] = parse_snapshot(read_observation(path), str(path))
    if not allow_incomplete and not all(view['scan_complete'] for view in views.values()):
        return dict(report_written=False, scans_complete=False, no_deletion_performed=True,
                    reason='Wait for BOTH result.json records AND BOTH root-total rows',
                    source_files={name: view['source_file'] for name, view in views.items()})
    path = _latest(out, 'cp1-new-observation-*.json')
    training = training_summary(read_observation(path) if path else None, str(path) if path else None)
    text, summary = render(views, training, time.time())
    summary.update(report_written=True, report_path=str(out / REPORT_NAME), summary_path=str(out / SUMMARY_NAME))
    for target, content in ((out / REPORT_NAME, text),
                           (out / SUMMARY_NAME, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n')):
        temporary = target.with_name(target.name + '.partial-' + str(os.getpid()))
        with temporary.open('w') as stream:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    return summary


def self_test():
    root = '/data/user/jxu957/experiments'
    document = dict(uid=14090, epoch=20, **{'claim.json': dict(epoch=10,
        command=['du', '-x', '-B1', '--max-depth=2', root]),
        'rows': [dict(bytes=10000, path=root + '/a'), dict(bytes=3000, path=root + '/a/features'),
                 dict(bytes=24096, path=root), dict(bytes=10000, path=root + '/b')],
        'errors': dict(bytes=0, lines=[])})
    incomplete = parse_snapshot(document, 'fixture')
    assert not incomplete['scan_complete'], 'A root row without result.json is incomplete'
    assert incomplete['root_minus_direct_subtotal_bytes'] == 4096
    document['result.json'] = dict(epoch=19, rc=1)
    error = "du: cannot read directory '" + root + "/a/private': Permission denied"
    document['errors'] = dict(bytes=len((error + '\n').encode()), lines=[error])
    primary = parse_snapshot(document, 'fixture-jxu')
    secondary_doc = json.loads(json.dumps(document)); secondary_doc['uid'] = 21930
    secondary_doc['rows'][0]['bytes'] = 12000
    secondary_doc['rows'][2]['bytes'] = 26096
    secondary = parse_snapshot(secondary_doc, 'fixture-mou')
    assert primary['scan_complete'] and primary['permission_incomplete']
    assert primary['all_error_bytes_present'] and primary['affected_top_directories'][root + '/a'] == 1
    training = training_summary(None, None)
    text, summary = render(dict(jxu=primary, mou=secondary), training, 21)
    assert error in text and '/a/features' not in text  # relative second-level label is displayed
    assert 'features' in text and 'jxu957 (UID14090)' in text
    assert summary['scans_complete'] and not summary['full_namespace_without_permission_gaps']
    assert summary['account_views_never_added'] and summary['no_deletion_performed']
    assert summary['account_views']['jxu']['root_total_bytes'] == 24096
    missing_root = json.loads(json.dumps(document)); missing_root['rows'].pop(2)
    assert not parse_snapshot(missing_root, 'fixture')['scan_complete']
    print('SELF_TEST_PASS: completion, metadata subtotal, permission gaps, cross-UID isolation, depth2 rendering')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--allow-incomplete', action='store_true', help='Explicitly emit a clearly marked partial report')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    result = build(args.out, allow_incomplete=args.allow_incomplete)
    print(json.dumps({key: result.get(key) for key in
        ('report_written', 'scans_complete', 'permission_incomplete', 'no_deletion_performed',
         'report_path', 'summary_path', 'reason', 'source_files') if key in result}, ensure_ascii=False))


if __name__ == '__main__':
    main()
