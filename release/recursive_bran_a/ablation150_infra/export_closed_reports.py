"""Export accepted small milestone reports, without reading training assets.

Standalone stdlib program for Python stdin on the current h100-11 study. The
fixed watcher namespaces and filename whitelist are intentional: this is not a
generic remote file reader. Original bytes are exported except the tree receipt,
whose private-provenance file pointer is removed and clearly marked projected.
"""
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import stat
import sys
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014/analysis')
AGGREGATE = BASE / 'aggregate-watch-b95c4f26'
TREE = BASE / 'tree-summary-b8a28622'
SCIENCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
MILESTONES = (5, 50, 100, 150)
LABELS = ('C1-root-only', 'C2-random-final', 'budget-5', 'budget-10',
          'budget-20', 'position-early', 'position-uniform', 'position-late',
          'generation-1', 'generation-2', 'generation-4')
UNIQUE = ('C1-root-only', 'C2-random-final', 'budget-5', 'budget-20',
          'position-early', 'position-late', 'generation-2')
FILE_LIMIT = 256 * 1024
RAW_LIMIT = 3 * 1024 * 1024
OUTPUT_LIMIT = 4 * 1024 * 1024
PRIVATE_KEYS = {'question_id', 'question_ids', 'sample_id', 'sample_ids',
                'token_ids', 'generated_ids', 'input_ids', 'response_ids',
                'original_qa', 'prompt', 'prompts', 'response', 'responses',
                'trajectory', 'trajectories', 'password', 'access_token',
                'secret', 'credentials', 'provenance-private'}
CSV_FIELDS = ['condition', 'outer', 'hits', 'em', 'nq_hits', 'hotpot_hits',
              'adam_each_rank', 'skips_each_rank', 'generated_tokens_recorded',
              'allocated_gpu_hours_recorded']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_small(path, namespace):
    path, namespace = Path(path), Path(namespace)
    if not path.is_absolute() or path != path.resolve() or not path.is_relative_to(namespace):
        raise ValueError('Report path escapes the exact analysis namespace')
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > FILE_LIMIT:
        raise ValueError('Only bounded regular small report files may be read')
    with path.open('rb') as handle:
        raw = handle.read(FILE_LIMIT + 1)
    after = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if signature(before) != signature(after) or len(raw) != before.st_size:
        raise ValueError('Report changed during the bounded read; retry a fresh snapshot')
    return raw


def public_json(raw):
    value = json.loads(raw)

    def check(item):
        if isinstance(item, dict):
            if PRIVATE_KEYS.intersection(item):
                raise ValueError('Public report contains a forbidden private-data field')
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Report has a nonfinite scalar')

    check(value)
    return value


def dev(value):
    if (set(value) != {'all', 'nq', 'hotpotqa'} or value['all']['count'] != 512
            or value['nq']['count'] != 256 or value['hotpotqa']['count'] != 256
            or value['nq']['hits'] + value['hotpotqa']['hits'] != value['all']['hits']):
        raise ValueError('Report lacks complete native dev512 aggregate coverage')
    for row in value.values():
        if (type(row['hits']) is not int or not 0 <= row['hits'] <= row['count']
                or not math.isclose(row['em'], row['hits'] / row['count'], abs_tol=1e-12)):
            raise ValueError('Native aggregate integer hits/EM differ')


def export():
    files, raw_total, errors, states = [], 0, [], {}

    def add(name, raw, accepted_sha=None, projection=None, source_sha=None):
        nonlocal raw_total
        if accepted_sha is not None and digest(raw) != accepted_sha:
            raise ValueError('Small file differs from its persisted acceptance SHA')
        raw_total += len(raw)
        if raw_total > RAW_LIMIT:
            raise ValueError('Small-report total output limit exceeded')
        files.append(dict(path=name, bytes=len(raw), sha256=digest(raw),
                          base64=base64.b64encode(raw).decode('ascii'),
                          accepted_source_sha256=accepted_sha or source_sha,
                          projection=projection, original_bytes=projection is None))

    for kind, namespace, state_name, protocol in (
            ('aggregate', AGGREGATE, 'watch-state.json', 'recursive_bran_ablation150_aggregate_watch_v1'),
            ('tree', TREE, 'tree-summary-state.json', 'recursive_ablation150_tree_summary_v1')):
        try:
            raw_state = read_small(namespace / state_name, namespace)
        except FileNotFoundError:
            states[kind] = dict(present=False)
            continue
        state = json.loads(raw_state)
        if state.get('protocol') != protocol or state.get('scientific_source_commit') != SCIENCE:
            raise ValueError('Watcher belongs to another protocol/scientific source')
        reports = state['reports']
        allowed = LABELS if kind == 'aggregate' else UNIQUE
        if set(reports) - set(allowed):
            raise ValueError('Watcher contains an unrequested report label')
        states[kind] = dict(present=True, time=state['time'], status=state['status'],
                            state_sha256=digest(raw_state), error_count=len(state.get('errors', [])))
        for label in allowed:
            if set(reports.get(label, {})) - {str(outer) for outer in MILESTONES}:
                raise ValueError('Watcher contains an unscheduled report milestone')
            for outer in MILESTONES:
                pointer = reports.get(label, {}).get(str(outer))
                if not pointer:
                    continue
                folder = namespace / ('milestones' if kind == 'aggregate' else '') / label / f'outer-{outer:03d}'
                try:
                    if Path(pointer['directory']) != folder:
                        raise ValueError('Accepted report pointer differs from its exact directory')
                    relative = f'{kind}/{label}/outer-{outer:03d}'
                    if kind == 'aggregate':
                        raw = read_small(folder / 'aggregate.json', namespace)
                        result = public_json(raw)
                        if (result.get('protocol') != 'recursive_bran_ablation150_closed_milestone_v1'
                                or result.get('milestone_report_complete') is not True
                                or result.get('requested_label') != label or result.get('milestone') != outer
                                or set(result['conditions']) != {'A-original', label}):
                            raise ValueError('Aggregate is not the accepted closed milestone')
                        for condition in result['conditions'].values():
                            if set(condition['milestones']) != {str(outer)}:
                                raise ValueError('Aggregate contains unexpected milestones')
                            dev(condition['milestones'][str(outer)]['dev512'])
                        companion = {}
                        for name in ('curve.csv', 'findings-zh.md'):
                            companion[name] = read_small(folder / name, namespace)
                        rows = csv.DictReader(io.StringIO(companion['curve.csv'].decode('utf-8')))
                        if rows.fieldnames != CSV_FIELDS:
                            raise ValueError('Public CSV schema differs')
                        rows = list(rows)
                        if (len(rows) != 2 or {row['condition'] for row in rows} != {'A-original', label}
                                or any(int(row['outer']) != outer for row in rows)):
                            raise ValueError('CSV is not the corresponding aggregate milestone')
                        add(relative + '/aggregate.json', raw, pointer['aggregate_sha256'])
                        for name, content in companion.items():
                            add(relative + '/' + name, content)
                    else:
                        raw_complete = read_small(folder / 'complete.json', namespace)
                        if digest(raw_complete) != pointer['complete_sha256']:
                            raise ValueError('Tree complete receipt differs from persisted acceptance')
                        accepted = json.loads(raw_complete)
                        raw = read_small(folder / 'summary.json', namespace)
                        summary = public_json(raw)
                        if (accepted.get('protocol') != protocol or accepted['group'] != label
                                or accepted['outer'] != outer or summary.get('protocol') != protocol
                                or summary['group'] != label or summary['outer'] != outer
                                or summary['source_commit'] != SCIENCE
                                or summary['tree']['status'] != 'COMPLETE_CLOSED_TREE512'
                                or summary['tree']['questions'] != 512
                                or summary['native_boundary']['sha256'] != accepted['boundary_sha256']
                                or any(summary[key] != accepted[key] for key in ('run_id', 'config_sha256'))
                                or len(raw) != accepted['files']['summary.json']['bytes']):
                            raise ValueError('Tree summary differs from its accepted closed512 boundary')
                        dev(summary['dev512'])
                        add(relative + '/summary.json', raw, accepted['files']['summary.json']['sha256'])
                        public_receipt = {key: accepted[key] for key in (
                            'protocol', 'group', 'outer', 'run_id', 'config_sha256', 'boundary_sha256', 'goal_complete')}
                        public_receipt['files'] = {'summary.json': accepted['files']['summary.json']}
                        raw_receipt = (json.dumps(public_receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
                        add(relative + '/export-acceptance.json', raw_receipt,
                            projection='Existing complete.json with private-provenance pointer removed',
                            source_sha=pointer['complete_sha256'])
                except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as error:
                    errors.append(dict(kind=kind, group=label, outer=outer, error=str(error)))
    return dict(protocol='recursive_ablation150_small_closed_reports_export_v1', time=time.time(),
                scientific_source_commit=SCIENCE, goal_complete=False, states=states,
                raw_bytes=raw_total, file_count=len(files), files=files, errors=errors,
                scope='Existing accepted small reports only; no replay, paired rows, bootstrap, models, '
                      'GPU, HTTP, signals, locks, Node writes, raw QA/trajectories or tensor reads')


def main():
    try:
        result = export()
        raw = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(raw) > OUTPUT_LIMIT:
            raise ValueError('Encoded report output exceeds 4 MiB')
        sys.stdout.buffer.write(raw + b'\n')
        sys.stdout.buffer.flush()
        return 2 if result['errors'] else 0
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(protocol='recursive_ablation150_small_closed_reports_export_v1',
                              status='EXPORT_ATTENTION', error=str(error), training_action='NONE')), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
