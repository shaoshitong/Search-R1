"""Merge accepted small report capsules locally; never query or change training."""
import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from release.recursive_bran_a.ablation150_reports_v1.report import write_report

SOURCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
MILESTONES = (5, 50, 100, 150)
LABELS = ('C1-root-only', 'C2-random-final', 'budget-5', 'budget-10',
          'budget-20', 'position-early', 'position-uniform', 'position-late',
          'generation-1', 'generation-2', 'generation-4')
REUSE = {'budget-10': 'A-original', 'position-uniform': 'A-original',
         'generation-1': 'C1-root-only', 'generation-4': 'A-original'}
AGGREGATE = re.compile(r'aggregate/([^/]+)/outer-(\d{3})/aggregate\.json\Z')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(raw):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, parse_constant=reject)


def dev_signature(row):
    dev = row['dev512']
    values = []
    for key, count in (('all', 512), ('nq', 256), ('hotpotqa', 256)):
        item = dev[key]
        hits, em = item['hits'], item['em']
        if (type(item['count']) is not int or item['count'] != count
                or type(hits) is not int or not 0 <= hits <= count
                or isinstance(em, bool) or not isinstance(em, (int, float))
                or not math.isfinite(em) or em != hits/count):
            raise ValueError('Incomplete native dev512')
        values.append(hits)
    if values[0] != values[1]+values[2]:
        raise ValueError('Dataset hits do not add to dev512')
    return tuple(values)


def collect(capsule_path):
    raw = Path(capsule_path).read_bytes()
    if len(raw) > 4*1024*1024:
        raise ValueError('Capsule exceeds the published small-export limit')
    capsule = load(raw)
    if (capsule['protocol'] != 'recursive_ablation150_small_closed_reports_export_v1'
            or capsule['scientific_source_commit'] != SOURCE or capsule['errors']
            or capsule['file_count'] != len(capsule['files'])):
        raise ValueError('Wrong source, protocol or incomplete export')
    report = dict(protocol='recursive_ablation150_combined_closed_aggregates_v1',
                  reference_condition='A-original', conditions={}, comparisons={},
                  point_provenance={}, goal_complete=False,
                  capsule_sha256=sha(raw), capsule_time=capsule['time'],
                  scientific_source_commit=SOURCE)
    seen, total = set(), 0
    for entry in capsule['files']:
        name = entry['path']
        path = PurePosixPath(name)
        if (not isinstance(name, str) or path.is_absolute() or '..' in path.parts
                or str(path) != name or name in seen):
            raise ValueError('Duplicate or unsafe capsule path')
        seen.add(name)
        data = base64.b64decode(entry['base64'], validate=True)
        if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
            raise ValueError('Capsule member size/SHA differs: ' + name)
        total += len(data)
        match = AGGREGATE.fullmatch(name)
        if not match:
            continue
        label, outer_text = match.groups()
        outer, key = int(outer_text), str(int(outer_text))
        if label not in LABELS or outer not in MILESTONES:
            raise ValueError('Unexpected condition or evaluation milestone')
        item = load(data)
        if (entry.get('original_bytes') is not True
                or entry.get('accepted_source_sha256') != sha(data)
                or item['protocol'] != 'recursive_bran_ablation150_closed_milestone_v1'
                or item['requested_label'] != label or item['milestone'] != outer
                or item['milestone_report_complete'] is not True
                or item['reference_condition'] != 'A-original'
                or item['reuse_of'] != REUSE.get(label)
                or item['generated_new_training'] is not (label not in REUSE)):
            raise ValueError('Aggregate identity or reuse contract differs')
        for condition_id in ('A-original', label):
            condition = item['conditions'][condition_id]
            row = condition['milestones'][key]
            if row['outer'] != outer:
                raise ValueError('Milestone row outer differs')
            signature = dev_signature(row)
            dest = report['conditions'].setdefault(condition_id, dict(
                condition_id=condition_id, run_id=condition['run_id'],
                config_sha256=condition['config_sha256'],
                world_size=condition['world_size'], sampled_memory=None, milestones={}))
            if any(dest[k] != condition[k] for k in ('run_id', 'config_sha256', 'world_size')):
                raise ValueError('Condition identity differs between milestones')
            if key in dest['milestones']:
                if condition_id != 'A-original' or dev_signature(dest['milestones'][key]) != signature:
                    raise ValueError('Conflicting duplicate aggregate point')
            else:
                dest['milestones'][key] = copy.deepcopy(row)
        report['point_provenance'].setdefault(label, {})[key] = dict(
            capsule_member=name, aggregate_sha256=sha(data),
            reuse_of=item['reuse_of'], generated_new_training=item['generated_new_training'],
            native_evidence=item.get('native_evidence'),
            reference_provenance=item.get('reference_provenance'),
            sampled_memory=item['conditions'][label].get('sampled_memory'))
        a = report['conditions']['A-original']['milestones'][key]['dev512']['all']
        b = report['conditions'][label]['milestones'][key]['dev512']['all']
        report['comparisons'].setdefault(label, {})[key] = dict(
            status='COMPLETE_AGGREGATE', aggregate_difference_pp=100*(b['em']-a['em']),
            scope='Same-outer aggregate EM only; reused labels are not independent runs')
    if total != capsule['raw_bytes'] or total > 3*1024*1024:
        raise ValueError('Capsule total size differs')
    missing = [dict(condition=label, outer=n) for label in LABELS for n in MILESTONES
               if str(n) not in report['point_provenance'].get(label, {})]
    report.update(planned_label_points=44, accepted_label_points=44-len(missing),
                  missing_label_points=missing, analysis_complete=not missing,
                  completion_scope='All milestone aggregates alone do not prove training terminal, archive or whole goal completion',
                  limitations=['Original cost coverage flags remain unchanged; boundary-only values are not complete prefix costs.',
                               'Aliases reuse A or C1 and must not be summed as independent training cost.',
                               'Native receipt/scalar evidence is not independent tensor reload.',
                               'Single-seed aggregate EM does not prove causal or statistically reliable superiority.'])
    report['conditions'] = {cid: report['conditions'][cid] for cid in ('A-original',)+LABELS
                            if cid in report['conditions']}
    for condition in report['conditions'].values():
        condition['milestones'] = dict(sorted(condition['milestones'].items(), key=lambda pair: int(pair[0])))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.capsule)
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve previous snapshots')
    write_report(report, args.output)
    findings = args.output/'findings-zh.md'
    text = findings.read_text()
    text += '\n已接受计划标签节点：%d/44；尚缺%d。复用标签不代表新增独立训练。\n' % (
        report['accepted_label_points'], len(report['missing_label_points']))
    text += '\n训练终态、归档及整个goal完成情况须另行验收；此表不将goal标记完成。\n'
    findings.write_text(text)
    print(json.dumps(dict(output=str(args.output), accepted_label_points=report['accepted_label_points'],
                          missing_label_points=len(report['missing_label_points']), goal_complete=False)))


if __name__ == '__main__':
    main()
