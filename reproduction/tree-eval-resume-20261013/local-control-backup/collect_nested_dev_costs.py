#!/usr/bin/env python3
"""Read only completed dev5..N control summaries (N=120,125,..200); no JSONL/PT.

The default120 CLI remains compatible. New immutable supplements require actual
FULLN/CUMN/dev512 evidence and reuse SHA-bound earlier control summaries.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess

ORIGINAL_CUM120_SHA = '2a233058c17789412f46615d39f25b0b05ed340080a52e38304d2f03363afcb8'
ORIGINAL_SUPPLEMENT120_SHA = '58734a871e022e1d8de6c628aab8a2ffe010b13ca2c57352a5f61d3c7f58f57b'
CHILD = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
PARENT = 'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4'
CONFIGS = {CHILD: '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3',
           PARENT: '22beaec8aa2b449b496754aacfd9b1e7c89009c9c27a8eff3aef92b768b28b59'}
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
REMOTE = r'''
import hashlib,json,os,time
from pathlib import Path
pins=PIN_DATA
os.nice(15);started=time.time();cpu=time.process_time();items=[]
for pin in pins:
 path=Path(pin['path'])
 if path.is_symlink() or path.suffix!='.json' or path.parent!=Path(pin['run'])/'output/validation-cost':raise ValueError('Fixed own small control JSON path differs')
 with path.open('rb') as stream:
  before=os.fstat(stream.fileno())
  if not 0<before.st_size<=100000:raise ValueError('Small control summary size budget')
  raw=stream.read();after=os.fstat(stream.fileno())
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Completed control summary changed during read')
 value=json.loads(raw)
 for key in ['code_commit','config_sha256','data_identity','model_identity','job_id']:
  if value.get(key)!=pin[key]:raise ValueError('Actual control summary identity differs: '+key)
 if not isinstance(value.get('run'),str) or Path(value['run']).resolve(strict=True)!=(Path(pin['run'])/'output').resolve(strict=True) or value.get('actual_completed_outer')!=pin['outer'] or value.get('status')!='VALIDATION_COST_COMPLETE' or value.get('evaluated_rows')!=512 or value.get('dataset_rows')!=512 or value.get('phase')!=pin['phase']:raise ValueError('Actual completed dev outer/run/coverage differs')
 m=value.get('metrics') or {}
 if m.get('val/qa_count/all')!=512 or m.get('val/qa_count/nq')!=256 or m.get('val/qa_count/hotpotqa')!=256 or m.get('val/f1/all')!=pin['F1'] or m.get('val/em/all')!=pin['EM']:raise ValueError('Actual dev metrics differ from immutable completion evidence')
 if pin['ledger_sha256'] is not None and value.get('ledger_sha256')!=pin['ledger_sha256']:raise ValueError('Actual control ledger SHA differs from original completed dev')
 items.append(dict(outer=pin['outer'],raw_summary_utf8=raw.decode('utf8'),source=dict(path=str(path),resolved_path=str(path.resolve(strict=True)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=before.st_mtime_ns,stable_read=True)))
result=dict(status='ACTUAL_COMPLETED_DEV_CONTROL_SUMMARIES_READ',observed_epoch=time.time(),items=items,remote_nice=os.getpriority(os.PRIO_PROCESS,0),collector_process_cpu_seconds=time.process_time()-cpu,collector_wall_seconds=time.time()-started,remote_files_written=False,jsonl_or_tensor_or_model_read=False)
encoded=json.dumps(result,ensure_ascii=False,allow_nan=False)
if len(encoded.encode())>800000:raise ValueError('Small control-summary batch budget')
print(encoded)
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def publish(base, mirrors, name, raw):
    if len(raw) > 100000 or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Small relative audit output required')
    for folder in [base] + mirrors:
        path = folder / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Immutable control-summary output already differs: ' + name)
    for folder in [base] + mirrors:
        path = folder / name;path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_name(path.name + '.tmp');tmp.write_bytes(raw);os.replace(tmp, path)
        if path.read_bytes() != raw:
            raise ValueError('Small control summary mirror readback differs')
    return dict(path=name, bytes=len(raw), sha256=sha(raw), both_mirrors_byte_equal=True)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def read_bound(base, mirrors, reference):
    name = reference.get('path', reference.get('file'))
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Explicit relative immutable summary reference required')
    path = base / name
    if path.stat().st_size > 100000:
        raise ValueError('Small immutable summary/part required')
    raw = path.read_bytes()
    if len(raw) != reference['bytes'] or sha(raw) != reference['sha256']:
        raise ValueError('Immutable summary/part SHA or bytes differs: ' + name)
    if any((mirror / name).read_bytes() != raw for mirror in mirrors):
        raise ValueError('Immutable summary/part mirror bytes differ: ' + name)
    return raw


def rows_from_summary(base, mirrors, summary):
    value = summary['rows']
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or value.get('storage') != 'complete_original_field_parts' or value.get('container_type') != 'list':
        raise ValueError('Unknown complete original row storage')
    rows = []
    for reference in value['parts']:
        part = json.loads(read_bound(base, mirrors, reference))
        if reference.get('json_selector') != 'value' or not isinstance(part.get('value'), list):
            raise ValueError('Complete original row part selector differs')
        rows.extend(part['value'])
    if len(rows) != value['original_entries']:
        raise ValueError('Complete original row count differs')
    return rows


def control_row(pin, raw, meta, saved):
    value = json.loads(raw)
    for key in ['code_commit', 'config_sha256', 'data_identity', 'model_identity', 'job_id']:
        if value.get(key) != pin[key]:
            raise ValueError('Original control summary identity differs: ' + key)
    if sha(raw) != meta['sha256'] or len(raw) != meta['bytes'] or meta['path'] != pin['path'] or meta.get('stable_read') is not True:
        raise ValueError('Original completed control JSON SHA/bytes/path differs')
    resolved_output = str(Path(meta['resolved_path']).parent.parent)
    if value.get('run') not in (pin['run'] + '/output', resolved_output):
        raise ValueError('Original control run/output alias differs')
    if value.get('actual_completed_outer') != pin['outer'] or value.get('status') != 'VALIDATION_COST_COMPLETE' or value.get('evaluated_rows') != 512 or value.get('dataset_rows') != 512 or value.get('phase') != pin['phase']:
        raise ValueError('Original completed dev outer/coverage differs')
    metrics = value.get('metrics') or {}
    if metrics.get('val/qa_count/all') != 512 or metrics.get('val/qa_count/nq') != 256 or metrics.get('val/qa_count/hotpotqa') != 256 or metrics.get('val/f1/all') != pin['F1'] or metrics.get('val/em/all') != pin['EM']:
        raise ValueError('Original dev metrics differ from immutable completion')
    if pin['ledger_sha256'] is not None and value.get('ledger_sha256') != pin['ledger_sha256']:
        raise ValueError('Original dev ledger SHA differs')
    if not isinstance(value.get('ledger_sha256'), str) or len(value['ledger_sha256']) != 64:
        raise ValueError('Actual original ledger SHA is absent')
    return dict(outer=pin['outer'], code_commit=pin['code_commit'], config_sha256=pin['config_sha256'],
        run=pin['run'], phase=value['phase'], actual_completed_outer=value['actual_completed_outer'],
        evaluated_rows=value['evaluated_rows'], counters=value.get('counters'), source_control_summary=meta,
        local_original_summary=saved, ledger_sha256=value.get('ledger_sha256'),
        start_epoch=value.get('start_epoch'), end_epoch=value.get('end_epoch'),
        wall_seconds=value.get('wall_seconds'), trainer_process_cpu_seconds=value.get('trainer_process_cpu_seconds'),
        pad_token_id=value.get('pad_token_id'), eos_token_id=value.get('eos_token_id'),
        observer_source_sha256=value.get('observer_source_sha256'), cost_count_scope=value.get('cost_count_scope'),
        limitations=value.get('limitations'), precise_engine_emitted_tokens=None, scoring_call_event_total=None)


def load_prior(base, mirrors, through, pins):
    name = 'DEV5_%d_NATIVE_COST_SUPPLEMENT.json' % through
    raw = (base / name).read_bytes()
    if through == 120 and sha(raw) != ORIGINAL_SUPPLEMENT120_SHA:
        raise ValueError('Original immutable dev5..120 supplement SHA differs')
    reference = dict(path=name, bytes=len(raw), sha256=sha(raw))
    read_bound(base, mirrors, reference)
    summary = json.loads(raw)
    if summary.get('status') != 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK' or summary.get('completed_dev_outers') != list(range(5, through + 1, 5)):
        raise ValueError('Prior completed supplement coverage differs')
    source = summary['source_cumulative']
    if source['path'] != 'CUMULATIVE_COST_THROUGH%d.json' % through:
        raise ValueError('Prior supplement CUM outer differs')
    read_bound(base, mirrors, source)
    rows = rows_from_summary(base, mirrors, summary)
    if [row.get('outer') for row in rows] != summary['completed_dev_outers']:
        raise ValueError('Prior supplement rows differ')
    pin_map = {pin['outer']:pin for pin in pins}
    for row in rows:
        saved = row['local_original_summary']
        if saved['path'] != 'dev_native_cost_summaries/DEV_OUTER%04d_SUMMARY.json' % row['outer']:
            raise ValueError('Prior original control filename differs')
        original = read_bound(base, mirrors, saved)
        if control_row(pin_map[row['outer']], original, row['source_control_summary'], saved) != row:
            raise ValueError('Prior scalar row differs from preserved original summary')
    return rows, reference


def publish_rows(base, mirrors, through, summary):
    if len(encoded(summary)) <= 100000:
        return
    rows = summary['rows'];chunks = [];chunk = []
    for row in rows:
        trial = chunk + [row]
        if len(encoded(dict(value=trial))) > 90000 and chunk:
            chunks.append(chunk);chunk = [row]
        else:
            chunk = trial
        if len(encoded(dict(value=chunk))) > 90000:
            raise ValueError('One original scalar row exceeds small part budget')
    if chunk:
        chunks.append(chunk)
    parts = []
    for number, chunk in enumerate(chunks):
        name = 'dev_native_cost_summaries/DEV5_%d_ROWS_PART%03d.json' % (through, number)
        info = publish(base, mirrors, name, encoded(dict(value=chunk)))
        parts.append(dict(file=name, bytes=info['bytes'], sha256=info['sha256'], json_selector='value'))
    summary['rows'] = dict(storage='complete_original_field_parts', container_type='list',
        original_entries=len(rows), parts=parts)


def completed_inputs(base, through):
    cum_name = 'CUMULATIVE_COST_THROUGH%d.json' % through
    full_name = 'FULL_OUTER%d_PROGRESS_RECEIPT.json' % through
    if not (base / cum_name).exists() or not (base / full_name).exists():
        return None
    raw = (base / cum_name).read_bytes()
    if len(raw) > 100000:
        raise ValueError('Small completed cumulative descriptor required')
    cum_sha = sha(raw)
    if through == 120 and cum_sha != ORIGINAL_CUM120_SHA:
        raise ValueError('Original immutable CUM120 SHA differs')
    cumulative = json.loads(raw);full = json.loads((base / full_name).read_bytes())
    if cumulative.get('through_effective_outer') != through or full.get('completed_outer') != through:
        raise ValueError('Actual CUM/FULL completed outer differs')
    dev = full.get('actual_dev512');devs = cumulative.get('completed_dev_evaluations')
    if not isinstance(dev, dict) or dev.get('outer') != through or dev.get('status') != 'VALIDATION_COST_COMPLETE' or dev.get('rows') != 512 or not isinstance(devs, list) or [x.get('outer') for x in devs] != list(range(5, through + 1, 5)):
        return None
    if full.get('source_commit') != CHILD or full.get('config_sha256') != CONFIGS[CHILD]:
        raise ValueError('Completed source/config differs from fixed protocol')
    run_id = 'tree-nested-branch-fresh200-not-slurm;systemd:' + CHILD
    if full.get('run') != '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/' + run_id or full.get('unit') != 'sst-tree-nested-branch-fresh200-778b2b43.service':
        raise ValueError('Completed run/unit differs from original task')
    spec = importlib.util.spec_from_file_location('dev_cost_local_completed_inputs', Path(__file__).with_name('append_nested_local_costs.py'))
    appender = importlib.util.module_from_spec(spec);spec.loader.exec_module(appender)
    evidence = appender.actual_inputs(base, base / full_name)
    identity = evidence['native'].get('identity', {})
    if any(identity.get(key) != expected for key, expected in dict(code_commit=CHILD, config_sha256=CONFIGS[CHILD], model_identity=MODEL, data_identity=DATA, completed_step=through, actual_update_limit=200, run_id=run_id, job_id='not-slurm;systemd:' + CHILD).items()):
        raise ValueError('Actual native source/config/model/data/completed identity differs')
    metrics = dev.get('metrics') or {};last = devs[-1]
    if metrics.get('val/qa_count/all') != 512 or metrics.get('val/qa_count/nq') != 256 or metrics.get('val/qa_count/hotpotqa') != 256 or dev.get('path') != last['ledger'] or dev.get('phase') != last['phase'] or dev.get('ledger_sha256') != last['ledger_sha256'] or metrics.get('val/f1/all') != last['F1'] or metrics.get('val/em/all') != last['EM']:
        raise ValueError('Actual FULL dev512 differs from immutable cumulative dev')
    full_raw = (base / full_name).read_bytes()
    return cumulative, dict(path=cum_name, bytes=len(raw), sha256=cum_sha), dict(path=full_name, bytes=len(full_raw), sha256=sha(full_raw)), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--ssh-helper', type=Path, required=True)
    parser.add_argument('--through-outer', type=int, default=120, choices=range(120, 201, 5),
        help='Only an actually completed dev512 boundary; default120 preserves the old CLI.')
    args = parser.parse_args();base = args.base.resolve(strict=True);through = args.through_outer
    actual = completed_inputs(base, through)
    if actual is None:
        print(json.dumps(dict(status='PENDING_ACTUAL_COMPLETED_DEV_EVIDENCE', through_outer=through,
            required=['CUMULATIVE_COST_THROUGH%d.json' % through, 'FULL_OUTER%d_PROGRESS_RECEIPT.json with actual COMPLETE dev512' % through],
            remote_operations=False, outputs_written=False)), flush=True)
        return
    cumulative, cumulative_ref, full_ref, evidence = actual
    mirrors = [Path(x) for x in cumulative['mirrors']]
    devs = cumulative['completed_dev_evaluations'];pins = []
    for dev in devs:
        outer = dev['outer'];source = PARENT if outer == 5 else CHILD
        if dev['source_commit'] != source or dev['rows'] != 512 or Path(dev['ledger']).name != dev['ledger']:
            raise ValueError('Immutable dev source/rows/file differs')
        run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + source
        pins.append(dict(outer=outer, code_commit=source, config_sha256=CONFIGS[source], model_identity=MODEL,
            data_identity=DATA, job_id='not-slurm;systemd:' + source, run=run,
            path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'],
            ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM']))
    stem = 'DEV5_%d_NATIVE_COST' % through;rows = [];prior_ref = None
    if (base / (stem + '_SUPPLEMENT.json')).exists():
        rows, existing_ref = load_prior(base, mirrors, through, pins)
        old = json.loads((base / existing_ref['path']).read_bytes())
        if old['source_cumulative'] != cumulative_ref:
            raise ValueError('Existing immutable supplement CUM binding differs')
        if through > 120 and old.get('source_full_receipt') is None:
            raise ValueError('New immutable supplement must bind its actual FULL receipt')
        if old.get('source_full_receipt') is not None and old['source_full_receipt'] != full_ref:
            raise ValueError('Existing immutable supplement FULL binding differs')
        md_raw = (base / (stem + '_SUPPLEMENT.md')).read_bytes()
        if any((mirror / (stem + '_SUPPLEMENT.md')).read_bytes() != md_raw for mirror in mirrors):
            raise ValueError('Existing immutable supplement Markdown mirror differs')
        print(json.dumps(dict(status='ALREADY_EXISTS_BYTES_VERIFIED', supplement=existing_ref, controls=len(rows),
            original_outputs_unchanged=True, remote_operations=False)), flush=True)
        return
    for prior_through in range(through - 5, 119, -5):
        if (base / ('DEV5_%d_NATIVE_COST_SUPPLEMENT.json' % prior_through)).exists():
            rows, prior_ref = load_prior(base, mirrors, prior_through, pins)
            break
    reused_outers = [row['outer'] for row in rows]
    missing_pins = [pin for pin in pins if pin['outer'] not in reused_outers]
    if os.environ.get('TREE_MGMT_CONTROL_PATH') != '/tmp/tree-mgmt-3fw9a1ab/control':
        raise ValueError('Explicit original task management ControlPath required')
    spec = importlib.util.spec_from_file_location('dev_cost_retained18_route', args.ssh_helper)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    code = REMOTE.replace('PIN_DATA', repr(missing_pins))
    try:
        response = helper.run('env CUDA_VISIBLE_DEVICES= python3 -B -c ' + shlex.quote(code), target='h100-10', timeout=60)
        if response.returncode:
            raise RuntimeError('Read-only dev summary collection unavailable: ' + response.stderr[-2000:])
        result = json.loads(response.stdout)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        failure = dict(status='DEV_NATIVE_COST_SOURCE_READBACK_UNAVAILABLE_NO_TRAINING_FAULT_INFERRED',
            observed_utc=datetime.now(timezone.utc).isoformat(), through_outer=through,
            error_type=type(exc).__name__, error=str(exc), source_cumulative=cumulative_ref,
            training_operations=False, automatic_retry=False, ssh_helper_sha256=sha(args.ssh_helper.read_bytes()))
        if isinstance(exc, subprocess.TimeoutExpired):
            failure['error'] = 'Existing helper read-only SSH exceeded its timeout; no automatic retry.'
            failure['timeout_seconds'] = exc.timeout
        item = publish(base, mirrors, stem + '_COLLECTION_ERROR.json', encoded(failure))
        print(json.dumps(item), flush=True)
        raise SystemExit(1)
    if result.get('status') != 'ACTUAL_COMPLETED_DEV_CONTROL_SUMMARIES_READ' or result.get('remote_nice') != 15 or [x['outer'] for x in result.get('items', [])] != [pin['outer'] for pin in missing_pins]:
        raise ValueError('Actual completed control-summary response differs')
    for pin, item in zip(missing_pins, result['items']):
        raw = item['raw_summary_utf8'].encode();meta = item['source']
        saved = dict(path='dev_native_cost_summaries/DEV_OUTER%04d_SUMMARY.json' % pin['outer'],
            bytes=len(raw), sha256=sha(raw), both_mirrors_byte_equal=True)
        row = control_row(pin, raw, meta, saved)
        publish(base, mirrors, saved['path'], raw)
        rows.append(row)
    if [row['outer'] for row in rows] != list(range(5, through + 1, 5)):
        raise ValueError('Complete original dev cost row coverage differs')
    keys = sorted({key for row in rows for key in (row['counters'] or {})})
    def aggregate(values):
        known = [value for value in values if numeric(value)]
        total = math.fsum(known) if known else None
        if isinstance(total, float) and total.is_integer():total = int(total)
        return dict(total=total if len(known) == len(values) else None, known_sum=total,
            missing_outers=[row['outer'] for row, value in zip(rows, values) if not numeric(value)])
    summary = dict(summary_schema='tree_dev_native_cost_supplement_v2',
        status='ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK', observed_epoch=result['observed_epoch'],
        through_outer=through, source_cumulative=cumulative_ref, source_full_receipt=full_ref,
        source_completed_evidence_bindings=evidence['bindings'],
        model_identity=MODEL, data_identity=DATA,
        split_identity=dict(evaluated_rows=512, nq_rows=256, hotpotqa_rows=256,
            independent_split_file_sha256=None, scope='Actual official dataset identity and per-dataset counts; no independently captured split-file SHA inferred.'),
        prior_immutable_supplement=prior_ref, reused_original_control_outers=reused_outers,
        newly_read_remote_control_outers=[pin['outer'] for pin in missing_pins],
        cache_scope='Reused controls retain their original source/capture metadata and SHA; this new observation does not claim a new remote read of those controls.',
        completed_dev_outers=[row['outer'] for row in rows], rows=rows,
        native_counter_totals={key:aggregate([(row['counters'] or {}).get(key) for row in rows]) for key in keys},
        original_validation_observer_wall_seconds=aggregate([row['wall_seconds'] for row in rows]),
        original_validation_trainer_process_cpu_seconds=aggregate([row['trainer_process_cpu_seconds'] for row in rows]),
        prior_fullmetric_testing_phase_wall_seconds=cumulative['completed_dev_testing_phase_wall_seconds'],
        newly_bound_dev_ledger_sha_outers=[row['outer'] for row, dev in zip(rows, devs) if dev['ledger_sha256'] is None and row['ledger_sha256']],
        precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        scopes=['submitted_prefill_token_events are submitted repeated events, not unique KV/cached tokens/FLOPs.',
            'returned_nonpad_model_token_events exclude PAD and exclude EOS when PAD==EOS; not precise engine-emitted tokens.',
            'http_attempts/completed/failed are actual batched HTTP calls; http_query_events are individual queries, not HTTP calls.',
            'Original counters keep their original meanings; missing fields stay null. No mean is expanded into a total.',
            'Validation observer wall and process CPU are contained in outer/service scopes; never add them again to immutable CUM or whole-service CPU.',
            'No scoring-call event counter in this observer; do not infer zero or an evaluated-row count.',
            'Dev5 belongs to ba9/22be; dev10..%d belong to778/355; effective ancestry retained.' % through],
        read_only=True, remote_files_written=False, jsonl_rollout_tensor_or_model_read=False,
        collector_source_sha256=sha(Path(__file__).read_bytes()), remote_program_sha256=sha(code.encode()),
        ssh_helper_sha256=sha(args.ssh_helper.read_bytes()), collector_cpu_seconds=result['collector_process_cpu_seconds'],
        collector_cpu_scope='This small new readback process only; not added to validation/service CPU.',
        remote_nice=result['remote_nice'], immutable_cumulative_rewritten=False,
        old_dev5_120_outputs_rewritten=False, STATUS_REPORT_FUP_or_training_modified=False, tests_or_review_performed=False)
    publish_rows(base, mirrors, through, summary)
    info = publish(base, mirrors, stem + '_SUPPLEMENT.json', encoded(summary))
    md = ('# 已完成dev5–%d原生控制计数补充\n\n' % through
        + '绑定实际FULL/CUM/dev512身份；复用%d份SHA验证历史原件，只读%d份新增COMPLETE小控制JSON。'
          'dev5属于ba9/22be，后续属于778/355。原字段超限时完整分块，不删必需标量。\n\n' % (len(reused_outers), len(missing_pins))
        + 'submitted prefill为重复提交事件；returned nonpad在PAD==EOS时排除EOS，不称精确engine emitted。'
          'HTTP calls与query events分开；评分调用和精确engine emitted仍null。'
          '验证wall/process CPU已包含于整轮/服务，不再相加。旧5–120、历史CUM与状态不改；只保存小摘要，无JSONL/PT读取。\n')
    md_info = publish(base, mirrors, stem + '_SUPPLEMENT.md', md.encode())
    if sha((base / cumulative_ref['path']).read_bytes()) != cumulative_ref['sha256']:
        raise ValueError('Original immutable cumulative snapshot changed')
    print(json.dumps(dict(supplement=info, markdown=md_info, controls=len(rows),
        reused_original_control_outers=reused_outers, newly_read_remote_control_outers=[pin['outer'] for pin in missing_pins],
        native_counter_totals=summary['native_counter_totals'], original_cumulative_unchanged=True,
        remote_files_written=False)), flush=True)


if __name__ == '__main__':
    main()
