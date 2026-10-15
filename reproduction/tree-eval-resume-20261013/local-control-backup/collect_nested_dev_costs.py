#!/usr/bin/env python3
"""Read only fixed completed dev5..120 control summaries; never read JSONL/PT."""
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

CUM = 'CUMULATIVE_COST_THROUGH120.json'
CUM_SHA = '2a233058c17789412f46615d39f25b0b05ed340080a52e38304d2f03363afcb8'
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
result=dict(status='ACTUAL_DEV5_120_CONTROL_SUMMARIES_READ',observed_epoch=time.time(),items=items,remote_nice=os.getpriority(os.PRIO_PROCESS,0),collector_process_cpu_seconds=time.process_time()-cpu,collector_wall_seconds=time.time()-started,remote_files_written=False,jsonl_or_tensor_or_model_read=False)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--ssh-helper', type=Path, required=True)
    args = parser.parse_args();base = args.base.resolve(strict=True)
    raw = (base / CUM).read_bytes()
    if sha(raw) != CUM_SHA:
        raise ValueError('Immutable completed CUM120 SHA differs')
    cumulative = json.loads(raw);mirrors = [Path(x) for x in cumulative['mirrors']]
    devs = cumulative['completed_dev_evaluations']
    if [x['outer'] for x in devs] != list(range(5, 121, 5)):
        raise ValueError('Exactly24 actual completed dev5..120 records required')
    pins = []
    for dev in devs:
        outer = dev['outer'];source = PARENT if outer == 5 else CHILD
        if dev['source_commit'] != source or dev['rows'] != 512 or Path(dev['ledger']).name != dev['ledger']:
            raise ValueError('Immutable dev source/rows/file differs')
        run = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + source
        pins.append(dict(outer=outer, code_commit=source, config_sha256=CONFIGS[source], model_identity=MODEL,
            data_identity=DATA, job_id='not-slurm;systemd:' + source, run=run,
            path=run + '/output/validation-cost/' + dev['ledger'], phase=dev['phase'],
            ledger_sha256=dev['ledger_sha256'], F1=dev['F1'], EM=dev['EM']))
    if os.environ.get('TREE_MGMT_CONTROL_PATH') != '/tmp/tree-mgmt-3fw9a1ab/control':
        raise ValueError('Explicit original task management ControlPath required')
    spec = importlib.util.spec_from_file_location('dev_cost_retained18_route', args.ssh_helper)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    code = REMOTE.replace('PIN_DATA', repr(pins))
    try:
        response = helper.run('env CUDA_VISIBLE_DEVICES= python3 -B -c ' + shlex.quote(code), target='h100-10', timeout=60)
        if response.returncode:
            raise RuntimeError('Read-only dev summary collection unavailable: ' + response.stderr[-2000:])
        result = json.loads(response.stdout)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        failure = dict(status='DEV_NATIVE_COST_SOURCE_READBACK_UNAVAILABLE_NO_TRAINING_FAULT_INFERRED',
            observed_utc=datetime.now(timezone.utc).isoformat(), error_type=type(exc).__name__, error=str(exc),
            source_cumulative_sha256=CUM_SHA, training_operations=False, automatic_retry=False,
            ssh_helper_sha256=sha(args.ssh_helper.read_bytes()))
        if isinstance(exc, subprocess.TimeoutExpired):
            failure['error'] = 'Existing helper read-only SSH exceeded its timeout; no automatic retry.'
            failure['timeout_seconds'] = exc.timeout
        item = publish(base, mirrors, 'DEV5_120_NATIVE_COST_COLLECTION_ERROR.json', encoded(failure))
        print(json.dumps(item), flush=True)
        raise SystemExit(1)
    if result.get('status') != 'ACTUAL_DEV5_120_CONTROL_SUMMARIES_READ' or result.get('remote_nice') != 15 or [x['outer'] for x in result.get('items', [])] != list(range(5, 121, 5)):
        raise ValueError('Actual completed control-summary response differs')
    rows = []
    for pin, item in zip(pins, result['items']):
        raw = item['raw_summary_utf8'].encode();value = json.loads(raw);meta = item['source']
        if sha(raw) != meta['sha256'] or len(raw) != meta['bytes'] or meta['path'] != pin['path']:
            raise ValueError('Actual source control JSON SHA/bytes differs')
        saved = publish(base, mirrors, 'dev_native_cost_summaries/DEV_OUTER%04d_SUMMARY.json' % pin['outer'], raw)
        rows.append(dict(outer=pin['outer'], code_commit=pin['code_commit'], config_sha256=pin['config_sha256'],
            run=pin['run'], phase=value['phase'], actual_completed_outer=value['actual_completed_outer'],
            evaluated_rows=value['evaluated_rows'], counters=value.get('counters'), source_control_summary=meta,
            local_original_summary=saved, ledger_sha256=value.get('ledger_sha256'),
            start_epoch=value.get('start_epoch'), end_epoch=value.get('end_epoch'),
            wall_seconds=value.get('wall_seconds'), trainer_process_cpu_seconds=value.get('trainer_process_cpu_seconds'),
            pad_token_id=value.get('pad_token_id'), eos_token_id=value.get('eos_token_id'),
            observer_source_sha256=value.get('observer_source_sha256'), cost_count_scope=value.get('cost_count_scope'),
            limitations=value.get('limitations'), precise_engine_emitted_tokens=None, scoring_call_event_total=None))
    keys = sorted({k for row in rows for k in (row['counters'] or {})})
    def aggregate(values):
        known = [v for v in values if numeric(v)]
        total = math.fsum(known) if known else None
        if isinstance(total, float) and total.is_integer():total = int(total)
        return dict(total=total if len(known) == len(values) else None, known_sum=total,
            missing_outers=[row['outer'] for row, v in zip(rows, values) if not numeric(v)])
    summary = dict(summary_schema='tree_dev5_120_native_cost_supplement_v1',
        status='ACTUAL_COMPLETED_DEV_CONTROL_SUMMARY_COST_READBACK', observed_epoch=result['observed_epoch'],
        source_cumulative=dict(path=CUM, bytes=(base / CUM).stat().st_size, sha256=CUM_SHA),
        completed_dev_outers=[row['outer'] for row in rows], rows=rows,
        native_counter_totals={key:aggregate([(row['counters'] or {}).get(key) for row in rows]) for key in keys},
        original_validation_observer_wall_seconds=aggregate([row['wall_seconds'] for row in rows]),
        original_validation_trainer_process_cpu_seconds=aggregate([row['trainer_process_cpu_seconds'] for row in rows]),
        prior_fullmetric_testing_phase_wall_seconds=cumulative['completed_dev_testing_phase_wall_seconds'],
        newly_bound_dev_ledger_sha_outers=[row['outer'] for row, dev in zip(rows, devs) if dev['ledger_sha256'] is None and row['ledger_sha256']],
        precise_engine_emitted_tokens=None, scoring_call_event_total=None,
        scopes=['submitted_prefill_token_events are submitted repeated events,not uniqueKV/cached tokens/FLOPs.',
            'returned_nonpad_model_token_events exclude PAD and excludeEOS when PAD==EOS;not exactengineemitted.',
            'http_attempts/completed/failed are actual batchedHTTPcalls;http_query_events are individual queryevents,notHTTPcallcount.',
            'Original summary counters retain original meanings and missing fields remain null. No means expanded into totals.',
            'Validation observerwall and processCPU are already contained in outer/service scopes;not added again to immutableCUM120 or whole serviceCPU.',
            'No scoring-call event counter in this observer;do not infer zero or evaluatedrowcount.',
            'Dev5 belongs toba9/22be;dev10..120 belong to778/355;effective ancestry retained.'],
        read_only=True, remote_files_written=False, jsonl_rollout_tensor_or_model_read=False,
        collector_source_sha256=sha(Path(__file__).read_bytes()), remote_program_sha256=sha(code.encode()),
        ssh_helper_sha256=sha(args.ssh_helper.read_bytes()), collector_cpu_seconds=result['collector_process_cpu_seconds'],
        collector_cpu_scope='This small readback process only;not added to validation/service CPU.',
        remote_nice=result['remote_nice'], immutable_CUM120_rewritten=False,
        STATUS_REPORT_FUP_or_training_modified=False, tests_or_review_performed=False)
    info = publish(base, mirrors, 'DEV5_120_NATIVE_COST_SUPPLEMENT.json', encoded(summary))
    md = ('# 已完成dev5–120原生控制计数补充\n\n'
        '只读24份实际COMPLETE小控制summary，绑定source/config/model/data/outer与原文件SHA。'
        'dev5属于ba9/22be，后续属于778/355；原summary字节及独立补充双镜像一致。\n\n'
        '计数保持原名：submitted_prefill_token_events、returned_nonpad_model_token_events、http_attempts、'
        'http_query_events等。PAD==EOS时nonpad排除EOS，不能称精确engine emitted；queryevents不冒充HTTPcalls。'
        '缺失评分调用计数、精确engine emitted仍null，不从均值/评测题数推算。\n\n'
        'observer验证wall与trainer processCPU保留独立范围，已经包含于整轮/服务，不再相加。'
        'CUM120保持原SHA；只保存小控制JSON，未读JSONL/rollout/tensor/model，无训练或状态操作。\n')
    md_info = publish(base, mirrors, 'DEV5_120_NATIVE_COST_SUPPLEMENT.md', md.encode())
    if sha((base / CUM).read_bytes()) != CUM_SHA:
        raise ValueError('Original immutable CUM120 changed')
    print(json.dumps(dict(supplement=info, markdown=md_info, controls=len(rows),
        native_counter_totals=summary['native_counter_totals'], newly_bound_dev_ledger_sha_outers=summary['newly_bound_dev_ledger_sha_outers'],
        wall_seconds=summary['original_validation_observer_wall_seconds'],
        trainer_process_cpu_seconds=summary['original_validation_trainer_process_cpu_seconds'],
        original_CUM120_unchanged=True, remote_files_written=False)), flush=True)


if __name__ == '__main__':
    main()
