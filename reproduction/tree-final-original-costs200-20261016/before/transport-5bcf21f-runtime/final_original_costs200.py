#!/usr/bin/env python3
"""Gated original-file cost200 collection and pure local aggregation.

Source preparation is not actual200 evidence. Neither phase trains/evaluates,
recaptures native state, invokes source1ed, writes canonical state, or creates
historical polls/FULLs/intermediate CUM snapshots. Old190 is byte protected.

Future CLI (actual byte/SHA-bound request/manifest only):
  python3 -B work/tree-speed-repair-20261008/final_original_costs200.py prepare-pins \
    --workspace "$WS" --pins "$ACTUAL_REQUEST" --pins-bytes "$BYTES" --pins-sha256 "$SHA"
  python3 -B work/tree-speed-repair-20261008/final_original_costs200.py collect \
    --workspace "$WS" --pins "$ACTUAL_PINS" --pins-bytes "$BYTES" --pins-sha256 "$SHA"
  python3 -B work/tree-speed-repair-20261008/final_original_costs200.py aggregate \
    --workspace "$WS" --pins "$ACTUAL_PINS" --pins-bytes "$BYTES" --pins-sha256 "$SHA" \
    --capture-bytes "$ACTUAL_CAPTURE_BYTES" --capture-sha256 "$ACTUAL_CAPTURE_SHA"

prepare-pins and collect each use one separately counted read-only helper call.
O_EXCL acquisition forbids duplicate submissions. The complete bounded response
is durably framed before dependent output rendering. recover-prepare and
recover-collect accept --response-bytes/--response-sha256, read only that saved
response and never call SSH. aggregate uses saved SHA-bound scalar parts only.
See INPUT_CONTRACT.md. Rolling native200 is count evidence, not HF acceptance.
"""
import argparse
import ast
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import uuid
import zlib


CHILD = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INV = 'a313bda919834d24b4c5d491698d39ce'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:'+CHILD
BASE_NAME = 'outputs/tree-nested-branch-handoff-20261013'
STEM = 'FINAL200_ORIGINAL_COSTS'
CONTROL_PATH = '/tmp/tree-mgmt-3fw9a1ab/control'
APP = 'work/tree-speed-repair-20261008/nested-local-cost-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/append_nested_local_costs.py'
BACK = str(Path(APP).with_name('backfill_nested_candidate_costs.py'))
CORE = 'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
HELPER = 'work/tree-speed-repair-20261008/ssh_retained18.py'
GUARD = 'work/tree-speed-repair-20261008/nested-native5-recovery-source/search_r1/baselines/tree_guard_continue.py'
RANK_WRITER = 'work/tree-speed-repair-20261008/nested-native5-recovery-source/search_r1/baselines/tree_rank_audit.py'
STATIC = {APP: (68224, '6c27a48be242ad8288f285f332e0b141cdad94aa1302205423b7c5aa68d3245d'),
          BACK: (33500, 'f64872751d87d497f46b860d27be5c64147d37051df4b605f04f5b40ab77c4d4'),
          CORE: (39098, '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'),
          HELPER: (4411, '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'),
          GUARD: (8516, '3a46fe46fc2afa078e582d473f711b820839dd37e58b21a68c09b22cad61e574'),
          RANK_WRITER: (10694, '4ef35ac5e0ab0089a669fd459a342678cd9dd46f00e4cd51c917f191e56c183b')}
PROTECTED = {
 'CUMULATIVE_COST_THROUGH190.json': (84345, '46b9bf8796a670e5f19b2153dc3115f1060c766019d636694e45b023555bca0a'),
 'COST_OUTER190_LOCAL_RECEIPT.json': (69542, 'f28f58af3c804a178bd4525ad5e191c2069dd6fb84addab747d95c1c60bd1612'),
 'LEARNING_CURVE_COMPLETE_THROUGH190.json': (1194660, '5a03840da2fa65bd1c00c0465148796db36bba0262448b631157d6e98dcf92da'),
 'LEARNING_CURVE_COMPLETE_THROUGH190.csv': (208230, '9379ba15d35a210095a6efc970506c97c786c7ecb9c6c7606035136a88f75a6b'),
 'DEV5_190_NATIVE_COST_SUPPLEMENT.json': (7131, '818265f0703c9b44b8aeb2d57bb42aa8d2a3ff80a1119e531a4b9bf2f1046cb2'),
 'DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json': (81037, '236e318fa205a3746ebc9eac6fbef5097b221f2e0bb072cb47c91236538256e4')}
LOCAL, RAW, GENERATED = {}, {}, {}


class AttemptAlreadyClaimed(ValueError):
    pass


def aggregate_values(outers, values, scope, known_values=None, record_counts=None, required_counts=None):
    """Complete totals and partial known record sums have different coverage."""
    need(len(outers)==len(values),'Aggregate outer/value coverage')
    known_values=values if known_values is None else known_values
    need(len(known_values)==len(outers),'Partial known-value coverage')
    known=math.fsum(v for v in known_values if number(v))
    if known.is_integer():known=int(known)
    complete=all(number(v) for v in values)
    value=math.fsum(values) if complete else None
    if isinstance(value,float) and value.is_integer():value=int(value)
    result=dict(total=value,known_sum=known,known_outers=[n for n,v in zip(outers,values) if number(v)],
                missing_outers=[n for n,v in zip(outers,values) if not number(v)],complete=complete,scope=scope,
                partial_known_outers=[n for n,v,k in zip(outers,values,known_values) if not number(v) and number(k)])
    if record_counts is not None:
        need(len(record_counts)==len(outers) and len(required_counts)==len(outers),'Record-level coverage')
        result.update(field_record_counts_by_outer=dict(zip(map(str,outers),record_counts)),
                      required_record_counts_by_outer=dict(zip(map(str,outers),required_counts)),
                      known_record_count=sum(v for v in record_counts if type(v) is int),
                      required_record_count=sum(v for v in required_counts) if all(type(v) is int for v in required_counts) else None)
    return result


def development_record(row,metric):
    value=row['value']
    return dict(outer=row['outer'],source_commit=row['source_commit'],ledger=Path(row['source']['path']).name,
                ledger_sha256=value['ledger_sha256'],rows=512,phase=value['phase'],
                testing_phase_wall_seconds=metric.get('timing_s/testing'),F1=value['metrics']['val/f1/all'],
                EM=value['metrics']['val/em/all'],original_control_source=row['source'])


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def enc(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def locate(value):
    p = Path(value)
    need(p.parts and '..' not in p.parts, 'Invalid local path')
    p = p if p.is_absolute() else W/p if p.parts[0] in ('outputs', 'work') else B/p
    p = p.resolve(strict=True)
    need(p.is_relative_to(W) and p.is_file(), 'Local source leaves workspace')
    return p


def read(ref, mirrors=False):
    need(type(ref.get('bytes')) is int and 0 < ref['bytes'] <= 4*1024*1024 and
         re.fullmatch('[0-9a-f]{64}', ref.get('sha256', '')), 'Actual local byte/SHA descriptor required')
    p = locate(ref.get('path') or ref['file']);raw = p.read_bytes()
    need((len(raw), sha(raw)) == (ref['bytes'], ref['sha256']), 'Local source SHA/bytes differ: '+str(p))
    if mirrors:
        need(p.is_relative_to(B), 'Mirrored evidence outside BASE')
        for root in M:
            need((root/p.relative_to(B)).read_bytes() == raw, 'Local input mirror differs')
    name = str(p.relative_to(W))
    if name in RAW:
        need(RAW[name] == raw, 'Local source changed during invocation')
    LOCAL[name] = dict(path=name, bytes=len(raw), sha256=sha(raw), mirror_checked=mirrors)
    RAW[name] = raw
    return raw


def bound(ref, mirrors=False):
    return strict(read(ref, mirrors))


def ref(path):
    p = locate(path);raw=p.read_bytes()
    return dict(path=str(p.relative_to(W)), bytes=len(raw), sha256=sha(raw))


def publish(name, raw):
    need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Generated own output name')
    for root in [B]+M:
        p=root/name
        need(not p.exists() or p.read_bytes() == raw, 'Immutable output exists with different bytes: '+name)
    for root in [B]+M:
        p=root/name;p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tmp=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
            with tmp.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:
                os.link(tmp,p)
            except FileExistsError:
                need(p.read_bytes()==raw,'Concurrent immutable publication differs: '+name)
            finally:
                tmp.unlink()
        need(p.read_bytes()==raw, 'Published byte readback differs')
    GENERATED[name]=raw
    return ref(B/name)


def claim_attempt(name, value):
    """Authoritative primary O_EXCL wins before any helper submission."""
    path=B/name;path.parent.mkdir(parents=True,exist_ok=True);raw=enc(value)
    try:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        existing=path.read_bytes()
        raise AttemptAlreadyClaimed('Attempt already claimed; no helper call. Existing bytes/SHA '+str(len(existing))+'/'+sha(existing))
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    GENERATED[name]=raw
    # Mirror failure also leaves the primary claim; never release or retry it.
    return publish(name,raw)


def persist_response(name, header, stdout, stderr):
    payload=stdout.encode('utf-8');errors=stderr.encode('utf-8')
    need(len(payload)<=32*1024*1024 and len(errors)<=32768,'Bounded audit response; no raw assets')
    header=dict(header,stdout_bytes=len(payload),stdout_sha256=sha(payload),
                stderr_bytes=len(errors),stderr_sha256=sha(errors))
    first=enc(header);need(len(first)<=32768,'Response framing header budget')
    raw=first+payload+errors
    # Primary is durable before mirrors and before parsing/rendering any outputs.
    return publish(name,raw)


def saved_response(name,size,digest):
    path=B/name;raw=path.read_bytes()
    need(0<len(raw)<=32*1024*1024+65536 and (len(raw),sha(raw))==(size,digest),'Saved actual response byte/SHA')
    first,body=raw.split(b'\n',1);header=strict(first);length=header['stdout_bytes']
    stdout,stderr=body[:length],body[length:]
    need((len(stdout),sha(stdout))==(length,header['stdout_sha256']) and
         (len(stderr),sha(stderr))==(header['stderr_bytes'],header['stderr_sha256']),'Saved response framing')
    return header,stdout.decode('utf-8'),stderr.decode('utf-8')


def retain_error(phase,exc,response_name=None):
    path=B/response_name if response_name else None
    durable=ref(path) if path and path.exists() else None
    value=dict(schema='tree_final200_original_cost_read_error_v2',phase=phase,
               observed_utc=datetime.now(timezone.utc).isoformat(),error_type=type(exc).__name__,error=str(exc)[:2000],
               durable_response=durable,local_recovery_available=durable is not None,
               automatic_retry=False,training_fault_inferred=False,remote_replay_allowed=False)
    name=STEM+'_READ_ERROR.json'
    if (B/name).exists():name=STEM+'_READ_ERROR_'+uuid.uuid4().hex+'.json'
    publish(name,enc(value))


def checks():
    observations=[]
    for i in range(2):
        for name, q in LOCAL.items():
            raw=(W/name).read_bytes();need(raw==RAW[name], 'Protected original changed')
            if q['mirror_checked']:
                for root in M:
                    need((root/(W/name).relative_to(B)).read_bytes()==raw, 'Protected original mirror changed')
        for name,raw in GENERATED.items():
            for root in [B]+M:
                need((root/name).read_bytes()==raw, 'Generated scalar output changed')
        observations.append(dict(pass_number=i+1, observed_utc=datetime.now(timezone.utc).isoformat(),
                                 files=len(LOCAL), mirrored_files=sum(q['mirror_checked'] for q in LOCAL.values()),
                                 generated_three_copy_files=len(GENERATED),
                                 generated_metadata_primary_bytes=sum(len(raw) for raw in GENERATED.values()),
                                 generated_metadata_three_copy_bytes=3*sum(len(raw) for raw in GENERATED.values())))
    return observations


def expand(value):
    if not isinstance(value, dict) or value.get('storage')!='complete_original_field_parts':
        return value
    out={} if value['container_type']=='mapping' else []
    for q in value['parts']:
        item=bound(q, True)
        need(item.get('schema')=='cumulative_scalar_field_part_v1' and item.get('original_field'), 'Complete part schema')
        part=item['value']
        if isinstance(out, dict):
            need(isinstance(part, dict) and not set(out).intersection(part), 'Duplicate part keys')
            out.update(part)
        else:
            need(isinstance(part, list), 'Complete list part');out.extend(part)
    need(len(out)==value['original_entries'], 'Complete part entry count')
    return out


def split_product(value, prefix):
    """Same complete-field protocol; no trimming to fit a byte budget."""
    result=copy.deepcopy(value);parts={}
    for key in sorted(result, key=lambda k:len(enc(result[k])), reverse=True):
        if len(enc(result))<=100000:
            break
        data=result[key]
        if not isinstance(data,(dict,list)):
            continue
        current={} if isinstance(data,dict) else [];chunks=[]
        for k,v in data.items() if isinstance(data,dict) else enumerate(data):
            candidate=copy.deepcopy(current)
            if isinstance(data,dict):candidate[k]=v
            else:candidate.append(v)
            if len(enc(dict(schema='cumulative_scalar_field_part_v1', original_field=key, value=candidate)))>100000 and current:
                chunks.append(current);current={k:v} if isinstance(data,dict) else [v]
            else:current=candidate
        if current:chunks.append(current)
        refs=[]
        for i,chunk in enumerate(chunks):
            name=f'final_original_costs200/{prefix}_{key}_PART{i:03d}.json'
            raw=enc(dict(schema='cumulative_scalar_field_part_v1', original_field=key, value=chunk))
            need(len(raw)<=100000, 'One complete scalar evidence item exceeds100k; no truncation')
            parts[name]=raw;refs.append(dict(file=name,bytes=len(raw),sha256=sha(raw),json_selector='value'))
        result[key]=dict(storage='complete_original_field_parts',container_type='mapping' if isinstance(data,dict) else 'list',
                         original_entries=len(data),parts=refs)
    need(len(enc(result))<=100000, 'Complete scalar descriptor exceeds100k')
    return parts, result


def inputs(args):
    manifest=bound(dict(path=str(args.pins),bytes=args.pins_bytes,sha256=args.pins_sha256))
    preparing=args.phase in ('prepare-pins','recover-prepare')
    need(manifest['schema']==('tree_final200_original_prepare_request_v1' if preparing else 'tree_final200_original_cost_input_manifest_v1') and
         manifest['status']==('ACTUAL200_ORIGINAL_PREPARATION_INPUTS_READY' if preparing else 'ACTUAL200_ORIGINAL_FILE_INPUTS_FROZEN'),
         'Actual200 preparation request or complete frozen manifest required')
    for key,value in dict(source_commit=CHILD,config_sha256=CONFIG,model_identity=MODEL,data_identity=DATA,
                          unit=UNIT,run=RUN,actual_training_pid=503255,actual_training_invocation_id=INV).items():
        need(manifest.get(key)==value, 'Actual200 manifest identity: '+key)
    need(locate(manifest['source_command']['path'])==Path(__file__).resolve(), 'Executing source pin')
    read(manifest['source_command']);read(manifest['remote_source'])
    need(locate(manifest['remote_source']['path'])==Path(__file__).with_name('final_original_costs200_remote.py').resolve(),
         'Remote source must be adjacent frozen file')
    for name,(size,digest) in STATIC.items():
        read(dict(path=name,bytes=size,sha256=digest))
    for name,(size,digest) in PROTECTED.items():
        bound(dict(path=str(B/name),bytes=size,sha256=digest),True) if name.endswith('.json') else read(dict(path=str(B/name),bytes=size,sha256=digest),True)
    prior=bound(dict(path=str(B/'CUMULATIVE_COST_THROUGH190.json'),bytes=PROTECTED['CUMULATIVE_COST_THROUGH190.json'][0],
                     sha256=PROTECTED['CUMULATIVE_COST_THROUGH190.json'][1]),True)
    prior={k:expand(v) for k,v in prior.items()}
    native=bound(manifest['actual200']['native_metadata'],True)
    poll=bound(manifest['actual200']['progress_poll'])
    resource=bound(manifest['actual200']['resource'])
    core=module(W/CORE,'final200_published_metadata_gates')
    need(poll.get('run')==RUN and poll.get('unit')==UNIT and poll.get('source_commit')==CHILD and
         poll.get('node')=='h100-10' and poll['progress']['actual_completed_outer_from_attempt_results']==200 and
         not poll['progress']['errors'], 'Actual200 original progress')
    entries=[int(n) for n in re.findall(r'epoch\s+\d+,\s*step\s+(\d+)\b',poll['logs']['stage.log'])]
    need(not entries or max(entries)<=200,'No actual training201 entry in this final inventory')
    core.validate_restore(poll)
    record,guard,completed,totals=core.validate_native(native,poll['progress'],200)
    metric=core.validate_metric(poll,200);core.validate_ranks(poll,metric,200)
    need(metric is not None,'Wait for original complete metric200 before preparation/collection; native rolling200 alone is insufficient')
    need(record['driver']['next_step']==201, 'Native200 saved cursor')
    state=resource.get('training_systemd') or resource.get('systemd') or (resource.get('state') or {}).get('training_unit_observation')
    state=state if isinstance(state,dict) else dict(line.split('=',1) for line in state.splitlines() if '=' in line)
    need(state.get('InvocationID')==INV and state.get('MainPID') in ('0','503255'), 'Original capture-time unit identity')
    if manifest['actual200'].get('full_receipt') is not None:
        full=bound(manifest['actual200']['full_receipt'],True)
        need(full['completed_outer']==200 and full['source_commit']==CHILD and full['native_manifest_sha256']==native['manifest_sha256']
             and full.get('actual_next_outer_entered') is None, 'Actual terminal FULL200 only; no actual201')
    captured={}
    for q in prior['candidate_cost_backfill_join']['inputs']:
        part=bound(dict(path=str(B/q['file']),bytes=(B/q['file']).stat().st_size,sha256=q['sha256']),True)
        row=part['rollouts'][0]
        captured[q['outer']]=dict(bytes=q['audit_file_bytes'],sha256=q['audit_file_sha256'],
                                  audit_file=Path(row['audit_file']).name)
    oldcost=bound(dict(path=str(B/'COST_OUTER190_LOCAL_RECEIPT.json'),bytes=PROTECTED['COST_OUTER190_LOCAL_RECEIPT.json'][0],
                       sha256=PROTECTED['COST_OUTER190_LOCAL_RECEIPT.json'][1]),True)
    oldcandidate=bound(oldcost['actual_candidate190'],True);audit=oldcandidate['source_files']['audit']
    captured[190]=dict(bytes=audit['bytes'],sha256=audit['sha256'],audit_file=Path(audit['path']).name)
    failed=manifest.get('prior_failed_preparation_response')
    if failed is not None:
        need(locate(failed['path'])==B/(STEM+'_PREPARE_RESPONSE.txt'),'Exact preserved failed preparation response')
        read(failed,True)
        failed_header,_,_=saved_response(STEM+'_PREPARE_RESPONSE.txt',failed['bytes'],failed['sha256'])
        need(failed_header['phase']=='prepare-pins' and failed_header['exit_code']!=0 and
             failed_header['actual_remote_calls']==1,'Explicit known failed preparation, never unknown replay')
    if preparing:
        tag=manifest.get('preparation_attempt_tag','')
        need(args.prepare_attempt_tag==tag and tag in ('','REPAIR001'),'Bound authorized preparation attempt tag')
        need(not tag or failed is not None,'Repair qualification requires preserved known failed response')
    if not preparing:
        need(manifest['remote_pins']['schema']=='tree_final200_original_remote_pins_v1','Explicit own remote inputs')
        need([q.get('outer') for q in manifest['remote_pins']['periodic_controls']]==[195,200],
             'Only original195+200 controls are newly read; reuse protected38')
        remote_rows=[row for run in manifest['remote_pins']['runs'] for row in run['rows']]
        need(all(run['rank_writer_sha256']==STATIC[RANK_WRITER][1] for run in manifest['remote_pins']['runs']),
             'All three frozen source commits have the same actual rank writerSHA')
        need([row.get('outer') for row in remote_rows]==list(range(1,201)),'Freeze each original accepted outer once')
        for row in remote_rows:
            if 3<=row['outer']<=190:
                q=captured[row['outer']]
                need((row['audit'].get('bytes'),row['audit'].get('sha256'))==(q['bytes'],q['sha256']),
                     'Keep prior3..190 original audit capture SHA/bytes exactly')
            if row['outer'] in (1,2):
                need(row['audit'].get('historical_capture_sha256') is None,'No invented historical auditSHA for1/2')
        if manifest.get('preparation_response') is not None:
            q=manifest['preparation_response'];name=Path(q['path']).name
            need(re.fullmatch(STEM+r'_PREPARE(?:_REPAIR001)?_RESPONSE\.txt',name) and locate(q['path'])==B/name,'Actual preparation response filename')
            header,_,_=saved_response(name,q['bytes'],q['sha256'])
            need(header['phase']=='prepare-pins' and header['exit_code']==0 and header['actual_remote_calls']==1 and
                 manifest.get('preparation_remote_calls')==1+int(failed is not None),'Successful preparation and preserved failed calls are separate actual invocations')
        else:
            need(manifest.get('preparation_remote_calls',0)==0,'Do not infer a preparation call without its response')
    cross=B/'ORIGINAL_POLL_TRAINING_METRICS191_196_READBACK_20261016.json'
    if cross.exists():
        bound(dict(path=str(cross),bytes=61377,sha256='e3f2ae25a3f2b4cd1bd5b378bf446add723ed6b49c8c76941cc8f1d43ae84415'))
    owner=bound(dict(path=str(B/'DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json'),
                     bytes=PROTECTED['DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json'][0],
                     sha256=PROTECTED['DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json'][1]),True)
    need(owner['controls']==38 and owner['same_entry_executor_attested_cumulative_main_invocations']==
         owner['same_entry_executor_attested_cumulative_remote_incremental_invocations']==7,'Actual prior190 collector boundary')
    cached_rows=[]
    for q in owner['rows_part_refs']:
        part=bound(q,True);need(isinstance(part.get('value'),list),'Original190 dev rows part')
        cached_rows.extend(part['value'])
    need([row['outer'] for row in cached_rows]==list(range(5,191,5)), 'Original190 controls exact38 coverage')
    cached_controls=[]
    for old,row in zip(owner['local_original_control_refs'],cached_rows):
        raw=read(old,True);value=strict(raw);meta=row['source_control_summary']
        need(old['outer']==row['outer'] and (len(raw),sha(raw))==(meta['bytes'],meta['sha256']) and
             value['ledger_sha256']==row['ledger_sha256'], 'Protected original38 source/capture mismatch')
        cached_controls.append(dict(outer=row['outer'],source_commit=row['code_commit'],source=meta,
                                    raw_summary_utf8=raw.decode(),value=value,reused_original_control=True))
    return manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured


def remote_fragment(raw,mode):
    """Keep exact source segments needed for this mode, reducing inline argv."""
    source=raw.decode();tree=ast.parse(source)
    omit={'prepare'} if mode=='collect' else {'collect','candidates','rank_record','physical_attempt','metric_lines'}
    return '\n\n'.join(ast.get_source_segment(source,node) for node in tree.body
                       if not isinstance(node,ast.FunctionDef) or node.name not in omit)+'\n'


def transport_program(program):
    packed=base64.b64encode(zlib.compress(program.encode(),9)).decode()
    command='env CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
    helper=module(W/HELPER,'final200_original_existing_route');argv=helper.route(command,'h100-10')
    transport=dict(program_bytes=len(program.encode()),target_command_bytes=len(command.encode()),
                   management_command_bytes=len(argv[-1].encode()),all_local_argv_bytes=sum(len(x.encode())+1 for x in argv))
    need(transport['target_command_bytes']<=32768 and transport['management_command_bytes']<=65536 and
         transport['all_local_argv_bytes']<=80000,'Short compressed own-route budget; no SSH attempted')
    return helper,command,transport


def finish_collect(args,values,header,stdout,stderr,response_name):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured=values
    need(header['phase']=='collect' and header['exit_code']==0 and header['actual_remote_calls']==1 and header['input_manifest']==ref(args.pins),
         'Bound successful original collector response required')
    need(header['source_command']==manifest['source_command'] and header['remote_source']==manifest['remote_source'],
         'Saved response source differs from actual frozen inputs')
    result=strict(stdout)
    need(result['status']=='ORIGINAL200_RECORDS_READ' and result['remote_files_written'] is False and
         result['historical_live_poll_or_FULL_created'] is False,'Original response identity')
    rows=result['rows'];need([row['outer'] for row in rows]==list(range(1,201)),'Original200 rows')
    cross=str((B/'ORIGINAL_POLL_TRAINING_METRICS191_196_READBACK_20261016.json').relative_to(W))
    if cross in RAW:
        original=strict(RAW[cross]);need(original['schema']=='tree_original_poll_metric_scalar_crosscheck_v1','Original local crosscheck schema')
        for row in original['rows']:
            source=bound(row['source_poll'])
            need(row['metric'] in source['iteration_metrics'] and rows[row['outer']-1]['metric']==row['metric'],
                 'Fresh original191..196 line differs from original poll scalar')
    need([row['outer'] for row in result['controls']]==[195,200], 'Only two new original periodic controls')
    result['newly_read_original_control_outers']=[195,200]
    result['reused_original_control_outers']=[row['outer'] for row in cached_controls]
    result['controls']=cached_controls+result['controls']
    need([row['outer'] for row in result['controls']]==list(range(5,201,5)), 'Actual complete40 control sources')
    for row in result['controls']:
        for key in ('val/f1/all','val/em/all','val/qa_count/all','val/qa_count/nq','val/qa_count/hotpotqa'):
            need(row['value']['metrics'][key]==rows[row['outer']-1]['metric'].get(key),
                 'Original dev raw control and fresh original step scalar disagree')
    direct={str(rank):sum(row['rank_records'][rank]['successful_optimizer_calls'] for row in rows) for rank in range(8)}
    need(direct==completed,'All original per-outer rank sums versus actualnative200 credited totals')
    child_attempts={row['attempt_id'] for row in rows if row['source_commit']==CHILD}
    extras={row['value'].get('attempt_id') for row in result['noncredited_records'] if row['source_commit']==CHILD}
    extras.discard(None)
    need(not child_attempts.intersection(extras),'Credited attempt included again as noncredited')
    need(child_attempts|extras==set(range(6,guard['training_attempts']+1)),
         'Original child credited/refused/partial attempt inventory incomplete')
    need(rows[-1]['metric']==core_metric(poll),'Original metric200 differs from actual poll')
    native200=[row for row in result['native_metadata'] if row['outer']==200]
    need(native200 and any(row['source']['sha256']==native['manifest_sha256'] for row in native200),
         'Fresh original native200 manifest differs from successful metadata capture')
    actualdev=[d for d in poll['development'] if d.get('outer')==200 and d.get('status')=='VALIDATION_COST_COMPLETE']
    need(len(actualdev)==1 and actualdev[0]['rows']==512 and
         actualdev[0]['ledger_sha256']==result['controls'][-1]['value']['ledger_sha256'] and
         actualdev[0]['metrics']==result['controls'][-1]['value']['metrics'],
         'Original COMPLETE control200 differs from actual source-bound poll')
    result.update(input_manifest=ref(args.pins),generator=ref(Path(__file__)),remote_generator=manifest['remote_source'],
                  published_candidate_function_source=dict(path=BACK,bytes=STATIC[BACK][0],sha256=STATIC[BACK][1]),
                  original_read_only_helper=dict(path=HELPER,bytes=STATIC[HELPER][0],sha256=STATIC[HELPER][1]),
                  remote_program_sha256=header['remote_program_sha256'],remote_calls=1,remote_exit_code=header['exit_code'],
                  stdout_bytes=len(stdout.encode()),stdout_sha256=sha(stdout.encode()),
                  stderr_bytes=len(stderr.encode()),stderr_sha256=sha(stderr.encode()),
                  transport=header['transport'],durable_response=ref(B/response_name),preparation_remote_calls_separate=manifest.get('preparation_remote_calls',0),observer_cost_added_to_training=False)
    parts,descriptor=split_product(result,'CAPTURE')
    for name,raw in parts.items():publish(name,raw)
    product=publish(STEM+'_CAPTURE.json',enc(descriptor))
    for control in result['controls'][-2:]:
        raw=control['raw_summary_utf8'].encode()
        need((len(raw),sha(raw))==(control['source']['bytes'],control['source']['sha256']),
             'Original control UTF8 is not a reconstructed JSON')
        publish(f"final_original_costs200/DEV_OUTER{control['outer']:04d}_ORIGINAL.json",raw)
    for row in result['rows']:
        n=row['outer'];identity=dict(source_commit=row['source_commit'],run=row['run'],
                                    unit='sst-tree-nested-branch-fresh200-'+row['source_commit'][:8]+'.service',
                                    job_id='not-slurm;systemd:'+row['source_commit'])
        scalar=dict(schema='tree_final200_original_scalar_source_v1',evidence_mode='RETROSPECTIVE_SOURCE_READBACK',
                    observed_epoch=result['observed_epoch'],outer=n,**identity,iteration_metrics=[row['metric']],
                    metric_line_sources=[dict(outer=n,**row['metric_line'])],
                    collection_manifest=ref(args.pins),capture_product=product,
                    historical_observation_epoch=None,historical_live_poll_created=False)
        publish(f'final_original_costs200/SCALARS_OUTER{n:04d}.json',enc(scalar))
        metadata=dict(summary_schema='final200_original_rollout_metadata_v1',run_directory=row['run'],
                      rollouts=[{k:row[k] for k in ('global_step','audit_file','audit_file_bytes','audit_file_sha256')}],
                      capture_product=product,evidence_mode='RETROSPECTIVE_SOURCE_READBACK')
        publish(f'final_original_costs200/ROLLOUT_METADATA_OUTER{n:04d}.json',enc(metadata))
    readbacks=checks()
    readback=B/(STEM+'_CAPTURE_READBACK.json')
    if not readback.exists():
        publish(readback.name,enc(dict(product=product,input_manifest=ref(args.pins),
                source_inputs=list(LOCAL.values()),protected_original_readbacks=readbacks,remote_calls=1,
                training_or_evaluation_operations=False,full_native_recapture=False)))
    else:
        previous_raw=readback.read_bytes();previous=strict(previous_raw)
        need(previous['product']==product,'Existing completed capture readback differs')
        publish(readback.name,previous_raw)
    checks()
    print(json.dumps(dict(status='ORIGINAL200_CAPTURED_NOT_FINAL_ACCEPTANCE',product=product,remote_calls=1,
                          source1ed_calls=0,actual_training201_claimed=False)))


def collect(args, values):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured=values
    need(os.environ.get('TREE_MGMT_CONTROL_PATH')==CONTROL_PATH,'Explicit original task ControlPath required')
    tree=ast.parse(RAW[BACK].decode())
    literal=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and
                 any(isinstance(name,ast.Name) and name.id=='REMOTE' for name in node.targets))
    functions=literal[literal.index('fields='):literal.index("run=Path(pin['run'])")]
    guard_source=RAW[GUARD].decode();guard_tree=ast.parse(guard_source)
    functions+='\n'+ast.get_source_segment(guard_source,next(node for node in guard_tree.body
                               if isinstance(node,ast.FunctionDef) and node.name=='validate_ledger'))+'\n'
    program=functions+'\n'+remote_fragment(read(manifest['remote_source']),'collect')+'\nPINS='+repr(manifest['remote_pins'])+"\n_RESULT=collect(PINS)\n_ENCODED=json.dumps(_RESULT,ensure_ascii=False,separators=(',',':'),allow_nan=False)\nneed(len(_ENCODED.encode())<=32*1024*1024,'Complete scalar output exceeds32MiB; no raw output emitted')\nprint(_ENCODED)\n"
    helper,command,transport=transport_program(program);checks()
    response_name=STEM+'_REMOTE_RESPONSE.txt'
    try:
        claim_attempt(STEM+'_ATTEMPT.json',dict(schema='tree_final200_original_cost_once_attempt_v2',
            started_utc=datetime.now(timezone.utc).isoformat(),input_manifest=ref(args.pins),transport=transport,
            remote_program_sha256=sha(program.encode()),authorized_remote_calls=1,phase='collect',automatic_retry=False))
    except AttemptAlreadyClaimed:
        raise
    except Exception as exc:
        retain_error('collect-claim-no-helper',exc);raise
    try:
        started=datetime.now(timezone.utc).isoformat();response=helper.run(command,target='h100-10',timeout=900)
        header=dict(schema='tree_final200_original_actual_response_v1',phase='collect',exit_code=response.returncode,
                    started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),input_manifest=ref(args.pins),
                    source_command=manifest['source_command'],remote_source=manifest['remote_source'],
                    remote_program_sha256=sha(program.encode()),transport=transport,actual_remote_calls=1)
        saved=persist_response(response_name,header,response.stdout,response.stderr)
        header,stdout,stderr=saved_response(response_name,saved['bytes'],saved['sha256'])
        finish_collect(args,values,header,stdout,stderr,response_name)
    except Exception as exc:
        retain_error('collect',exc,response_name);raise


def finish_prepare(args,values,header,stdout,stderr,response_name):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured=values
    need(header['phase']=='prepare-pins' and header['exit_code']==0 and header['actual_remote_calls']==1 and header['input_manifest']==ref(args.pins),
         'Actual successful separate preparation response required')
    need(header['source_command']==manifest['source_command'] and header['remote_source']==manifest['remote_source'],
         'Saved preparation source differs')
    prepared=strict(stdout)
    need(prepared['status']=='ORIGINAL200_PATHS_PREPARED_NOT_COST_ACCEPTANCE' and prepared['remote_files_written'] is False,
         'Prepare response status')
    result=copy.deepcopy(manifest);result.update(schema='tree_final200_original_cost_input_manifest_v1',
        status='ACTUAL200_ORIGINAL_FILE_INPUTS_FROZEN',remote_pins=prepared['remote_pins'],
        preparation_response=ref(B/response_name),preparation_remote_calls=1+int(manifest.get('prior_failed_preparation_response') is not None),
        preparation_current_remote_calls=1,preparation_prior_failed_remote_calls=int(manifest.get('prior_failed_preparation_response') is not None),collector_remote_calls_before_collection=0)
    raw=enc(result);need(len(raw)<=4*1024*1024,'Complete prepared manifest budget; no truncated paths')
    output=publish(STEM+'_INPUT_PINS.json',raw);checks()
    print(json.dumps(dict(status='ORIGINAL200_INPUT_PINS_PREPARED_NOT_COLLECTED',input_manifest=output,
          prepare_current_remote_calls=1,prepare_total_remote_calls=result['preparation_remote_calls'],collect_remote_calls=0,source1ed_calls=0)))


def prepare_pins(args,values):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured=values
    need(os.environ.get('TREE_MGMT_CONTROL_PATH')==CONTROL_PATH,'Explicit original task ControlPath required')
    seed=dict(schema='tree_final200_original_prepare_seed_v1',captured_audits=captured,
              rank_writer_sha256=STATIC[RANK_WRITER][1],old_control_names=[Path(q['source']['path']).name for q in cached_controls])
    program=remote_fragment(read(manifest['remote_source']),'prepare-pins')+'\nSEED='+repr(seed)+"\n_RESULT=prepare(SEED)\n_ENCODED=json.dumps(_RESULT,ensure_ascii=False,separators=(',',':'),allow_nan=False)\nneed(len(_ENCODED.encode())<=4*1024*1024,'Complete preparation output budget; no paths truncated')\nprint(_ENCODED)\n"
    helper,command,transport=transport_program(program);checks()
    prepare_stem=STEM+'_PREPARE'+('_'+args.prepare_attempt_tag if args.prepare_attempt_tag else '')
    response_name=prepare_stem+'_RESPONSE.txt'
    claim_attempt(prepare_stem+'_ATTEMPT.json',dict(schema='tree_final200_original_prepare_once_attempt_v1',
        started_utc=datetime.now(timezone.utc).isoformat(),input_manifest=ref(args.pins),transport=transport,
        remote_program_sha256=sha(program.encode()),authorized_remote_calls=1,phase='prepare-pins',automatic_retry=False))
    try:
        started=datetime.now(timezone.utc).isoformat();response=helper.run(command,target='h100-10',timeout=900)
        header=dict(schema='tree_final200_original_actual_response_v1',phase='prepare-pins',exit_code=response.returncode,
                    started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),input_manifest=ref(args.pins),
                    source_command=manifest['source_command'],remote_source=manifest['remote_source'],
                    remote_program_sha256=sha(program.encode()),transport=transport,actual_remote_calls=1)
        saved=persist_response(response_name,header,response.stdout,response.stderr)
        header,stdout,stderr=saved_response(response_name,saved['bytes'],saved['sha256'])
        finish_prepare(args,values,header,stdout,stderr,response_name)
    except Exception as exc:
        retain_error('prepare-pins',exc,response_name);raise


def recover(args,values):
    need(args.response_bytes is not None and args.response_sha256 is not None,'Bind the actual durable saved response')
    preparing=args.phase=='recover-prepare'
    name=(STEM+'_PREPARE'+('_'+args.prepare_attempt_tag if args.prepare_attempt_tag else '')+'_RESPONSE.txt') if preparing else STEM+'_REMOTE_RESPONSE.txt'
    try:
        header,stdout,stderr=saved_response(name,args.response_bytes,args.response_sha256)
        publish(name,(B/name).read_bytes())
        if preparing:finish_prepare(args,values,header,stdout,stderr,name)
        else:finish_collect(args,values,header,stdout,stderr,name)
        publish(STEM+'_LOCAL_RECOVERY_'+uuid.uuid4().hex+'.json',enc(dict(observed_utc=datetime.now(timezone.utc).isoformat(),
            phase=args.phase,response=ref(B/name),actual_remote_calls_this_recovery=0,source1ed_calls=0)))
        checks()
    except Exception as exc:
        retain_error(args.phase,exc,name);raise


def core_metric(poll):
    found=[m for m in poll['iteration_metrics'] if m.get('step')==200]
    need(found and all(m==found[0] for m in found),'Actual original complete metric200 required for cost summary')
    return found[0]


def aggregate(args, values):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls,captured=values
    capture_path=B/(STEM+'_CAPTURE.json')
    need(args.capture_bytes is not None and args.capture_sha256 is not None,'Bind the actual collect result before aggregation')
    capture=bound(dict(path=str(capture_path),bytes=args.capture_bytes,sha256=args.capture_sha256),True)
    capture={k:expand(v) for k,v in capture.items()}
    need(capture['input_manifest']==ref(args.pins) and capture['remote_calls']==1 and capture['remote_exit_code']==0,
         'Actual frozen original capture required')
    rows=capture['rows'];need([r['outer'] for r in rows]==list(range(1,201)),'Unique original200 rows')
    def total(values, scope, **coverage):
        return aggregate_values([r['outer'] for r in rows],values,scope,**coverage)
    sampling={}
    def cost_value(row,key):
        value=row['controller'].get('costs',{}).get(key)
        return value if number(value) else row['metric'].get('rollout/'+key)
    for key in ['generated_tokens','prefill_tokens','retrieval_calls','trajectories','padded_rows','raw_emitted_tokens',
                'raw_retained_tokens','raw_discarded_tail_tokens','raw_inherited_leaf_tokens',
                'semantic_prefix_scan_cpu_seconds','semantic_prefix_scan_wall_seconds']:
        sampling[key]=total([cost_value(r,key) for r in rows],
                            'Whole original Controller.generate scalar events; includes DP padding; branch subset is not added again.')
    for key in ['scoring_calls','scoring_wall_seconds']:
        sampling[key]=total([r['metric'].get('rollout/'+key) for r in rows],
                            'Original whole rollout scoring metric; never inferred from candidates, trajectories or means.')
    for key,old in prior['effective_training_sampling'].items():
        if old.get('total') is not None and key in sampling:
            values=[r['metric'].get('rollout/'+key) if key in ('scoring_calls','scoring_wall_seconds') else cost_value(r,key)
                    for r in rows[:190]]
            need(all(number(v) for v in values) and
                 math.isclose(math.fsum(values),old['total'],rel_tol=1e-12,abs_tol=1e-9),
                 'Fresh original1..190 costs disagree with preservedCUM190: '+key)
    branch={key:total([r['branch']['candidate_round_controller_ledger'].get(key) for r in rows],
                     'Unique original branch round ledger; subset of whole Controller; partial known records retained.',
                     known_values=[r['branch']['candidate_round_ledger_known_sums'].get(key) for r in rows],
                     record_counts=[r['branch']['candidate_round_ledger_field_record_counts'].get(key) for r in rows],
                     required_counts=[r['branch']['unique_round_count'] for r in rows])
            for key in ['generated_tokens','prefill_tokens','retrieval_calls','scoring_calls','scoring_wall_seconds']}
    candidate={key:total([r['branch']['candidate_outcomes'][key]['candidate_count'] for r in rows],
                         'Original outcome count; fallback overlaps accepted, never a third additive bucket.')
               for key in ['all','accepted','discarded','fallback']}
    outcomes={}
    for key in ['all','accepted','discarded','fallback']:
        fields=sorted({name for r in rows for name in r['branch']['candidate_outcomes'][key]['field_sums']})
        outcomes[key]=dict(field_sums={name:total([r['branch']['candidate_outcomes'][key]['field_sums'].get(name) for r in rows],
                         'Original candidate event/amortized scalar; partial known records retained; prefix not new generation.',
                         known_values=[r['branch']['candidate_outcomes'][key]['field_known_sums'].get(name) for r in rows],
                         record_counts=[r['branch']['candidate_outcomes'][key]['field_record_counts'].get(name) for r in rows],
                         required_counts=[r['branch']['candidate_outcomes'][key]['candidate_count'] for r in rows]) for name in fields})
    timings={key:total([r['metric'].get('timing_s/'+key) for r in rows],
                       'Original step phase. Dev/gen/ref/update/save are included in step wall, not added again.')
             for key in ['step','gen','ref','adv','update_actor','testing','stage_testing','save_checkpoint']}
    writes={str(rank):sum(r['rank_records'][rank]['successful_optimizer_calls'] for r in rows) for rank in range(8)}
    need(writes==completed,'Original rank versus native200 credited totals')
    devs=[]
    for row in capture['controls']:
        devs.append(development_record(row,rows[row['outer']-1]['metric']))
    need([row['outer'] for row in devs]==list(range(5,201,5)),'Original40 dev records; retain acceptance/periodic phases')
    wall=timings['step']['total']
    cumulative=dict(summary_schema='tree_final200_original_cumulative_cost_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        through_effective_outer=200,target_outer=200,origin='fresh',evidence_mode='RETROSPECTIVE_SOURCE_READBACK',
        effective_source_segments=[{**s,'last_outer':200 if s['code_commit']==CHILD else s['last_outer']} for s in prior['effective_source_segments']],
        rows_counted_once=[dict(outer=r['outer'],source_commit=r['source_commit'],attempt_id=r['attempt_id'],
                               row_id=r['source_commit']+':'+str(r['outer'])+':'+str(r['attempt_id'])) for r in rows],
        immutable_metadata_input_hashes={name:q for name,q in LOCAL.items()},original_capture=ref(capture_path),
        effective_training_sampling=sampling,effective_extra_candidates=candidate,effective_branch_controller_costs=branch,
        candidate_outcome_costs_partial=outcomes,stage_timing_seconds=timings,complete_outer_wall_seconds=wall,
        controller_rollout_wall_seconds=total([r['controller'].get('wall_seconds') for r in rows],'Original Controller wall; step subset.'),
        controller_trainer_process_cpu_seconds=total([r['controller'].get('trainer_process_cpu_seconds') for r in rows],
                                                    'Controller trainer-process CPU only; not worker/retriever/service CPU.'),
        actual_total_service_cpu_seconds=None,own_unit_service_CPUUsageNSec_asof=state.get('CPUUsageNSec'),
        own_unit_CPU_scope='Same InvocationID original capture-time counter; as-of service scope includes startup/dev/waits; no whole-ancestry terminal total inferred.',
        successful_optimizer_accounting=dict(successful_calls_per_rank_through200=completed,total_actual_calls_per_rank=totals,
            partial_optimizer_calls_per_rank=guard['partial_optimizer_calls_per_rank'],direct_observed_calls_per_rank=writes,
            source='Original200 per-outer rank deltas reconcile finalnative200; ancestors already included, no repeated cumulative sum.',
            independent_processed_opportunities_per_outer=[dict(outer=r['outer'],values=[v['opportunities_processed'] for v in r['rank_records']]) for r in rows],
            independent_skipped_opportunities_per_outer=[dict(outer=r['outer'],values=[v['opportunities_skipped'] for v in r['rank_records']]) for r in rows],
            original_fresh_attempt_physical_opportunities_separate=True,no_4000_or_all20_assumption=True),
        completed_dev_evaluations=devs,completed_dev_testing_phase_wall_seconds=math.fsum(d['testing_phase_wall_seconds'] for d in devs) if all(number(d['testing_phase_wall_seconds']) for d in devs) else None,
        dev_phase_already_in_outer_wall=True,pending_full_metric_outers=[],
        physical8gpu_shared_capacity_upper_bound=dict(physical_gpu_count=8,effective_outer_wall_seconds=wall,
            capacity_gpu_seconds_upper_bound=8*wall if number(wall) else None,actual_gpu_active_seconds=None,
            exclusive_billable_gpu_hours=None,gpu_energy_kwh=None,scope='Shared physical capacity upper bound; no own continuous utilization or billing attribution.'),
        historical_failed_costs=prior['historical_failed_costs'],historical_bootstrap_costs=prior['historical_bootstrap_costs'],
        historical_whole_service_segment_context=prior['historical_whole_service_segment_context'],
        original_noncredited_record_readback=capture['noncredited_records'],
        unavailable_historical_native_JSON=capture['native_metadata_missing'],
        metadata_readback_overhead=dict(process_cpu_seconds=capture['observer_process_cpu_seconds'],wall_seconds=capture['observer_wall_seconds'],added_to_training_or_service=False),
        no_combined_effective_plus_failed_or_segment_grand_total=True,missing_fields_are_null=True,
        missing_intermediate_CUM_or_live_FULL_not_a_training_or_final_evaluation_gate=True,
        source1ed_invoked_by_this_entry=False,final51713_completion_claimed=False,goal_complete=False,mirrors=list(map(str,M)))
    products,descriptor=split_product(cumulative,'COST')
    for name,raw in products.items():publish(name,raw)
    cost_product=publish(STEM+'_COST_TOTALS.json',enc(descriptor))
    summary=dict(schema='tree_final200_original_cost_summary_v1',status='ORIGINAL200_COST_AGGREGATED_NOT_FINAL_ACCEPTANCE',
                 original_capture=ref(capture_path),original_cost_totals=cost_product,
                 source_inputs=list(LOCAL.values()),source1ed_main_calls=0,source1ed_remote_calls=0,
                 all200_original_effective_rows=200,original_native_metadata_records=len(capture['native_metadata']),
                 original_rank_records=1600,complete_field_parts=len(products),
                 final_original_collector_main_calls=1,final_original_collector_remote_calls=1,
                 local_aggregation_main_calls=1,local_aggregation_remote_calls=0,
                 source1ed_prior190_actual_main_remote_preserved=[7,7],source1ed_new_calls=0,
                 prepare_pins_remote_calls=manifest.get('preparation_remote_calls',0),
                 native200_rolling_is_optimizer_count_evidence_only=True,HF_tokenizer_or_permanent_native_acceptance_claimed=False,
                 historical_live_FULL_or_poll_created=False,canonical_or_BOOK_modified=False,
                 actual_next_training201_claimed=False,independent_final_acceptance_pending=True,
                 protected_original_readbacks=checks())
    parts,descriptor=split_product(summary,'SUMMARY')
    for name,raw in parts.items():publish(name,raw)
    product=publish(STEM+'.json',enc(descriptor))
    checks()
    print(json.dumps(dict(status=summary['status'],product=product,
                  original_cost_totals=summary['original_cost_totals'],source1ed_calls=0)))


def main():
    global W,B,M
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare-pins','collect','recover-prepare','recover-collect','aggregate'])
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--pins',required=True,type=Path)
    parser.add_argument('--pins-bytes',required=True,type=int)
    parser.add_argument('--pins-sha256',required=True)
    parser.add_argument('--capture-bytes',type=int)
    parser.add_argument('--capture-sha256')
    parser.add_argument('--prepare-attempt-tag',default='')
    parser.add_argument('--response-bytes',type=int)
    parser.add_argument('--response-sha256')
    args=parser.parse_args();need(__debug__,'Do not disable actual evidence gates with -O')
    W=args.workspace.resolve(strict=True);B=W/BASE_NAME
    M=[W/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
       Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    values=inputs(args)
    if args.phase=='prepare-pins':prepare_pins(args,values)
    elif args.phase=='collect':collect(args,values)
    elif args.phase.startswith('recover-'):recover(args,values)
    else:aggregate(args,values)


if __name__=='__main__':
    main()
