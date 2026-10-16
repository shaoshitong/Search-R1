#!/usr/bin/env python3
"""Gated original-file cost200 collection and pure local aggregation.

Source preparation is not actual200 evidence. Neither phase trains/evaluates,
recaptures native state, invokes source1ed, writes canonical state, or creates
historical polls/FULLs/intermediate CUM snapshots. Old190 is byte protected.

Future CLI (each phase once, actual byte/SHA-bound manifest only):
  python3 -B work/tree-speed-repair-20261008/final_original_costs200.py collect \
    --workspace "$WS" --pins "$ACTUAL_PINS" --pins-bytes "$BYTES" --pins-sha256 "$SHA"
  python3 -B work/tree-speed-repair-20261008/final_original_costs200.py aggregate \
    --workspace "$WS" --pins "$ACTUAL_PINS" --pins-bytes "$BYTES" --pins-sha256 "$SHA" \
    --capture-bytes "$ACTUAL_CAPTURE_BYTES" --capture-sha256 "$ACTUAL_CAPTURE_SHA"

collect uses one readonly existing retained18 helper call. An immutable attempt
marker forbids blind replay after an unknown result. aggregate uses only saved
SHA-bound scalar parts and cannot call SSH. See INPUT_CONTRACT.md for actual
original file refs, optional preexisting capture hashes and source preparation.
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
            tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(raw);os.replace(tmp,p)
        need(p.read_bytes()==raw, 'Published byte readback differs')
    GENERATED[name]=raw
    return ref(B/name)


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
                                 generated_three_copy_files=len(GENERATED)))
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
    need(manifest['schema']=='tree_final200_original_cost_input_manifest_v1' and
         manifest['status']=='ACTUAL200_ORIGINAL_FILE_INPUTS_FROZEN', 'Actual200 manifest required')
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
    need(record['driver']['next_step']==201, 'Native200 saved cursor')
    state=resource.get('training_systemd') or resource.get('systemd') or (resource.get('state') or {}).get('training_unit_observation')
    state=state if isinstance(state,dict) else dict(line.split('=',1) for line in state.splitlines() if '=' in line)
    need(state.get('InvocationID')==INV and state.get('MainPID') in ('0','503255'), 'Original capture-time unit identity')
    if manifest['actual200'].get('full_receipt') is not None:
        full=bound(manifest['actual200']['full_receipt'],True)
        need(full['completed_outer']==200 and full['source_commit']==CHILD and full['native_manifest_sha256']==native['manifest_sha256']
             and full.get('actual_next_outer_entered') is None, 'Actual terminal FULL200 only; no actual201')
    need(manifest['remote_pins']['schema']=='tree_final200_original_remote_pins_v1', 'Explicit own remote inputs')
    need([q.get('outer') for q in manifest['remote_pins']['periodic_controls']]==[195,200],
         'Only original195+200 controls are newly read; reuse protected38')
    remote_rows=[row for run in manifest['remote_pins']['runs'] for row in run['rows']]
    need(all(run['rank_writer_sha256']==STATIC[RANK_WRITER][1] for run in manifest['remote_pins']['runs']),
         'All three frozen source commits have the same actual rank writerSHA')
    need([row.get('outer') for row in remote_rows]==list(range(1,201)), 'Freeze every original effective outer path once')
    captured={q['outer']:(q['audit_file_bytes'],q['audit_file_sha256']) for q in prior['candidate_cost_backfill_join']['inputs']}
    oldcost=bound(dict(path=str(B/'COST_OUTER190_LOCAL_RECEIPT.json'),bytes=PROTECTED['COST_OUTER190_LOCAL_RECEIPT.json'][0],
                       sha256=PROTECTED['COST_OUTER190_LOCAL_RECEIPT.json'][1]),True)
    oldcandidate=bound(oldcost['actual_candidate190'],True)
    audit=oldcandidate['source_files']['audit'];captured[190]=(audit['bytes'],audit['sha256'])
    for row in remote_rows:
        if 3<=row['outer']<=190:
            need((row['audit'].get('bytes'),row['audit'].get('sha256'))==captured[row['outer']],
                 'Keep prior3..190 original audit capture SHA/bytes exactly')
        if row['outer'] in (1,2):
            need(row['audit'].get('historical_capture_sha256') is None,
                 'No invented historical capture-time auditSHA for1/2')
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
    return manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls


def collect(args, values):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls=values
    marker=B/(STEM+'_ATTEMPT.json')
    need(all(not (root/marker.name).exists() for root in [B]+M), 'Original read attempted already; inspect saved result, never blind retry')
    need(os.environ.get('TREE_MGMT_CONTROL_PATH')==CONTROL_PATH,'Explicit original task ControlPath required')
    tree=ast.parse(RAW[BACK].decode())
    literal=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and
                 any(isinstance(name,ast.Name) and name.id=='REMOTE' for name in node.targets))
    functions=literal[literal.index('fields='):literal.index("run=Path(pin['run'])")]
    need('def grouping(' in functions and 'def ledger(' in functions and 'def raw_meta(' in functions, 'Published candidate pure functions')
    guard_source=RAW[GUARD].decode();guard_tree=ast.parse(guard_source)
    functions+='\n'+ast.get_source_segment(guard_source,next(node for node in guard_tree.body
                               if isinstance(node,ast.FunctionDef) and node.name=='validate_ledger'))+'\n'
    remote_raw=read(manifest['remote_source'])
    program=functions+'\n'+remote_raw.decode()+'\nPINS='+repr(manifest['remote_pins'])+"\nprint(json.dumps(collect(PINS),ensure_ascii=False,separators=(',',':'),allow_nan=False))\n"
    packed=base64.b64encode(zlib.compress(program.encode(),9)).decode()
    command='env CUDA_VISIBLE_DEVICES= python3 -B -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
    helper=module(W/HELPER,'final200_original_existing_route')
    argv=helper.route(command,'h100-10')
    transport=dict(program_bytes=len(program.encode()),target_command_bytes=len(command.encode()),
                   management_command_bytes=len(argv[-1].encode()),all_local_argv_bytes=sum(len(x.encode())+1 for x in argv))
    need(transport['target_command_bytes']<=32768 and transport['management_command_bytes']<=65536 and
         transport['all_local_argv_bytes']<=80000,'Short compressed own-route budget; no SSH attempted')
    checks()
    publish(marker.name,enc(dict(schema='tree_final200_original_cost_once_attempt_v1',started_utc=datetime.now(timezone.utc).isoformat(),
                                input_manifest=ref(args.pins),transport=transport,remote_program_sha256=sha(program.encode()),
                                authorized_remote_calls=1,automatic_retry=False,training_operations=False)))
    try:
        response=helper.run(command,target='h100-10',timeout=900)
        need(response.returncode==0,'Read-only original-file helper failed: '+response.stderr[-1500:])
        need(len(response.stdout.encode())<=32*1024*1024,'Complete scalar response budget; no truncation')
        result=strict(response.stdout)
        need(result['status']=='ORIGINAL200_RECORDS_READ' and result['remote_files_written'] is False and
             result['historical_live_poll_or_FULL_created'] is False,'Original response identity')
        rows=result['rows'];need([row['outer'] for row in rows]==list(range(1,201)),'Original200 rows')
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
    except Exception as exc:
        publish(STEM+'_READ_ERROR.json',enc(dict(schema='tree_final200_original_cost_read_error_v1',
                 observed_utc=datetime.now(timezone.utc).isoformat(),error_type=type(exc).__name__,error=str(exc)[:2000],
                 automatic_retry=False,training_fault_inferred=False,main_replay_allowed=False)))
        raise
    result.update(input_manifest=ref(args.pins),generator=ref(Path(__file__)),remote_generator=manifest['remote_source'],
                  published_candidate_function_source=dict(path=BACK,bytes=STATIC[BACK][0],sha256=STATIC[BACK][1]),
                  original_read_only_helper=dict(path=HELPER,bytes=STATIC[HELPER][0],sha256=STATIC[HELPER][1]),
                  remote_program_sha256=sha(program.encode()),remote_calls=1,remote_exit_code=response.returncode,
                  stdout_bytes=len(response.stdout.encode()),stdout_sha256=sha(response.stdout.encode()),
                  stderr_bytes=len(response.stderr.encode()),stderr_sha256=sha(response.stderr.encode()),
                  transport=transport,observer_cost_added_to_training=False)
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
    publish(STEM+'_CAPTURE_READBACK.json',enc(dict(product=product,input_manifest=ref(args.pins),
            source_inputs=list(LOCAL.values()),protected_original_readbacks=readbacks,remote_calls=1,
            training_or_evaluation_operations=False,full_native_recapture=False)))
    checks()
    print(json.dumps(dict(status='ORIGINAL200_CAPTURED_NOT_FINAL_ACCEPTANCE',product=product,remote_calls=1,
                          source1ed_calls=0,actual_training201_claimed=False)))


def core_metric(poll):
    found=[m for m in poll['iteration_metrics'] if m.get('step')==200]
    need(found and all(m==found[0] for m in found),'Actual original complete metric200 required for cost summary')
    return found[0]


def aggregate(args, values):
    manifest,prior,native,poll,resource,guard,completed,totals,state,cached_controls=values
    capture_path=B/(STEM+'_CAPTURE.json')
    need(args.capture_bytes is not None and args.capture_sha256 is not None,'Bind the actual collect result before aggregation')
    capture=bound(dict(path=str(capture_path),bytes=args.capture_bytes,sha256=args.capture_sha256),True)
    capture={k:expand(v) for k,v in capture.items()}
    need(capture['input_manifest']==ref(args.pins) and capture['remote_calls']==1 and capture['remote_exit_code']==0,
         'Actual frozen original capture required')
    rows=capture['rows'];need([r['outer'] for r in rows]==list(range(1,201)),'Unique original200 rows')
    def total(values, scope):
        known=[(r['outer'],v) for r,v in zip(rows,values) if number(v)]
        missing=[r['outer'] for r,v in zip(rows,values) if not number(v)]
        value=math.fsum(v for _,v in known)
        if value.is_integer():value=int(value)
        return dict(total=value if not missing else None,known_sum=value,known_outers=[n for n,_ in known],
                    missing_outers=missing,complete=not missing,scope=scope)
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
                     'Unique original branch round ledger; subset of whole Controller; missing fields remain null.')
            for key in ['generated_tokens','prefill_tokens','retrieval_calls','scoring_calls','scoring_wall_seconds']}
    candidate={key:total([r['branch']['candidate_outcomes'][key]['candidate_count'] for r in rows],
                         'Original outcome count; fallback overlaps accepted, never a third additive bucket.')
               for key in ['all','accepted','discarded','fallback']}
    outcomes={}
    for key in ['all','accepted','discarded','fallback']:
        fields=sorted({name for r in rows for name in r['branch']['candidate_outcomes'][key]['field_sums']})
        outcomes[key]=dict(field_sums={name:total([r['branch']['candidate_outcomes'][key]['field_sums'].get(name) for r in rows],
                         'Original candidate event/amortized scalar; inherited prefix is not new generation.') for name in fields})
    timings={key:total([r['metric'].get('timing_s/'+key) for r in rows],
                       'Original step phase. Dev/gen/ref/update/save are included in step wall, not added again.')
             for key in ['step','gen','ref','adv','update_actor','testing','stage_testing','save_checkpoint']}
    writes={str(rank):sum(r['rank_records'][rank]['successful_optimizer_calls'] for r in rows) for rank in range(8)}
    need(writes==completed,'Original rank versus native200 credited totals')
    devs=[]
    for row in capture['controls']:
        n=row['outer'];v=row['value'];m=rows[n-1]['metric']
        devs.append(dict(outer=n,source_commit=row['source_commit'],ledger=Path(row['source']['path']).name,
                         ledger_sha256=v['ledger_sha256'],rows=512,phase='periodic',
                         testing_phase_wall_seconds=m.get('timing_s/testing'),F1=v['metrics']['val/f1/all'],
                         EM=v['metrics']['val/em/all'],original_control_source=row['source']))
    need([row['outer'] for row in devs]==list(range(5,201,5)),'Original40 periodic dev records')
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
    parser.add_argument('phase',choices=['collect','aggregate'])
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--pins',required=True,type=Path)
    parser.add_argument('--pins-bytes',required=True,type=int)
    parser.add_argument('--pins-sha256',required=True)
    parser.add_argument('--capture-bytes',type=int)
    parser.add_argument('--capture-sha256')
    args=parser.parse_args();need(__debug__,'Do not disable actual evidence gates with -O')
    W=args.workspace.resolve(strict=True);B=W/BASE_NAME
    M=[W/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
       Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    values=inputs(args)
    if args.phase=='collect':collect(args,values)
    else:aggregate(args,values)


if __name__=='__main__':
    main()
