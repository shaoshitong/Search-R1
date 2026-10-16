#!/usr/bin/env python3
"""Future root dev195 gate builder; source preparation is not execution evidence.

Never runs source1ed main, owner main, evaluation, training or canonical writers.
The frozen owner's complete local verify_gate_inputs prefix runs before the one
optional original-control helper call. The full owner contract runs in memory
before publication, then against the published immutable gate/pending bytes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

BASE = 'outputs/tree-nested-branch-handoff-20261013'
OWNER = 'work/tree-speed-repair-20261008/dev195_owner_local_receipt.py'
# Actual frozen owner source pin; it does not assert future190/195 success.
# Current real source1ed success is through185 (main6/remote6).
OWNER_BYTES = 54241
OWNER_SHA = '1939adf9d5e1168a67821f758d6df106787a964efba511077d420568aa60d7fa'
ENTRY = '1edaf50bb8e14de51db936e0859e369792011233'
COLLECTOR = 'work/tree-speed-repair-20261008/nested-dev-cumulative-parts-source/reproduction/tree-eval-resume-20261013/local-control-backup/collect_nested_dev_costs.py'
COLLECTOR_SHA = '545bc2cb668da2b672ee3a859c031171927f3b2b6a2aaa3d368bee7d09b732d0'
HELPER = 'work/tree-speed-repair-20261008/ssh_retained18.py'
HELPER_SHA = '8065815f4452923bcb96fbdc12b96b246b7834f85f1bc2a9f75623912c157021'
CORE = 'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup/sync_nested_native_progress.py'
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
CONTROL_PATH = '/tmp/tree-mgmt-3fw9a1ab/control'
GATE = 'DEV195_ROOT_OWNER_GATE_PINS.json'
PENDING = 'DEV195_SOURCE_BOUND_PENDING_READBACK.json'
CAPTURE = 'DEV195_ORIGINAL_CONTROL_READY_CAPTURE.json'

# One fixed own control file, two reads inside one helper invocation. No retries,
# directory traversal, raw rollout, tensor, model, index or credential contents.
REMOTE = r'''
import hashlib,json,os,stat,time
from pathlib import Path
pin=PIN_DATA
os.nice(15)
connection=os.environ.get('SSH_CONNECTION','').split()
if len(connection)!=4 or connection[2] not in ('10.33.0.18','127.0.0.1'):raise ValueError('Original18/loopback SSH endpoint identity differs')
path=Path(pin['path']);alias='/dataset/shaoshitong/search-r1-rl/tree-grpo/'
retained='/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/'
expected=retained+str(path)[len(alias):]
if not str(path).startswith(alias) or path.is_symlink() or path.suffix!='.json' or path.parent!=Path(pin['run'])/'output/validation-cost':raise ValueError('Fixed own control path differs')
resolved=str(path.resolve(strict=True))
if resolved!=expected:raise ValueError('Shared retained18 resolved control path differs')
with path.open('rb') as stream:
 before=os.fstat(stream.fileno())
 if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=100000:raise ValueError('Small regular control file required')
 raw=stream.read(100001);middle=os.fstat(stream.fileno());stream.seek(0);again=stream.read(100001);after=os.fstat(stream.fileno())
last=path.stat()
identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
if raw!=again or any(identity(value)!=identity(before) for value in (middle,after,last)) or str(path.resolve(strict=True))!=resolved:raise ValueError('Original control two reads unstable')
value=json.loads(raw)
for key in ('code_commit','config_sha256','model_identity','data_identity','job_id'):
 if value.get(key)!=pin[key]:raise ValueError('Original control identity differs: '+key)
if value.get('run') not in (pin['run']+'/output',str(Path(resolved).parent.parent)):raise ValueError('Original control run differs')
if value.get('training_protocol')!='tree_fresh200_five_leaf_v1' or value.get('physical_gpus')!=8 or value.get('slurm_reserved_gpus')!=0 or value.get('allocation_kind')!='user_authorized_shared_systemd' or value.get('resource_protocol')!='training_memory_headroom_shared_v1':raise ValueError('Original shared control protocol differs')
if value.get('actual_completed_outer')!=195 or value.get('global_step')!=195 or value.get('status')!='VALIDATION_COST_COMPLETE' or value.get('phase')!='periodic' or value.get('dataset_rows')!=512 or value.get('evaluated_rows')!=512:raise ValueError('Original dev195 completion/coverage differs')
if value.get('metrics')!=pin['metrics'] or value.get('ledger_sha256')!=pin['ledger_sha256']:raise ValueError('Original dev195 metrics or inner ledger identity differs')
ready=dict(outer=195,raw_summary_utf8=raw.decode('utf8'),source=dict(path=str(path),resolved_path=resolved,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime_ns=before.st_mtime_ns,stable_read=True))
print(json.dumps(dict(status='ACTUAL_DEV195_ORIGINAL_CONTROL_TWO_STABLE_READS',control_ready=ready,actual_control_reads=2,observed_epoch=time.time(),remote_files_written=False,collector_entry_invoked=False,SSH_server_ip=connection[2]),ensure_ascii=False,allow_nan=False))
'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def pin(path, raw, workspace):
    return dict(path=str(path.resolve().relative_to(workspace)), bytes=len(raw), sha256=sha(raw))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixed(path, size, digest):
    raw = path.read_bytes()
    need(len(raw) == size and sha(raw) == digest, 'Frozen dependency differs: ' + str(path))
    return raw


def read_bound(path, size, digest, workspace):
    path = path.resolve(strict=True)
    need(path.is_relative_to(workspace), 'Local input outside workspace')
    need(type(size) is int and 0 < size <= 100000 and re.fullmatch(r'[a-f0-9]{64}', digest or ''), 'Actual small input pin required')
    raw = path.read_bytes()
    need(len(raw) == size and sha(raw) == digest, 'Actual input bytes/SHA differ')
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-bytes', required=True, type=int)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--source-bytes', required=True, type=int)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--control-capture', type=Path, help='Reuse an existing original control-ready JSON; requires request.control_capture bytes/SHA pin; no SSH')
    args = parser.parse_args()
    need(__debug__, 'Required owner gates cannot run with python -O')
    w = args.workspace.resolve(strict=True);b = w / BASE
    own_raw = fixed(Path(__file__).resolve(), args.source_bytes, args.source_sha256)
    request_raw = read_bound(args.request, args.request_bytes, args.request_sha256, w)
    request = json.loads(request_raw)
    need(request['schema'] == 'dev195_root_gate_build_request_v1', 'Actual root request schema differs')
    need(request['collector_source_commit'] == ENTRY and request['collector_source_sha256'] == COLLECTOR_SHA, 'Exact source1ed identity required')
    executor = request['collector_executor_agent']
    need(isinstance(executor, str) and executor.startswith('/root/'), 'Root-authorized sole executor required')
    att = request['sole_executor_attestation']
    need(att['executor_agent'] == executor and att['sole_executor_authorized'] is True and isinstance(att['observed_utc'], str) and att['observed_utc'], 'Actual executor/root attestation required')
    for key in ('main_invocations_this_outer', 'remote_incremental_invocations_this_outer', 'preceding_local_pending_invocations'):
        need(type(att[key]) is int and att[key] == 0, 'Collector already invoked or unproven: ' + key)
    need(att['preceding_pending_remote_operations'] is False, 'No earlier PENDING remote operation must be attested')
    need(request['authorized_main_invocations'] == request['authorized_remote_incremental_invocations'] == 1, 'Unique future main1/remote1 authorization required')
    need(type(OWNER_BYTES) is int and OWNER_BYTES > 0 and re.fullmatch(r'[a-f0-9]{64}', OWNER_SHA or ''), 'Frozen final195 owner source pin required: actual frozen owner195 source pin is not installed')
    owner_raw = fixed(w / OWNER, OWNER_BYTES, OWNER_SHA)
    fixed(w / COLLECTOR, 27911, COLLECTOR_SHA);fixed(w / HELPER, 4411, HELPER_SHA);fixed(w / CORE, 39098, CORE_SHA)
    owner = module('dev195_root_builder_frozen_owner', w / OWNER)
    need(owner.W == w, 'Frozen owner workspace identity differs')
    mod = module('dev195_root_builder_collector_pure_functions', w / COLLECTOR)
    core = module('dev195_root_builder_core_pure_functions', w / CORE)
    mirrors = owner.M
    need(all(location.is_dir() for location in [b] + mirrors), 'Existing BASE/two mirrors required')
    absent = ('DEV5_195_NATIVE_COST_SUPPLEMENT.json', 'DEV5_195_NATIVE_COST_SUPPLEMENT.md',
              'DEV5_195_NATIVE_COST_COLLECTION_ERROR.json', 'DEV5_195_NATIVE_COST_COLLECTOR_EXECUTION.json',
              'DEV5_195_NATIVE_COST_LOCAL_RECEIPT.json', 'DEV195_LOCAL_PREFLIGHT_INPUT_PINS.json',
              'DEV5_195_NATIVE_COST_LOCAL_PREFLIGHT_INPUT_PINS.json', 'DEV5_195_NATIVE_COST_PRE_EXECUTION_INPUT_PINS.json',
              'DEV5_195_NATIVE_COST_COLLECTOR_EXECUTION_START.json', GATE, PENDING)
    def absence():
        need(not any((location / name).exists() for location in [b] + mirrors for name in absent),
             'Collector/preflight/gate products already exist; verify and reuse, never blind retry')
    absence()
    # Only request-bound actual prior190 artifacts may establish source1ed
    # cumulative7/PENDING0, 38 controls, 190 training points and 228 CSV rows.
    # The frozen owner195 contract requires 39 current controls and CUM195.
    # Subsequent curve195 must contain195 train/39 dev/234 CSV rows. True
    # next196 stage entry is independently required, not saved cursor196.
    # All such local conditions must pass before any optional remote read.
    prior_documents = owner.verify_prior_inputs(request['prior_inputs'], mod, core)
    prior = prior_documents['DEV5_190_NATIVE_COST_LOCAL_RECEIPT.json']
    prior_owner_ref = request['prior_inputs']['owner_receipt']
    need(prior['outer'] == 190 and prior['collector_source_commit'] == ENTRY and prior['collector_source_sha256'] == COLLECTOR_SHA, 'Actual prior190 owner source differs')
    need(prior['same_entry_executor_attested_cumulative_main_invocations'] == prior['same_entry_executor_attested_cumulative_remote_incremental_invocations'] == 7
         and prior['preceding_local_pending_invocations'] == 0 and prior['remote_collector_exit_code'] == 0, 'Prior source1ed cumulative7/PENDING0 evidence differs')
    now = datetime.now(timezone.utc).isoformat()
    gate = dict(schema='dev195_root_owner_gate_pins_v1', status='ACTUAL_DEV195_FULL_NATIVE_CUM_COST_CONTROL_READY',
                outer=195, observed_utc=now, collector_source_commit=ENTRY, collector_source_sha256=COLLECTOR_SHA,
                collector_executor_agent=executor, authorized_main_invocations=1, authorized_remote_incremental_invocations=1,
                source_pins=request['source_pins'], prior_inputs=request['prior_inputs'])
    need(set(gate['source_pins']) == set(owner.FUTURE_NAMES), 'Exactly actual FULL/native/CUM/COST195 descriptors required')
    virtual = {str((b / GATE).resolve()): encode(gate)}
    actual_read = owner.read
    def virtual_read(name, digest=None, size=None):
        path = str(owner.locate(name).resolve())
        if path in virtual:
            raw = virtual[path]
            need(digest is None or sha(raw) == digest, 'In-memory prospective metadata SHA differs')
            need(size is None or len(raw) == size, 'In-memory prospective metadata size differs')
            return json.loads(raw)
        return actual_read(name, digest, size)
    owner.read = virtual_read
    # Execute only this frozen owner's pure local contract prefix, in its isolated
    # imported module. Exact source SHA and split anchors are fixed above. No main.
    text = owner_raw.decode('utf8');start = text.index('def verify_gate_inputs(args, mod, core):\n')
    stop = text.index("    pending = bound(gate['pending_evidence'])\n", start)
    prefix = text[start:stop].replace('def verify_gate_inputs(', 'def verify_local_completion_inputs(', 1)
    prefix += '    return gate, full, native, rawcum, cumulative, cost\n'
    exec(prefix, owner.__dict__)
    local_args = SimpleNamespace(root_pins=b / GATE, root_pins_bytes=len(virtual[str((b / GATE).resolve())]), root_pins_sha256=sha(virtual[str((b / GATE).resolve())]))
    _, full, native, rawcum, cumulative, cost = owner.verify_local_completion_inputs(local_args, mod, core)
    actual = full['actual_dev512'];dev = cumulative['completed_dev_evaluations'][-1]
    control_pin = dict(owner.dev_pin(mod, dev), metrics=actual['metrics'])
    capture_path = args.control_capture.resolve(strict=True) if args.control_capture else b / CAPTURE
    capture_ref = request.get('control_capture')
    cache_mode = args.control_capture is not None or capture_path.exists()
    if cache_mode:
        need(isinstance(capture_ref, dict) and owner.locate(capture_ref['path']).resolve() == capture_path.resolve(), 'Actual cached control descriptor required; no automatic remote retry')
        capture_raw = read_bound(capture_path, capture_ref['bytes'], capture_ref['sha256'], w)
        capture = json.loads(capture_raw);ready = capture.get('control_ready', capture)
        read_scope = 'Reused original capture bytes and original stable-read metadata; no fresh SSH/read claimed.'
    else:
        need(capture_ref is None, 'Requested cached control is absent; refuse remote fallback')
        need(os.environ.get('TREE_MGMT_CONTROL_PATH') == CONTROL_PATH, 'Original task ControlPath required')
        helper = module('dev195_root_builder_existing_retained18_helper', w / HELPER)
        code = REMOTE.replace('PIN_DATA', repr(control_pin))
        response = helper.run('env CUDA_VISIBLE_DEVICES= python3 -B -c ' + shlex.quote(code), target='h100-10', timeout=60)
        need(response.returncode == 0, 'Original control read failed; no retry: ' + response.stderr[-1200:])
        capture = json.loads(response.stdout)
        need(capture['status'] == 'ACTUAL_DEV195_ORIGINAL_CONTROL_TWO_STABLE_READS' and capture['actual_control_reads'] == 2
             and capture['remote_files_written'] is False and capture['collector_entry_invoked'] is False, 'Original control capture result differs')
        ready = capture['control_ready'];capture_raw = encode(capture)
        read_scope = 'One existing retained18 helper invocation, two actual byte/mtime/inode-stable reads of the fixed original dev195 control; collector main/PENDING not called.'
    need(ready['outer'] == 195 and isinstance(ready['raw_summary_utf8'], str), 'Actual original control-ready shape differs')
    original_raw = ready['raw_summary_utf8'].encode('utf8');meta = ready['source']
    alias = '/dataset/shaoshitong/search-r1-rl/tree-grpo/';retained = '/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/'
    need(meta['path'] == control_pin['path'] and meta['path'].startswith(alias)
         and meta['resolved_path'] == retained + meta['path'][len(alias):] and meta['stable_read'] is True
         and type(meta.get('mtime_ns')) is int and meta['mtime_ns'] > 0
         and meta['bytes'] == len(original_raw) and meta['sha256'] == sha(original_raw), 'Actual raw file path/resolved/bytes/SHA/stability differs')
    value = json.loads(original_raw)
    need(value['metrics'] == actual['metrics'] and value['training_protocol'] == 'tree_fresh200_five_leaf_v1'
         and value['physical_gpus'] == 8 and value['slurm_reserved_gpus'] == 0
         and value['allocation_kind'] == 'user_authorized_shared_systemd'
         and value['resource_protocol'] == 'training_memory_headroom_shared_v1', 'Original shared dev protocol differs')
    mod.control_row(control_pin, original_raw, meta, dict(path='dev_native_cost_summaries/DEV_OUTER0195_SUMMARY.json', bytes=len(original_raw), sha256=sha(original_raw)))
    # Preserve a successful small capture before any later gate failure, enabling
    # an explicitly SHA-bound local reuse instead of repeating SSH.
    if not cache_mode:
        need(len(capture_raw) <= 100000, 'Small capture budget exceeded')
        mod.publish(b, mirrors, CAPTURE, capture_raw)
    pending = dict(schema='dev195_source_bound_pending_readback_v1', outer=195, source_commit=ENTRY,
        source_sha256=COLLECTOR_SHA, executor_agent=executor, observed_utc=datetime.now(timezone.utc).isoformat(),
        preceding_local_pending_invocations=0, preceding_pending_remote_operations=False,
        collector_main_invocations_this_outer=0, collector_remote_incremental_invocations_this_outer=0,
        original_dev195_control_complete=True, dev_score_pending=False,
        observations=[dict(observation='Root and uniquely authorized executor attest source1ed195 main0/remote0/PENDING0 before the unique authorized future invocation.', sole_executor_attestation=att),
                      dict(observation='Exact request-bound prior190 artifacts and immutable owner receipt corroborate same source1ed cumulative main7/remote7 and prior PENDING0.', prior_owner_receipt=prior_owner_ref, prior_inputs=gate['prior_inputs']),
                      dict(observation='All four actual FULL/native/CUM/COST195 completion inputs and normal next196 entry passed the frozen owner local contract before control capture.', source_pins=gate['source_pins']),
                      dict(observation=read_scope, control_source=meta, remote_files_written=False)],
        evidence_scope='Root/sole-executor source-bound invocation attestations, prior immutable artifacts and local product absence; no independent global process audit. PENDING0 concerns collector calls only: original dev195 scoring is COMPLETE.',
        builder_source=pin(Path(__file__), own_raw, w), root_build_request=pin(args.request, request_raw, w))
    pending_raw = encode(pending)
    gate.update(pending_evidence=pin(b / PENDING, pending_raw, w), control_ready=ready,
                gate_scope='Actual FULL/native/CUM/COST195 and original dev195 COMPLETE raw control; source1ed sole executor main0/remote0/PENDING0 before authorized main1/remote1. No ROOT_FULL195 dependency, collector invocation or canonical mutation.')
    gate_raw = encode(gate)
    need(len(pending_raw) <= 100000 and len(gate_raw) <= 100000, 'Small gate/pending metadata budget exceeded')
    virtual[str((b / PENDING).resolve())] = pending_raw;virtual[str((b / GATE).resolve())] = gate_raw
    local_args.root_pins_bytes = len(gate_raw);local_args.root_pins_sha256 = sha(gate_raw)
    owner.verify_gate_inputs(local_args, mod, core)
    absence()
    need(Path(__file__).read_bytes() == own_raw and args.request.read_bytes() == request_raw, 'Frozen builder/request changed')
    owner.two_pass()
    mod.publish(b, mirrors, PENDING, pending_raw);mod.publish(b, mirrors, GATE, gate_raw)
    owner.read = actual_read
    owner.verify_gate_inputs(local_args, mod, core)
    owner.two_pass()
    for number in (1, 2):
        for name, raw in ((PENDING, pending_raw), (GATE, gate_raw)):
            need(all((location / name).read_bytes() == raw for location in [b] + mirrors), 'Actual gate/pending mirror byte readback differs')
    print(json.dumps(dict(status='ACTUAL_DEV195_ROOT_OWNER_GATE_PUBLISHED_NO_COLLECTOR_INVOKED',
          gate=pin(b / GATE, gate_raw, w), pending=pin(b / PENDING, pending_raw, w),
          source1ed195_main_invocations=0, source1ed195_remote_incremental_invocations=0,
          preceding_local_pending_invocations=0, original_dev_control_complete=True,
          control_capture_reused=cache_mode, two_mirror_two_byte_readbacks=True,
          collector_eval_training_or_canonical_invoked=False), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('DEV195_ROOT_GATE_NOT_PUBLISHED_OR_REUSE_REQUIRED: ' + str(error))
