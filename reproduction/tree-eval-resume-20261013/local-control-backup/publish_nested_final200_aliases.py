#!/usr/bin/env python3
"""Publish only local final200 current metadata from an existing finalizer proof.

No collector, evaluator, tensor reader, training action or network client exists
in this entry. Missing final200 evidence is PENDING and writes nothing.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CORE_SHA = '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
FINALIZER_SHA = '61385504e4bb389cdba508c1b74f84fe6464e1d4f8ec11989e8cfe0c230dc541'
FULL = 'FULL_OUTER200_PROGRESS_RECEIPT.json'
PROOF = 'OUTER200_TRAINING_COMPLETE_FINAL_PHASE_ACTUAL_PROGRESS.json'
RECEIPT = 'CURRENT_FINAL200_ALIAS_RECONCILIATION.json'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
EXTERNAL_MIRROR = Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')
MAX_BYTES = 12 * 1024**2


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key, including nested objects: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + item)))


def dependencies():
    paths = [HERE/'sync_nested_native_progress.py', HERE/'finalize_nested_progress.py']
    for path, sha in zip(paths, (CORE_SHA, FINALIZER_SHA)):
        if digest(path.read_bytes()) != sha:
            raise ValueError('Pinned local dependency changed: ' + path.name)
    spec = importlib.util.spec_from_file_location('sync_nested_native_progress', paths[0])
    core = importlib.util.module_from_spec(spec)
    sys.modules['sync_nested_native_progress'] = core
    spec.loader.exec_module(core)
    spec = importlib.util.spec_from_file_location('final200_pinned_finalizer', paths[1])
    finalizer = importlib.util.module_from_spec(spec);spec.loader.exec_module(finalizer)
    return core, finalizer, paths


def metadata_source(core, commit, dependency_paths):
    core.need(isinstance(commit, str) and re.fullmatch('[0-9a-f]{40}', commit),
              'Explicit metadata source commit required', 'REJECTED_INPUT_NO_WRITE')
    git = ['git', '-C', str(HERE)]
    def run(arguments):
        return subprocess.run(git+arguments, check=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE).stdout
    root = Path(run(['rev-parse', '--show-toplevel']).decode().strip()).resolve()
    core.need(run(['rev-parse', '--verify', commit+'^{commit}']).decode().strip() == commit,
              'Metadata source commit does not resolve exactly', 'REJECTED_INPUT_NO_WRITE')
    pins = []
    for path in [Path(__file__).resolve(), *dependency_paths]:
        raw = path.read_bytes();relative = path.relative_to(root).as_posix()
        core.need(run(['show', commit+':'+relative]) == raw,
                  'Metadata commit does not bind executing file: '+relative, 'REJECTED_INPUT_NO_WRITE')
        pins.append(dict(path=str(path), repository_path=relative, bytes=len(raw), sha256=digest(raw)))
    return dict(commit=commit, files=pins)


class Inputs:
    """Stable strict JSON, exact SHA/size and both local mirror readbacks."""
    def __init__(self, core, workspace, mirrors):
        self.core, self.workspace = core, workspace.resolve()
        self.base = (self.workspace/core.BASE_REL).resolve()
        self.mirrors, self.pins, self.cache = mirrors, {}, {}
        self.closure_done, self.expanding, self.expanded_parts = set(), set(), set()

    def path(self, value, root=None):
        p = Path(value)
        self.core.need('..' not in p.parts, 'Parent traversal forbidden', 'REJECTED_INPUT_NO_WRITE')
        if not p.is_absolute():
            p = (root or self.workspace)/p
        self.core.need(not p.is_symlink(), 'Symlink metadata forbidden', 'REJECTED_INPUT_NO_WRITE')
        p = p.resolve()
        self.core.need(p.is_relative_to(self.base) and p.suffix == '.json',
                       'Only local audit-base JSON evidence allowed: '+str(p), 'REJECTED_INPUT_NO_WRITE')
        return p

    def read(self, value, root=None):
        p = self.path(value, root)
        self.core.need(p.is_file(), 'Required local metadata unavailable: '+str(p))
        with p.open('rb') as stream:
            before = os.fstat(stream.fileno())
            self.core.need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES,
                           'Ordinary compact metadata required', 'REJECTED_INPUT_NO_WRITE')
            raw = stream.read();after = os.fstat(stream.fileno())
        signature = lambda x:(x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        self.core.need(signature(before) == signature(after) == signature(p.stat()),
                       'Input changed during read', 'CONCURRENT_UPDATE_NO_WRITE')
        data = strict_json(raw)
        self.core.need(isinstance(data, dict), 'Metadata object required', 'REJECTED_INPUT_NO_WRITE')
        pin = dict(path=str(p), bytes=len(raw), sha256=digest(raw))
        if str(p) in self.pins:
            self.core.need(self.pins[str(p)] == pin, 'Contradictory input bindings', 'REJECTED_INPUT_NO_WRITE')
        for mirror in self.mirrors:
            target = mirror/p.relative_to(self.base)
            self.core.need(target.is_file() and target.read_bytes() == raw,
                           'Evidence mirror bytes differ: '+str(target), 'REJECTED_INPUT_NO_WRITE')
        self.pins[str(p)] = pin;self.cache[str(p)] = data
        return data, pin

    def bound(self, descriptor, root=None):
        c = self.core
        c.need(isinstance(descriptor, dict) and c.sha64(descriptor.get('sha256'))
               and type(descriptor.get('bytes')) is int and descriptor['bytes'] > 0,
               'Complete SHA/byte descriptor required', 'REJECTED_INPUT_NO_WRITE')
        name = descriptor.get('path') or descriptor.get('file')
        c.need(isinstance(name, str) and name, 'Descriptor path missing', 'REJECTED_INPUT_NO_WRITE')
        data, pin = self.read(name, root)
        c.need((pin['sha256'], pin['bytes']) == (descriptor['sha256'], descriptor['bytes']),
               'Bound evidence bytes differ: '+name, 'REJECTED_INPUT_NO_WRITE')
        return data, pin

    def expand(self, value, field):
        """Recursively close all complete-original-field parts, including nesting."""
        c = self.core
        if not isinstance(value, dict) or value.get('storage') != 'complete_original_field_parts':
            if isinstance(value, dict):
                return {k:self.expand(v, field+'.'+k) for k,v in value.items()}
            if isinstance(value, list):
                return [self.expand(v, field+'['+str(i)+']') for i,v in enumerate(value)]
            return value
        kind, count, parts = (value.get(k) for k in ('container_type', 'original_entries', 'parts'))
        c.need(kind in ('mapping', 'list') and type(count) is int and count >= 0
               and isinstance(parts, list) and (parts or count == 0),
               'Invalid complete-field descriptor: '+field, 'REJECTED_INPUT_NO_WRITE')
        result = {} if kind == 'mapping' else [];seen = set()
        for ref in parts:
            c.need(isinstance(ref, dict) and ref.get('json_selector') == 'value',
                   'Part selector differs: '+field, 'REJECTED_INPUT_NO_WRITE')
            path = self.path(ref.get('file', ''), self.base)
            c.need(path.is_relative_to(self.base/'cumulative_cost_blocks')
                   and str(path) not in seen and str(path) not in self.expanding
                   and str(path) not in self.expanded_parts,
                   'Duplicate, cyclic or escaped cumulative part', 'REJECTED_INPUT_NO_WRITE')
            seen.add(str(path));self.expanding.add(str(path));self.expanded_parts.add(str(path))
            part, _ = self.bound(ref, self.base)
            c.need(part.get('schema') == 'cumulative_scalar_field_part_v1'
                   and part.get('original_field') == field,
                   'Part schema/original field differs: '+field, 'REJECTED_INPUT_NO_WRITE')
            chunk = part.get('value')
            c.need(isinstance(chunk, dict if kind == 'mapping' else list),
                   'Part container differs: '+field, 'REJECTED_INPUT_NO_WRITE')
            if kind == 'mapping':
                c.need(not result.keys() & chunk.keys(), 'Repeated mapping entry across parts', 'REJECTED_INPUT_NO_WRITE')
                result.update(chunk)
            else:
                result.extend(chunk)
            self.expanding.remove(str(path))
        c.need(len(result) == count, 'Complete-field original entry count differs', 'REJECTED_INPUT_NO_WRITE')
        return self.expand(result, field)

    def closure(self, value, root=None):
        """Close declared local manifests; never follow remote source_files/PT."""
        root = root or self.base;c = self.core
        if (isinstance(value, dict) and c.sha64(value.get('sha256')) and 'bytes' in value
                and (value.get('path') or value.get('file'))):
            data, pin = self.bound(value, root)
            if pin['path'] not in self.closure_done:
                self.closure_done.add(pin['path']);self.closure(data, self.base)
            return
        if isinstance(value, list):
            for item in value:self.closure(item, root)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ('source_files', 'derived_input_snapshots'):
                    # Remote originals and explicitly mutable derived snapshots
                    # are observations, not immutable local dependency manifests.
                    continue
                if isinstance(item, dict) and c.sha64(item.get('sha256')) and 'bytes' in item:
                    name = item.get('path') or item.get('file')
                    if name is None and isinstance(key, str) and key.endswith('.json'):
                        item = dict(item, file=key);name=key
                    if name is not None:
                        data, pin = self.bound(item, root)
                        if pin['path'] not in self.closure_done:
                            self.closure_done.add(pin['path']);self.closure(data, self.base)
                        continue
                self.closure(item, root)

    def unchanged(self):
        for pin in self.pins.values():
            p = Path(pin['path']);raw = p.read_bytes()
            self.core.need(len(raw) == pin['bytes'] and digest(raw) == pin['sha256'],
                           'Input changed before publication', 'CONCURRENT_UPDATE_NO_WRITE')
            for m in self.mirrors:
                self.core.need((m/p.relative_to(self.base)).read_bytes() == raw,
                               'Input mirror changed', 'CONCURRENT_UPDATE_NO_WRITE')


def identity(c, value):
    c.need(all(value.get(k) == v for k,v in dict(source_commit=c.SOURCE,
           config_sha256=c.CONFIG, unit=c.UNIT, run=c.RUN).items()),
           'Exact original source/config/unit/run differs', 'REJECTED_INPUT_NO_WRITE')


def poll_identity(c, poll, resource):
    c.need(all(poll.get(k) == v for k,v in dict(source_commit=c.SOURCE,
           unit=c.UNIT, job_id=c.JOB, run=c.RUN, node='h100-10', experiment_origin='fresh').items()),
           'Original poll identity differs', 'REJECTED_INPUT_NO_WRITE')
    ps, rs, active = c.validate_unit(poll, resource, 200)
    c.need(ps.get('InvocationID') == rs.get('InvocationID') == c.INVOCATION,
           'Both actual systemd observations require exact Invocation', 'REJECTED_INPUT_NO_WRITE')
    restore, _ = c.validate_restore(poll)
    progress = poll.get('progress') or {}
    c.need(progress.get('actual_completed_outer_from_attempt_results') == 200 and not progress.get('errors'),
           'Actual200 credit/error boundary pending')
    entries = [int(n) for n in re.findall(r'epoch\s+\d+,\s*step\s+(\d+)\b', (poll.get('logs') or {}).get('stage.log') or '')]
    c.need(not entries or max(entries) <= 200, 'Actual training201 entry is outside this publisher', 'REJECTED_INPUT_NO_WRITE')
    c.need(not any(r.get('outer_step', 0) > 200 for r in poll.get('last_rank_summaries', [])),
           'Rank evidence past200 is outside this publisher', 'REJECTED_INPUT_NO_WRITE')
    watcher = resource.get('state') or {}
    c.need(not watcher or (watcher.get('training_source_commit') == c.SOURCE
           and watcher.get('training_unit') == c.UNIT and watcher.get('run_directory') == c.RUN
           and watcher.get('expected_training_invocation_id') == c.INVOCATION
           and not watcher.get('last_errors') and not watcher.get('audit_error_count')),
           'Source-bound resource watcher differs or has audit errors', 'REJECTED_INPUT_NO_WRITE')
    c.need(c.number(poll.get('epoch')) and c.number(resource.get('epoch')),
           'Actual poll/resource times required')
    return progress, ps, rs, active, restore


def actual_final_phase(c, poll, resource, active):
    """Strong terminal/handoff evidence precedes retained final-stage log text.

    This is local validation only. The frozen core/finalizer are not changed.
    Active MainPID identifies the original supervisor; an actual body terminal
    result does not imply that its service/cgroup has already disappeared.
    """
    ps, rs, observed_active = c.validate_unit(poll, resource, 200)
    c.need(observed_active == active and ps.get('InvocationID') == rs.get('InvocationID') == c.INVOCATION,
           'Final-phase original Invocation/active state differs', 'REJECTED_INPUT_NO_WRITE')
    c.need(all(poll.get(k) == v for k,v in dict(source_commit=c.SOURCE,unit=c.UNIT,job_id=c.JOB,run=c.RUN).items()),
           'Final-phase original source/job/unit/run differs', 'REJECTED_INPUT_NO_WRITE')
    c.need((poll.get('progress') or {}).get('actual_completed_outer_from_attempt_results') == 200
           and not (poll.get('progress') or {}).get('errors'), 'Final-phase actual200 progress differs')
    expected_cgroup = '/system.slice/'+c.UNIT
    for observed in (ps, rs):
        if 'ControlGroup' in observed:
            allowed = (expected_cgroup,) if active else (expected_cgroup, '')
            c.need(observed['ControlGroup'] in allowed,
                   'Final-phase observation is not the exact own cgroup', 'REJECTED_INPUT_NO_WRITE')
        tasks = observed.get('TasksCurrent')
        if not active and tasks is not None and re.fullmatch(r'\d+', str(tasks).strip()):
            c.need(int(str(tasks).strip()) == 0, 'Inactive original unit still reports tasks', 'REJECTED_INPUT_NO_WRITE')
    own = resource.get('own_cgroup') or {}
    c.need(isinstance(own, dict), 'Own cgroup observation must be metadata', 'REJECTED_INPUT_NO_WRITE')
    pids = own.get('pids.current')
    if pids is not None:
        c.need(type(pids) in (str, int) and re.fullmatch(r'\d+', str(pids).strip()),
               'Own cgroup process count is invalid', 'REJECTED_INPUT_NO_WRITE')
        pids = int(str(pids).strip())
        c.need(active or pids == 0, 'Inactive original unit still has own cgroup tasks', 'REJECTED_INPUT_NO_WRITE')
    c.need(own.get('own_cgroup_live_descendants_empty', True) is not False,
           'Explicit live own descendants contradict terminal evidence', 'REJECTED_INPUT_NO_WRITE')
    result = poll.get('run_result')
    if result is not None:
        c.need(isinstance(result, dict) and result.get('code_commit') == c.SOURCE and result.get('job_id') == c.JOB,
               'Run-result original source/job differs', 'REJECTED_INPUT_NO_WRITE')
        for key, expected in (('config_sha256',c.CONFIG),('run',c.RUN),('run_directory',c.RUN),
                              ('unit',c.UNIT),('invocation_id',c.INVOCATION),('InvocationID',c.INVOCATION)):
            c.need(key not in result or result[key] == expected,
                   'Run-result explicit '+key+' differs', 'REJECTED_INPUT_NO_WRITE')
    if result and result.get('status') == 'ACTUAL_HUMAN_AUTHORIZED_DIRECT_FRESH200_TRAINING_COMPLETE':
        c.need(result.get('actual_completed_outer') == 200 and result.get('training_body_exit_code') == 0
               and result.get('own_cgroup_live_residual_empty') is True and not result.get('cleanup_error'),
               'Actual normal200 terminal/exit/reported own cleanup differs', 'REJECTED_INPUT_NO_WRITE')
        residual = result.get('cleanup_residual', {})
        c.need(isinstance(residual, dict) and all(value == 'Z' for value in residual.values()),
               'Reported live cleanup residual contradicts terminal result', 'REJECTED_INPUT_NO_WRITE')
        return dict(kind='actual_training_terminal', evidence=result, normal_exit_claimed=True)
    handoff = resource.get('final_handoff')
    if handoff is not None:
        authority = handoff.get('authority') if isinstance(handoff, dict) else None
        c.need(isinstance(handoff, dict) and handoff.get('status') == 'EXPLICIT_NESTED_TRAIN200_FINAL_EVALUATION_HANDOFF_AUTHORIZED'
               and handoff.get('training_source_commit') == c.SOURCE and handoff.get('training_unit') == c.UNIT
               and handoff.get('training_run') == c.RUN and handoff.get('actual_training_completed_outer') == 200
               and handoff.get('exact_executor_resume') is False and handoff.get('training_stop_or_resume_by_wrapper') is False
               and isinstance(authority, dict) and authority.get('scope') == 'only_final_evaluation_after_completed_training200'
               and all(isinstance(authority.get(key), str) and authority[key].strip()
                       for key in ('human_instruction', 'original_instruction_time')),
               'Exact final-only handoff/source/authority differs', 'REJECTED_INPUT_NO_WRITE')
        for key, expected in (('config_sha256',c.CONFIG),('training_job_id',c.JOB),
                              ('invocation_id',c.INVOCATION),('training_invocation_id',c.INVOCATION)):
            c.need(key not in handoff or handoff[key] == expected,
                   'Handoff explicit '+key+' differs', 'REJECTED_INPUT_NO_WRITE')
        c.need(not active and all(item.get('MainPID') == '0' for item in (ps,rs)),
               'Authorized handoff requires the actually terminal original unit', 'REJECTED_INPUT_NO_WRITE')
        failed_progress = (result or {}).get('actual_progress_evidence') or {}
        c.need(result and result.get('status') == 'FAILED'
               and failed_progress.get('actual_completed_outer_from_attempt_results') == 200
               and c.counts(failed_progress.get('completed_actual_optimizer_calls_per_rank'), 'Handoff terminal calls')
                   == c.counts((poll.get('progress') or {}).get('completed_actual_optimizer_calls_per_rank'), 'Handoff poll calls'),
               'Handoff must retain the real failed final-phase supervisor and actual200 accounting', 'REJECTED_INPUT_NO_WRITE')
        # The existing probe's zero pids.current is an actual own-scope count.
        # If the cgroup already disappeared (null), require separate captured
        # own-descendant evidence; never interpret null as an empty cgroup.
        terminal = resource.get('final_handoff_original_terminal') or {}
        terminal_state = terminal.get('systemd') or {}
        captured_empty = (terminal.get('unit') == c.UNIT and terminal.get('actual_completed_training_outer') == 200
                          and terminal_state.get('MainPID') == '0'
                          and terminal_state.get('ActiveState') in ('inactive', 'failed')
                          and terminal_state.get('ControlGroup', '') in (expected_cgroup, '')
                          and terminal_state.get('InvocationID',c.INVOCATION) == c.INVOCATION
                          and terminal_state.get('own_cgroup_live_descendants_empty') is True)
        residual = result.get('cleanup_residual', {})
        reported_empty = (result.get('own_cgroup_live_residual_empty') is True and not result.get('cleanup_error')
                          and isinstance(residual,dict) and all(state == 'Z' for state in residual.values()))
        c.need(pids == 0 or captured_empty or reported_empty,
               'Handoff own cgroup empty state is unobserved', 'PENDING_FINAL_PHASE_NO_WRITE')
        return dict(kind='explicit_authorized_final_only_handoff', evidence=handoff, normal_exit_claimed=False)
    return c.final_phase(poll, resource, active)


def recorded_final_phase(c, poll, resource, active, recorded):
    """Validate immutable as-of phase separately from stronger phase selection."""
    actual = actual_final_phase(c, poll, resource, active)
    try:
        original = c.final_phase(poll, resource, active)
    except c.Observation:
        # Only the known retained-log ordering defect may be bypassed. All
        # original fixed identities/200/native/rank/cost gates remain outside
        # this function. Never invent a different old proof or rewrite it.
        stage = (poll.get('logs') or {}).get('stage.log') or ''
        c.need(not active and actual['kind'] in ('actual_training_terminal','explicit_authorized_final_only_handoff')
               and ('Final validation dataset size: 51713' in stage or 'Final validation metrics:' in stage)
               and recorded == actual,
               'Recorded final phase differs beyond the known old ordering defect', 'REJECTED_INPUT_NO_WRITE')
        return actual, dict(scope='Strong actual terminal/handoff revalidated; frozen core cannot pass retained final log with inactive unit.',
                            original_recorded_phase=recorded,old_core_retained_log_ordering_bypassed=True)
    c.need(recorded == original, 'Immutable original as-of final phase differs', 'REJECTED_INPUT_NO_WRITE')
    return actual, dict(scope='Immutable phase matches frozen core at its bound observation; stronger terminal/handoff priority is separate.',
                        original_recorded_phase=recorded,old_core_retained_log_ordering_bypassed=False)


def validate(c, finalizer, inputs):
    b = inputs.base
    proof, proof_pin = inputs.read(b/PROOF)
    c.need(proof.get('status') == 'ACTUAL_FULL200_NATIVE200_TWO_OBSERVATIONS_LOCAL_SUMMARY_FINALIZED'
           and proof.get('source_commit') == c.SOURCE and proof.get('config_sha256') == c.CONFIG
           and proof.get('unit') == c.UNIT and proof.get('invocation_id') == c.INVOCATION
           and proof.get('completed_outer') == 200 and proof.get('active_outer') is None
           and proof.get('same_original_instance') is True and proof.get('loaded_native_restore_outer') == 5
           and proof.get('current_validator_dependency_sha256') == CORE_SHA
           and proof.get('finalizer_sha256') == FINALIZER_SHA
           and proof.get('goal_status') == 'active' and proof.get('goal_complete') is False
           and proof.get('no_new_restore_performed') is True and proof.get('errors') == []
           and proof.get('new_training_submitted') is False and proof.get('running_source_hot_modified') is False,
           'Finalizer200 proof/source/instance differs', 'REJECTED_INPUT_NO_WRITE')
    full, full_pin = inputs.bound(proof.get('full_outer_receipt'))
    c.need(Path(full_pin['path']) == b/FULL, 'Exact FULL200 filename required', 'REJECTED_INPUT_NO_WRITE')
    identity(c, full)
    c.need(full.get('local_control_script_sha256') == CORE_SHA and full.get('completed_outer') == 200
           and full.get('restored_outer') == 5 and full.get('actual_next_outer_entered') is None
           and full.get('actual_training_invocation_id') == c.INVOCATION and full.get('goal_status') == 'active',
           'FULL200 boundary/original source differs', 'REJECTED_INPUT_NO_WRITE')
    bindings = full.get('input_bindings') or {}
    first, first_pin = inputs.bound(bindings.get('poll'))
    resource1, resource1_pin = inputs.bound(bindings.get('resource'))
    native, native_pin = inputs.bound(bindings.get('native_readback'))
    original, original_pin = inputs.bound(proof.get('native_readback'))
    c.need(original_pin == native_pin and original == native, 'Native proof/FULL pins differ', 'REJECTED_INPUT_NO_WRITE')
    snapshots = proof.get('snapshots')
    c.need(isinstance(snapshots, list) and len(snapshots) == 4, 'Two poll/resource bindings required')
    polls = [inputs.bound(ref) for ref in snapshots]
    c.need(polls[0][1] == first_pin and polls[2][1] == resource1_pin,
           'Proof original observation differs from FULL', 'REJECTED_INPUT_NO_WRITE')
    second, second_pin = polls[1];resource, resource_pin = polls[3]
    p1, ps1, rs1, active1, restore = poll_identity(c, first, resource1)
    progress, ps, rs, active, _ = poll_identity(c, second, resource)
    c.need(second['epoch'] > first['epoch'] and proof.get('observed_epochs') == [first['epoch'],second['epoch']],
           'Two actual ordered observations differ', 'REJECTED_INPUT_NO_WRITE')
    metric = c.validate_metric(first, 200);ranks = c.validate_ranks(first, metric, 200)
    record, guard, completed, totals = c.validate_native(native, p1, 200)
    c.need(native.get('identity', {}).get('model_identity') == MODEL
           and native.get('identity', {}).get('data_identity') == DATA,
           'Native model/data identity differs', 'REJECTED_INPUT_NO_WRITE')
    c.need(c.counts(progress.get('completed_actual_optimizer_calls_per_rank'), 'Second credited calls') == completed
           and progress.get('training_attempts_seen') == guard['training_attempts']
           and progress.get('all_skipped_attempts') == guard['refused_attempts'],
           'Second actual persisted accounting differs', 'REJECTED_INPUT_NO_WRITE')
    c.need(full.get('iteration_metrics') == metric and full.get('full_iteration_metric_pending') == (metric is None)
           and full.get('per_rank_successful_adam') == proof.get('successful_adam_per_rank') == completed
           and full.get('per_rank_actual_total_adam') == totals
           and full.get('native_manifest_sha256') == proof.get('native_manifest_sha256') == native['manifest_sha256']
           and full.get('native_driver_next_step') == record['driver']['next_step'] == 201
           and full.get('actual_training_pid') == int(rs1['MainPID']) and proof.get('main_pid') == int(rs['MainPID']),
           'FULL/proof actual metric/native/phase/PID differs', 'REJECTED_INPUT_NO_WRITE')
    _, full_phase_scope = recorded_final_phase(c, first, resource1, active1, full.get('final_phase_evidence'))
    if metric is None:
        c.need(all(r['attempt_id'] == guard['training_attempts'] for r in ranks),
               'Actual200 rank attempt differs', 'REJECTED_INPUT_NO_WRITE')
    c.need(full.get('last_rank_summaries') == ranks, 'FULL rank returns differ', 'REJECTED_INPUT_NO_WRITE')
    final, proof_phase_scope = recorded_final_phase(c, second, resource, active, proof.get('final_phase_evidence'))
    c.need(proof.get('run_result') == second.get('run_result'),
           'Actual final phase/terminal/handoff proof differs', 'REJECTED_INPUT_NO_WRITE')
    sizes1, sizes2 = first.get('own_output_files') or {}, second.get('own_output_files') or {}
    growth = {name:sizes2[name]-sizes1[name] if c.integer(sizes1.get(name)) and c.integer(sizes2.get(name)) else None
              for name in ('stage.log', 'retriever.log')}
    c.need(proof.get('log_byte_growth') == growth and proof.get('continued_progress_evidence') is None,
           'Actual final200 observations disagree', 'REJECTED_INPUT_NO_WRITE')
    costs, cost_pin = inputs.bound(proof.get('cumulative_cost'))
    c.need(Path(cost_pin['path']) == b/'CUMULATIVE_COST_THROUGH200.json', 'Exact CUM200 required')
    costs = {k:inputs.expand(v, k) for k,v in costs.items()}
    c.need(costs.get('through_effective_outer') == costs.get('target_outer') == 200 and costs.get('origin') == 'fresh',
           'Cumulative200 boundary differs', 'REJECTED_INPUT_NO_WRITE')
    rows = costs.get('rows_counted_once') or []
    c.need(len(rows) == 200 and sorted(row.get('outer') for row in rows) == list(range(1,201)),
           'Exact effective1..200 coverage required', 'REJECTED_INPUT_NO_WRITE')
    c.need(all(row.get('source_commit') == ('0befbaf99fc44942105831b3fbec94e4bf1b8ccb' if row['outer'] == 1 else
           'ba9f6627ec9a4e9749830d7b6acca53f6feec6b4' if row['outer'] <= 5 else c.SOURCE) for row in rows),
           'Cumulative ancestry differs', 'REJECTED_INPUT_NO_WRITE')
    accounting = costs.get('successful_optimizer_accounting') or {}
    c.need(accounting.get('successful_calls_per_rank_through200') == completed
           and accounting.get('direct_observed_calls_per_rank') == completed,
           'CUM actual Adam counters differ', 'REJECTED_INPUT_NO_WRITE')
    immutable = costs.get('immutable_metadata_input_hashes')
    c.need(isinstance(immutable, dict) and immutable, 'Full immutable cost manifest required')
    for pin in (full_pin, first_pin, resource1_pin, native_pin):
        ref = immutable.get(Path(pin['path']).relative_to(b).as_posix())
        c.need(isinstance(ref, dict) and ref.get('sha256') == pin['sha256'] and ref.get('bytes') == pin['bytes'],
               'CUM does not bind original FULL/poll/native/resource', 'REJECTED_INPUT_NO_WRITE')
    inputs.closure(immutable);inputs.closure(costs.get('resolved_fixed_cost_inputs') or {})
    inputs.closure(costs.get('candidate_cost_backfill_join') or {})
    inputs.closure(full.get('input_bindings'));inputs.closure(proof.get('attachments') or {})
    index, index_pin = inputs.bound(proof.get('candidate_index'))
    candidate, candidate_pin = inputs.bound(proof.get('candidate_current_supplement'))
    rates, rates_pin = inputs.bound(proof.get('current_branch_fallback_rates'))
    registry, registry_pin = inputs.bound(proof.get('candidate_registry'))
    c.need(Path(index_pin['path']) == b/'CANDIDATE_COST_EXTENSION_OUTER0200.json'
           and Path(candidate_pin['path']) == b/'candidate_cost_backfill/CANDIDATE_OUTER0200.json'
           and Path(rates_pin['path']) == b/'BRANCH_FALLBACK_RATES_OUTER0200.json'
           and Path(registry_pin['path']) == b/'CANDIDATE_COST_EXTENSION_INPUTS.json',
           'Exact current200 candidate/rates/registry paths required', 'REJECTED_INPUT_NO_WRITE')
    c.need(all(v.get('source_commit') == c.SOURCE and v.get('config_sha256') == c.CONFIG for v in (index,candidate,rates))
           and index.get('run_directory') == c.RUN and index.get('actual_completed_cost_outers') == [200]
           and candidate.get('requested_outer') == rates.get('outer') == 200
           and candidate.get('status') == 'ACTUAL_COMPLETED_CANDIDATE_SCALARS_AGGREGATED',
           'Actual candidate200 identity/completion differs', 'REJECTED_INPUT_NO_WRITE')
    parts = index.get('parts');entries = [row for row in registry.get('indexes', []) if row.get('outer') == 200]
    c.need(isinstance(parts, list) and len(parts) == 1 and parts[0].get('outer') == 200 and len(entries) == 1,
           'Exactly one actual candidate200 index/part required', 'REJECTED_INPUT_NO_WRITE')
    for ref, expected, root in [(parts[0],candidate_pin,b), (entries[0],index_pin,b),
             (rates.get('source_part'),candidate_pin,None), (rates.get('source_full_outer_receipt'),full_pin,None),
             (rates.get('cumulative_snapshot'),cost_pin,None)]:
        _, actual = inputs.bound(ref, root)
        c.need(actual == expected, 'Candidate/rates/FULL/CUM join differs', 'REJECTED_INPUT_NO_WRITE')
    inputs.closure(registry);inputs.closure(index);inputs.closure(candidate.get('source_pin') or {})
    c.need(proof.get('candidate_cost_join_scope') == rates.get('cumulative_boundary'),
           'Candidate cost scope differs', 'REJECTED_INPUT_NO_WRITE')
    # Validate the original finalizer's own complete top-level part manifest as
    # well as our recursive closure. Nested part pins are additionally retained.
    _, original_parts = finalizer.expand_cost_fields(inputs.cache[cost_pin['path']],cost_pin,inputs.bound)
    c.need(proof.get('cumulative_cost_field_parts') == original_parts,
           'Finalizer cost part manifest differs', 'REJECTED_INPUT_NO_WRITE')
    actual_dev, latest, seconds, _ = c.dev_evidence(second, {}, c.validate_metric(second,200),200)
    if actual_dev is None:
        actual_dev, latest, seconds, _ = c.dev_evidence(first, {}, metric,200)
    if actual_dev is not None:latest = actual_dev
    # Missing current200 normal dev is null/pending; historical195 never becomes
    # current or latest generic200 score. Historical numbered fields are kept.
    if actual_dev is None:latest = None;seconds = None
    if metric is None:seconds = None
    curve_pin = (proof.get('attachments') or {}).get('learning_curve_complete')
    if curve_pin is not None:
        curve, curve_pin = inputs.bound(curve_pin)
        end = (curve.get('training_outer_range') or [None])[-1]
        c.need(c.integer(end) and end <= 200 and curve.get('target_outer') == 200,
               'Bound curve boundary differs', 'REJECTED_INPUT_NO_WRITE')
    return dict(proof=proof,proof_pin=proof_pin,full=full,full_pin=full_pin,first_pin=first_pin,
        poll=second,poll_pin=second_pin,resource=resource,resource_pin=resource_pin,native=native,native_pin=native_pin,
        record=record,guard=guard,completed=completed,totals=totals,restore=restore,ranks=ranks,metric=metric,
        final=final,phase_validation=dict(full_asof=full_phase_scope,finalizer_asof=proof_phase_scope),
        active=active,ps=ps,rs=rs,costs=costs,cost_pin=cost_pin,index_pin=index_pin,
        candidate_pin=candidate_pin,rates=rates,rates_pin=rates_pin,registry_pin=registry_pin,
        dev=actual_dev,dev_seconds=seconds,curve_pin=curve_pin)


def aliases(c, d, metadata, now, receipt_path):
    p, metric, dev = d['poll'], d['metric'], d['dev']
    completed, totals = d['completed'], d['totals']
    restored = c.counts(d['restore']['driver']['completed_optimizer_calls_per_rank'], 'Actual restored calls')
    child = {r:totals[r]-restored[r] for r in totals}
    credited_child = {r:completed[r]-restored[r] for r in completed}
    c.need(all(n >= 0 for n in [*child.values(),*credited_child.values()]), 'Child counts below restore boundary')
    kind = d['final']['kind'];pending = metric is None
    phase = {'actual_training_terminal':'TRAIN200_COMPLETE_ACTUAL_TRAINING_TERMINAL_FINAL_WORK_PENDING',
             'explicit_authorized_final_only_handoff':'TRAIN200_COMPLETE_AUTHORIZED_FINAL_HANDOFF_PENDING'}.get(
             kind, 'TRAIN200_COMPLETE_ACTUAL_FINAL_PHASE_PENDING')
    def scalar(name):return metric.get(name) if metric else None
    wall, dev_seconds = scalar('timing_s/step'), d['dev_seconds']
    ordinary = wall-dev_seconds if wall is not None and dev_seconds is not None else None
    variance_names = ('intra_adv_population_variance','group_actual_adv_population_variance',
        'within_root_raw_reward_population_variance','group_raw_reward_population_variance','mixed_reward_question_fraction')
    variances = {'tree_variance/'+name:scalar('tree_variance/'+name) for name in variance_names}
    costs = {name:scalar(name) for name in ('rollout/candidate_attempts','rollout/candidate_accepted',
        'rollout/candidate_fallbacks','rollout/candidate_discarded','rollout/generated_tokens','rollout/prefill_tokens',
        'rollout/retrieval_calls','timing_s/gen','timing_s/update_actor')}
    direct = p.get('direct_rank_records');direct = int(direct) if c.integer(direct) and direct >= 0 else None
    scores = dict(outer=200 if dev else None,f1_percent=100*dev['metrics']['val/f1/all'] if dev else None,
                  em_percent=100*dev['metrics']['val/em/all'] if dev else None,current_outer_dev_complete=bool(dev))
    value = dict(status=phase,phase=phase,current_phase=phase,training_phase=phase,
        observed_current_training_phase=phase,current_observation_status=phase,source_commit=c.SOURCE,
        config_sha256=c.CONFIG,unit=c.UNIT,run=c.RUN,target_outer=200,main_pid=int(d['rs']['MainPID']),
        pid=int(d['rs']['MainPID']),invocation_id=c.INVOCATION,invocation=c.INVOCATION,
        original_training_pid=int(c.PID),goal_status='active',goal_active=True,goal_complete=False,
        formal_effect_complete=False,normal_training_continues=False,
        new_completed_outer=195,new_completed_outer_scope='Actual direct child6..200 (195 completed outers); effective1..200 includes actual cold-restored native5. No training201.',
        last_completed_sampling_outer=200,latest_sampling_completed_outer=200,
        active_outer_optimizer_scope='Actual training update limit200 completed; no active/next training outer. Final phase is separately bound.',
        successful_adam_per_rank=completed,per_rank_successful_adam=completed,actual_successful_adam_per_rank=completed,
        actual_total_adam_per_rank=totals,per_rank_actual_total_adam=totals,last_confirmed_chain_adam_per_rank=completed,
        child_successful_adam_per_rank=credited_child,actual_child_recorded_physical_Adam_calls=sum(child.values()),
        actual_physical_rank_optimizer_calls=sum(child.values()),actual_physical_rank_optimizer_calls_all_history=None,
        all_historical_physical_Adam_calls=None,effective_boundary_rank_calls=sum(totals.values()),
        actual_physical_rank_optimizer_calls_scope='Actual direct child6..200 total native per-rank calls minus actual restored100/rank; includes recorded child partial writes.',
        actual_physical_rank_optimizer_calls_all_history_scope='Unknown failed-ancestor physical partial calls remain null.',
        effective_boundary_rank_calls_scope='Actual native200 totals include restored native5 plus direct child6..200 and recorded child partial writes; excludes failed-ancestor physical calls.',
        successful_adam_per_rank_scope='Actual credited native200 per-rank counters; no 4000 assumption.',
        child_successful_adam_per_rank_scope='Actual native200 credited counters minus actual cold-restored100/rank.',
        actual_current_restore_outer=5,restored_outer=5,loaded_native_restore_outer=5,native_restore_step=5,
        inherited_completed_outer=5,actual_current_restore_adam_per_rank=restored,
        actual_current_restore_scope='Actual cold-loaded native5/100 each remains the only cold restore; saved native200 is writer metadata.',
        actual_child_counters_known=True,effective_counters_known=True,
        actual_training_progress=p['progress'],all_skipped_attempts=d['guard']['refused_attempts'],
        direct_rank_record_count=direct,direct_child_rank_record_count=direct,direct_child_rank_records=direct,
        inherited_rank_records=40,inherited_rank_record_count=40,
        effective_rank_records=direct+40 if direct is not None else None,
        effective_chain_rank_record_count=direct+40 if direct is not None else None,
        actual_outer_successful_writes_per_rank={str(r['rank']):r['actual_successful_writes'] for r in d['ranks']},
        actual_outer_processed_opportunities_per_rank={str(r['rank']):r['actual_opportunities_processed'] for r in d['ranks']},
        actual_outer_skipped_opportunities_per_rank={str(r['rank']):r['actual_opportunities_skipped'] for r in d['ranks']},
        full_metric_processed_opportunities=scalar('actor/optimizer_opportunities_processed'),
        full_metric_skipped_opportunities=scalar('actor/optimizer_opportunities_skipped'),no_all20_writes_or_zero_skip_assumption=True,
        native_saved_driver_next_step=201,native_saved_cursor_is_not_live_next_outer_entry=True,
        native_saved_cursor_scope='Actual saved native200 driver next_step201 is a cursor only; training201 was not entered.',
        native_manifest_sha256=d['native']['manifest_sha256'],latest_native_manifest_sha256=d['native']['manifest_sha256'],
        native_manifest_history_receipt_sha256=d['native']['receipt_sha256'],latest_native_history_receipt_sha256=d['native']['receipt_sha256'],
        latest_native_metadata_scope='Actual native200 writer/archive JSON metadata; no tensor read or cold reload.',
        current_native_metadata_readback=d['native_pin'],last_rank_summaries=d['ranks'],
        checkpoint=p.get('native_manifests'),recent_iteration_metrics=p.get('iteration_metrics'),
        current_native_manifest_history=d['resource'].get('state') or None,
        native_manifest_history=d['resource'].get('state') or None,
        native_watcher_systemd=d['resource'].get('watcher_systemd'),
        legacy_unbound_native_collector_observation=p.get('native_manifest_history'),
        completed_outer_rollout_cost=[v for v in p.get('rollout_cost', []) if v.get('candidate_outer') == 200],
        current_iteration_metrics=metric,current_outer_full_metric=metric,current_training_variances=variances,
        current_variances=variances,last_complete_training_variances=variances,current_candidate_cost_scalars=costs,
        current_generation_wall_seconds=scalar('timing_s/gen'),current_outer_wall_seconds=wall,
        latest_full_outer_wall_seconds=wall,last_complete_outer_wall_seconds=wall,
        latest_full_outer_wall_minutes=wall/60 if wall is not None else None,
        latest_full_outer_wall_scope='Actual FULL200 timing only; missing current200 metric remains null.',
        current_ordinary_training_seconds_excluding_dev=ordinary,latest_ordinary_training_seconds_excluding_dev=ordinary,
        current_training_archive_wall_seconds_excluding_dev=ordinary,latest_training_archive_wall_seconds_excluding_dev=ordinary,
        training_archive_wall_seconds_excluding_current_dev=ordinary,latest_completed_dev512_wall_seconds=dev_seconds,
        latest_training_variance_outer=200 if metric else None,latest_training_variance_scope='Actual current200 metric; unobserved fields null.',
        current_reward_mean=scalar('critic/rewards/mean'),current_score_mean=scalar('critic/score/mean'),
        current_dev512=dev,actual_dev512=dev,current_dev512_outer=200 if dev else None,
        current_dev512_wall_seconds=dev_seconds,normal_dev512_wall_seconds=dev_seconds,
        current_dev512_f1=dev['metrics']['val/f1/all'] if dev else None,current_dev512_em=dev['metrics']['val/em/all'] if dev else None,
        current_dev512_f1_percent=scores['f1_percent'],current_dev512_em_percent=scores['em_percent'],
        latest_dev512=dev,latest_actual_dev512=dev,latest_dev512_outer=200 if dev else None,
        latest_confirmed_dev_outer=200 if dev else None,latest_confirmed_dev_f1=dev['metrics']['val/f1/all'] if dev else None,
        latest_confirmed_dev_em=dev['metrics']['val/em/all'] if dev else None,latest_actual_dev512_scores=scores,
        latest_dev512_f1_percent=scores['f1_percent'],latest_dev512_em_percent=scores['em_percent'],
        latest_dev512_source_commit=c.SOURCE if dev else None,normal_dev512_pending=dev is None,
        normal_dev512_outer200_pending=dev is None,dev200_result_pending=dev is None,
        current_dev_boundary_scope='Only actual normal dev200 is current/latest here; older numbered dev evidence remains historical.',
        current_full_outer_progress_receipt=FULL,current_full_outer_progress_receipt_sha256=d['full_pin']['sha256'],
        last_full_outer_progress_receipt=FULL,last_full_outer_progress_receipt_sha256=d['full_pin']['sha256'],
        last_full_receipt=d['full_pin']['path'],last_full_receipt_sha256=d['full_pin']['sha256'],last_progress_receipt=FULL,
        current_completed_progress_evidence=d['proof_pin']['path'],current_completed_progress_sha256=d['proof_pin']['sha256'],
        current_actual_progress_proof=d['proof_pin'],actual_progress_receipt=d['proof_pin'],
        evidence=d['poll_pin']['path'],last_actual_poll=d['poll_pin']['path'],last_confirmed_poll=d['poll_pin']['path'],
        last_actual_poll_sha256=d['poll_pin']['sha256'],last_current_progress_poll=d['poll_pin']['path'],last_rank_observation=d['first_pin']['path'],
        current_resource_observation=d['resource_pin'],last_confirmed_resource_epoch=d['resource']['epoch'],
        current_systemd=d['ps'],latest_guard=p.get('latest_guard'),same_instance_confirmed=True,same_instance_repoll_required=False,
        current_remote_state='KNOWN_AT_ACTUAL_BOUND_FINAL200_POLL',current_remote_health_known=True,current_observation_unknown=False,
        current_observation_error=None,current_failure_evidence=None,current_observation_failure_evidence=None,
        training_terminal_proven=kind == 'actual_training_terminal',run_result=p.get('run_result'),
        current_final_phase_evidence=d['final'],final_phase_evidence=d['final'],
        original_unit_active_at_bound_poll=d['active'],
        training_body_terminal_does_not_establish_unit_exit=d['active'] and d['final']['kind'] == 'actual_training_terminal',
        original_full_final_phase_evidence=d['full'].get('final_phase_evidence'),
        original_finalizer_final_phase_evidence=d['proof'].get('final_phase_evidence'),
        final_phase_validation_scope=d['phase_validation'],
        current_final_phase_source_poll=d['poll_pin'],current_final_phase_resource=d['resource_pin'],
        current_service_CPUUsageNSec_at_bound_poll=d['rs'].get('CPUUsageNSec'),
        current_service_CPU_scope='Raw actual resource observation only; terminal whole-service CPU is not accepted by this metadata publisher.',
        final_service_CPU_seconds=None,actual_total_service_cpu_seconds=None,
        per_outer_costs_through=200,latest_cost_complete_outer=200,cumulative_cost_summary=d['cost_pin']['path'],
        cumulative_cost_summary_sha256=d['cost_pin']['sha256'],latest_cost_complete_scope=d['rates'].get('cumulative_boundary'),
        current_branch_fallback_rates_receipt=d['rates_pin']['path'],current_branch_fallback_rates_receipt_sha256=d['rates_pin']['sha256'],
        current_candidate_cost_actual_rates=d['rates'],initial_state_fallback_rates=d['rates'],
        candidate_cost_extension_latest_complete_outer=200,candidate_cost_extension_latest_part=d['candidate_pin']['path'],
        candidate_cost_extension_latest_part_sha256=d['candidate_pin']['sha256'],candidate_cost_extension_registry=d['registry_pin']['path'],
        candidate_cost_extension_registry_sha256=d['registry_pin']['sha256'],current_candidate_cost_join_scope=d['rates'].get('cumulative_boundary'),
        current_cost_coverage=dict(immutable_cumulative_join_through=200,separate_candidate_supplement_outer=200,
            candidate_supplement_for_next_outer=None,final_complete_cost_accepted=False,scope=d['rates'].get('cumulative_boundary')),
        current_cost_local_readback=dict(immutable_cost=d['cost_pin'],candidate_supplement=d['candidate_pin']),
        current_cost_summary_publication_pending_outer=None,current_cost_accounting_backfill_pending=None,
        effective_source_segments=d['costs'].get('effective_source_segments'),
        current_final200_alias_publication=dict(path=str(receipt_path),metadata_source_commit=metadata['commit'],
            script_sha256=metadata['files'][0]['sha256'],proof=d['proof_pin']),
        final200_alias_publisher_actual_source_commit=metadata['commit'],final200_alias_publisher_actual_sha256=metadata['files'][0]['sha256'],
        final200_alias_publisher_actual_completed_outer=200,
        independent_cold_reload_performed=False,exact_executor_resume=False,submitted_this_turn=False,
        new_training_submitted=False,running_source_hot_modified=False,restart_or_submit_performed=False,
        training_action='none',current_observation_training_action='none',
        final200_51713_HF_tokenizer_acceptance_complete=False,final_evaluation_complete=False,
        state_scope='Actual final200 training updates/native writer and actual final-phase proof. Generic next/active aliases are null.',
        canonical_field_authority_scope='Generic aliases now refer to actual200; numbered historical evidence is retained. Final51713/HF/tokenizer/cost acceptance/cleanup and the overall goal remain unfinished.',
        health='Actual200 completed; final phase '+kind+' at the original Invocation. Goal active; no training201 or final acceptance claim.',
        updated_at=now,recorded_at_utc=now)
    for key in ('completed_outer','current_outer','actual_completed_outer','actual_optimizer_completed_outer',
        'last_confirmed_outer','last_confirmed_chain_outer','last_confirmed_optimizer_outer','last_durable_completed_outer',
        'latest_native_checkpoint_outer','last_confirmed_native_saved_outer','latest_saved_native_outer','last_full_outer','last_complete_outer'):
        value[key] = 200
    for key in ('last_complete_full_metric_outer','last_confirmed_full_metric_outer','full_metric_last_seen','current_metric_outer'):
        value[key] = 200 if metric is not None else None
    for key in ('current_iteration_metric_pending','current_full_outer_metric_pending','current_outer_full_metric_pending',
                'full_iteration_metric_pending','full_iteration_log_pending'):
        value[key] = pending
    for key in ('observed_epoch','last_observed_epoch','last_confirmed_epoch','last_actual_poll_epoch','last_current_progress_poll_epoch'):
        value[key] = p['epoch']
    for key in ('active_outer','active_outer_in_progress','actual_next_outer_entered','actual_next_outer','current_next_outer',
        'last_confirmed_active_outer','latest_active_sampling_outer','last_active_sampling_outer','expected_next_outer',
        'next_outer','next_outer_sampling_complete','active_outer_sampling_complete','active_outer_sampling_complete_confirmed',
        'active_outer_actual_partial_Adam_calls','current_partial_optimizer_calls_per_rank','active_outer_advantage_variance',
        'active_outer_optimizer_completed','active_outer_optimizer_completion_confirmed','active_outer_full_update_unconfirmed',
        'current_active_rollout_progress','current_live_sampling_progress','current_live_validation_progress',
        'current_actual_optimizer_progress','active_outer_rollout_cost','current_next_outer_progress',
        'current_next_outer_progress_evidence','current_rollout_cost','latest_rollout_cost','current_latest_sampling_interval_log_growth',
        'active_outer_in_validation','expected_next_dev_outer','next_outer_scoring','current_active_outer_full_metric_pending'):
        value[key] = None
    for key in ('current_active_sampling_cost_scope','current_next_outer_scope','expected_next_outer_scope','current_latest_sampling_interval_scope'):
        value[key] = 'Training update limit200 completed; no active or next training outer is claimed.'
    for key in ('score_updated','dev_score_updated','new_score_updated','dev_score_updated_this_turn','score_updated_this_turn'):
        value[key] = bool(dev and d['full'].get('dev_score_updated_this_turn'))
    value['score_update_scope'] = 'Existing actual dev200 record only; this entry performs no evaluation.'
    for key, name in [('latest_actual_total_adv_population_variance','group_actual_adv_population_variance'),
            ('latest_intra_adv_population_variance','intra_adv_population_variance'),
            ('latest_within_root_raw_reward_population_variance','within_root_raw_reward_population_variance'),
            ('latest_mixed_reward_question_fraction','mixed_reward_question_fraction')]:
        value[key] = scalar('tree_variance/'+name)
    return value


def layout(c, workspace, mirrors):
    b=workspace/c.BASE_REL
    json_paths=[workspace/'outputs/STATUS.json',workspace/'outputs/tree-nested-branch/STATUS.json',
                b/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json',b/'CURRENT_OBSERVER_CONNECTION_STATE.json']
    reports=[workspace/'outputs/REPORT.md',workspace/'outputs/tree-nested-branch/REPORT.md']
    targets=json_paths+reports
    links={m/p.name:p for m in mirrors for p in [*json_paths[1:],reports[1]]}
    return json_paths,reports,targets,links


def current(c, path, raw, epoch):
    if path.suffix != '.json':return
    value = strict_json(raw)
    c.need(isinstance(value,dict) and value.get('source_commit') == c.SOURCE and value.get('goal_status') == 'active'
           and value.get('goal_complete') is not True and value.get('goal_active',True) is not False,
           'Current metadata source/unfinished goal differs', 'STALE_SNAPSHOT_NO_WRITE')
    for key in ('config_sha256','unit','run'):
        expected={'config_sha256':c.CONFIG,'unit':c.UNIT,'run':c.RUN}[key]
        c.need(value.get(key,expected) == expected, 'Current original identity differs', 'STALE_SNAPSHOT_NO_WRITE')
    for key in ('current_outer','completed_outer','actual_completed_outer','last_confirmed_outer','latest_native_checkpoint_outer'):
        c.need(value.get(key) is None or (c.number(value[key]) and value[key] <= 200),
               'Current metadata is newer than selected boundary', 'STALE_SNAPSHOT_NO_WRITE')
    for key in ('observed_epoch','last_observed_epoch','last_confirmed_epoch','last_actual_poll_epoch','last_current_progress_poll_epoch'):
        c.need(value.get(key) is None or (c.number(value[key]) and value[key] <= epoch),
               'Current metadata has a newer observation', 'STALE_SNAPSHOT_NO_WRITE')


def render(c, d, metadata, now, receipt_path, prior, canonical, reports, links):
    common=aliases(c,d,metadata,now,receipt_path);outputs={}
    # Clear any other *generic* live aliases without altering explicitly
    # numbered/historical keys. This includes keys introduced after this release.
    live = re.compile(r'^(?:active_outer|actual_next_outer|current_next_outer|expected_next_outer|next_outer|latest_active_sampling_outer|last_active_sampling_outer|last_confirmed_active_outer)(?:$|_)')
    for path in canonical:
        old=strict_json(prior[path]);value=copy.deepcopy(old)
        retired={k:v for k,v in old.items() if (k in common and common[k] != v) or
                 (live.match(k) and not re.search(r'\d',k) and k not in common)}
        archive=value.setdefault('historical_current_alias_reconciliations',{})
        c.need(isinstance(archive,dict) and 'final200' not in archive,
               'Unknown preexisting final200 alias archive', 'REJECTED_INPUT_NO_WRITE')
        archive['final200']=dict(scope='Prior generic metadata, retained as historical evidence.',fields=retired)
        for key in old:
            if live.match(key) and not re.search(r'\d',key) and key not in common:value[key]=None
        value.update(copy.deepcopy(common));outputs[path]=encoded(value)
    outputs[canonical[1]]=outputs[canonical[0]]
    report=(f'Current nested Tree200: actual completed training updates200; native200 writer metadata complete.\n\n'
        f'Actual final phase: {d["final"]["kind"]}; PID at the bound observation {d["rs"]["MainPID"]}, '
        f'original PID {c.PID}, Invocation {c.INVOCATION}. Source {c.SOURCE}; config {c.CONFIG}.\n\n'
        f'Two actual poll epochs {d["proof"]["observed_epochs"]}. All generic active/next training aliases are null. '
        'Saved driver next_step201 is a cursor only; no training201 entry is claimed.\n\n'
        f'Actual credited Adam/rank {d["completed"]}; totals with recorded partial writes {d["totals"]}. '
        'Direct child counts subtract the actual cold-restored100/rank. Cold load remains native5. '
        'Unknown all-history physical optimizer calls remain null.\n\n'
        f'Current FULL200 metric {"pending; timing/variance scalars null" if d["metric"] is None else "present"}; '
        f'normal dev200 {"complete" if d["dev"] else "pending; current/latest dev scores null"}. '
        f'Current outer seconds {common["current_outer_wall_seconds"]}; dev seconds {d["dev_seconds"]}.\n\n'
        f'Existing immutable CUM200 {d["cost_pin"]["path"]}; candidate200 {d["candidate_pin"]["path"]}; '
        f'rates {d["rates_pin"]["path"]}. {d["rates"].get("cumulative_boundary", "")}\n\n'
        'All required local manifests and recursive cumulative parts are SHA/byte checked with both mirrors. '
        'Historical numbered evidence is retained. No CUM200 rewrite, evaluation, tensor read, cold reload, training action or remote operation.\n\n'
        'Goal remains active. Full51713 scoring, final HF/tokenizer proof, independent acceptance, final resource/CPU acceptance and cleanup remain separate unfinished work.\n').encode()
    for path in reports:outputs[path]=report
    for path,source in links.items():outputs[path]=outputs[source]
    return outputs


def publish(c, d, metadata, inputs, workspace, mirrors):
    b=inputs.base;canonical,reports,targets,links=layout(c,workspace,mirrors)
    receipt_path=b/RECEIPT;receipt_paths=[receipt_path,*(m/RECEIPT for m in mirrors)]
    input_pins=sorted(inputs.pins.values(),key=lambda p:p['path'])
    binding=dict(metadata_source=metadata,input_pins=input_pins,pinned_core_sha256=CORE_SHA,pinned_finalizer_sha256=FINALIZER_SHA)
    fingerprint=digest(encoded(binding));snapshot=b/('final200_alias_prior_'+fingerprint[:20]);manifest_path=snapshot/'SNAPSHOT_MANIFEST.json'
    all_targets=[*targets,*links]
    exists=[p.exists() for p in receipt_paths]
    rerun=any(exists)
    c.need(not rerun or all(exists), 'Partial final200 receipt publication; preserve backup for recovery', 'PARTIAL_PUBLICATION_NO_WRITE')
    if rerun:
        receipt_raw=receipt_path.read_bytes();receipt=strict_json(receipt_raw)
        c.need(all(p.read_bytes() == receipt_raw for p in receipt_paths)
               and receipt.get('input_fingerprint') == fingerprint and receipt.get('bindings') == binding,
               'Existing receipt mirrors/original evidence/source differ', 'REJECTED_INPUT_NO_WRITE')
        now=receipt.get('created_at_utc');manifest_raw=manifest_path.read_bytes()
        c.need(receipt.get('backup') == dict(path=str(manifest_path),bytes=len(manifest_raw),sha256=digest(manifest_raw)),
               'Backup manifest differs from receipt', 'REJECTED_INPUT_NO_WRITE')
        manifest=strict_json(manifest_raw);rows=manifest.get('files')
        c.need(manifest.get('schema') == 'nested_final200_alias_backup_v1'
               and manifest.get('input_fingerprint') == fingerprint and manifest.get('created_at_utc') == now
               and isinstance(rows,list) and len(rows) == 14 and {r.get('path') for r in rows} == {str(p) for p in all_targets},
               'Backup whitelist/source binding differs', 'REJECTED_INPUT_NO_WRITE')
        prior={}
        for i,(path,row) in enumerate(zip(all_targets,rows)):
            member=snapshot/f'{i:02d}-{path.name}.gz';stored=member.read_bytes()
            c.need(row.get('path') == str(path) and row.get('stored_path') == str(member)
                   and row.get('stored_bytes') == len(stored) and row.get('stored_sha256') == digest(stored)
                   and int.from_bytes(stored[4:8],'little') == 0,
                   'Backup closed gzip bytes differ', 'REJECTED_INPUT_NO_WRITE')
            raw=gzip.decompress(stored)
            c.need(row.get('bytes') == len(raw) and row.get('sha256') == digest(raw),
                   'Backup original bytes differ', 'REJECTED_INPUT_NO_WRITE')
            prior[path]=raw
    else:
        c.need(not snapshot.exists(), 'Partial prior attempt exists; preserve backup for recovery', 'PARTIAL_PUBLICATION_NO_WRITE')
        now=datetime.now(timezone.utc).isoformat();prior={p:p.read_bytes() for p in all_targets}
    for path,raw in prior.items():
        c.need(len(raw) <= MAX_BYTES, 'Current metadata exceeds compact limit', 'REJECTED_INPUT_NO_WRITE')
        current(c,path,raw,d['poll']['epoch'])
    c.need(prior[canonical[0]] == prior[canonical[1]], 'Root/nested STATUS bytes differ', 'REJECTED_INPUT_NO_WRITE')
    for path,source in links.items():
        c.need(prior[path] == prior[source], 'Prior canonical/mirror bytes differ', 'REJECTED_INPUT_NO_WRITE')
    outputs=render(c,d,metadata,now,receipt_path,prior,canonical,reports,links)
    expected_outputs=[dict(path=str(p),bytes=len(raw),sha256=digest(raw)) for p,raw in outputs.items()]
    if not rerun:
        rows=[]
        for i,(path,raw) in enumerate(prior.items()):
            stored=gzip.compress(raw,mtime=0);member=snapshot/f'{i:02d}-{path.name}.gz'
            rows.append(dict(path=str(path),bytes=len(raw),sha256=digest(raw),stored_path=str(member),
                             stored_bytes=len(stored),stored_sha256=digest(stored)))
        manifest_raw=encoded(dict(schema='nested_final200_alias_backup_v1',created_at_utc=now,
                                 input_fingerprint=fingerprint,files=rows,global_atomic_transaction=False))
    expected= dict(schema='nested_final200_alias_publication_v1',status='ACTUAL_FINAL200_LOCAL_METADATA_ALIASES_PUBLISHED',
        created_at_utc=now,source_commit=c.SOURCE,config_sha256=c.CONFIG,unit=c.UNIT,run=c.RUN,
        original_training_pid=int(c.PID),actual_bound_main_pid=int(d['rs']['MainPID']),invocation_id=c.INVOCATION,
        completed_outer=200,active_outer=None,actual_next_outer_entered=None,native_saved_driver_next_step=201,
        input_fingerprint=fingerprint,bindings=binding,outputs=expected_outputs,
        backup=dict(path=str(manifest_path),bytes=len(manifest_raw),sha256=digest(manifest_raw)),
        current_metric_pending=d['metric'] is None,actual_cold_load_outer=5,
        actual_adam_per_rank=d['totals'],final_phase_evidence=d['final'],
        actual_training201_claimed=False,two_mirrors_byte_equal=True,immutable_evidence_modified=False,
        CUM200_rewritten=False,source_training_modified=False,tensor_contents_read=False,
        independent_cold_reload_performed=False,remote_actions=False,global_atomic_transaction=False,
        final51713_HF_resource_CPU_acceptance_claimed=False,goal_status='active',goal_complete=False)
    expected_raw=encoded(expected)
    if rerun:
        c.need(receipt_raw == expected_raw, 'Existing receipt is not the recomputed original publication', 'REJECTED_INPUT_NO_WRITE')
        for _ in range(2):
            inputs.unchanged()
            c.need(all(p.is_file() and p.read_bytes() == raw for p,raw in outputs.items())
                   and all(p.read_bytes() == expected_raw for p in receipt_paths),
                   'Partial/tampered metadata publication; no idempotent PASS', 'PARTIAL_PUBLICATION_NO_WRITE')
        return dict(status='IDEMPOTENT_FINAL200_ORIGINAL_EVIDENCE_AND_BYTES_VERIFIED',receipt=str(receipt_path),writes=0)
    with (b/'.final200_alias_publication.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs.unchanged()
        c.need(all(p.read_bytes() == raw for p,raw in prior.items()) and not any(p.exists() for p in receipt_paths)
               and not snapshot.exists(), 'Concurrent target publication', 'CONCURRENT_UPDATE_NO_WRITE')
        snapshot.mkdir()
        for i,(path,raw) in enumerate(prior.items()):
            member=snapshot/f'{i:02d}-{path.name}.gz';member.write_bytes(gzip.compress(raw,mtime=0))
            c.need(gzip.decompress(member.read_bytes()) == raw, 'Backup exact original byte readback failed')
        manifest_path.write_bytes(manifest_raw)
        c.need(manifest_path.read_bytes() == manifest_raw, 'Backup manifest readback failed')
        try:
            for path,raw in outputs.items():
                c.need(path.read_bytes() == prior[path], 'Target changed before replace', 'CONCURRENT_UPDATE_NO_WRITE')
                c.atomic_replace(path,raw)
            # Receipt is the last stage; its presence alone never establishes PASS.
            for _ in range(2):
                inputs.unchanged()
                c.need(all(p.read_bytes() == raw for p,raw in outputs.items()), 'Two-pass output readback differs')
            for path in receipt_paths:
                c.need(not path.exists(), 'Receipt appeared concurrently', 'CONCURRENT_UPDATE_NO_WRITE')
                c.atomic_replace(path,expected_raw)
            for _ in range(2):
                c.need(all(p.read_bytes() == raw for p,raw in outputs.items())
                       and all(p.read_bytes() == expected_raw for p in receipt_paths), 'Final publication readback differs')
        except Exception:
            # Never restore over an unknown concurrent value. A failed rollback
            # preserves the closed backup and cannot be mistaken for a success.
            c.need(all(p.read_bytes() in (prior[p],outputs[p]) for p in outputs),
                   'Unknown concurrent bytes; manual recovery required', 'PARTIAL_PUBLICATION_NO_WRITE')
            for path,raw in prior.items():c.atomic_replace(path,raw)
            for path in receipt_paths:
                if path.exists():
                    c.need(path.read_bytes() == expected_raw, 'Unknown concurrent receipt; manual recovery required')
                    path.unlink()
            raise
    return dict(status=expected['status'],receipt=str(receipt_path),receipt_sha256=digest(expected_raw),
                backup=str(manifest_path),modified_metadata_documents=14,mirrors=2,
                two_readbacks=True,remote_actions=False,goal_status='active',goal_complete=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--metadata-source-commit',required=True)
    args=parser.parse_args()
    c,finalizer,dependency_paths=dependencies()
    workspace=args.workspace.resolve()
    mirrors=[workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',EXTERNAL_MIRROR]
    c.need(all(p.is_dir() for p in mirrors), 'Both existing local mirrors required', 'REJECTED_INPUT_NO_WRITE')
    metadata=metadata_source(c,args.metadata_source_commit,dependency_paths)
    inputs=Inputs(c,workspace,mirrors);data=validate(c,finalizer,inputs)
    print(json.dumps(publish(c,data,metadata,inputs,workspace,mirrors)))


if __name__ == '__main__':
    try:main()
    except Exception as error:
        print(json.dumps(dict(status=getattr(error,'status','LOCAL_INPUT_OR_IO_ERROR_NO_SUCCESS'),
            reason=str(error),successful_publication_claimed=False,remote_actions=False,automatic_retry=False)))
        raise SystemExit(2)
