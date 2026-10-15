"""Publish five local metadata aliases after an already reconciled FULL6..199.

Usage: python3 publish_nested_metadata_scope.py --outer N \
  --second-poll /workspace/outputs/tree-nested-branch-handoff-20261013/POLL.json \
  --workspace /workspace

The poll must be the original second poll bound by the FULL progress proof and
current alias reconciliation. All evidence, ten JSON copies and four unchanged
REPORT copies are checked before writing. Fourteen prior documents are closed
as mtime=0 gzip with SHA/size and exact decompression readback. Only the five
FIELDS below change; no counter, cost, score, curve, report, or training changes.
An existing immutable receipt is rejected rather than overwritten. Outer200 is
outside this entry's scope and cannot manufacture a live outer201 entry.
Writes use atomic replacement per file, not a global multi-file transaction.
Ordinary write errors attempt guarded rollback while holding the shared lock.
After interruption, use --recover-snapshot SNAPSHOT_DIR --workspace /workspace;
recovery refuses any document outside this publication's original/intended bytes.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ALIAS_SHA = 'f8775e2673d2844125a6683622880066cf870a3e6070a1ac847f5bb62ed8aafc'
FIELDS = ('last_completed_sampling_outer', 'active_outer_optimizer_scope',
          'new_completed_outer_scope', 'last_confirmed_poll', 'current_outer_full_metric')


def dependencies():
    import hashlib
    path = HERE/'reconcile_nested_full_current_aliases.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != ALIAS_SHA:
        raise ValueError('Pinned published alias helper changed')
    spec = importlib.util.spec_from_file_location('metadata_scope_published_alias', path)
    alias = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alias)
    core, _, paths = alias.dependencies()
    return alias, core, [path, *paths]


def rank_map(core, value):
    core.need(isinstance(value, dict) and set(value) == {str(i) for i in range(8)}
              and all(type(v) is int and v >= 0 for v in value.values()),
              'Actual eight-rank Adam map is required', 'REJECTED_INPUT_NO_WRITE')
    return value


def identity(core, value):
    core.need(all(value.get(k) == v for k, v in dict(source_commit=core.SOURCE,
              config_sha256=core.CONFIG, unit=core.UNIT, run=core.RUN).items()),
              'Fixed source/config/unit/run differs', 'REJECTED_INPUT_NO_WRITE')


def current_state(core, value, n, completed, totals, poll_epoch):
    identity(core, value)
    core.need(value.get('goal_status') == 'active', 'Goal is not active', 'REJECTED_INPUT_NO_WRITE')
    for key in ('actual_optimizer_completed_outer', 'completed_outer', 'actual_completed_outer',
                'current_outer'):
        core.need(value.get(key) == n, 'Current completed boundary differs: '+key,
                  'REJECTED_INPUT_NO_WRITE')
    core.need(value.get('active_outer') == n+1 and value.get('actual_next_outer_entered') == n+1,
              'Current aliases must already confirm actual next entry', 'REJECTED_INPUT_NO_WRITE')
    for key, expected in (('actual_total_adam_per_rank', totals),
                          ('successful_adam_per_rank', completed),
                          ('actual_successful_adam_per_rank', completed)):
        core.need(rank_map(core, value.get(key)) == expected, 'Current actual Adam map differs: '+key,
                  'REJECTED_INPUT_NO_WRITE')
    core.need(value.get('actual_current_restore_outer') == 5
              and value.get('actual_current_restore_adam_per_rank') == {str(i):100 for i in range(8)},
              'Actual cold-loaded native5/100 boundary differs', 'REJECTED_INPUT_NO_WRITE')
    core.need(value.get('current_partial_optimizer_calls_per_rank') is None
              and value.get('active_outer_actual_partial_Adam_calls') is None
              and value.get('active_outer_optimizer_completed') is None
              and value.get('active_outer_optimizer_completion_confirmed') is False,
              'Next optimizer state is no longer unknown', 'REJECTED_INPUT_NO_WRITE')
    for key in ('observed_epoch', 'last_observed_epoch', 'last_actual_poll_epoch',
                'last_actual_second_snapshot_epoch', 'last_confirmed_epoch',
                'last_current_progress_poll_epoch'):
        epoch = value.get(key)
        core.need(type(epoch) in (int, float) and 0 <= epoch <= poll_epoch,
                  'Selected second poll would regress observation: '+key, 'REJECTED_INPUT_NO_WRITE')


def layout(workspace, core):
    base = workspace/core.BASE_REL
    canonical = [workspace/'outputs/STATUS.json', workspace/'outputs/tree-nested-branch/STATUS.json',
                 base/'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json', base/'CURRENT_OBSERVER_CONNECTION_STATE.json']
    reports = [workspace/'outputs/REPORT.md', workspace/'outputs/tree-nested-branch/REPORT.md']
    mirrors = [workspace/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
               Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
    return base, canonical, reports, mirrors


def recover_snapshot(core, workspace, snapshot):
    """Caller holds the shared lock; preflight every byte before restoring any."""
    snapshot = snapshot.resolve();manifest_path = snapshot/'SNAPSHOT_MANIFEST.json'
    manifest, _ = core.held(manifest_path)
    n = manifest.get('outer');base, canonical, reports, mirrors = layout(workspace, core)
    core.need(type(n) is int and 6 <= n <= 199
              and snapshot.parent == base.resolve()
              and re.fullmatch(fr'metadata_scope_prior_outer{n}_\d{{8}}T\d{{12}}Z', snapshot.name),
              'Recovery snapshot path or outer is outside the whitelist', 'RECOVERY_REFUSED_NO_WRITE')
    identity(core, manifest)
    core.need(manifest.get('schema') == 'metadata_scope_recovery_manifest_v1'
              and manifest.get('source_script_sha256') == core.digest(Path(__file__).read_bytes()),
              'Recovery script/source binding differs', 'RECOVERY_REFUSED_NO_WRITE')
    full_pin = manifest.get('full') or {};full_path = base/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json'
    full, actual_full_pin = core.held(full_path)
    core.need(full_pin == actual_full_pin and full.get('completed_outer') == n,
              'Recovery immutable FULL binding differs', 'RECOVERY_REFUSED_NO_WRITE')
    identity(core, full)
    json_paths = canonical + [m/p.name for m in mirrors for p in canonical[1:]]
    report_paths = reports + [m/'REPORT.md' for m in mirrors]
    allowed = {str(p.resolve()):p for p in json_paths+report_paths}
    files = manifest.get('files')
    core.need(isinstance(files, list) and len(files) == 14
              and {row.get('source') for row in files} == set(allowed),
              'Recovery document whitelist differs', 'RECOVERY_REFUSED_NO_WRITE')
    original = {};current = {}
    for index, row in enumerate(files):
        path = allowed[row['source']];member = snapshot/f'{index:02d}-{path.name}.gz'
        core.need(row.get('stored_gzip_path') == str(member) and row.get('gzip_mtime') == 0,
                  'Recovery gzip path differs', 'RECOVERY_REFUSED_NO_WRITE')
        stored = member.read_bytes()
        core.need(len(stored) == row.get('stored_bytes') and core.digest(stored) == row.get('stored_sha256')
                  and int.from_bytes(stored[4:8], 'little') == 0,
                  'Recovery closed gzip bytes differ', 'RECOVERY_REFUSED_NO_WRITE')
        raw = gzip.decompress(stored)
        core.need(len(raw) == row.get('original_bytes') and core.digest(raw) == row.get('original_sha256'),
                  'Recovery original bytes differ', 'RECOVERY_REFUSED_NO_WRITE')
        observed = path.read_bytes();pin = (len(observed), core.digest(observed))
        permitted = {(row['original_bytes'], row['original_sha256'])}
        if path in json_paths:permitted.add((row['intended_bytes'], row['intended_sha256']))
        core.need(pin in permitted, 'Unknown current document; manual recovery required: '+str(path),
                  'RECOVERY_REFUSED_NO_WRITE')
        original[path] = raw;current[path] = observed
    receipt_path = base/f'CURRENT_METADATA_SCOPE_RECONCILIATION_OUTER{n}.json'
    receipts = [receipt_path, *(m/receipt_path.name for m in mirrors)]
    receipt_pins = manifest.get('receipt_files')
    core.need(isinstance(receipt_pins, list) and len(receipt_pins) == 3
              and {row.get('path') for row in receipt_pins} == {str(p) for p in receipts},
              'Recovery receipt whitelist differs', 'RECOVERY_REFUSED_NO_WRITE')
    existing_receipts = {}
    for row in receipt_pins:
        path = Path(row['path'])
        if path.exists():
            raw = path.read_bytes()
            core.need(len(raw) == row.get('bytes') and core.digest(raw) == row.get('sha256'),
                      'Unknown receipt; manual recovery required: '+str(path), 'RECOVERY_REFUSED_NO_WRITE')
            value = json.loads(raw)
            core.need(value.get('source_script_sha256') == manifest['source_script_sha256']
                      and value.get('completed_outer') == n and value.get('full') == full_pin
                      and value.get('backup', {}).get('path') == str(manifest_path)
                      and value.get('backup', {}).get('files_sha256') == core.digest(core.json_bytes(files)),
                      'Receipt does not belong to this publication', 'RECOVERY_REFUSED_NO_WRITE')
            existing_receipts[path] = raw
    core.need(all(p.read_bytes() == raw for p,raw in current.items())
              and all(p.read_bytes() == raw for p,raw in existing_receipts.items()),
              'Documents changed during recovery preflight', 'RECOVERY_REFUSED_NO_WRITE')
    for path in json_paths:
        core.need(path.read_bytes() == current[path], 'Document changed during recovery; manual recovery required')
        core.atomic_replace(path, original[path])
    for path, raw in existing_receipts.items():
        core.need(path.read_bytes() == raw, 'Receipt changed during recovery; manual recovery required')
        path.unlink()
    core.need(all(p.read_bytes() == raw for p,raw in original.items()) and not any(p.exists() for p in receipts),
              'Recovery incomplete; preserve snapshot for manual recovery')
    return dict(status='ORIGINAL_TEN_JSON_RESTORED_REPORTS_UNCHANGED', snapshot=str(snapshot),
                removed_matching_receipts=len(existing_receipts), global_atomic_transaction=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outer', type=int)
    parser.add_argument('--second-poll', type=Path)
    parser.add_argument('--recover-snapshot', type=Path)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    alias, core, dependency_paths = dependencies()
    workspace = args.workspace.resolve()
    if args.recover_snapshot is not None:
        core.need(args.outer is None and args.second_poll is None,
                  'Recovery uses only --recover-snapshot and --workspace', 'REJECTED_INPUT_NO_WRITE')
        with dependency_paths[1].open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            result = recover_snapshot(core, workspace, args.recover_snapshot)
        print(json.dumps(result));return
    core.need(args.outer is not None and args.second_poll is not None,
              'Ordinary publication requires --outer and --second-poll', 'REJECTED_INPUT_NO_WRITE')
    n = args.outer
    core.need(6 <= n <= 199, 'Only ordinary FULL6..199 with an actual next entry is supported',
              'REJECTED_INPUT_NO_WRITE')
    base, canonical, reports, mirrors = layout(workspace, core)
    inputs = alias.Inputs(workspace, core)
    for path in (Path(__file__).resolve(), *dependency_paths):
        raw = path.read_bytes()
        inputs.pins.append(dict(path=str(path), bytes=len(raw), sha256=core.digest(raw)))
    full, full_pin = inputs.read(base/f'FULL_OUTER{n}_PROGRESS_RECEIPT.json')
    proof, proof_pin = inputs.read(base/f'OUTER{n}_FULL_NEXT{n+1}_ACTUAL_PROGRESS.json')
    reconciled, alias_pin = inputs.read(base/f'CURRENT_ALIAS_RECONCILIATION_OUTER{n}.json')
    poll, poll_pin = inputs.read(args.second_poll)
    identity(core, full)
    core.need(full.get('completed_outer') == n and full.get('actual_next_outer_entered') == n+1
              and full.get('restored_outer') == 5 and full.get('goal_status') == 'active'
              and full.get('full_iteration_metric_pending') is False
              and full.get('actual_training_pid') == int(core.PID)
              and full.get('actual_training_invocation_id') == core.INVOCATION,
              'Published FULL boundary or original instance differs', 'REJECTED_INPUT_NO_WRITE')
    metric = full.get('iteration_metrics')
    core.need(isinstance(metric, dict) and metric.get('step') == n,
              'Published FULL iteration metric differs', 'REJECTED_INPUT_NO_WRITE')
    totals = rank_map(core, full.get('per_rank_actual_total_adam'))
    completed = rank_map(core, full.get('per_rank_successful_adam'))
    core.need(proof.get('source_commit') == core.SOURCE and proof.get('config_sha256') == core.CONFIG
              and proof.get('unit') == core.UNIT and proof.get('main_pid') == int(core.PID)
              and proof.get('invocation_id') == core.INVOCATION and proof.get('same_original_instance') is True
              and proof.get('completed_outer') == n and proof.get('active_outer') == n+1
              and proof.get('goal_status') == 'active' and not proof.get('errors')
              and proof.get('run_result') is None, 'Completed progress proof differs', 'REJECTED_INPUT_NO_WRITE')
    core.need(reconciled.get('source_commit') == core.SOURCE and reconciled.get('outer') == n
              and reconciled.get('actual_next_outer_entered') == n+1
              and reconciled.get('status') == 'ACTUAL_FULL_LOCAL_ALIASES_RECONCILED'
              and reconciled.get('actual_adam_per_rank') == totals,
              'Ordinary alias reconciliation must already be published', 'REJECTED_INPUT_NO_WRITE')
    for descriptor, expected in ((proof.get('full_outer_receipt'), full_pin),
                                 (reconciled.get('full'), full_pin),
                                 (reconciled.get('proof'), proof_pin),
                                 (reconciled.get('poll'), poll_pin)):
        _, held = inputs.bound(descriptor)
        core.need(held == expected, 'Immutable input binding differs', 'REJECTED_INPUT_NO_WRITE')
    snapshots = proof.get('snapshots')
    core.need(isinstance(snapshots, list) and len(snapshots) == 4,
              'Original second poll/resource bindings required', 'REJECTED_INPUT_NO_WRITE')
    _, second_pin = inputs.bound(snapshots[1])
    core.need(second_pin == poll_pin, 'CLI poll is not the actual bound second poll', 'REJECTED_INPUT_NO_WRITE')
    resource, resource_pin = inputs.bound(snapshots[3])
    _, alias_resource_pin = inputs.bound(reconciled.get('resource'))
    core.need(resource_pin == alias_resource_pin, 'Second resource binding differs', 'REJECTED_INPUT_NO_WRITE')
    progress = alias.identity(core, poll, n)
    core.need(poll.get('native_load_receipt', {}).get('provenance', {}).get('child_config_sha256') == core.CONFIG
              and progress.get('completed_actual_optimizer_calls_per_rank') == completed,
              'Original poll config or completed Adam differs', 'REJECTED_INPUT_NO_WRITE')
    core.validate_unit(poll, resource, n)
    core.validate_restore(poll)
    core.need(core.validate_metric(poll, n) == metric, 'Original FULL metric differs', 'REJECTED_INPUT_NO_WRITE')
    native, native_pin = inputs.bound(reconciled.get('native'))
    _, _, actual_completed, actual_totals = core.validate_native(native, progress, n)
    core.need(actual_completed == completed and actual_totals == totals
              and native.get('manifest_sha256') == full.get('native_manifest_sha256'),
              'Published native actual Adam/manifest differs', 'REJECTED_INPUT_NO_WRITE')
    entries = [int(x) for x in re.findall(r'epoch\s+\d+,\s*step\s+(\d+)\b',
                                        (poll.get('logs') or {}).get('stage.log') or '')]
    core.need(entries and n+1 in entries and max(entries) == n+1,
              'Original poll lacks an actual next-stage entry', 'REJECTED_INPUT_NO_WRITE')
    core.need(not any(row.get('outer_step', 0) > n for row in poll.get('last_rank_summaries', [])),
              'Next rank optimizer update is already published', 'REJECTED_INPUT_NO_WRITE')
    prior = {};states = []
    for path in canonical:
        value, pin = inputs.read(path);raw = path.read_bytes()
        core.need(core.digest(raw) == pin['sha256'], 'Current JSON changed during read', 'CONCURRENT_UPDATE_NO_WRITE')
        current_state(core, value, n, completed, totals, poll['epoch'])
        prior[path] = raw;states.append(value)
    core.need(prior[canonical[0]] == prior[canonical[1]], 'Root/nested STATUS bytes differ', 'REJECTED_INPUT_NO_WRITE')
    for path in reports:
        raw = path.read_bytes();core.need(len(raw) <= 12*1024**2, 'Prior REPORT exceeds local metadata limit')
        prior[path] = raw
    for mirror in mirrors:
        for source in (canonical[1], canonical[2], canonical[3], reports[1]):
            path = mirror/source.name;raw = path.read_bytes()
            core.need(raw == prior[source], 'Current mirror bytes differ: '+str(path), 'REJECTED_INPUT_NO_WRITE')
            prior[path] = raw
    core.need(len(prior) == 14, 'Fourteen prior documents required', 'REJECTED_INPUT_NO_WRITE')
    receipt_path = base/f'CURRENT_METADATA_SCOPE_RECONCILIATION_OUTER{n}.json'
    receipt_paths = [receipt_path, *(m/receipt_path.name for m in mirrors)]
    core.need(not any(p.exists() for p in receipt_paths), 'Immutable metadata receipt already exists',
              'EXISTING_RECEIPT_NO_WRITE')
    updates = dict(last_completed_sampling_outer=n,
        active_outer_optimizer_scope=f'Actual{n+1} sampling entered; completed or partial optimizer writes for{n+1} remain unknown.',
        new_completed_outer_scope=f'Effective actual1..{n}; current child6..{n} with actually cold-loaded native5 preserved. FULL{n} complete.',
        last_confirmed_poll=poll_pin['path'], current_outer_full_metric=metric)
    outputs = {};before = [];after = []
    for path, old in zip(canonical, states):
        value = copy.deepcopy(old);value.update(copy.deepcopy(updates))
        core.need(all(value[k] == old[k] for k in old if k not in FIELDS)
                  and set(value)-set(old) <= set(FIELDS), 'Only five metadata fields may change')
        raw = core.json_bytes(value);outputs[path] = raw
        before.append(dict(file=dict(path=str(path), bytes=len(prior[path]), sha256=core.digest(prior[path])),
                           fields={k:({'metric_step':old[k].get('step')} if k == 'current_outer_full_metric'
                                   and isinstance(old.get(k), dict) else old.get(k)) for k in FIELDS}))
        after.append(dict(path=str(path), bytes=len(raw), sha256=core.digest(raw)))
    for mirror in mirrors:
        for source in canonical[1:]:outputs[mirror/source.name] = outputs[source]
    now = datetime.now(timezone.utc);stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
    snapshot = base/f'metadata_scope_prior_outer{n}_{stamp}';manifest_path = snapshot/'SNAPSHOT_MANIFEST.json'
    with dependency_paths[1].open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs.unchanged()
        core.need(all(p.read_bytes() == raw for p, raw in prior.items()),
                  'Current documents changed during preparation', 'CONCURRENT_UPDATE_NO_WRITE')
        core.need(not any(p.exists() for p in receipt_paths), 'Metadata receipt appeared concurrently', 'CONCURRENT_UPDATE_NO_WRITE')
        snapshot.mkdir();manifest = []
        for index, (path, raw) in enumerate(prior.items()):
            member = snapshot/f'{index:02d}-{path.name}.gz';member.write_bytes(gzip.compress(raw, mtime=0))
            stored = member.read_bytes();core.need(gzip.decompress(stored) == raw, 'Prior gzip must restore exact original bytes')
            intended = outputs.get(path, raw)
            manifest.append(dict(source=str(path), original_bytes=len(raw), original_sha256=core.digest(raw),
                intended_bytes=len(intended), intended_sha256=core.digest(intended),
                stored_gzip_path=str(member), stored_bytes=len(stored), stored_sha256=core.digest(stored),
                gzip_mtime=0, exact_original_bytes_readback=True))
        script_sha = core.digest(Path(__file__).read_bytes())
        receipt = dict(schema='current_metadata_scope_reconciliation_v1', created_at_utc=now.isoformat(),
            completed_outer=n, active_outer=n+1, source_commit=core.SOURCE, config_sha256=core.CONFIG,
            unit=core.UNIT, run=core.RUN, source_script_sha256=script_sha,
            pinned_alias_sha256=ALIAS_SHA, pinned_core_sha256=alias.CORE_SHA,
            pinned_finalizer_sha256=alias.FINALIZER_SHA, input_pins=inputs.pins, full=full_pin,
            proof=proof_pin, alias_reconciliation=alias_pin, second_poll=poll_pin, second_resource=resource_pin,
            native=native_pin, observed_epoch=poll['epoch'], before=before, after=after,
            backup=dict(path=str(manifest_path), files_sha256=core.digest(core.json_bytes(manifest)),
                        closed_original_documents=14),
            updated_fields=list(FIELDS), current_metric_step=n, latest_completed_sampling=n,
            current_active_optimizer_unknown=True, two_mirrors_byte_equal=True,
            actual_cold_load_outer=5, source_training_mutation=False, counters_cost_score_curve_reports_modified=False,
            global_atomic_transaction=False, new_eval_review_test=False, remote_actions=False,
            goal_status='active', goal_complete=False)
        receipt_raw = core.json_bytes(receipt)
        for path in receipt_paths:outputs[path] = receipt_raw
        manifest_raw = core.json_bytes(dict(schema='metadata_scope_recovery_manifest_v1',
            created_at_utc=now.isoformat(), outer=n, source_commit=core.SOURCE, config_sha256=core.CONFIG,
            unit=core.UNIT, run=core.RUN, source_script_sha256=script_sha, full=full_pin, files=manifest,
            receipt_files=[dict(path=str(p), bytes=len(receipt_raw), sha256=core.digest(receipt_raw))
                           for p in receipt_paths], global_atomic_transaction=False))
        manifest_path.write_bytes(manifest_raw)
        core.need(manifest_path.read_bytes() == manifest_raw, 'Backup manifest readback differs')
        try:
            inputs.unchanged()
            for path, raw in outputs.items():
                if path in prior:
                    core.need(path.read_bytes() == prior[path], 'Target changed before replacement: '+str(path))
                else:
                    core.need(not path.exists(), 'Receipt appeared before replacement: '+str(path))
                core.atomic_replace(path, raw)
            core.need(all(p.read_bytes() == raw for p, raw in outputs.items()), 'Metadata output readback differs')
            core.need(all(p.read_bytes() == prior[p] for p in reports)
                      and all((m/'REPORT.md').read_bytes() == prior[m/'REPORT.md'] for m in mirrors),
                      'REPORT bytes must remain unchanged')
        except Exception as write_error:
            try:
                rollback = recover_snapshot(core, workspace, snapshot)
            except Exception as rollback_error:
                rollback = dict(status='ROLLBACK_REFUSED_OR_INCOMPLETE_MANUAL_RECOVERY_REQUIRED',
                                error=str(rollback_error), snapshot=str(snapshot))
            print(json.dumps(dict(status='PUBLICATION_WRITE_FAILED', error=str(write_error),
                                  rollback=rollback, backup=str(manifest_path))))
            raise
    print(json.dumps(dict(status='LOCAL_FIVE_METADATA_FIELDS_AND_TWO_MIRRORS_PUBLISHED',
        receipt=dict(path=str(receipt_path), bytes=len(receipt_raw), sha256=core.digest(receipt_raw)),
        backup=dict(path=str(manifest_path), bytes=len(manifest_raw), sha256=core.digest(manifest_raw)),
        modified_files=len(outputs), global_atomic_transaction=False, remote_actions=False)))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status='METADATA_SCOPE_PUBLICATION_FAILED', error=str(exc), remote_actions=False)))
        raise SystemExit(1)
