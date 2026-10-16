#!/usr/bin/env python3
"""Local source preparation: retire two completed180 pending live pointers.

Future invocation only after this exact source is normally published:
 python3 -B work/tree-speed-repair-20261008/dev180_completed_pending_live_pointer_repair.py \
   --input-pins <actual DEV180_COMPLETED_PENDING_POINTER_REPAIR_INPUT_PINS.json> \
   --input-pins-bytes <actual bytes> --input-pins-sha256 <actual SHA256>

Requires fixed actual FULL180/closure180/pending180/old-helper evidence and all
14 exact current prestate files. Only two top-level live values become null;
one explicit historical pending reference is inserted. All remaining original
JSON text is retained, and REPORT bytes stay exact. No original training code,
collector, evaluation, remote resource, counter, score or goal is modified.

The original helper rejects any existing receipt. Its three old current receipt
slots are vacated only after their exact raw bytes have been archived to three
new clearly historical files and read back twice. This preserves the original
as-of receipt, not a rewritten immutable artifact. This entry does not run the
helper. Root must run the unchanged original helper LAST afterward, generate a
new actual ROOTFULL manifest, and retain the old failed manifest as history.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
SOURCE = W / 'work/tree-speed-repair-20261008/dev180_completed_pending_live_pointer_repair.py'
CONTROL = W / 'work/tree-speed-repair-20261008/nested-native5-controls-source/reproduction/tree-eval-resume-20261013/local-control-backup'
HELPER = CONTROL / 'publish_nested_metadata_scope.py'
CORE = CONTROL / 'sync_nested_native_progress.py'
FIELDS = ('current_live_validation_progress', 'current_actual_optimizer_progress')
HISTORY = 'historical_outer180_pending_evidence_completed_by_FULL180'
CANONICAL = [W / 'outputs/STATUS.json', W / 'outputs/tree-nested-branch/STATUS.json',
             B / 'CURRENT_NATIVE_RECOVERY_FOLLOWUP.json', B / 'CURRENT_OBSERVER_CONNECTION_STATE.json']
JSONS = CANONICAL + [m / p.name for m in M for p in CANONICAL[1:]]
REPORTS = [W / 'outputs/REPORT.md', W / 'outputs/tree-nested-branch/REPORT.md', *[m / 'REPORT.md' for m in M]]
ALL = JSONS + REPORTS
CURRENT_HELPER = [B / 'CURRENT_METADATA_SCOPE_RECONCILIATION_OUTER180.json',
                  *[m / 'CURRENT_METADATA_SCOPE_RECONCILIATION_OUTER180.json' for m in M]]
HISTORICAL_HELPER_NAME = 'HISTORICAL_METADATA_SCOPE_RECONCILIATION_OUTER180_BEFORE_COMPLETED_PENDING_LIVE_POINTER_REPAIR.json'
HISTORICAL_HELPER = [B / HISTORICAL_HELPER_NAME, *[m / HISTORICAL_HELPER_NAME for m in M]]
OUTPUT_NAME = 'OUTER180_COMPLETED_PENDING_LIVE_POINTER_REPAIR_RECEIPT.json'
OUTPUTS = [B / OUTPUT_NAME, *[m / OUTPUT_NAME for m in M]]
FIXED = {
    'full': ('FULL_OUTER180_PROGRESS_RECEIPT.json', 32033, '731f90f8354a6ce1fb6e528ee56bb67e9611b9bbaf99b28fc20708a219dc8c47'),
    'canonical_closure': ('OUTER180_CURVE_DEVCOST_CURRENT_ALIAS_CLOSURE.json', 83094, '392f43cb3cb8c762c3a9e32f9867e77895afc074626b78af0959fbff545c8fe5'),
    'completed_pending_evidence': ('OUTER180_UPDATES_NATIVE_DEV_PENDING_ACTUAL_2288_2289.json', 16737, 'edeadcc69373ee85d97eae510e6ff8cae77cf7240190e355a17bd84636b344a4'),
    'old_helper_receipt': ('CURRENT_METADATA_SCOPE_RECONCILIATION_OUTER180.json', 15491, '557e285f881beda6ae262f713d52bebc63633ace97b3b2703278c5271b92d1d0'),
    'failed_root_full_manifest': ('OUTER180_ROOT_FULL_ACTUAL_INPUT_PINS_2293_2299.json', 9148, 'c00f0db39dbdd05ed3e35f38d1dcf1de628a6f966df4dfe6f35f58fd3b236bf9'),
    'canonical_input_manifest': ('DEV180_CANONICAL_ACTUAL_INPUT_PINS_2293_2299.json', 11874, 'a9e3c908d577904c549a634c25a0f16cf82d694572e4499ddc19f10e0ac4f227'),
}
HELD = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def locate(name):
    assert isinstance(name, str) and name and '://' not in name
    p = Path(name)
    assert p.parts and '..' not in p.parts
    p = p if p.is_absolute() else W / p if p.parts[0] in ('outputs', 'work') else B / p
    p = p.resolve(strict=True)
    assert p.is_relative_to(W) or any(p.is_relative_to(m) for m in M)
    assert p.is_file() and p.suffix in ('.json', '.md', '.py', '.txt', '.csv', '.gz')
    return p


def rawpin(p, raw=None):
    raw = p.read_bytes() if raw is None else raw
    return dict(path=str(p), bytes=len(raw), sha256=digest(raw))


def exact(ref, expected, mirror=False):
    assert isinstance(ref, dict) and type(ref.get('bytes')) is int and ref['bytes'] > 0
    assert re.fullmatch('[0-9a-f]{64}', ref.get('sha256', ''))
    p = locate(ref.get('path', ref.get('file')))
    assert p == expected.resolve(strict=True)
    raw = p.read_bytes()
    assert len(raw) == ref['bytes'] and digest(raw) == ref['sha256'], 'Actual bound bytes changed: ' + str(p)
    assert p not in HELD or HELD[p] == raw
    HELD[p] = raw
    if mirror:
        assert p.is_relative_to(B)
        for m in M:
            q = m / p.relative_to(B)
            observed = q.read_bytes()
            assert observed == raw, 'Actual immutable mirror differs: ' + str(q)
            HELD[q] = observed
    return raw


def unchanged(skip=()):
    excluded = set(skip)
    for p, raw in HELD.items():
        if p not in excluded:
            assert p.read_bytes() == raw, 'Held local input changed: ' + str(p)


def patch_top_values(raw, pending_ref):
    """Patch only two original literal spans and insert one historical field."""
    text = raw.decode('utf8')
    decoder = json.JSONDecoder(object_pairs_hook=unique)
    pos = 0
    while pos < len(text) and text[pos].isspace(): pos += 1
    assert text[pos] == '{'
    pos += 1
    spans = {}
    last_end = None
    while True:
        while text[pos].isspace(): pos += 1
        if text[pos] == '}': break
        key, end = decoder.raw_decode(text, pos)
        assert isinstance(key, str) and key not in spans
        pos = end
        while text[pos].isspace(): pos += 1
        assert text[pos] == ':'
        pos += 1
        while text[pos].isspace(): pos += 1
        start = pos
        value, end = decoder.raw_decode(text, pos)
        spans[key] = (start, end, value)
        last_end = end
        pos = end
        while text[pos].isspace(): pos += 1
        if text[pos] == ',':
            pos += 1
        else:
            assert text[pos] == '}'
            break
    assert not text[pos + 1:].strip() and last_end is not None
    assert HISTORY not in spans, 'Historical field already exists; no repeat repair'
    changes = []
    for key in FIELDS:
        start, end, value = spans[key]
        assert value == pending_ref, 'Current live field is not the fixed actual pending180 evidence'
        changes.append((start, end, 'null'))
    historical = json.dumps(pending_ref, ensure_ascii=False, indent=2, allow_nan=False)
    historical = historical.replace('\n', '\n  ')
    insertion = ',\n  ' + json.dumps(HISTORY) + ': ' + historical
    changes.append((last_end, last_end, insertion))
    revised = text
    for start, end, replacement in sorted(changes, reverse=True):
        revised = revised[:start] + replacement + revised[end:]
    old = decode(raw)
    result = revised.encode('utf8')
    new = decode(result)
    assert all(new[key] is None for key in FIELDS) and new[HISTORY] == pending_ref
    assert {k: v for k, v in old.items() if k not in FIELDS} == {k: v for k, v in new.items() if k not in (*FIELDS, HISTORY)}
    assert set(new) == set(old) | {HISTORY}
    return result, [dict(start=start, end=end, replacement_bytes=len(replacement.encode())) for start, end, replacement in changes]


def atomic_replace(p, raw):
    temp = p.with_name('.' + p.name + '.completed180-repair-' + str(os.getpid()))
    assert not temp.exists()
    try:
        with temp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, p)
    finally:
        if temp.exists(): temp.unlink()


def create(p, raw):
    with p.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    assert p.read_bytes() == raw


def main():
    assert __debug__, 'Required actual guards cannot be disabled with -O'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-pins', required=True)
    parser.add_argument('--input-pins-bytes', type=int, required=True)
    parser.add_argument('--input-pins-sha256', required=True)
    args = parser.parse_args()
    mp = locate(args.input_pins)
    mraw = exact(dict(path=str(mp), bytes=args.input_pins_bytes, sha256=args.input_pins_sha256), mp)
    manifest = decode(mraw)
    assert manifest['schema'] == 'dev180_completed_pending_live_pointer_repair_input_pins_v1'
    assert manifest['outer'] == 180 and Path(manifest['workspace_root']).resolve() == W
    assert set(manifest['references']) == set(FIXED)
    assert set(manifest['sources']) == {'repair', 'original_helper', 'original_core', 'unchanged_root_full_template'}
    exact(manifest['sources']['repair'], SOURCE)
    exact(manifest['sources']['original_helper'], HELPER)
    exact(manifest['sources']['original_core'], CORE)
    assert (len(HELD[HELPER]), digest(HELD[HELPER])) == (24931, 'fc2f3f70b562eebbbd6c6607aa75846b4a59da5c91dca1271de1bfb92851f29c')
    assert digest(HELD[CORE]) == '676dca88489fb4d3c84891603990fc96256a14fb7da8ac2bc5c2def053e016c1'
    root_template = W / 'work/tree-speed-repair-20261008/tree_through180_full_root_readback.template.command.txt'
    exact(manifest['sources']['unchanged_root_full_template'], root_template)
    assert digest(HELD[root_template]) == '23348b3e04550fe5a832b6f55cd68b1343746ebaa00a5aca0278cf192c014e80'
    docs = {}
    for key, (name, size, sha) in FIXED.items():
        ref = manifest['references'][key]
        assert ref['bytes'] == size and ref['sha256'] == sha
        docs[key] = decode(exact(ref, B / name, mirror=key not in ('failed_root_full_manifest', 'canonical_input_manifest')))
    full, closure, pending, helper = [docs[k] for k in ('full', 'canonical_closure', 'completed_pending_evidence', 'old_helper_receipt')]
    pending_ref = rawpin(B / FIXED['completed_pending_evidence'][0])
    assert full['status'] == 'ACTUAL_FULL_OUTER180_NATIVE180_COMPLETE_NEXT181_ENTERED'
    assert full['completed_outer'] == 180 and full['actual_next_outer_entered'] == 181 and full['full_iteration_metric_pending'] is False
    assert full['latest_dev512_outer'] == full['actual_dev512']['outer'] == 180 and full['actual_dev512']['rows'] == 512
    assert full['actual_training_pid'] == 503255 and full['actual_training_invocation_id'] == 'a313bda919834d24b4c5d491698d39ce'
    assert closure['completed_outer'] == closure['latest_dev_outer'] == closure['curve_actual_through'] == 180 and closure['active_outer'] == 181
    assert pending['actual_optimizer_completed_outer'] == pending['latest_saved_native_outer'] == 180 and pending['outer181_entered_claimed'] is False
    assert pending['status'] == 'ACTUAL_NATIVE180_DEV180_RESULT_PENDING' and pending['dev180_result'] is None
    assert full['source_commit'] == pending['source_commit'] == closure['actual_source_commit'] == helper['source_commit']
    assert full['config_sha256'] == pending['config_sha256'] == closure['actual_config_sha256'] == helper['config_sha256']
    assert helper['completed_outer'] == 180 and helper['active_outer'] == 181 and helper['full'] == manifest['references']['full']
    # Existing helper as-of backup and all original gzip members remain immutable.
    backup_path = locate(helper['backup']['path'])
    backup_raw = backup_path.read_bytes();HELD[backup_path] = backup_raw
    backup = decode(backup_raw)
    assert len(backup['files']) == helper['backup']['closed_original_documents'] == 14
    assert digest(encode(backup['files'])) == helper['backup']['files_sha256']
    for row in backup['files']:
        path = locate(row['stored_gzip_path'])
        stored = path.read_bytes();HELD[path] = stored
        assert len(stored) == row['stored_bytes'] and digest(stored) == row['stored_sha256'] and int.from_bytes(stored[4:8], 'little') == 0
        original = gzip.decompress(stored)
        assert len(original) == row['original_bytes'] and digest(original) == row['original_sha256']
    prestate = manifest['canonical_prestate']
    assert len(ALL) == len(set(ALL)) == len(prestate) == 14
    assert {locate(r['path']) for r in prestate} == set(ALL)
    prior = {locate(ref['path']): exact(ref, locate(ref['path'])) for ref in prestate}
    outputs, edits = {}, {}
    for p in JSONS:
        state = decode(prior[p])
        assert state['completed_outer'] == 180 and state['active_outer'] == 181 and state['latest_dev512_outer'] == 180
        assert state['goal_status'] == 'active' and state['consecutive_blocked_goal_turns'] == 0
        assert state['per_rank_actual_total_adam'] == full['per_rank_actual_total_adam']
        assert state['current_dev512'] == full['actual_dev512'] and state['current_training_variances'] == full['current_training_variances']
        assert state['actual_current_restore_outer'] == 5 and state['actual_current_restore_adam_per_rank'] == {str(i): 100 for i in range(8)}
        outputs[p], edits[p] = patch_top_values(prior[p], pending_ref)
    for p in REPORTS: outputs[p] = prior[p]
    assert prior[CANONICAL[0]] == prior[CANONICAL[1]] and outputs[CANONICAL[0]] == outputs[CANONICAL[1]]
    for m in M:
        for p in CANONICAL[1:]:
            assert prior[m / p.name] == prior[p] and outputs[m / p.name] == outputs[p]
        assert prior[m / 'REPORT.md'] == prior[REPORTS[0]] == prior[REPORTS[1]]
    assert not any(p.exists() for p in [*HISTORICAL_HELPER, *OUTPUTS]), 'Existing repair/history output must not be overwritten'
    assert not any(B.glob('ROOT_FULL180_PUBLICATION_READBACK_*.json')), 'ROOTFULL180 already exists; retain it and reject repair'
    old_helper_raw = HELD[CURRENT_HELPER[0]]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot = B / ('outer180_completed_pending_live_pointer_repair_prior_' + stamp)
    removed = []
    repaired = []
    receipt_raw = None
    with CORE.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unchanged()
        assert not any(p.exists() for p in [*HISTORICAL_HELPER, *OUTPUTS])
        snapshot.mkdir()
        rows = []
        for i, p in enumerate(ALL):
            member = snapshot / (f'{i:02d}-' + p.name + '.gz')
            stored = gzip.compress(prior[p], mtime=0)
            create(member, stored)
            assert gzip.decompress(member.read_bytes()) == prior[p]
            rows.append(dict(source=str(p), original_bytes=len(prior[p]), original_sha256=digest(prior[p]),
                intended_bytes=len(outputs[p]), intended_sha256=digest(outputs[p]),
                stored_gzip_path=str(member), stored_bytes=len(stored), stored_sha256=digest(stored), gzip_mtime=0))
        snapshot_manifest = snapshot / 'SNAPSHOT_MANIFEST.json'
        recovery = dict(schema='outer180_completed_pending_live_pointer_repair_snapshot_v1', outer=180,
            input_manifest=rawpin(mp), repair_source=rawpin(SOURCE), files=rows,
            original_helper_receipt=manifest['references']['old_helper_receipt'],
            original_helper_current_slots=[str(p) for p in CURRENT_HELPER],
            historical_helper_archive_paths=[str(p) for p in HISTORICAL_HELPER], global_atomic_transaction=False)
        create(snapshot_manifest, encode(recovery))
        for p in HISTORICAL_HELPER: create(p, old_helper_raw)
        archived_observations = []
        for number in (1, 2):
            for p in HISTORICAL_HELPER: assert p.read_bytes() == old_helper_raw
            for p in CURRENT_HELPER: assert p.read_bytes() == old_helper_raw
            archived_observations.append(dict(pass_number=number, observed_utc=datetime.now(timezone.utc).isoformat(),
                                             original_current_copies=3, exact_historical_archive_copies=3))
            if number == 1: time.sleep(1)
        try:
            unchanged()
            for p in JSONS:
                assert p.read_bytes() == prior[p]
                atomic_replace(p, outputs[p]);repaired.append(p)
            for p in ALL: assert p.read_bytes() == outputs[p]
            # Vacate only the three exact old as-of receipt slots after archive proof.
            for p in CURRENT_HELPER:
                assert p.read_bytes() == old_helper_raw
                p.unlink();removed.append(p)
            assert not any(p.exists() for p in CURRENT_HELPER)
            after_observations = []
            for number in (1, 2):
                unchanged(skip=[*ALL, *CURRENT_HELPER])
                for p in ALL: assert p.read_bytes() == outputs[p]
                for p in HISTORICAL_HELPER: assert p.read_bytes() == old_helper_raw
                assert not any(p.exists() for p in CURRENT_HELPER)
                after_observations.append(dict(pass_number=number, observed_utc=datetime.now(timezone.utc).isoformat(),
                    canonical_and_report_files=14, changed_JSON_files=10, reports_unchanged=4, mirror_roots=2,
                    archived_original_helper_copies=3, old_helper_current_slots_vacated=3))
                if number == 1: time.sleep(1)
            receipt = dict(schema='outer180_completed_pending_live_pointer_repair_receipt_v1',
                status='LOCAL_COMPLETED180_PENDING_LIVE_POINTERS_REPAIRED_HELPER_LAST_REQUIRED',
                created_at_utc=datetime.now(timezone.utc).isoformat(), completed_outer=180, active_outer=181,
                actual_input_manifest=rawpin(mp), repair_source=rawpin(SOURCE), fixed_actual_references=manifest['references'],
                changed_live_fields=list(FIELDS), new_historical_field=HISTORY, historical_completed180_pending_evidence=pending_ref,
                all_other_original_JSON_text_outside_value_spans_and_history_insertion_preserved=True,
                all_other_counter_score_cost_curve_cold_restore_phase_goal_objects_unchanged=True,
                original14_gzip_snapshot_manifest=rawpin(snapshot_manifest), original14_exact_gzip_rows=rows,
                pointer_literal_edits=[dict(path=str(p), edits=edits[p]) for p in JSONS],
                before=[rawpin(p, prior[p]) for p in ALL], after=[rawpin(p, outputs[p]) for p in ALL],
                original_helper_asof_receipt_bytes_preserved_in_history=[rawpin(p, old_helper_raw) for p in HISTORICAL_HELPER],
                original_helper_asof_snapshot_manifest=rawpin(backup_path), original_helper_current_slots_vacated=[str(p) for p in CURRENT_HELPER],
                original_helper_archive_two_readbacks=archived_observations, actual_after_two_readbacks=after_observations,
                unchanged_original_helper_source=rawpin(HELPER), original_helper_rerun_performed_by_this_entry=False,
                original_helper_LAST_still_required=True, fresh_ROOTFULL_input_manifest_and_readback_still_required=True,
                failed_ROOTFULL_manifest_retained=True, canonical180_actual_manifest_and_closure_retained=True,
                ROOTFULL_guards_modified=False, global_atomic_transaction=False, collector_invocations=0, SSH_invocations=0,
                remote_actions=False, training_or_evaluation_modified=False, tensor_contents_read=False,
                test_compile_review_performed=False, goal_status='active', goal_complete=False)
            receipt_raw = encode(receipt)
            assert len(receipt_raw) < 256000
            for p in OUTPUTS: create(p, receipt_raw)
            for number in (1, 2):
                for p in OUTPUTS: assert p.read_bytes() == receipt_raw
                for p in ALL: assert p.read_bytes() == outputs[p]
                for p in HISTORICAL_HELPER: assert p.read_bytes() == old_helper_raw
                assert not any(p.exists() for p in CURRENT_HELPER)
                if number == 1: time.sleep(1)
        except Exception:
            # Preserve snapshot/history even on failure. Roll back only held bytes.
            for p in reversed(repaired):
                now = p.read_bytes()
                assert now in (prior[p], outputs[p]), 'Concurrent update; manual recovery required: ' + str(p)
                if now != prior[p]: atomic_replace(p, prior[p])
            for p in removed:
                assert not p.exists(), 'Receipt reappeared; manual recovery required: ' + str(p)
                create(p, old_helper_raw)
            if receipt_raw is not None:
                for p in OUTPUTS:
                    if p.exists():
                        assert p.read_bytes() == receipt_raw
                        p.unlink()
            raise
    print(json.dumps(dict(status=receipt['status'], receipt=rawpin(OUTPUTS[0]), copies=3, receipt_readback_passes=2,
                          historical_helper=rawpin(HISTORICAL_HELPER[0]), helper_current_slots_absent=True,
                          helper_LAST_still_required=True, snapshot=rawpin(snapshot_manifest), remote_actions=False)))


if __name__ == '__main__':
    main()
