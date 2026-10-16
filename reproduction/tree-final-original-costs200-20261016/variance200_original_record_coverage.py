#!/usr/bin/env python3
"""Future immutable original-scalar coverage through effective outer200.

SOURCE PREPARATION ONLY. Do not run until the actual input manifest is frozen.
This is metadata extraction, not training, evaluation, review, a test, or SSH.
The original778 loop completes periodic dev512, saves200, and logs its training
metrics before counter201 switches to final51713. A rolling native200 save alone
does not prove those original metric records exist. FULL200 is deliberately not
required: its early final-phase snapshot may precede the original metric line.

Future invocation, with the manifest's actual local bytes/SHA:
  python3 -B work/tree-speed-repair-20261008/variance200_original_record_coverage.py \
    --input-manifest "$ACTUAL_MANIFEST" --input-manifest-bytes "$ACTUAL_BYTES" \
    --input-manifest-sha256 "$ACTUAL_SHA256"

Manifest schema: tree_nested_variance200_original_record_input_manifest_v1; status
ACTUAL_OUTER200_ORIGINAL_SCALAR_INPUTS_FROZEN; fixed experiment identity fields
source_commit/config_sha256/model_identity/data_identity/unit/run,
actual_training_pid/actual_training_invocation_id; prior_coverage descriptor;
scalar_records exactly [{outer:172..200, source:{path,bytes,sha256}}], where each
source_kind is ORIGINAL_POLL or RETROSPECTIVE_ORIGINAL_FILE. The latter must be an
explicit final_original_costs200 scalar artifact binding the raw original stage line,
line SHA/offset and actual collection time; it is never a historical live poll;
actual200 contains SHA/byte descriptors native_metadata, progress_poll, resource,
dev200_periodic_control, and original_control_source. The last descriptor is the
remote original control's path/resolved_path/bytes/sha256/stable_read observation,
not a remote path followed by this program. All other descriptors name existing
ordinary local JSON metadata inside BASE. Future pins/metrics are never preset.

All15 required series retain direct original recorded values. Additional raw
question-group reward variance and denominator/alias fields are also recorded.
No scalar is reconstructed from audit hashes, a mean, fabricated zero, final
outputs, or a later step. A derived scalar would need separate explicit origin;
this entry accepts no derived replacement. Dev512 and final51713 are disjoint.
Native cursor201 is a saved boundary and is never actual training201 entry.
Only the two named new small products are written, to BASE and both mirrors;
all original evidence remains byte-identical through two real readback rounds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time


W = Path(__file__).resolve().parents[2]
B = W / 'outputs/tree-nested-branch-handoff-20261013'
M = [W / 'work/tree-grpo/reproduction/baselines/tree-validation-20261002/tree-nested-branch',
     Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/tree-nested-branch')]
SCRIPT = W / 'work/tree-speed-repair-20261008'
FROZEN = SCRIPT / 'nested-native5-recovery-source'
C = '778b2b43dad90fc0c1c5d95d77c2326286011b73'
CONFIG = '355edf5232dc963deb526281f7e1e61b98eaaa73b0bffc5e7855ef3d262799b3'
MODEL = '744d9f616373b3890cf815412560e34041e21cd4915c026967d8cc13a423aea0'
DATA = '0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
RUN = '/dataset/shaoshitong/search-r1-rl/tree-grpo/runs/tree-nested-branch-fresh200-not-slurm;systemd:' + C
UNIT = 'sst-tree-nested-branch-fresh200-778b2b43.service'
INV = 'a313bda919834d24b4c5d491698d39ce'
PRIOR_NAME = 'METRIC_RECORD_COVERAGE_THROUGH171_LOCAL.json'
PRIOR_BYTES = 152283
PRIOR_SHA = '7b9eea902588edb97c4d76887e666697c0099bef217148bf5c6cf73f71fc4b90'
PRIOR_SOURCE = SCRIPT / 'variance171_record_coverage.py'
NAME = 'METRIC_RECORD_COVERAGE_THROUGH200_ORIGINAL_RECORDS'
OUTPUT = NAME + '.json'
READBACK = NAME + '_READBACKS.json'
MAX_BYTES = 4 * 1024 * 1024
MAX_PRODUCT_BYTES = 512 * 1024
FIELDS = {
    'training_reward_mean': 'critic/rewards/mean',
    'training_score_mean': 'critic/score/mean',
    'raw_within_tree_reward_population_variance': 'tree_variance/within_root_raw_reward_population_variance',
    'pure_intra_adv_population_variance': 'tree_variance/intra_adv_population_variance',
    'total_actual_adv_population_variance': 'tree_variance/group_actual_adv_population_variance',
    'five_leaf_same_reward_question_fraction': 'tree_variance/same_reward_question_fraction',
    'five_leaf_mixed_reward_question_fraction': 'tree_variance/mixed_reward_question_fraction',
    'all_reward_one_question_count': 'tree_variance/exact_reward_one_question_count',
    'all_reward_zero_question_count': 'tree_variance/exact_reward_zero_question_count',
    'root0_same_reward_fraction': 'tree_variance/root0_same_reward_fraction',
    'root0_mixed_reward_fraction': 'tree_variance/root0_mixed_reward_fraction',
    'root1_same_reward_fraction': 'tree_variance/root1_same_reward_fraction',
    'root1_mixed_reward_fraction': 'tree_variance/root1_mixed_reward_fraction',
    'all_roots_same_reward_tree_fraction': 'tree_variance/same_reward_tree_fraction',
    'all_roots_mixed_reward_tree_fraction': 'tree_variance/mixed_reward_tree_fraction',
}
SERIES = ['question_count', 'tree_count', 'mixed_question_count',
          'exact_reward_one_tree_count', 'exact_reward_zero_tree_count', 'mixed_tree_count',
          'root0_exact_reward_one_count', 'root0_exact_reward_zero_count', 'root0_mixed_count',
          'root1_exact_reward_one_count', 'root1_exact_reward_zero_count', 'root1_mixed_count']
EXTRA = ['tree_variance/group_raw_reward_population_variance',
         'tree_variance/within_root_normalized_component_population_variance',
         'tree_variance/empty_loss_row_count']
SOURCE_PINS = {
    'search_r1/baselines/advantages.py': (10150, '470b84aba5dcb675eb5cadb04e755e4cb6c8a43d916b1bb66d7846cac1a03509'),
    'search_r1/baselines/controller.py': (30282, 'f78d55d98450614a1a8dac99535277f5cf8a3153f980d8c6f047fea5173b42d1'),
    'search_r1/baselines/tree_audit.py': (15243, '50e7fc4860661b4bd66b7c7acade6c8ec057c499dd6b036ffe0ae240be7fd93d'),
    'verl/trainer/ppo/ray_trainer.py': (67989, '92227516ae5e53171afacbd2f64f4201add03413f4cc6f1d5fa9851dfd2953a0'),
}
FILES, RAW = {}, {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strict(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    def constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def local(name):
    p = Path(name)
    assert '..' not in p.parts, str(p)
    if not p.is_absolute():
        p = W / p if p.parts and p.parts[0] in ('outputs', 'work') else B / p
    assert not p.is_symlink(), str(p)
    p = p.resolve(strict=True)
    assert p.is_relative_to(B) and p.suffix == '.json', str(p)
    return p


def read_bytes(p, size=None, expected=None, mirrors=False):
    assert p.is_file() and not p.is_symlink() and stat.S_ISREG(p.stat().st_mode), str(p)
    with p.open('rb') as stream:
        before = stream.fileno()
        start = os.fstat(before)
        assert 0 < start.st_size <= MAX_BYTES, str(p)
        raw = stream.read()
        end = os.fstat(before)
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    assert signature(start) == signature(end) == signature(p.stat()), 'changed during read: ' + str(p)
    assert size is None or type(size) is int and len(raw) == size, 'bytes: ' + str(p)
    assert expected is None or digest(expected) and sha(raw) == expected, 'SHA: ' + str(p)
    ref = dict(path=str(p.relative_to(W)), bytes=len(raw), sha256=sha(raw), mirror_checked=mirrors)
    if str(p) in FILES:
        assert RAW[str(p)] == raw and FILES[str(p)] == ref, 'changed repeat read: ' + str(p)
    if mirrors:
        for mirror in M:
            assert (mirror / p.relative_to(B)).read_bytes() == raw, 'input mirror: ' + str(p)
    FILES[str(p)], RAW[str(p)] = ref, raw
    return raw


def read(name, size=None, expected=None, mirrors=False):
    p = local(name)
    value = strict(read_bytes(p, size, expected, mirrors))
    assert isinstance(value, dict), str(p)
    return value


def bound(ref, mirrors=False):
    assert isinstance(ref, dict) and type(ref.get('bytes')) is int and ref['bytes'] > 0 and digest(ref.get('sha256'))
    name = ref.get('path') or ref.get('file')
    assert isinstance(name, str) and name
    return read(name, ref['bytes'], ref['sha256'], mirrors)


def reference(name):
    p = local(name)
    assert str(p) in FILES
    return {k: v for k, v in FILES[str(p)].items() if k != 'mirror_checked'}


def metric(source, outer):
    records = source.get('iteration_metrics')
    records = [records] if isinstance(records, dict) else records
    assert isinstance(records, list)
    found = [r for r in records if isinstance(r, dict) and r.get('step') == outer]
    assert found and all(r == found[0] for r in found), 'missing/conflicting original step: ' + str(outer)
    return found[0]


def original_record_source(row):
    """Validate true original polls or explicitly labelled original-file extraction."""
    n = row['outer']; source = bound(row['source']); poll_identity(source)
    kind = row.get('source_kind', 'ORIGINAL_POLL')
    if kind == 'ORIGINAL_POLL':
        assert local(row['source']['path']).name.startswith('NATIVE5_CONTINUE_POLL_')
        assert source.get('schema') != 'tree_final200_original_scalar_source_v1'
        return source, metric(source, n), dict(source_kind=kind, original_poll=reference(row['source']['path']),
                                             retrospective_original_source=None)
    assert kind == 'RETROSPECTIVE_ORIGINAL_FILE'
    assert source['schema'] == 'tree_final200_original_scalar_source_v1'
    assert source['evidence_mode'] == 'RETROSPECTIVE_SOURCE_READBACK' and source['outer'] == n
    assert number(source['observed_epoch']) and source['historical_observation_epoch'] is None
    assert source['historical_live_poll_created'] is False
    lines = source['metric_line_sources']; assert len(lines) == 1 and lines[0]['outer'] == n
    line = lines[0]; raw = line['raw_line_utf8'].encode('utf-8')
    assert line['path'] == RUN + '/stage.log' and line['line_sha256'] == sha(raw)
    assert type(line['byte_start']) is int and line['byte_start'] >= 0
    assert type(line['byte_end']) is int and line['byte_end'] - line['byte_start'] == len(raw)
    assert type(line['line_number']) is int and line['line_number'] > 0 and raw.endswith(b'\n')
    prefix = line['prefix']; assert prefix['path'] == line['path'] and prefix['stable_read'] is True
    assert type(prefix['bytes']) is int and prefix['bytes'] >= line['byte_end'] and digest(prefix['sha256'])
    assert prefix['read_scope'] == 'stable exact prefix read twice'
    parsed = {}
    for key, value in re.findall(r'([a-zA-Z0-9_./-]+):(-?[0-9]+(?:[.][0-9]+)?(?:[eE][+-]?[0-9]+)?)(?=\s|$)', line['raw_line_utf8']):
        value = float(value); assert number(value) and (key not in parsed or parsed[key] == value)
        parsed[key] = value
    m = metric(source, n); assert parsed == m and m['step'] == n
    capture = bound(source['capture_product'], True)
    assert capture['schema'] == 'tree_final200_original_remote_readback_v1' and capture['status'] == 'ORIGINAL200_RECORDS_READ'
    assert capture['remote_calls'] == 1 and capture['remote_exit_code'] == 0 and capture['remote_files_written'] is False
    assert capture['input_manifest'] == source['collection_manifest'] and capture['observed_epoch'] == source['observed_epoch']
    manifest = bound(source['collection_manifest'])
    assert manifest['schema'] == 'tree_final200_original_cost_input_manifest_v1'
    def source_path(ref):
        p = Path(ref['path']); assert '..' not in p.parts
        p = (p if p.is_absolute() else W / p).resolve(strict=True)
        assert p.is_relative_to(W)
        return p
    assert source_path(manifest['source_command']) == source_path(capture['generator'])
    assert all(manifest['source_command'][k] == capture['generator'][k] for k in ('bytes', 'sha256'))
    assert manifest['remote_source'] == capture['remote_generator']
    for producer in (manifest['source_command'], manifest['remote_source']):
        read_bytes(source_path(producer), producer['bytes'], producer['sha256'])
    rows = capture['rows']
    if isinstance(rows, dict):
        assert rows['storage'] == 'complete_original_field_parts' and rows['container_type'] == 'list'
        expanded = []
        for ref in rows['parts']:
            part = bound(ref, True)
            assert part['schema'] == 'cumulative_scalar_field_part_v1' and part['original_field'] == 'rows'
            assert isinstance(part['value'], list); expanded.extend(part['value'])
        assert len(expanded) == rows['original_entries']; rows = expanded
    assert [r['outer'] for r in rows] == list(range(1, 201))
    selected = rows[n - 1]
    assert selected['source_commit'] == C and selected['run'] == RUN
    assert selected['metric'] == m and selected['metric_line'] == {k:v for k,v in line.items() if k != 'outer'}
    return source, m, dict(source_kind=kind, original_poll=None,
                          retrospective_original_source=reference(row['source']['path']),
                          actual_original_file_line={k:v for k,v in line.items() if k != 'raw_line_utf8'},
                          raw_line_utf8_preserved_in_bound_original_source=True,
                          original_capture=reference(source['capture_product']['path']),
                          retrospective_observed_epoch=source['observed_epoch'], historical_live_observation_claimed=False)


def poll_identity(source):
    for key, value in dict(source_commit=C, unit=UNIT, run=RUN,
                           job_id='not-slurm;systemd:' + C).items():
        assert source[key] == value, key


def check_relations(m, outer):
    keys = set(FIELDS.values()) | {'tree_variance/' + s for s in SERIES} | set(EXTRA)
    assert all(number(m.get(k)) for k in keys), 'missing/nonfinite original scalar: ' + str(outer)
    assert all(m[k] >= 0 for k in keys if 'variance' in k or k.endswith('_count'))
    assert all(0 <= m[k] <= 1 for k in keys if k.endswith('_fraction'))
    for k in keys:
        if k.endswith('_count'):
            assert m[k] == int(m[k]), 'noninteger count: ' + k
    v = lambda name: m['tree_variance/' + name]
    def equal(a, b):
        assert number(a) and number(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (outer, a, b)
    q, t = v('question_count'), v('tree_count')
    equal(q, 512); equal(t, 1024)
    equal(v('same_reward_question_fraction') + v('mixed_reward_question_fraction'), 1)
    equal(v('exact_reward_one_question_count') + v('exact_reward_zero_question_count') + v('mixed_question_count'), q)
    equal(v('mixed_reward_question_fraction'), v('mixed_question_count') / q)
    equal(v('same_reward_question_fraction'), (v('exact_reward_one_question_count') + v('exact_reward_zero_question_count')) / q)
    equal(v('same_reward_tree_fraction') + v('mixed_reward_tree_fraction'), 1)
    equal(v('exact_reward_one_tree_count') + v('exact_reward_zero_tree_count') + v('mixed_tree_count'), t)
    equal(v('mixed_reward_tree_fraction'), v('mixed_tree_count') / t)
    equal(v('intra_adv_population_variance'), v('within_root_normalized_component_population_variance'))
    for slot in (0, 1):
        prefix = 'root' + str(slot) + '_'
        equal(v(prefix + 'same_reward_fraction') + v(prefix + 'mixed_reward_fraction'), 1)
        equal(v(prefix + 'exact_reward_one_count') + v(prefix + 'exact_reward_zero_count') + v(prefix + 'mixed_count'), q)
        equal(v(prefix + 'mixed_reward_fraction'), v(prefix + 'mixed_count') / q)
        equal(v(prefix + 'same_reward_fraction'), (v(prefix + 'exact_reward_one_count') + v(prefix + 'exact_reward_zero_count')) / q)
    equal(v('mixed_tree_count'), v('root0_mixed_count') + v('root1_mixed_count'))
    assert 0 <= v('empty_loss_row_count') <= 2560


def systemd(value):
    assert isinstance(value, (str, dict))
    return value if isinstance(value, dict) else dict(line.split('=', 1) for line in value.splitlines() if '=' in line)


def actual200(inputs, last_metric):
    assert isinstance(inputs, dict)
    native = bound(inputs['native_metadata'], True)
    poll = bound(inputs['progress_poll'])
    resource = bound(inputs['resource'])
    control = bound(inputs['dev200_periodic_control'], True)
    poll_identity(poll)
    progress = poll['progress']
    assert progress['actual_completed_outer_from_attempt_results'] == 200 and not progress['errors']
    assert metric(poll, 200) == last_metric, '200 progress poll must retain the exact original200 metrics'
    states = (systemd(poll['systemd']), systemd(resource.get('training_systemd') or resource.get('systemd')
              or (resource.get('state') or {}).get('training_unit_observation')))
    assert states[0].get('InvocationID', INV) == states[1]['InvocationID'] == INV
    for state in states:
        assert state.get('MainPID') in ('0', '503255')
        if state.get('ActiveState') == 'active' and state.get('SubState') == 'running':
            assert state['MainPID'] == '503255'
    entries = [int(n) for n in re.findall(r'epoch\s+\d+,\s*step\s+(\d+)\b', poll['logs']['stage.log'])]
    assert not entries or max(entries) <= 200, 'actual training201 entry is outside this inventory'
    # A source-bound snapshot may be taken before or after the authorized stop.
    # Systemd liveness is preserved as an observation, never counted as progress.
    receipt, identity = native['receipt'], native['identity']
    assert native['source_commit'] == receipt['source_commit'] == C and native['outer'] == 200
    for key, value in dict(code_commit=C, config_sha256=CONFIG, model_identity=MODEL, data_identity=DATA,
                           completed_step=200, actual_update_limit=200, run_id=Path(RUN).name,
                           job_id='not-slurm;systemd:' + C).items():
        assert identity[key] == value, key
    assert receipt['completed_outer'] == 200 and receipt['status'] == 'COMPLETE_NATIVE_MANIFEST_METADATA_CAPTURED'
    assert native['archive_matches_original_manifest'] is True and native['manifest_status'] == 'COMPLETE_NATIVE_ROLLING_STATE_SAVED'
    assert native['manifest_sha256'] == native['recorded_original_manifest_sha256'] == receipt['history_manifest_sha256'] == receipt['source_manifest_sha256']
    assert digest(native['manifest_sha256']) and digest(native['receipt_sha256'])
    for item in (native, receipt):
        assert item['tensor_contents_read'] is False and item['independent_cold_reload_performed'] is False
    workers = receipt['workers']
    assert len(workers) == 8 and all(type(w['rank']) is int for w in workers) and {w['rank'] for w in workers} == set(range(8))
    totals = {}
    for worker in workers:
        value = worker['actual_total_optimizer_calls']
        assert type(value) is int and value >= 0 and worker['observed_adam_steps'] == [value]
        assert worker['scheduler_last_epoch'] == 200 and worker['full_worker_state_finite'] is True
        totals[str(worker['rank'])] = value
    assert progress['completed_actual_optimizer_calls_per_rank'] == totals
    assert receipt['driver']['next_step'] == 201
    ranks = poll['last_rank_summaries']
    assert len(ranks) == 8 and {r['rank'] for r in ranks} == set(range(8))
    for rank in ranks:
        assert rank['outer_step'] == 200
        assert type(rank['successful_optimizer_calls']) is int and rank['successful_optimizer_calls'] > 0
        assert type(rank['skipped_empty_minibatches']) is int and rank['skipped_empty_minibatches'] >= 0
        assert rank['finite_grad_norms'] == rank['successful_optimizer_calls']
        assert type(rank['nonzero_grad_norms']) is int and 0 <= rank['nonzero_grad_norms'] <= rank['finite_grad_norms']
        # Original778 rank records do not emit every opportunity denominator.
        # Missing optional fields remain unknown rather than being supplied by
        # the FULL publisher's derived rank-view aliases.
        for key in ('actual_opportunities_processed', 'actual_opportunities_skipped'):
            assert key not in rank or rank[key] is None or type(rank[key]) is int and rank[key] >= 0
    for key in ('actor/optimizer_steps', 'actor/optimizer_opportunities_processed',
                'actor/optimizer_opportunities_skipped', 'actor/skipped_empty_minibatches'):
        assert number(last_metric.get(key)) and last_metric[key] >= 0
    for key, value in dict(code_commit=C, config_sha256=CONFIG, model_identity=MODEL,
                           data_identity=DATA, job_id='not-slurm;systemd:' + C).items():
        assert control[key] == value, key
    assert control['actual_completed_outer'] == 200 and control['status'] == 'VALIDATION_COST_COMPLETE'
    assert control['phase'] == 'periodic' and control['evaluated_rows'] == control['dataset_rows'] == 512
    assert Path(control['run']).name == 'output' and Path(control['run']).parent.name == Path(RUN).name
    metrics = control['metrics']
    assert metrics['val/qa_count/all'] == 512 and metrics['val/qa_count/nq'] == metrics['val/qa_count/hotpotqa'] == 256
    assert all(number(metrics[k]) and 0 <= metrics[k] <= 1 for k in ('val/f1/all', 'val/em/all'))
    assert digest(control['ledger_sha256'])
    assert any(dev.get('outer') == 200 and dev.get('phase') == 'periodic'
               and dev.get('ledger_sha256') == control['ledger_sha256'] and dev.get('metrics') == metrics
               for dev in poll['development'])
    original = inputs['original_control_source']
    assert original['stable_read'] is True and digest(original['sha256']) and type(original['bytes']) is int
    control_ref = reference(inputs['dev200_periodic_control']['path'])
    assert original['bytes'] == control_ref['bytes'] and original['sha256'] == control_ref['sha256']
    for key in ('path', 'resolved_path'):
        assert isinstance(original[key], str) and original[key].startswith(('/dataset/', '/mnt/'))
    assert Path(original['path']).parent == Path(RUN) / 'output/validation-cost'
    assert Path(original['resolved_path']).parent.name == 'validation-cost'
    assert control['run'] in (RUN + '/output', str(Path(original['resolved_path']).parent.parent))
    for name in FIELDS.values():
        assert number(last_metric[name])
    return dict(native_metadata=reference(inputs['native_metadata']['path']),
                progress_poll=reference(inputs['progress_poll']['path']), resource=reference(inputs['resource']['path']),
                dev200_periodic_control=control_ref, original_control_source=original,
                native_manifest_sha256=native['manifest_sha256'], native_history_receipt_sha256=native['receipt_sha256'],
                actual_total_adam_per_rank=totals, last_actual_rank_summaries=ranks,
                original200_optimizer_metric_scalars={key:last_metric[key] for key in
                    ('actor/optimizer_steps', 'actor/optimizer_opportunities_processed',
                     'actor/optimizer_opportunities_skipped', 'actor/skipped_empty_minibatches')},
                absent_rank_opportunity_fields_remain_unknown=True,
                no_4000_or_all20_or_zero_skip_assumption=True, native_saved_scheduler_last_epoch=200,
                native_saved_driver_next_step=201, saved_cursor_is_not_actual201_entry=True,
                systemd_observations=states, original_dev200_rows=512, original_dev200_phase='periodic',
                original_dev200_inner_ledger_sha256=control['ledger_sha256'],
                original_metric200_logged_before_original_full51713=True,
                old_slow_final_FULL200_or_final_completion_required=False,
                tensor_contents_read=False, independent_cold_reload_performed=False)


def unchanged(rounds=2):
    out = []
    for n in range(rounds):
        for name, ref in FILES.items():
            p = Path(name); raw = p.read_bytes()
            assert raw == RAW[name] and len(raw) == ref['bytes'] and sha(raw) == ref['sha256'], name
            if ref['mirror_checked']:
                for mirror in M:
                    assert (mirror / p.relative_to(B)).read_bytes() == raw, name
        observation = dict(round=n + 1, observed_utc=datetime.now(timezone.utc).isoformat(),
                           unique_inputs=len(FILES), original_bytes_preserved=True)
        out.append(observation); print(json.dumps(observation))
        if n + 1 < rounds:
            time.sleep(1)
    return out


def encode(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def publish(value):
    raw = encode(value)
    assert 0 < len(raw) <= MAX_PRODUCT_BYTES
    targets = [folder / OUTPUT for folder in [B] + M]
    assert not any((folder / name).exists() for folder in [B] + M for name in (OUTPUT, READBACK))
    for target in targets:
        assert target.parent.is_dir()
        with target.open('xb') as stream:
            stream.write(raw)
    observations = []
    for n in range(2):
        for target in targets:
            assert target.read_bytes() == raw
        unchanged(1)
        observations.append(dict(round=n + 1, observed_utc=datetime.now(timezone.utc).isoformat(),
                                 product_copies=3, source_bytes_and_mirrors_preserved=True))
        if n == 0:
            time.sleep(1)
    receipt = encode(dict(schema='variance200_record_coverage_publication_readback_v1',
                          product=dict(path=str(targets[0]), bytes=len(raw), sha256=sha(raw)),
                          observations=observations, remote_operations=False, training_modified=False))
    for folder in [B] + M:
        with (folder / READBACK).open('xb') as stream:
            stream.write(receipt)
    for folder in [B] + M:
        assert (folder / READBACK).read_bytes() == receipt
    print(json.dumps(dict(status=value['status'], path=str(targets[0]), bytes=len(raw), sha256=sha(raw),
                          publication_readback=READBACK, readback_bytes=len(receipt), readback_sha256=sha(receipt))))


def main():
    if not __debug__:
        raise RuntimeError('Required actual-byte conditions cannot be disabled with python -O')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', required=True, type=Path)
    parser.add_argument('--input-manifest-bytes', required=True, type=int)
    parser.add_argument('--input-manifest-sha256', required=True)
    args = parser.parse_args()
    assert args.input_manifest_bytes > 0 and digest(args.input_manifest_sha256)
    assert not any((folder / name).exists() for folder in [B] + M for name in (OUTPUT, READBACK))
    manifest = read(str(args.input_manifest), args.input_manifest_bytes, args.input_manifest_sha256)
    assert manifest['schema'] == 'tree_nested_variance200_original_record_input_manifest_v1'
    assert manifest['status'] == 'ACTUAL_OUTER200_ORIGINAL_SCALAR_INPUTS_FROZEN'
    for key, value in dict(source_commit=C, config_sha256=CONFIG, model_identity=MODEL, data_identity=DATA,
                           unit=UNIT, run=RUN, actual_training_pid=503255, actual_training_invocation_id=INV).items():
        assert manifest[key] == value, key
    prior = bound(manifest['prior_coverage'], True)
    assert local(manifest['prior_coverage']['path']) == B / PRIOR_NAME
    assert manifest['prior_coverage']['bytes'] == PRIOR_BYTES and manifest['prior_coverage']['sha256'] == PRIOR_SHA
    assert prior['schema'] == 'tree_nested_metric_record_coverage_through171_local_v1'
    assert prior['required_field_gaps'] == {} and set(prior['coverage']) == set(FIELDS)
    assert prior['failed_or_duplicate_attempts_counted'] is False and prior['latest_values_used_for_earlier_outers'] is False
    assert prior['actual_local_relations']['relation_failures'] == prior['actual_local_relations']['FULL_original_and_compact_or_curve_mismatches'] == []
    old_catalog = prior['source_catalog']
    assert isinstance(old_catalog, dict) and old_catalog
    alias_asof=[]
    for source_id,ref in old_catalog.items():
        if source_id=='s002':
            expected=dict(path='PER_OUTER_METRICS_COMPACT.json',bytes=101394,
                          sha256='c5f201b38087718b17f7a8434e84b0814a0e1aaef94ec364b2c1efbeea6247f0')
            assert ref==expected
            declaration=manifest['prior_mutable_alias_asof']
            assert declaration['catalog_id']==source_id and declaration['historical_descriptor']==expected
            current=declaration['current_observation'];assert local(current['path'])==B/expected['path']
            bound(current,True)
            alias_asof.append(dict(catalog_id=source_id,historical_descriptor=expected,current_observation=current,
                original_old_alias_bytes_available=False,old_alias_not_reconstructed=True,
                scope='Only this exact mutable legacy index is as-of; all361 immutable sources and original171 values/bindings remain strict.'))
        else:
            bound(ref)
    prior_source = read_bytes(PRIOR_SOURCE, 23741, 'b0b1d374ce85915bb7b8e82f2ad0475f1ce1c3ab2e8325cf19b6b4a38394165e')
    implementation = []
    for name, (size, expected) in SOURCE_PINS.items():
        p = FROZEN / name; raw = read_bytes(p, size, expected)
        implementation.append(dict(source_commit=C, repository_path=name, bytes=len(raw), sha256=sha(raw)))
    self_raw = read_bytes(Path(__file__).resolve())
    records, bindings = {}, []
    rows = manifest['scalar_records']
    assert isinstance(rows, list) and [r['outer'] for r in rows] == list(range(172, 201))
    for row in rows:
        n = row['outer']; source, m, origin = original_record_source(row)
        check_relations(m, n)
        assert m['training/actual_outer_updates'] == n
        records[n] = m
        bindings.append(dict(outer=n, source_commit=C, **origin,
                             selector='iteration_metrics[step=' + str(n) + ']',
                             value_origin='direct_original_recorded_scalar', derived_replacement=False))
    fresh_capture=bound(source['capture_product'],True);fresh_rows=fresh_capture['rows']
    if isinstance(fresh_rows,dict):
        descriptor=fresh_rows;fresh_rows=[]
        for ref in descriptor['parts']:
            part=bound(ref,True);assert part['schema']=='cumulative_scalar_field_part_v1' and part['original_field']=='rows'
            fresh_rows.extend(part['value'])
        assert len(fresh_rows)==descriptor['original_entries']
    assert [r['outer'] for r in fresh_rows]==list(range(1,201))
    crosscheck_fields=list(FIELDS.values())+['tree_variance/'+key for key in SERIES]+EXTRA
    for binding in prior['per_outer_source_bindings']:
        n=binding['outer'];old_source=bound(old_catalog[binding['metric_source']])
        if n==1:
            candidates=[item['metrics'] for item in old_source['stage_log']['outer1_complete_metric_lines'] if item['metrics'].get('step')==1]
            assert candidates and all(item==candidates[0] for item in candidates);old_metric=candidates[0]
        else:old_metric=metric(old_source,n)
        fresh=fresh_rows[n-1]['metric']
        assert all(number(old_metric.get(key)) and old_metric[key]==fresh.get(key) for key in crosscheck_fields), ('prior171 original metric mismatch',n)
        for label,field in FIELDS.items():assert prior['coverage'][label]['values_in_outer1_to171_order'][n-1]==fresh[field]
        for key in SERIES:assert prior['original_denominator_and_count_series'][key][n-1]==fresh['tree_variance/'+key]
        if binding.get('original_metric_line_sha256') is not None:
            assert binding['original_metric_line_sha256']==fresh_rows[n-1]['metric_line']['line_sha256']
    for row in fresh_rows:check_relations(row['metric'],row['outer'])
    actual = actual200(manifest['actual200'], records[200])
    coverage = {}
    for label, field in FIELDS.items():
        old = prior['coverage'][label]
        assert old['original_field'] == field and old['actual_covered_outers'] == list(range(1, 172)) and old['unknown_outers'] == []
        values = old['values_in_outer1_to171_order']
        assert len(values) == 171 and all(number(v) for v in values)
        all_values = values + [records[n][field] for n in range(172, 201)]
        coverage[label] = dict(original_field=field, actual_covered_outers=list(range(1, 201)), unknown_outers=[],
                               values_in_outer1_to200_order=all_values,
                               value_origin='direct original scalar; prior1..171 retained exactly,172..200 explicitly bound to original polls or retrospective original-file lines')
    counts = {}
    for key in SERIES:
        values = prior['original_denominator_and_count_series'][key]
        assert len(values) == 171 and all(number(v) for v in values)
        counts[key] = values + [records[n]['tree_variance/' + key] for n in range(172, 201)]
    observations = unchanged()
    value = dict(schema='tree_nested_metric_record_coverage_through200_original_records_v1',
                 status='PASS_ACTUAL_ORIGINAL_SCALAR_COVERAGE_THROUGH200', observed_utc=datetime.now(timezone.utc).isoformat(),
                 actual_input_manifest=reference(str(args.input_manifest)), prior_coverage=reference(PRIOR_NAME),
                 prior1_to171_original_catalogued_sources_preserved=len(old_catalog)-len(alias_asof),
                 prior1_to171_catalogued_descriptors_retained=len(old_catalog),
                 prior_mutable_alias_asof=alias_asof,
                 original1_to171_crosscheck_against_preserved_metric_sources=dict(fields=30,values=5130,mismatches=0,
                    original_capture=reference(source['capture_product']['path']),all15_required_plus12_counts_plus3_extra=True),
                 prior1_to171_values_and_bindings_preserved_exactly=True,
                 prior_per_outer_source_bindings=prior['per_outer_source_bindings'],
                 prior_binding_source_ids_resolve_in='prior_coverage.source_catalog',
                 new_per_outer_source_bindings=bindings, coverage=coverage,
                 original_denominator_and_count_series=counts,
                 additional_original_recorded_fields_172_to200={field:[records[n][field] for n in range(172, 201)] for field in EXTRA},
                 additional_original_recorded_fields_1_to200={field:[row['metric'][field] for row in fresh_rows] for field in EXTRA},
                 actual200_boundary=actual, frozen778_implementation_sources=implementation,
                 definitions=prior['definitions'], historical_gap_scope=prior['historical_gap_scope'],
                 field_origin_policy='All15 required values and extra fields are original recorded scalars. No derived scalar replaces an absent original. Population variance and original sample-std normalization/epsilon/dtype roles stay distinct.',
                 persistence_scope='Original200 periodic512, then saved200 and logger step200, then original counter201 full51713. Original poll metric200 is required; native200 alone is insufficient. No original slow final completion or FULL200 metric loop is required.',
                 final_dev_scope='Original periodic dev200512 control is separately SHA-bound. The fixed validation512 and final51713 are disjoint; a final subset cannot impersonate periodic512.',
                 failed_or_duplicate_attempts_counted=False, later_step_values_used_for_earlier_outers=False,
                 original_relations_checked_for_new_outers=list(range(172, 201)), required_field_gaps={},
                 derived_scalar_replacements=[], input_readback_observations=observations,
                 source_catalog=sorted(FILES.values(), key=lambda x:x['path']),
                 generator=dict(path=str(Path(__file__).resolve()), bytes=len(self_raw), sha256=sha(self_raw)),
                 adapted_from_generator=dict(path=str(PRIOR_SOURCE), bytes=len(prior_source), sha256=sha(prior_source)),
                 mirrors=[str(p) for p in M], final200_model_HF_or_final51713_acceptance_claimed=False,
                 remote_operations=False, training_modified=False, evaluations_started=False,
                 tensor_model_raw_rollout_or_full_evaluation_output_read=False, tests_compile_or_review_performed=False,
                 canonical_or_existing_products_modified=False)
    publish(value)


if __name__ == '__main__':
    main()
