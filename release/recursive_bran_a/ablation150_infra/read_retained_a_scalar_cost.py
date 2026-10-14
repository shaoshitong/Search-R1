"""Read existing A final-ancestry scalar costs; never replay A or load tensors.

The original run supplies outer1..100; the accepted saved100 recovery supplies
outer101..150. Original discarded101..118/partial119 are excluded and their
physical rework cost remains explicitly outside the coverage of these sums.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
import time

ORIGINAL = Path('/dataset/shaoshitong/recursive-bran-a-h100-20261012')
RECOVERY = ORIGINAL / 'recovery-from100-20261013'
PACK = Path('/dataset/shaoshitong/recursive-ablation150-20261014/source-analysis-b95c4f26/'
            'release/recursive_bran_a/ablation150_aggregate_watch_v1/a-reference-pack.json')
PACK_SHA = 'c3fabcd0dc7830283e4e9f88360e59a7ac3d561cb7570f2d66e941669744f45c'
PARENT_SHA = 'bb0599ced9db88eaa5dfec21ead76d214a8a1f5b4a64aa7baa4b85a41ceebce2'
FINAL150_SHA = 'eac297e697caf21fe8b08c78ce0851d1bd9dd2b3fd9c47604495772b1d0ed944'
ORIGINAL_RUN = '1c806d15752d44a7ac99687f0737d7e3'
RECOVERY_RUN = 'd97047b5902642c883d24aa908945cda'
ORIGINAL_CONFIG = '653193a4eb06bd28667f78ac14a66a2d1c97200ca8199b8521dcc2d3de35b319'
RECOVERY_CONFIG = '43e28cc7ecc0d38a93324bf480aafb23a0c1ecdb979e9ea587d0358ae095233d'
ORIGINAL_SOURCE = '7966a1e76d6abddb4349d57c4727d69fe62fb16d'
RECOVERY_SOURCE = 'a1ec638575268747468b1a14ff8fd5f858ca0e1a'
SMALL_LIMIT = 4 * 1024 * 1024
LOG_LIMIT = 4 * 1024**3
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
STEP = re.compile(r'\bstep:(\d+) - ')
COUNT_FIELDS = {f'rollout/{name}' for name in (
    'generated_tokens', 'prefill_tokens', 'retrieval_calls', 'padded_rows',
    'context_mismatches', 'generation_rpc_calls', 'retrieval_http_calls',
    'raw_slots', 'raw_emitted_tokens', 'raw_retained_tokens',
    'raw_discarded_tail_tokens', 'raw_inherited_leaf_tokens', 'trajectories')}
ROLLOUT_SECONDS = {f'rollout/{name}' for name in (
    'generation_rpc_wall_seconds', 'retrieval_http_json_wall_seconds',
    'interaction_wall_seconds', 'semantic_prefix_scan_cpu_seconds',
    'semantic_prefix_scan_wall_seconds')}
ACCOUNTING = ('training/actual_adam_calls_per_rank', 'training/actual_outer_updates',
              'training/consumed_sampling_rounds', 'actor/scheduler_step_after_outer')
REQUIRED = ('rollout/raw_emitted_tokens', 'timing_s/step', 'timing_s/gen',
            'timing_s/ref', 'timing_s/update_actor', *ACCOUNTING)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def regular(path, limit):
    require(path.is_absolute() and path == path.resolve(), 'Unexpected symlink/relative evidence path')
    value = path.stat()
    require(stat.S_ISREG(value.st_mode) and value.st_size <= limit, 'Evidence is not a bounded regular file')
    return value


def small(path, expected=None, parse=True):
    before = regular(path, SMALL_LIMIT)
    raw = path.read_bytes()
    require(signature(before) == signature(path.stat()) and len(raw) == before.st_size,
            'Small retained evidence changed while reading')
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or digest == expected, 'Pinned retained small evidence SHA differs')
    proof = dict(path=str(path), bytes=len(raw), sha256=digest)
    return (json.loads(raw) if parse else raw), proof


def configuration(base, run, config_sha, source):
    receipt, receipt_proof = small(base / 'receipts/configuration.json')
    require((receipt['run_id'], receipt['config_sha256'], receipt['source_commit']) ==
            (run, config_sha, source), 'Actual A run/config/source differs from the retained chain')
    config_path = base / 'receipts/resolved.yaml'
    require(Path(receipt['config']) == config_path, 'Actual A resolved config path differs')
    _, config_proof = small(config_path, config_sha, parse=False)
    entry, entry_proof = small(base / 'receipts/train-start.json')
    require(entry['run_id'] == run, 'Actual A entry belongs to another run')
    return dict(run_id=run, config_sha256=config_sha, declared_source_commit=source,
                configuration_receipt=receipt_proof, resolved_config=config_proof,
                entry_receipt=entry_proof,
                entry={key: entry[key] for key in ('time', 'pid', 'run_id', 'job')},
                data_identity=receipt['data_identity'])


def identity_chain():
    original = configuration(ORIGINAL, ORIGINAL_RUN, ORIGINAL_CONFIG, ORIGINAL_SOURCE)
    recovery = configuration(RECOVERY, RECOVERY_RUN, RECOVERY_CONFIG, RECOVERY_SOURCE)
    parent_path = ORIGINAL / 'checkpoints/training_state/global_step_100/post-evaluation/manifest.json'
    parent, parent_proof = small(parent_path, PARENT_SHA)
    binding, binding_proof = small(RECOVERY / 'receipts/continuation-binding.json')
    migration, migration_proof = small(RECOVERY / 'output/sparse-schedule-migrated.json')
    require(parent['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
            and parent['boundary_phase'] == 'post-evaluation'
            and parent['identity']['completed_step'] == 100
            and parent['identity']['config_sha256'] == ORIGINAL_CONFIG
            and parent['identity']['code_commit'] == ORIGINAL_SOURCE, 'Actual saved100 parent differs')
    require(binding['protocol'] == 'recursive_a_sparse_schedule_continuation_v1'
            and binding['handoff_contract'] == 'h100_saved100_after_allocation_terminal_v1'
            and binding['parent_manifest'] == str(parent_path)
            and binding['parent_manifest_sha256'] == PARENT_SHA
            and binding['parent_config_sha256'] == ORIGINAL_CONFIG
            and binding['parent_identity'] == parent['identity']
            and binding['parent_completed_step'] == 100 and binding['parent_optimizer_calls'] == 2000,
            'Actual recovery binding does not point to the accepted parent')
    provenance = migration['provenance']
    require(migration['status'] == 'COLD_RECURSIVE_A_STATE_MIGRATED'
            and migration['driver_actual_state_equal'] is True and migration['next_outer'] == 101
            and provenance['parent_identity'] == parent['identity']
            and provenance['parent_manifest_sha256'] == PARENT_SHA
            and provenance['child_code_commit'] == RECOVERY_SOURCE
            and provenance['child_config_sha256'] == RECOVERY_CONFIG,
            'Actual recovery migration receipt is incomplete or mismatched')
    workers = migration['workers']
    require(len(workers) == 8 and {row['rank'] for row in workers} == set(range(8))
            and all(row['actual_cold_native_state_equal'] is True
                    and row['adam_steps'] == [2000] and row['scheduler_step'] == 100 for row in workers),
            'Actual native eight-rank saved100 restoration receipt differs')
    final, final_proof = small(
        RECOVERY / 'checkpoints/training_state/global_step_150/post-evaluation/manifest.json', FINAL150_SHA)
    require(final['status'] == 'COMPLETE_SAME_WORLD_STATE_SAVED'
            and final['boundary_phase'] == 'post-evaluation'
            and final['identity']['completed_step'] == 150
            and final['identity']['config_sha256'] == RECOVERY_CONFIG
            and final['identity']['code_commit'] == RECOVERY_SOURCE,
            'Actual saved150 final-ancestry boundary differs')
    require(original['data_identity'] == recovery['data_identity'] == parent['identity']['data_identity']
            == final['identity']['data_identity'], 'Actual A data identities differ across recovery')
    return dict(original=original, recovery=recovery, parent_post100=parent_proof,
                continuation_binding=binding_proof, migration=migration_proof,
                final_post150=final_proof, native_worker_driver_restore_receipt_accepted=True,
                independent_tensor_reload_performed=False)


def parse_scalar_line(raw):
    captured = raw.decode('utf-8', errors='replace').strip()
    clean = ANSI.sub('', captured)
    match = STEP.search(clean)
    if not match:
        return None
    values = {}
    for item in clean[match.end():].split(' - '):
        key, sep, text = item.rpartition(':')
        if not sep:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        if (key in COUNT_FIELDS or key in ROLLOUT_SECONDS or key in ACCOUNTING
                or key.startswith('timing_s/')):
            require(math.isfinite(value) and value >= 0, 'Nonfinite/negative native scalar: ' + key)
            if key in COUNT_FIELDS or key in ACCOUNTING:
                require(value == int(value), 'Noninteger native count: ' + key)
                value = int(value)
            require(key not in values or values[key] == value, 'Conflicting repeated metric in one line')
            values[key] = value
    if 'timing_s/step' not in values:
        return None  # A live partial/progress line never proves a closed outer.
    return int(match.group(1)), values, hashlib.sha256(captured.encode('utf-8')).hexdigest()


def log_rows(base, first, last):
    path = base / 'logs/training.log'
    before = regular(path, LOG_LIMIT)
    rows, repeated, source_hash = {}, 0, hashlib.sha256()
    with path.open('rb') as handle:
        for line_number, raw in enumerate(handle, 1):
            source_hash.update(raw)
            if b'step:' not in raw or not raw.endswith(b'\n'):
                continue
            marker = re.search(rb'\bstep:(\d+) - ', raw)
            if marker is None or not first <= int(marker.group(1)) <= last:
                continue  # Discarded/other-run ranges are never counted or scalar-validated.
            parsed = parse_scalar_line(raw)
            if parsed is None:
                continue
            outer, metrics, line_sha = parsed
            if not first <= outer <= last:
                continue
            require(metrics.get('training/actual_outer_updates') == outer
                    and metrics.get('training/consumed_sampling_rounds') == outer
                    and metrics.get('actor/scheduler_step_after_outer') == outer,
                    'Native scalar outer/accounting/scheduler identity differs')
            row = dict(outer=outer, metrics=metrics, source_line_number=line_number,
                       captured_line_sha256=line_sha,
                       raw_line_sha256=hashlib.sha256(raw).hexdigest())
            if outer in rows:
                require(rows[outer]['metrics'] == metrics, 'Conflicting retained closed scalar outer')
                repeated += 1
            else:
                rows[outer] = row
    require(signature(before) == signature(path.stat()), 'Retained source log changed during the read')
    return rows, dict(path=str(path), bytes=before.st_size, sha256=source_hash.hexdigest(),
                      device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns,
                      retained_outer_range=[first, last], identical_repeated_scalar_lines=repeated,
                      full_source_log_read=True,
                      line_hash_convention='captured: exact retained UTF-8 line stripped of surrounding whitespace '
                      '(ANSI retained), matching the existing reference reader; raw: exact line bytes')


def prefix(rows, outer, identities_accepted):
    present = sorted(index for index in rows if index <= outer)
    required = list(range(1, outer + 1))
    keys = sorted({key for index in present for key in rows[index]['metrics']})
    coverage = {key: sum(key in rows[index]['metrics'] for index in present) for key in keys}
    complete = identities_accepted and present == required and all(coverage.get(key) == outer for key in REQUIRED)
    sums = {key: sum(rows[index]['metrics'].get(key, 0) for index in present)
            for key in keys if key in COUNT_FIELDS or key in ROLLOUT_SECONDS or key.startswith('timing_s/')}
    return dict(outer=outer, status='COMPLETE_RETAINED_ANCESTRY_SCALARS' if complete else 'INCOMPLETE',
                observed_closed_outers=len(present), missing_outers=sorted(set(required) - set(present)),
                per_scalar_outer_coverage=coverage, recorded_prefix_sums=sums,
                physical_emitted_token_prefix_complete=complete,
                allocated_gpu_hours_from_recorded_outer_wall=(sums['timing_s/step'] * 8 / 3600
                    if complete else None),
                actual_adam_each_rank=(rows[outer]['metrics'].get('training/actual_adam_calls_per_rank')
                    if outer in rows else None), historical_rework_cost_covered=False)


def collect():
    errors, identities, rows, logs = [], None, {}, {}
    try:
        identities = identity_chain()
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as error:
        errors.append(dict(scope='ancestry_identity', error=str(error)))
    for label, base, first, last in (('original', ORIGINAL, 1, 100), ('recovery', RECOVERY, 101, 150)):
        try:
            captured, logs[label] = log_rows(base, first, last)
            rows.update(captured)
        except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as error:
            errors.append(dict(scope=label + '_retained_log', error=str(error)))
    matching, pack_proof = {}, None
    try:
        pack, pack_proof = small(PACK, PACK_SHA)
        require(pack['protocol'] == 'retained_A_native_aggregate_pack_v1'
                and pack['new_inference'] is False, 'Boundary reference is not the retained native pack')
        for outer in (5, 50, 100, 150):
            accepted = pack['milestones'][str(outer)]
            line = rows.get(outer)
            require(line is not None, 'Retained boundary scalar line is absent: ' + str(outer))
            expected_log = str((ORIGINAL if outer <= 100 else RECOVERY) / 'logs/training.log')
            require(accepted['source']['native_log'] == expected_log
                    and line['captured_line_sha256'] == accepted['source']['captured_line_sha256'],
                    'Exact captured native boundary line differs from the retained pack')
            for key, value in accepted['recorded_boundary_scalars'].items():
                if key in REQUIRED or key in COUNT_FIELDS or key in ROLLOUT_SECONDS or key.startswith('timing_s/'):
                    require(line['metrics'].get(key) == value, 'Boundary native scalar differs from retained pack')
            matching[str(outer)] = True
    except (FileNotFoundError, ValueError, KeyError, TypeError, OSError) as error:
        errors.append(dict(scope='retained_boundary_pack', error=str(error)))
    identity_accepted = identities is not None and len(matching) == 4 and not errors
    summaries = {str(outer): prefix(rows, outer, identity_accepted) for outer in (5, 50, 100, 150)}
    complete = summaries['150']['status'] == 'COMPLETE_RETAINED_ANCESTRY_SCALARS'
    return dict(protocol='retained_A_final_ancestry_scalar_cost_v1', time=time.time(),
                status='COMPLETE_RETAINED_ANCESTRY_SCALAR150' if complete else 'INCOMPLETE',
                goal_complete=False, identities=identities, logs=logs, retained_boundary_pack=pack_proof,
                boundary_pack_line_and_scalar_match=matching, prefixes=summaries,
                closed_rows=[dict(rows[index], entry_segment='original' if index <= 100 else 'recovery')
                             for index in sorted(rows)], errors=errors,
                scopes=dict(physical_tokens='Recorded raw_emitted_tokens, including padding/discarded tails by '
                            'native convention; not training-token counts or independent token reconstruction',
                            phase_walls='Existing native phase timers may overlap. Do not add them to total step. '
                            'Step includes scheduled dev/save when native does; no separate CPU/GPU utilization claim',
                            gpu_hours='Eight configured GPUs times recorded completed-outer step wall only; '
                            'not active GPU time, Slurm lease billing, startup, downtime or historical rework',
                            historical_rework='Original101..118 and partial119 were unpersisted and replayed after '
                            'saved100. Their physical cost and earlier failed launches are NOT covered by ancestry sums',
                            identity='Two actual entry/run/config/source segments joined only by the actual saved100 '
                            'native migration receipt; no common fabricated config or bitwise vLLM restoration claim',
                            actions='Existing log/receipt reads only; no A evaluation, QA/raw-text export, '
                            'model/tensor/weight reads, GPU/HTTP/signals/locks or Node writes'))


def main():
    result = collect()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result['status'] == 'COMPLETE_RETAINED_ANCESTRY_SCALAR150' else 2


if __name__ == '__main__':
    raise SystemExit(main())
