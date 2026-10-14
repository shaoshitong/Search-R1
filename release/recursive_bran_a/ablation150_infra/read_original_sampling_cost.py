"""Read only the closed C1 original outer1 log, before zero-write replay.

Standalone stdlib stdin entry. No model, checkpoint payload, raw trajectory,
GPU/HTTP interface, process signal, Node write or training lock is used. The
timestamp-less action records are bound to the one identified coordinator.
Recorded trajectory elapsed times overlap and are never GPU billing.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import time

BASE = Path('/dataset/shaoshitong/recursive-ablation150-20261014')
GROUP = BASE / 'C1-root-only'
RUN = '260d1debb6214d75b3d1a5954d88f524'
CONFIG_SHA = 'b2a47bdbd8d721579e983062eb3ee942cc1e4b6aa021e7c49861e3763cdb10de'
SCIENCE = '233b0dc95de3f7b797d5b424d3b7dcaafca5328f'
OWNER = (1590507, 0, 676184748)
COORDINATOR_PID = 1596968
OUTPUT = GROUP / 'output/attempt-70b2fb0ea0964047841802e98f78bf70'
LOG = GROUP / 'logs/training-attempt-1791983422634305636.log'
PRESERVED = BASE / 'preserved-zero-write-namespace-failure-1590507'
PARTIAL = GROUP / 'checkpoints/training_state/partial_attempt_000001_outer_000001/manifest.json'
PARTIAL_SHA = '7295d56549a1f34f11b4999baa7c36a0427c953d34b9dd3031cabfa2907af2e1'
SOURCE = BASE / 'source-entry-repair-72a09951'
SOURCE_FILES = {
    'search_r1/baselines/rollout.py': 'd841298ce23ac387f3e26fae4deb00b7a1b440c26a8dffbc27a495551bac4393',
    'search_r1/baselines/controller.py': '2e098c75bca005862481df2d78198a236121fb528d42ef0f18563a0e0adf66eb',
    'verl/trainer/ppo/ray_trainer.py': '5b80191b78e6d2b769cae897cfda29e89a9c35d3522e95925bf8e1cf9631a238',
}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
PID = re.compile(r'\(main_task pid=(\d+)\)')
COUNTERS = ('generated_tokens', 'prefill_tokens', 'retrieval_calls', 'padded_rows',
            'context_mismatches', 'generation_rpc_calls', 'retrieval_http_calls',
            'raw_slots', 'raw_emitted_tokens', 'raw_retained_tokens',
            'raw_discarded_tail_tokens')
SECONDS = ('generation_rpc_wall_seconds', 'retrieval_http_json_wall_seconds',
           'interaction_wall_seconds', 'semantic_prefix_scan_cpu_seconds',
           'semantic_prefix_scan_wall_seconds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, label, integer=False):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid recorded scalar: ' + label)
    require(not integer or value == int(value), 'Noninteger recorded count: ' + label)
    return int(value) if integer else float(value)


def stable(path, limit=2 * 1024 * 1024):
    require(not path.is_symlink() and path.resolve().is_relative_to(BASE.resolve()),
            'Evidence path escapes this exact experiment')
    before = path.stat()
    require(before.st_size <= limit, 'Bounded evidence exceeds allowed read size')
    raw = path.read_bytes()
    after = path.stat()
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    require(signature(before) == signature(after) and len(raw) == before.st_size,
            'Closed evidence changed during read')
    return raw, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     device=before.st_dev, inode=before.st_ino, mtime_ns=before.st_mtime_ns)


def evidence(path):
    raw, proof = stable(path)
    return json.loads(raw), proof


def identity(value):
    require((value.get('pid'), value.get('uid'), value.get('tick')) == OWNER,
            'Original owner identity differs')
    require(value.get('run_id') == RUN, 'Original run differs')


def collect():
    old, old_proof = evidence(PRESERVED / 'current-entry.json')
    identity(old)
    require(old.get('output') == str(OUTPUT) and old.get('group') == 'C1-root-only'
            and old.get('config_sha256') == CONFIG_SHA and old.get('source_commit') == SCIENCE,
            'Original source/config/output binding differs')
    terminal, terminal_proof = evidence(OUTPUT / 'entry-exit.json')
    identity(terminal)
    require(terminal.get('rc') == 1, 'Original actual failure terminal differs')
    wait, wait_proof = evidence(PRESERVED / 'actual-parent-wait.json')
    require(wait.get('pid') == OWNER[0] and wait.get('rc') == 1
            and wait.get('group') == 'C1-root-only', 'Original parent wait differs')
    partial, partial_proof = evidence(PARTIAL)
    require(partial_proof['sha256'] == PARTIAL_SHA, 'Original partial manifest SHA differs')
    require(partial.get('attempted_outer') == 1 and partial.get('completed_outer_updates') == 0
            and partial.get('status') == 'PARTIAL_OUTER_STATE_INCOMPLETE_CONTROLLED_STOP',
            'Original zero-write boundary differs')
    p_identity = partial.get('identity', {})
    require(p_identity.get('run_id') == RUN and p_identity.get('config_sha256') == CONFIG_SHA
            and p_identity.get('code_commit') == SCIENCE, 'Partial run/config/source differs')
    workers = partial.get('worker_outcome', {}).get('worker_states', [])
    require(len(workers) == 8 and {r.get('rank') for r in workers} == set(range(8))
            and all(r.get('committed_optimizer_calls') == 0 for r in workers),
            'Original eight-rank zero Adam evidence differs')
    require(partial.get('actor_inputs', {}).get('rows') == 2560,
            'Original prepared selected row count differs')
    source_proofs = []
    for relative, expected in SOURCE_FILES.items():
        _, proof = stable(SOURCE / relative)
        require(proof['sha256'] == expected, 'Published telemetry source SHA differs: ' + relative)
        source_proofs.append(proof)
    raw, log_proof = stable(LOG, 128 * 1024 * 1024)
    require(raw.endswith(b'\n'), 'Closed original log has an unfinished last line')
    lines = raw.decode('utf-8', errors='strict').splitlines()
    markers = ('ABLATION_ENTRY_START=', 'TREE_TIMING=', 'TREE_ACTION_TIMING=', 'TREE_DRIVER_STAGE=')
    parsed = []
    for line_number, original in enumerate(lines, 1):
        line = ANSI.sub('', original)
        for marker in markers:
            if marker not in line:
                continue
            require('repeated ' not in line, 'Ray log deduplication prevents full action coverage')
            row = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0]
            match = PID.search(line)
            parsed.append((line_number, marker, row, int(match.group(1)) if match else None))
            break
    starts = [(n, r) for n, m, r, _ in parsed if m == 'ABLATION_ENTRY_START=']
    require(len(starts) == 1 and starts[0][1] == old, 'Full log has a different/duplicate entry start')
    events = [(n, m, r) for n, m, r, pid in parsed if m != 'ABLATION_ENTRY_START='
              and pid == COORDINATOR_PID]
    require(events, 'Original coordinator telemetry missing')
    require(all(r.get('outer') == 1 for _, _, r in events), 'Log mixes coordinator outer identities')
    timed = [(n, r) for n, m, r in events if m == 'TREE_TIMING=']
    require(all(r.get('run_id') == RUN for _, r in timed), 'Timed coordinator run identity differs')
    compose_start = [(n, r) for n, r in timed if r.get('event') == 'compose_start']
    compose_end = [(n, r) for n, r in timed if r.get('event') == 'compose_complete']
    require(len(compose_start) == len(compose_end) == 1 and compose_start[0][0] < compose_end[0][0]
            and compose_start[0][1].get('rows') == compose_end[0][1].get('rows') == 2560,
            'Unique complete selected2560 compose boundary missing')
    compose = compose_end[0][1]
    require(terminal['time'] >= compose['unix_time'], 'Terminal precedes the closed sampling boundary')
    roots, branches, rpc_rows, actions = {}, {}, [], []
    sums = collections.Counter()
    pending_rpc = None
    for n, marker, row in events:
        if marker == 'TREE_TIMING=' and row.get('event') == 'generation_rpc_complete':
            require(n < compose_start[0][0] and pending_rpc is None,
                    'Duplicate/unpaired RPC or generation after compose')
            pending_rpc = row
            rpc_rows.append(row)
        elif marker == 'TREE_ACTION_TIMING=':
            require(n < compose_start[0][0] and pending_rpc is not None
                    and row.get('successful') is True
                    and row.get('active_requests') == pending_rpc.get('active_requests'),
                    'Action lacks its unique successful coordinator RPC binding')
            delta = row.get('cost_delta', {})
            require(delta.get('generation_rpc_calls') == 1, 'Action RPC delta differs')
            for key in COUNTERS:
                sums[key] += number(delta.get(key), key, integer=True)
            for key in SECONDS:
                sums[key] += number(delta.get(key), key)
            require(delta['raw_emitted_tokens'] == delta['generated_tokens']
                    and delta['raw_retained_tokens'] + delta['raw_discarded_tail_tokens'] == delta['raw_emitted_tokens'],
                    'Recorded emitted/retained/discarded conservation differs')
            require(abs(delta['generation_rpc_wall_seconds'] - pending_rpc['shared_batch_wall_seconds']) < 0.01,
                    'Recorded RPC time differs between action and timed marker')
            sums['action_wall_seconds'] += number(row.get('wall_seconds'), 'action wall')
            sums['action_driver_cpu_seconds'] += number(row.get('driver_cpu_seconds'), 'action CPU')
            actions.append((n, row))
            pending_rpc = None
        elif marker == 'TREE_TIMING=' and row.get('event') == 'trajectory_complete':
            require(n < compose_start[0][0], 'Trajectory terminal follows compose')
            target = roots if row.get('kind') == 'initial' else branches
            require(row.get('kind') in ('initial', 'recursive_branch') and row.get('trajectory_id') not in target,
                    'Repeated/unknown recorded trajectory terminal')
            number(row.get('source_index'), 'trajectory source', integer=True)
            require(0 <= row['source_index'] < 512, 'Trajectory source outside original512')
            if target is branches:
                attempt = number(row.get('attempt'), 'branch attempt', integer=True)
                require(1 <= attempt <= 10, 'Branch attempt exceeds the original budget')
                number(row.get('new_generated_tokens'), 'branch generated tokens', integer=True)
            target[row['trajectory_id']] = row
    require(pending_rpc is None and len(actions) == len(rpc_rows) > 0,
            'Incomplete successful action/RPC coverage')
    require(len(roots) == 512 and {r['source_index'] for r in roots.values()} == set(range(512)),
            'Complete unique initial512 coverage missing')
    require(not (set(roots) & set(branches)), 'Initial/branch trajectory identity collision')
    require(len({(r['source_index'], r['attempt']) for r in branches.values()}) == len(branches),
            'Duplicate original branch attempt coverage')
    require(sums['context_mismatches'] == 0, 'Original recorded context mismatch occurred')
    rpc_start = min(r['unix_time'] - number(r['shared_batch_wall_seconds'], 'RPC wall') for r in rpc_rows)
    rpc_end = max(r['unix_time'] for r in rpc_rows)
    compose_seconds = number(compose['wall_seconds'], 'compose wall')
    root_elapsed_sum = sum(number(r['wall_seconds'], 'root elapsed') for r in roots.values())
    branch_elapsed_sum = sum(number(r['wall_seconds'], 'branch elapsed') for r in branches.values())
    return dict(protocol='ablation150_original_zero_write_sampling_cost_v1', time=time.time(),
        status='COMPLETE_RECORDED_ACTION_COUNTS_WITH_TIMING_LIMITS', group='C1-root-only',
        run_id=RUN, config_sha256=CONFIG_SHA, scientific_source_commit=SCIENCE,
        original_outer=1, original_optimizer_calls_per_rank=0,
        reused_for_committed_outer1=True, additional_sampling_or_training=False,
        evidence=dict(entry=old_proof, terminal=terminal_proof, parent_wait=wait_proof,
                      zero_write_manifest=partial_proof, full_closed_log=log_proof,
                      telemetry_source=source_proofs, coordinator_pid=COORDINATOR_PID),
        coverage=dict(initial_trajectories=512, completed_nonroot_candidates=len(branches),
                      selected_training_rows=2560, successful_unique_action_batches=len(actions),
                      generation_rpc_records=len(rpc_rows), full_log_read=True),
        recorded_action_cost_sum=dict(sums),
        time_scopes=dict(first_rpc_start_unix_derived=rpc_start, last_rpc_end_unix=rpc_end,
            compose_complete_unix=compose['unix_time'], compose_seconds=compose_seconds,
            generation_RPC_window_wall_seconds=rpc_end-rpc_start,
            first_RPC_to_compose_end_wall_seconds=compose['unix_time']-rpc_start,
            initial_trajectory_elapsed_sum_seconds_overlapping=root_elapsed_sum,
            nonroot_trajectory_elapsed_sum_seconds_overlapping=branch_elapsed_sum,
            exact_native_timing_s_gen=None, measured_GPU_active_seconds=None,
            allocator_billed_GPU_seconds=None),
        limits=[
            'Action sums count physical emitted/prefill tokens including GPU divisibility padding; they are not selected leaf token totals.',
            'RPC start is derived from the recorded end timestamp and monotonic duration. The RPC window and first-RPC-to-compose interval omit pre-first-RPC setup and are not the exact native gen timer.',
            'Initial and nonroot trajectory elapsed sums overlap shared batch execution; do not add them to phase wall time or multiply them into GPU billing.',
            'CPU driver timing and shared RPC wall are recorded scalar clocks, not GPU kernel utilization or allocator billing.',
            'Sampling completed before a namespace programming failure at zero Adam writes; its prepared2560 was reused once. Do not count this evidence as a second training outer.',
        ], training_action='NONE', Node_writes=False)


def main():
    try:
        result = collect()
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(protocol='ablation150_original_zero_write_sampling_cost_v1', time=time.time(),
            status='INCOMPLETE', run_id=RUN, original_outer=1,
            reason=type(error).__name__ + ': ' + str(error), recovered_totals=None,
            training_action='NONE', Node_writes=False)
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
