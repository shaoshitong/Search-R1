"""Pure CPU private-trace readback for one already canonical-bound actual outer15.

Runs against original1843; no writer, model, generation, Slurm or actor calls.
Only hashes/counts are printed. The bounded pilot does not claim all200 coverage.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

ROOT = Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014')
BASE = ROOT.parent / 'qprobe-teacher-warmup200-4gpu-jxu957-recovery718650-20261014'
ORIGINAL = ROOT / 'source' / '1843babdd96b387dd10d91e3207c9ea9489b6941'
RAW_SHA = '717e0a81b02acbc38572d54b3b527acc962666110095105f97f14a10e9782953'
STEP = 15


def need(value, message):
    if not value:
        raise ValueError('ACTUAL_TRACE_READBACK: ' + message)


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def read_bound(path, expected_sha=None, load=None, mode=None):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'unsafe file path')
    with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid == 14090, 'wrong file type/owner')
        if mode is not None:
            need(stat.S_IMODE(before.st_mode) == mode, 'private file mode differs')
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(8*2**20), b''):
            h.update(block)
        digest = h.hexdigest()
        if expected_sha is not None:
            need(digest == expected_sha, 'fresh full file SHA differs')
        stream.seek(0)
        value = load(stream) if load else stream.read()
        after = os.fstat(stream.fileno())
        need(identity(before) == identity(after) == identity(path.lstat()), 'file identity changed while reading')
    return value, dict(path=str(path), sha256=digest, bytes=before.st_size)


def audit():
    need(os.getuid() == 14090 and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'actual CPU owner required')
    need(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
         'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')),
         'explicit CPU caps1 required')
    need(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'no bytecode writes required')
    os.chdir(str(ORIGINAL))
    sys.path.insert(0, str(ORIGINAL))
    import torch
    from verl import DataProto
    from search_g1.qprobe_runtime import _complete_generation
    from reproduction.common8.behavior_runtime import (actor_rows, cast_token_fields, row_hash, read_gate_receipt)
    from reproduction.common8.behavior_trace import (make_trace, audit_outer, canonical_bytes, document_sha256)
    from reproduction.common8.behavior_probability import verify
    from reproduction.common8.precision_contract import sampling_journal_only
    for name in ('reproduction.common8.behavior_runtime', 'reproduction.common8.behavior_trace',
                 'reproduction.common8.behavior_probability', 'search_g1.qprobe_runtime'):
        need(Path(sys.modules[name].__file__).resolve().is_relative_to(ORIGINAL), 'reader original source differs')
    run_bytes, manifest_file = read_bound(BASE/'logs/run-manifest.json')
    run = json.loads(run_bytes)
    need(run['source_commit'] == ORIGINAL.name and run['training_world_size'] == 4, 'producer source/topology differs')
    # Bind loaded pure-reader bytes to the original producer source map.
    for name in ('reproduction/common8/behavior_runtime.py', 'reproduction/common8/behavior_trace.py',
                 'reproduction/common8/behavior_probability.py', 'search_g1/qprobe_runtime.py'):
        need(hashlib.sha256((ORIGINAL/name).read_bytes()).hexdigest() == run['source_sha256'][name],
             'original reader source bytes differ')
    ledger_bytes, _ = read_bound(BASE/'logs/actual-updates.jsonl')
    matches = [r for r in map(json.loads, ledger_bytes.splitlines()) if r['step'] == STEP]
    need(len(matches) == 1, 'completed actual15 must occur exactly once')
    row = matches[0]
    aid = row['attempt_id']
    binding = read_gate_receipt(BASE, run, STEP, update_record=row)
    gate_bytes, gate_file = read_bound(BASE/'logs'/('behavior-probability-attempt-'+aid+'.json'),
                                     binding['proof_sha256'], mode=0o600)
    gate = json.loads(gate_bytes)
    allocation_bytes, allocation_file = read_bound(BASE/'logs/native-allocation-admission.json')
    allocation = json.loads(allocation_bytes)
    job = allocation['slurm_job_id']
    need(allocation['accepted'] is True and allocation['run_id'] == run['run_id'] and job == '719225',
         'actual retained native4 allocation differs')
    raw, raw_file = read_bound(BASE/'artifacts/question-probe/attempts'/aid/'real-rollout.pt', RAW_SHA,
                               lambda f: torch.load(f, map_location='cpu', weights_only=False))
    ident = raw['identity']
    snapshot = run['run_id']+':'+str(STEP)+':policy-'+aid
    need(ident['run_id'] == run['run_id'] and ident['source_sha256'] == run['source_sha256'] and
         ident['slurm_job_id'] == job and ident['logical_update'] == STEP and ident['attempt_id'] == aid and
         ident['policy_snapshot_id'] == gate['policy_snapshot_id'] == snapshot, 'raw/gate actual identity differs')
    _complete_generation(raw['output'], ident)
    parents = []
    for i, value in enumerate(raw['parents'].non_tensor_batch['index']):
        if hasattr(value, 'item'):
            value = value.item()
        parents.append(str(i)+'-'+document_sha256(value))
    need(len(parents) == 512 and parents == ident['parent_group_ids'], 'actual parent order/digests differ')
    ids = [run['run_id']+'-'+str(STEP)+'-attempt-'+aid+'-'+parent+'-sample-'+str(sample)
           for parent in parents for sample in range(5)]
    need(ids == ident['trajectory_ids'] == gate['trajectory_ids'], 'actual ordered G5 identities differ')
    saved = raw['output']
    out = DataProto.from_dict(tensors=dict(saved.batch.items()),
                             non_tensors=dict(saved.non_tensor_batch), meta_info=dict(saved.meta_info))
    cast_token_fields(out)
    need(out.non_tensor_batch['behavior_trajectory_id'].tolist() == ids, 'actual output trajectory order differs')
    sampling = gate['sampling']
    need(canonical_bytes(out.meta_info['behavior_sampling']) == canonical_bytes(sampling), 'actual sampler differs')
    rows = actor_rows(out)
    turns = out.non_tensor_batch['behavior_turns'].tolist()
    trajectories = [dict(trajectory_id=tid, parent_occurrence_index=i//5,
                         sample_index=i%5, turns=turns[i]) for i, tid in enumerate(ids)]
    bindings = dict(expected_run_id=run['run_id'], expected_source_sha256=run['source_sha256'],
                    expected_slurm_job_id=job, expected_logical_update=STEP, expected_parent_group_ids=parents,
                    expected_trajectory_ids=ids, expected_policy_snapshot_id=snapshot, expected_sampling=sampling)
    trace = make_trace(run_id=run['run_id'], source_sha256=run['source_sha256'], slurm_job_id=job,
                       logical_update=STEP, policy_snapshot_id=snapshot, sampling=sampling,
                       parent_group_ids=parents, trajectories=trajectories, actor_rows=rows)
    context = audit_outer(trace, actor_rows=rows, **bindings)
    need(sampling_journal_only(run) is True, 'only original explicit journal-only policy is supported')
    width = out.batch['responses'].shape[1]
    need(torch.equal(out.batch['sampling_mask'].cpu(), out.batch['info_mask'][:, -width:].cpu().bool()),
         'actual sampling mask differs')
    proof = verify(None, out.batch['sampling_log_probs'].detach().cpu(),
                   out.batch['info_mask'][:, -width:].detach().cpu(),
                   out.batch['attention_mask'][:, -width:].detach().cpu(),
                   trace=trace, actor_rows=rows, audit=context, require_hf_parity=False, **bindings)
    need(all(canonical_bytes(value) == canonical_bytes(gate[key]) for key, value in proof.items()),
         'independent full-token probability/context proof differs from actual actor gate')
    out.batch['old_log_probs'] = out.batch['sampling_log_probs'].clone()
    tensor_rows = {tid: row_hash(out, i) for i, tid in enumerate(ids)}
    upstream = {tid: parents[i//5].split('-', 1)[1] for i, tid in enumerate(ids)}
    need(tensor_rows == gate['row_tensor_sha256'] and upstream == gate['upstream_group_sha256'],
         'actual eight-field actor rows/parent hashes differ')
    private_dir = BASE/'artifacts/behavior-traces'
    s = private_dir.lstat()
    need(private_dir.resolve() == private_dir and stat.S_ISDIR(s.st_mode) and s.st_uid == 14090 and
         stat.S_IMODE(s.st_mode) == 0o700, 'actual private trace directory differs')
    rebuilt_bytes = canonical_bytes(trace)
    rebuilt_sha = hashlib.sha256(rebuilt_bytes).hexdigest()
    private_bytes, private_file = read_bound(private_dir/('attempt-'+aid+'.json'),
                                             rebuilt_sha, mode=0o600)
    need(private_bytes == rebuilt_bytes, 'rebuilt trace differs from actual original bytes')
    need(rebuilt_sha == context['trace_sha256'] == proof['trace_sha256'] == gate['trace_sha256'] ==
         gate['private_trace_sha256'] == row['behavior_trace_sha256'], 'full private trace SHA bindings differ')
    return dict(status='ACTUAL_RAW15_PRIVATE_TRACE_ALLTOKENS_AND_ACTOR_ROWS_EQUAL', epoch=time.time(),
                step=STEP, job=job, source_commit=ORIGINAL.name, source_manifest=manifest_file,
                raw_file=raw_file, gate_file=gate_file, allocation_file=allocation_file, private_trace_file=private_file,
                actual_private_trace_byte_equal=True, actual_all2560_ordered_ids_equal=True,
                actual_512_parent_digests_equal=True, actual_2560_actor_tensor_rows_equal=True,
                actual_all_retained_token_contexts_equal=True, actual_selected_logprob_projection_equal=True,
                included_actor_token_count=proof['included_actor_token_count'],
                actor_rows_sha256=proof['actor_rows_sha256'], audit_sha256=proof['audit_sha256'],
                tensor_rows_combined_sha256=document_sha256(tensor_rows),
                hf_forward_performed=False, model_weights_loaded=False, generated_trajectories=0,
                actor_or_head_updates_performed=0, evidence_files_written=0,
                scope='actual15 only; paired with prior raw15 canonical-gold/strict-EM readback; not all200/native final restore')


if __name__ == '__main__':
    print(json.dumps(audit()), flush=True)
