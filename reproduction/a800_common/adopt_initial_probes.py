"""Reuse authenticated official-model initial0 calibration after local startup failure.

Only fitted artifact bytes are inherited. The target gets its own version and
must still admit its real probe endpoint; no generation or training is claimed.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from reproduction.a800.probe_admission import calibrated_version
from reproduction.a800_common import qprobe_adoption_source_delta as qprobe_sources
from reproduction.common8 import adopt_initial_probes as artifacts
from reproduction.common8.formal_contract import assert_formal_run
from reproduction.common8.freeze_calibration_plan import _unstarted
from reproduction.common8.policy_calibration_plan import canonical_bytes
from reproduction.common8.prepare_run import verify_model
from reproduction.common8.precision_contract import BF16_PROFILE
from reproduction.common8.source_files import runtime_files
from reproduction.common8.verify_data import digest


LOCAL_HELPERS = frozenset((
    'reproduction/a800_common/adopt_initial_probes.py',
    'reproduction/a800_common/deploy.py',
    'reproduction/a800_common/prepare_run.py',
    'reproduction/a800_common/run_pipeline.py',
    'reproduction/a800_common/train.sh',
    'reproduction/common8/verify_full.py',
    'reproduction/common8/formal_inline_safety.py',
    'reproduction/common8/formal_inline_evaluation.py',
    'reproduction/common8/configuration.py',
    'reproduction/common8/formal_contract.py',
))
RAY_ENTRY = 'verl/trainer/main_ppo_eco.py'
BF16_SOURCE_DELTA = 'reproduction/a800_common/bf16-adoption-source-delta.json'
QPROBE_SOURCE_DELTA = qprobe_sources.DELTA_PATH
BF16_DELTA_SOURCE_FILES = frozenset((
    'verl/workers/actor/dp_actor.py',
    'reproduction/common8/behavior_probability.py',
    'reproduction/common8/behavior_runtime.py',
    'verl/trainer/ppo/ray_trainer.py',
    'reproduction/common8/precision_evidence.py',
    'reproduction/common8/precision_contract.py',
    'reproduction/common8/update_evidence.py',
))
ORIGINAL_HYDRA_OUTPUTS = {
    'outputs/2026-10-08/11-08-49/.hydra/config.yaml':
        'f68a9586ff09bac9b74a5613bf6295efe0d2a05324dd8feaf17fe0a412a3fbfa',
    'outputs/2026-10-08/11-08-49/.hydra/hydra.yaml':
        '0e739a6da28c0047c5a90e1aa57e996f16ed06fb825ca23b3717dc398afc4cad',
    'outputs/2026-10-08/11-08-49/.hydra/overrides.yaml':
        '6932b54712a16f3ee80bee87ed5a2a194963d9015c05e7697f29021067c7e934',
}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _namespace(value):
    path = Path(value)
    _check(path.is_absolute() and path == path.resolve() and path.is_dir()
           and not path.is_symlink() and path.stat().st_uid == os.getuid(),
           'Expected a canonical owned run directory')
    return path


def _read(path, base):
    return artifacts._read(artifacts._owned_file(Path(path), base))


def _source_map(base, run, provenance=None):
    actual = {str(path.relative_to(base / 'repo')): digest(path)
              for path in runtime_files(base / 'repo')}
    expected = run['source_sha256']
    _check(all(actual.get(name) == sha for name, sha in expected.items()),
           'A bound run source file changed or is missing')
    extras = {name: sha for name, sha in actual.items() if name not in expected}
    if extras:
        # This exact trio was read from the immutable original failed run. It
        # was emitted after prepare_run's disk snapshot; it is not new source.
        _check(run['run_id'] == '6de29bcb7be741cfa1bfacdf9e34efc9'
               and run['source_commit'] == 'f820ba6cb2db4f5cbe2ffc370cf515c3fe22d43c'
               and len(expected) == 380 and extras == ORIGINAL_HYDRA_OUTPUTS,
               'Unrecognized files outside the bound source snapshot')
        claim_path = artifacts._owned_file(base / 'logs/training-launch-claim.json', base)
        claim = artifacts._read(claim_path)
        ledger = _read(base / 'logs/process-ledger.json', base)
        trainer = ledger['jobs']['trainer']
        leader = trainer['leader']
        _check(claim['run_id'] == ledger['run_id'] == run['run_id']
               and claim['execution_id'] == run['execution_id']
               and claim['shell_pid'] == leader['pid']
               and claim['shell_start_ticks'] == leader['start_ticks']
               and leader['uid'] == os.getuid() and leader['sid'] == leader['pid']
               and trainer['arguments'] == ['bash', 'reproduction/a800_common/train.sh']
               and trainer['state'] == 'STOPPED'
               and type(trainer.get('exit_code')) is int and trainer['exit_code'] == 1,
               'Hydra files have no matching actual failed trainer launch')
        log_path = artifacts._owned_file(Path(trainer['log']), base)
        _check(log_path == base / 'logs/trainer.log'
               and 'OSError: AF_UNIX path length cannot exceed 107 bytes' in log_path.read_text()
               and not any((base / 'logs').glob('optimizer-rank-*.jsonl'))
               and not any((base / 'logs').glob('scheduler-rank-*.jsonl')),
               'Original failure was not Ray startup before optimizer activity')
        updates = base / 'logs/actual-updates.jsonl'
        _check(not updates.exists() or updates.stat().st_size == 0,
               'Original failed trainer has completed update evidence')
        mtimes = {}
        for name in extras:
            path = artifacts._owned_file(base / 'repo' / name, base)
            modified = path.stat().st_mtime
            _check(claim['claimed_unix'] - 2 <= modified <= log_path.stat().st_mtime + 2,
                   'Hydra output timestamp is outside the actual failed launch')
            mtimes[name] = modified
        proof = dict(role='authenticated_original_ray_failure_hydra_runtime_outputs',
            run_id=run['run_id'], source_commit=run['source_commit'],
            bound_source_map_sha256=hashlib.sha256(canonical_bytes(expected)).hexdigest(),
            runtime_output_sha256=extras, runtime_output_mtime=mtimes,
            training_launch_claim_sha256=digest(claim_path), trainer_log_sha256=digest(log_path),
            trainer_job_sha256=hashlib.sha256(canonical_bytes(trainer)).hexdigest(),
            trainer_pid=leader['pid'], trainer_start_ticks=leader['start_ticks'],
            claimed_unix=claim['claimed_unix'], failed_log_modified_unix=log_path.stat().st_mtime)
        if provenance is not None:
            provenance.update(proof)
    return dict(expected)


def _outside_ray_bootstrap(raw):
    native_hook = (b'def main_task(config):\n'
                   b'    from search_g1.qprobe_driver_gpu import bind_native_driver_gpu\n'
                   b'    bind_native_driver_gpu(config)\n')
    if native_hook in raw:
        _check(raw.count(native_hook) == 1, 'Repeated native question-Head driver routing hook')
        # Only this exact native-only entry hook is transport integration. The
        # helper's full bytes remain bound by the precise source delta below.
        raw = raw.replace(native_hook, b'def main_task(config):\n', 1)
    start = b'def main(config):\n'
    end = b'    try:\n        ray.get(main_task.remote(config))\n'
    _check(raw.count(start) == 1 and raw.count(end) == 1,
           'Unrecognized bounded Ray entry point')
    prefix, remainder = raw.split(start, 1)
    bootstrap, suffix = remainder.split(end, 1)
    _check(b"assert not ray.is_initialized()" in bootstrap
           and b"ray.init(address='local'" in bootstrap,
           'Private local Ray bootstrap is missing')
    return prefix + start + end + suffix


def _bf16_reuse(run, old):
    return ('precision_profile' not in old
            and run.get('precision_profile') == BF16_PROFILE
            and old['run_id'] == '6de29bcb7be741cfa1bfacdf9e34efc9'
            and old['source_commit'] == 'f820ba6cb2db4f5cbe2ffc370cf515c3fe22d43c')


def budget_changes_allowed(run, old):
    """Authenticate BF16 and the explicit fresh four-rank micro4 transition."""
    previous, current = old['budget'], run['budget']
    if canonical_bytes(previous) == canonical_bytes(current):
        return {}
    _check(_bf16_reuse(run, old), 'Initial calibration scientific budget changed')
    values = dict(compute_dtype=('float32', 'bfloat16'),
                  rollout_dtype=('float32', 'bfloat16'), allow_tf32=(False, None))
    if run.get('actor_micro_batch_per_rank')==4:
        _check(type(run['actor_micro_batch_per_rank']) is int
               and run.get('profile')=='common8' and run.get('execution_backend')=='ssh_local'
               and run.get('training_world_size')==old.get('training_world_size')==4
               and run.get('completed_updates')==0 and run.get('target_updates')==200
               and not run.get('full_state_restore'), 'Micro4 adoption requires a fresh four-rank actor')
        values.update(per_rank_actor_micro_batch=(2,4),ppo_micro_batch_size=(8,16),
                      objective_reduction=('original two-row micro masked_mean','original four-row micro masked_mean'))
    for name, (before, after) in values.items():
        _check(canonical_bytes(previous[name]) == canonical_bytes(before)
               and canonical_bytes(current[name]) == canonical_bytes(after),
               'Unapproved initial calibration budget transition: ' + name)
    _check(canonical_bytes({k: v for k, v in previous.items() if k not in values})
           == canonical_bytes({k: v for k, v in current.items() if k not in values}),
           'Initial calibration unrelated budget changed')
    return {name: dict(old=previous[name], new=current[name]) for name in values}


def source_changed_allowed(name, previous, current, run, old):
    """Authenticate exact published BF16 deltas, never a blanket core exception.

    This module runs from the target's bound repo during adoption and final
    acceptance. The delta document is itself bound by that target source map.
    """
    _check(previous == old['source_sha256'] and current == run['source_sha256'],
           'Source comparison is not the complete bound manifests')
    if run.get('question_probe') is not None:
        # The legacy BF16 document describes an earlier target. A question-
        # probe target instead binds every changed path to the pinned published
        # question-probe baseline plus explicitly assigned integration edits.
        # Local helpers do not bypass this complete comparison in this route.
        base = Path(__file__).resolve().parents[3]
        path = artifacts._owned_file(base / 'repo' / QPROBE_SOURCE_DELTA, base)
        _check(digest(path) == current.get(QPROBE_SOURCE_DELTA),
               'Question-probe adoption delta differs from the target source snapshot')
        changes = qprobe_sources.validate(artifacts._read(path), previous, current, run, old)
        if name == QPROBE_SOURCE_DELTA:
            _check(name not in previous, 'Question-probe delta replaced an original bound file')
            return True
        return name in changes
    if name in LOCAL_HELPERS:
        return True
    if not _bf16_reuse(run, old) or name not in BF16_DELTA_SOURCE_FILES | {BF16_SOURCE_DELTA}:
        return False
    base = Path(__file__).resolve().parents[3]
    path = artifacts._owned_file(base / 'repo' / BF16_SOURCE_DELTA, base)
    _check(digest(path) == current.get(BF16_SOURCE_DELTA),
           'BF16 adoption delta differs from the target source snapshot')
    delta = artifacts._read(path)
    _check(set(delta) == {'schema_version', 'role', 'original_run_id',
                         'original_source_commit', 'target_precision_profile', 'changes'}
           and type(delta['schema_version']) is int and delta['schema_version'] == 1
           and delta['role'] == 'a800_bf16_initial_probe_source_delta'
           and delta['original_run_id'] == old['run_id']
           and delta['original_source_commit'] == old['source_commit']
           and delta['target_precision_profile'] == run['precision_profile'],
           'BF16 adoption delta has an unbound origin or precision profile')
    changes = delta['changes']
    _check(type(changes) is dict and changes and set(changes) <= BF16_DELTA_SOURCE_FILES,
           'BF16 adoption delta includes an unapproved source path')
    for item, proof in changes.items():
        _check(type(proof) is dict and set(proof) == {'old_sha256', 'new_sha256'}
               and proof['old_sha256'] == previous.get(item)
               and proof['new_sha256'] == current.get(item)
               and proof['old_sha256'] != proof['new_sha256']
               and isinstance(proof['new_sha256'], str) and len(proof['new_sha256']) == 64
               and all(c in '0123456789abcdef' for c in proof['new_sha256'])
               and (proof['old_sha256'] is not None
                    or item == 'reproduction/common8/precision_contract.py'),
               'BF16 adoption source hashes do not bind the manifests: ' + item)
    if name == BF16_SOURCE_DELTA:
        _check(name not in previous, 'BF16 source delta replaced an original bound file')
        return True
    return name in changes


def _compatible(base, source, run, old, source_provenance=None):
    for candidate in (run, old):
        assert_formal_run(candidate)
        _check(candidate['execution_backend'] == 'ssh_local'
               and candidate['training_world_size'] == 4,
               'Only actual local four-card official fresh runs can reuse probes')
        for name in ('actor_checkpoint', 'reference_checkpoint', 'resume_path',
                     'inherited_policy_run_id', 'inherited_optimizer_run_id',
                     'inherited_scheduler_run_id'):
            _check(candidate.get(name) is None, 'Inherited training state: ' + name)
    _check(run['run_id'] != old['run_id']
           and run['execution_id'] != old['execution_id'],
           'Original and target must have separate fresh identities')
    for name in ('initial_model', 'reference_model', 'initial_model_sha256',
                 'model_revision', 'input_sha256', 'calibration_question_ids',
                 'retrieval_asset_paths', 'retrieval_encoder'):
        _check(canonical_bytes(run[name]) == canonical_bytes(old[name]),
               'Initial calibration scientific identity changed: ' + name)
    budget_changes_allowed(run, old)
    model = Path(run['initial_model'])
    _check(model == model.resolve()
           and model == Path('/data/search-g1/assets/qwen2.5-3b-instruct'),
           'Encoder must be the original official initial model')
    official = artifacts._owned_file(
        base / 'preparation/official-model-manifest-original.json', base)
    _check(verify_model(model, official) == run['initial_model_sha256'],
           'Actual official initial model fingerprint differs')
    for root, candidate in ((base, run), (source, old)):
        for name, sha in candidate['input_sha256'].items():
            path = artifacts._owned_file(root / 'data' / name, root)
            _check(digest(path) == sha, 'Actual shared input changed: ' + name)
    previous, current = _source_map(source, old, source_provenance), _source_map(base, run)
    changed = []
    for name in sorted(set(previous) | set(current)):
        if previous.get(name) == current.get(name):
            continue
        if name == RAY_ENTRY:
            before = artifacts._owned_file(source / 'repo' / name, source).read_bytes()
            after = artifacts._owned_file(base / 'repo' / name, base).read_bytes()
            _check(_outside_ray_bootstrap(before) == _outside_ray_bootstrap(after),
                   'Generation/reward/trainer bytes changed outside local Ray startup')
        else:
            _check(source_changed_allowed(name, previous, current, run, old),
                   'Scientific source changed: ' + name)
        changed.append(name)
    return changed


def _original_artifacts(source, old):
    live_path = artifacts._owned_file(source / 'artifacts/live/manifest.json', source)
    live = artifacts._read(live_path)
    version = calibrated_version(old, live)
    _check(live['round'] == 0 and old['initial_probe_version'] in (None, version),
           'Original artifact is not the official initial0 evaluator')
    _check(live['label_proof']['model'] == old['initial_model'],
           'Original calibration was generated by another encoder')
    heads = artifacts._owned_file(Path(live['commit']), source).parent
    _check(heads == artifacts._owned_file(Path(live['boundary']), source).parent
           and heads.name == 'heads', 'Original heads have different round directories')
    round_dir = heads.parent
    _check(round_dir.parent == source / 'artifacts'
           and round_dir.name.startswith('round_0-attempt-'),
           'Expected immutable original generated initial0 artifacts')
    archive = artifacts._owned_file(round_dir / 'version-manifest.json', source)
    _check(artifacts._read(archive) == live, 'Original live and archived version differ')
    report, proof = live['calibration_report'], live['label_proof']
    _check(report['require_all_cv_folds'] is True
           and report['configured_cv_folds'] == 2
           and report['boundary_unique_question_weighting'] is True
           and proof['question_ids'] == old['calibration_question_ids']
           and proof['group_size'] == 5
           and proof['calibration_protocol']['rollout_max_new'] == 500
           and proof['calibration_protocol']['correction_feedback'] is True
           and proof['evidence_deletion'] == 'typed_retrieval_ranges_only_preserve_model_and_correction'
           and len(proof['actual_retrieval_counts']) == 1500
           and 0 < proof['searched_valid_trajectory_count'] <= 1500,
           'Original artifacts do not bind genuine strict300/G5 calibration')
    files = {live_path: digest(live_path), archive: digest(archive)}
    files.update(artifacts._files(heads, dict(live['head_sha256'], **{
        'calibration-report.json': live['calibration_report_sha256']})))
    dump = round_dir / 'dumps'
    _check('calibration-label-proof.json' in live['dump_sha256']
           and all(name.endswith(('.npz', '.jsonl'))
                   or name == 'calibration-label-proof.json'
                   for name in live['dump_sha256']), 'Unexpected calibration dump files')
    files.update(artifacts._files(dump, live['dump_sha256']))
    _check(artifacts._read(heads / 'calibration-report.json') == report
           and artifacts._read(dump / 'calibration-label-proof.json') == proof,
           'Actual calibration report or label proof bytes differ')
    artifacts._calibration_arrays(dump, report, proof)
    artifacts._fitted_heads(heads, report)
    ledger = _read(source / 'logs/process-ledger.json', source)
    job = ledger['jobs']['initial-calibration']
    expected = [old['python_executable'], '-m', 'reproduction.common8.refit',
                '--step', '0', '--model', old['initial_model']]
    _check(ledger['run_id'] == old['run_id'] and job['state'] == 'STOPPED'
           and type(job.get('exit_code')) is int and job['exit_code'] == 0
           and job['arguments'] == expected
           and job['leader']['uid'] == os.getuid()
           and job['leader']['sid'] == job['leader']['pid']
           and job['leader']['start_ticks'] > 0,
           'No actual successfully completed original initial0 calibration child')
    log_path = artifacts._owned_file(Path(job['log']), source)
    _check(log_path == source / 'logs/initial-calibration.log'
           and ('SEARCH_G1_BOTH_PROBES_FITTED=' + version) in log_path.read_text(),
           'Original actual calibration log does not bind the fitted version')
    files[log_path] = digest(log_path)
    calibration_log = artifacts._owned_file(round_dir / 'calibration.log', source)
    files[calibration_log] = digest(calibration_log)
    _check(not list((source / 'logs').glob('optimizer-rank-*.jsonl')),
           'Original startup run already has optimizer evidence')
    updates = source / 'logs/actual-updates.jsonl'
    _check(not updates.exists() or updates.stat().st_size == 0,
           'Original startup run already has update evidence')
    return live, archive, heads, dump, files, job, calibration_log


def _calibration_job_projection(job):
    return dict(role='actual_original_initial0_calibration_child',
        state=job['state'], exit_code=job['exit_code'],
        leader={key: job['leader'][key] for key in ('pid', 'start_ticks', 'sid', 'uid')},
        log=job['log'], module='reproduction.common8.refit', step=0,
        model=job['arguments'][-1])


def adopt(base, source_base):
    """Publish copied initial0 heads; leave HTTP/resource admission prospective."""
    base, source = _namespace(base), _namespace(source_base)
    _check(base != source, 'Cannot write into the original calibration namespace')
    for name in ('logs', 'artifacts', 'preparation'):
        _namespace(base / name)
    run_path = artifacts._owned_file(base / 'logs/run-manifest.json', base)
    old_path = artifacts._owned_file(source / 'logs/run-manifest.json', source)
    run_raw, old_raw = run_path.read_bytes(), old_path.read_bytes()
    run, old = artifacts._read(run_path), artifacts._read(old_path)
    _unstarted(base, run)
    _check(run['initial_probe_version'] is None
           and not any((base / path).exists() or (base / path).is_symlink() for path in (
               'artifacts/live/manifest.json', 'logs/initial-probes-accepted.json',
               'logs/initial-probe-adoption-claim.json', 'logs/initial-probe-adoption.json')),
           'Target already fitted, admitted or claimed probe adoption')
    source_provenance = {}
    changes = _compatible(base, source, run, old, source_provenance)
    live, archive, heads, dump, files, job, calibration_log = _original_artifacts(source, old)
    token = uuid.uuid4().hex
    round_dir = base / 'artifacts' / ('round_0-adopted-' + token)
    origin = dict(run_id=old['run_id'], source_commit=old['source_commit'],
        execution_id=old['execution_id'], base=str(source), version=live['version'],
        run_manifest_sha256=hashlib.sha256(old_raw).hexdigest(),
        archive_sha256=digest(archive), calibration_job=_calibration_job_projection(job),
        calibration_job_sha256=hashlib.sha256(canonical_bytes(job)).hexdigest(),
        calibration_log_sha256=digest(calibration_log),
        head_sha256=live['head_sha256'], dump_sha256=live['dump_sha256'],
        encoder_sha256=live['encoder_sha256'], calibration_report_sha256=live['calibration_report_sha256'],
        source_fingerprint_validation=source_provenance)
    claim = dict(schema_version=1, role='a800_initial_probe_adoption_claim',
                 run_id=run['run_id'], source_run_id=old['run_id'], source_base=str(source),
                 target_round_dir=str(round_dir), claimed_unix=time.time())
    artifacts._write(base / 'logs/initial-probe-adoption-claim.json', canonical_bytes(claim))
    round_dir.mkdir(mode=0o700)
    for name in ('heads', 'dumps', 'provenance'):
        (round_dir / name).mkdir(mode=0o700)
    for source_dir, name in ((heads, 'heads'), (dump, 'dumps')):
        for item in sorted(source_dir.iterdir()):
            target = round_dir / name / item.name
            artifacts._write(target, item.read_bytes())
            _check(digest(target) == files[item], 'Copied original artifact differs')
    for name, data in (
        ('original-version-manifest.json', archive.read_bytes()),
        ('original-run-manifest.json', old_raw),
        ('original-calibration-job.json', canonical_bytes(_calibration_job_projection(job))),
        ('original-calibration.log', calibration_log.read_bytes()),
    ):
        artifacts._write(round_dir / 'provenance' / name, data)
    _check(all(digest(path) == sha for path, sha in files.items())
           and old_path.read_bytes() == old_raw
           and _read(source / 'logs/process-ledger.json', source)['jobs']['initial-calibration'] == job,
           'Original calibration changed during copy; preserve partial adoption claim')
    _check(run_path.read_bytes() == run_raw, 'Target identity changed during adoption')
    _source_map(base, run)
    source_provenance_after = {}
    _source_map(source, old, source_provenance_after)
    _check(source_provenance_after == source_provenance,
           'Original source/runtime provenance changed during adoption')
    _unstarted(base, run)
    new = dict(live, run_id=run['run_id'], version=run['run_id'] + ':0:' + token,
        encoder=run['initial_model'], commit=str(round_dir / 'heads/commit_probe.pkl'),
        boundary=str(round_dir / 'heads/boundary_probe.pkl'), published_unix=time.time(),
        adoption_origin=origin, target_source_commit=run['source_commit'],
        target_source_sha256=run['source_sha256'], new_calibration_generated=False)
    calibrated_version(run, new)
    artifacts._write(round_dir / 'version-manifest.json', canonical_bytes(new))
    live_dir = base / 'artifacts/live'
    live_dir.mkdir(exist_ok=True, mode=0o700)
    _namespace(live_dir)
    temporary = live_dir / ('manifest-adoption-' + token + '.tmp')
    artifacts._write(temporary, canonical_bytes(new))
    os.link(temporary, live_dir / 'manifest.json')
    temporary.unlink()
    fd = os.open(live_dir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    receipt = dict(schema_version=1, role='a800_initial_probe_adoption',
        run_id=run['run_id'], execution_id=run['execution_id'], source_commit=run['source_commit'],
        adopted=True, version=new['version'], origin=origin, allowed_startup_source_changes=changes,
        allowed_training_precision_budget_changes=budget_changes_allowed(run, old),
        training_precision_source_delta_sha256=(run['source_sha256'].get(BF16_SOURCE_DELTA)
            if _bf16_reuse(run, old) else None),
        question_probe_source_delta_sha256=(run['source_sha256'].get(QPROBE_SOURCE_DELTA)
            if run.get('question_probe') is not None else None),
        live_manifest_sha256=digest(live_dir / 'manifest.json'),
        target_archive=str(round_dir / 'version-manifest.json'),
        target_archive_sha256=digest(round_dir / 'version-manifest.json'),
        actual_original_calibration_exit_code=0, finite_artifacts_revalidated=True,
        new_calibration_generated=False, new_discovery_performed=False,
        inherited_training_state=False, production_admission=False,
        child_http_admission_required=True, checked_unix=time.time())
    artifacts._write(base / 'logs/initial-probe-adoption.json', canonical_bytes(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--source-base', required=True)
    args = parser.parse_args()
    print(json.dumps(adopt(args.base, args.source_base)), flush=True)


if __name__ == '__main__':
    main()
