"""Original reward-grounding fit/adopt lifecycle for the prefix-selection run.

The correctness selection head is unrelated to these boundary/commit heads.
An actual joint actor checkpoint is committed before its scheduled refit.  A
failed refit therefore resumes that joint checkpoint, rather than rerunning an
actor update or silently proceeding with an obsolete reward evaluator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from reproduction.prefix20_online.launch import read, once, publish


REFIT_ROUNDS = (50, 100, 150)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_round(completed_outer):
    if type(completed_outer) is not int or not 0 <= completed_outer <= 200:
        raise ValueError('Grounding clock requires complete outer0..200')
    return min(completed_outer // 50 * 50, 150)


def static_binding(config):
    """Scientific lifecycle binding; current evaluator version is dynamic."""
    if not config.get('grounding_refresh_enabled', False):
        # Older CPU fixtures and immutable historical checkpoints stay readable.
        # New production launch validation does not permit this legacy mode.
        return dict(mode='legacy_fixed', initial_probe_version=config['grounding_probe_version'])
    from .interaction_contract import INTERACTION_CONTRACT
    return dict(mode='original_boundary_and_commit_refit', rounds=[0, *REFIT_ROUNDS],
        initial_manifest_sha256=config['grounding_manifest_sha256'],
        calibration_sha256=config['grounding_calibration_sha256'],
        calibration_questions=300, layers=[21, 24, 27], commit_position='Xa',
        commit_label='flip', selection_correctness_head_is_separate=True,
        initial_round0='retained_real_fit_original_calibration_protocol',
        subsequent_refit_interaction=INTERACTION_CONTRACT)


def validate_config(config):
    if (config.get('grounding_refresh_enabled') is not True
            or config.get('grounding_refit_rounds') != list(REFIT_ROUNDS)
            or len(config.get('grounding_calibration_question_ids', [])) != 300
            or len(set(config['grounding_calibration_question_ids'])) != 300
            or not config.get('grounding_calibration_sha256')
            or not str(config.get('grounding_calibration_questions', '')).startswith(('/mnt/', '/dataset/'))
            or type(config.get('grounding_refit_visible_device', 7)) is not int
            or not 0 <= config.get('grounding_refit_visible_device', 7) < 8):
        raise ValueError('Bind the real 300-question grounding calibration and 50/100/150 lifecycle')
    return config


def _manifest_state(config, path, *, digest=None):
    path = Path(path)
    if digest is not None and sha(path) != digest:
        raise ValueError('Grounding manifest bytes differ from their immutable receipt')
    value = read(path)
    if (value['round'] not in (0, *REFIT_ROUNDS) or not value['version']
            or value['label_proof']['question_count'] != 300
            or value['calibration_report']['unique_question_count'] != 300
            or set(value['label_proof']['question_ids']) != set(config['grounding_calibration_question_ids'])
            or set(value['head_sha256']) != {'boundary_probe.pkl', 'commit_probe.pkl'}):
        raise ValueError('Use actual fitted boundary and commit heads on the bound 300 questions')
    return dict(schema_version=1, run_id=config['run_id'], round=value['round'],
        version=value['version'], manifest_path=str(path), manifest_sha256=sha(path),
        encoder=value['encoder'], source_run_id=value['run_id'],
        encoder_sha256=value['encoder_sha256'], head_sha256=value['head_sha256'],
        boundary=value['boundary'], commit=value['commit'],
        lifecycle=static_binding(config))


def initialize(base, config):
    """Import the real retained round0 fit, or recover this run's current fit."""
    validate_config(config)
    base = Path(base); latest = base / 'grounding/latest.json'
    if latest.is_file():
        return checked_state(config, read(latest), require_adopted=False)
    source = Path(config['grounding_manifest'])
    state = _manifest_state(config, source, digest=config['grounding_manifest_sha256'])
    if state['round'] != 0 or state['version'] != config['grounding_probe_version']:
        raise ValueError('Fresh official actor0 requires the real verified round0 grounding fit')
    expected = config['model_files_sha256']
    if any(expected.get(name) != digest for name, digest in state['encoder_sha256'].items()):
        raise ValueError('Round0 grounding encoder differs from the official actor initialization')
    root = base / 'grounding/rounds/round-000000'; root.mkdir(parents=True, exist_ok=True)
    path = root / 'version-manifest.json'
    if path.exists():
        if sha(path) != state['manifest_sha256']:
            raise ValueError('Existing imported round0 manifest requires reconciliation')
    else:
        with path.open('xb') as handle:
            handle.write(source.read_bytes()); handle.flush(); os.fsync(handle.fileno())
    state = dict(state, manifest_path=str(path), source_manifest=str(source),
                 imported_actual_round0_fit=True, freshly_fitted_in_this_run=False,
                 source_actor_version=config['run_id'] + '-official0')
    once(root / 'fit-complete.json', state)
    publish(latest, state)
    return state


def checked_state(config, state, *, require_adopted=True):
    expected = _manifest_state(config, state['manifest_path'], digest=state['manifest_sha256'])
    if (state['run_id'] != config['run_id'] or state['lifecycle'] != static_binding(config)
            or any(state[key] != expected[key] for key in
                   ('round', 'version', 'encoder', 'encoder_sha256', 'head_sha256', 'boundary', 'commit'))):
        raise ValueError('Grounding state and actual immutable fitted manifest differ')
    if require_adopted:
        proof = Path(state['adoption_receipt'])
        if sha(proof) != state['adoption_receipt_sha256']:
            raise ValueError('Grounding adoption receipt changed')
        adopted = read(proof)
        if (adopted['run_id'] != config['run_id'] or adopted['version'] != state['version']
                or adopted['round'] != state['round'] or adopted['manifest_sha256'] != state['manifest_sha256']
                or sorted(row['rank'] for row in adopted['replicas']) != list(range(8))
                or any(row.get('ok') is not True or row.get('version') != state['version']
                       or row.get('round') != state['round'] or row.get('encoder') != state['encoder']
                       for row in adopted['replicas'])):
            raise ValueError('All eight actual reward replicas must adopt the same fitted evaluator')
    return state


def publish_live(base, config, state):
    state = checked_state(config, state, require_adopted=False)
    live = Path(base) / 'grounding/live/manifest.json'
    live.parent.mkdir(parents=True, exist_ok=True)
    publish(live, read(state['manifest_path']))
    return live


def adopt(base, config, state, replicas):
    """Publish only after the caller has read all eight real /health responses."""
    state = checked_state(config, state, require_adopted=False)
    proof = dict(run_id=config['run_id'], round=state['round'], version=state['version'],
        manifest_sha256=state['manifest_sha256'], replicas=replicas, adopted_unix=time.time())
    directory = Path(base) / 'grounding/adoptions'; directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + '.json'); once(path, proof)
    state = dict(state, adoption_receipt=str(path), adoption_receipt_sha256=sha(path))
    checked_state(config, state)
    publish(Path(base) / 'grounding/latest.json', state)
    return state


def bind_attempt(config, state, latest_actor, outer):
    state = checked_state(config, state)
    if state['round'] != expected_round(outer - 1):
        raise ValueError('Scheduled grounding refit/adoption must finish before the next actor update')
    result = dict(config, grounding_probe_version=state['version'],
                  grounding_current_manifest=state['manifest_path'], grounding_state=state)
    validate_actor_transition(result, latest_actor, outer)
    return result


def validate_actor_transition(config, latest, outer):
    if not config.get('grounding_refresh_enabled', False):
        return dict(status='LEGACY_FIXED_GROUNDING_FIXTURE')
    state = checked_state(config, config['grounding_state'])
    if state['version'] != config['grounding_probe_version'] or state['round'] != expected_round(outer - 1):
        raise ValueError('Reward attempt is not using its actual scheduled adopted grounding version')
    if outer == 1:
        if latest is not None or state['round'] != 0:
            raise ValueError('Fresh actor0 uses round0 grounding and no old optimizer')
        return dict(status='REAL_ROUND0_GROUNDING_ADOPTED')
    if latest is None or latest['completed_outer'] != outer - 1:
        raise ValueError('Grounding transition must bind the immediately previous actor checkpoint')
    previous = latest['grounding_state']
    previous_version = latest['grounding_probe_version']
    if (previous_version != previous['version'] or previous['round'] != expected_round(outer - 2)):
        raise ValueError('Previous actor did not retain its actual reward-grounding clock')
    if previous_version != state['version']:
        if (outer - 1 not in REFIT_ROUNDS or state['round'] != outer - 1
                or state['source_actor_version'] != latest['actor_version']):
            raise ValueError('Only an actual 50/100/150 actor-bound refit may change the reward evaluator')
    elif previous['round'] != state['round']:
        raise ValueError('Grounding clock changed without changing its real fitted version')
    return dict(status='SCHEDULED_GROUNDING_VERSION_BOUND', round=state['round'], version=state['version'])


def validate_joint(config, joint):
    """Check the evaluator actually used by the committed actor update."""
    outer = joint['outer']; state = checked_state(config, joint['grounding'])
    actor = joint['actor']
    if (state['round'] != expected_round(outer - 1)
            or actor['grounding_probe_version'] != state['version']
            or actor['grounding_state'] != state
            or sha(joint['grounding_snapshot']) != joint['grounding_snapshot_sha256']
            or read(joint['grounding_snapshot']) != state):
        raise ValueError('Joint checkpoint lost its actual reward-evaluator version and snapshot')
    return state


def recovery_state(source, config, joint):
    """Prefer a completed/adopted due fit, otherwise retain its pending boundary."""
    state = validate_joint(config, joint)
    due = joint['outer']
    if due not in REFIT_ROUNDS:
        return state
    refreshed = Path(source) / 'outers' / ('outer-%06d' % due) / 'grounding-refresh.json'
    fitted = Path(source) / 'grounding/rounds' / ('round-%06d' % due) / 'fit-complete.json'
    if refreshed.is_file():
        result = read(refreshed)
        if (result['run_id'] != config['run_id'] or result['outer'] != due
                or result['required_before_outer'] != due + 1):
            raise ValueError('Recovered grounding transition belongs to another milestone')
        value = checked_state(config, result['grounding'])
    elif fitted.is_file():
        # A real completed fit may not have reached all-eight adoption before
        # interruption. Restart replicas with that actual fit, then adopt it;
        # do not refit, regenerate calibration or rerun the completed actor.
        value = checked_state(config, read(fitted), require_adopted=False)
    else:
        return state
    if value['round'] != due or value['source_actor_version'] != joint['actor']['actor_version']:
        raise ValueError('Recovered due grounding fit is not bound to the saved actor checkpoint')
    return value


def fit(base, outer):
    """Run the original two-head dump/train procedure against this actor checkpoint."""
    import hashlib
    import joblib
    import numpy as np
    import pyarrow.parquet as pq
    from reproduction.common8.verify_data import probe_question_id
    from reproduction.a800.recovery_binding import finite_fitted_values
    from search_g1.resource import evaluator_gpu_lock
    base = Path(base); config = validate_config(read(base / 'preparation/launch-config.json'))
    if outer not in REFIT_ROUNDS:
        raise ValueError('Only 50/100/150 refit a reward evaluator needed by later actor updates')
    joint = read(base / 'outers' / ('outer-%06d' % outer) / 'complete.json')
    actor = joint['actor']
    if (joint['run_id'] != config['run_id'] or joint['outer'] != outer
            or actor['completed_outer'] != outer or actor['actual_adam_steps_per_rank'] != [outer * 20] * 8):
        raise ValueError('Fit grounding only from this run actual complete actor milestone')
    root = base / 'grounding/rounds' / ('round-%06d' % outer)
    root.mkdir(parents=True, exist_ok=True)
    complete = root / 'fit-complete.json'; intent = root / 'fit-intent.json'
    if complete.is_file():
        value = checked_state(config, read(complete), require_adopted=False)
        if value['source_actor_version'] != actor['actor_version']:
            raise ValueError('Existing fit used another actor checkpoint')
        return value
    if intent.exists():
        raise RuntimeError('Existing interrupted grounding fit requires reconciliation; do not replay it blindly')
    calibration = Path(config['grounding_calibration_questions'])
    if sha(calibration) != config['grounding_calibration_sha256']:
        raise ValueError('Grounding calibration file changed')
    table = pq.read_table(calibration)
    question_ids = [probe_question_id(chat) for chat in table['prompt'].to_pylist()]
    if len(question_ids) != 300 or question_ids != config['grounding_calibration_question_ids']:
        raise ValueError('Use exactly the original ordered 300 calibration questions')
    encoder = Path(actor['hf_model_dir'])
    expected = {name: value['sha256'] for name, value in actor['hf_files'].items()}
    encoder_hashes = {name: sha(encoder / name) for name in expected}
    if encoder_hashes != expected:
        raise ValueError('Actual saved actor encoder differs before reward refitting')
    once(intent, dict(run_id=config['run_id'], outer=outer, source_actor_version=actor['actor_version'],
        source_actor_receipt=actor, calibration_sha256=config['grounding_calibration_sha256'],
        created_unix=time.time()))
    dump = root / 'dumps'; heads = root / 'heads'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config.get('grounding_refit_visible_device', 7)),
        SEARCH_G1_BASE=str(base / 'grounding/refit-resource'), RETRIEVER_URL=config['search_url'],
        NO_PROXY='127.0.0.1,localhost')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env.pop(key, None)
    if config.get('rollout_attention_backend'):
        env['VLLM_ATTENTION_BACKEND'] = config['rollout_attention_backend']
    python = str(Path(config['environment_root']) / 'bin/python'); repo = base / 'repo'
    commands = [
        [python, str(repo / 'ts_probe_dump.py'), '--model', str(encoder), '--data', str(calibration),
         '--tag', 'nq-hotpot-common', '--cap', '300', '--G', '5', '--layers', '21,24,27',
         '--gpu_mem', str(config.get('grounding_refit_gpu_memory_utilization', .25)),
         '--rollout_max_new', '500', '--correction_feedback', '--out', str(dump),
         '--interaction-config', str(base / 'preparation/launch-config.json')],
        [python, str(repo / 'ts_probe_train.py'), '--dumps', str(dump), '--layers', '21,24,27',
         '--commit-position', 'Xa', '--commit-label', 'flip', '--save', str(heads), '--require-all-cv-folds']]
    # No resident grounding replicas may overlap calibration; launch stops only
    # its own eight replicas before invoking this phase and restarts them after.
    with evaluator_gpu_lock(), (root / 'calibration.log').open('x') as log:
        for command in commands:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = read(heads / 'calibration-report.json'); proof = read(dump / 'calibration-label-proof.json')
    from .interaction_contract import INTERACTION_CONTRACT
    if (proof['question_count'] != 300 or report['unique_question_count'] != 300
            or set(proof['question_ids']) != set(question_ids)
            or report['require_all_cv_folds'] is not True or report['configured_cv_folds'] != 2
            or proof['calibration_protocol']['rollout_max_new'] != 500
            or proof['calibration_protocol']['correction_feedback'] is not True
            or proof['evidence_deletion'] != 'typed_retrieval_ranges_only_preserve_model_and_correction'
            or proof.get('interaction_contract') != INTERACTION_CONTRACT
            or not np.isfinite([report['commit_cv_auc'], report['boundary_cv_auc']]).all()
            or min(report['commit_class_counts']) <= 0 or min(report['boundary_class_counts']) <= 0):
        raise ValueError('Actual two-head fitting report/labels do not match the original grounding protocol')
    for name in ('commit_probe.pkl', 'boundary_probe.pkl'):
        fitted = joblib.load(heads / name)
        if list(fitted['clf'].classes_) != [0, 1] or not np.isfinite(fitted['auc']):
            raise ValueError('Actual binary grounding head was not fitted')
        finite_fitted_values(vars(fitted['clf']))
    if any(sha(encoder / name) != digest for name, digest in encoder_hashes.items()):
        raise ValueError('Actor encoder changed during actual grounding calibration')
    payload = dict(run_id=config['run_id'], round=outer,
        version='%s:%d:%s' % (config['run_id'], outer, uuid.uuid4().hex), encoder=str(encoder),
        commit=str(heads / 'commit_probe.pkl'), boundary=str(heads / 'boundary_probe.pkl'),
        encoder_sha256=encoder_hashes,
        head_sha256={name: hashlib.sha256((heads / name).read_bytes()).hexdigest()
                     for name in ('commit_probe.pkl', 'boundary_probe.pkl')},
        calibration_report=report, label_proof=proof,
        calibration_report_sha256=sha(heads / 'calibration-report.json'),
        dump_sha256={path.name: sha(path) for path in sorted(dump.iterdir()) if path.is_file()},
        source_actor_version=actor['actor_version'], calibration_sha256=config['grounding_calibration_sha256'],
        published_unix=time.time())
    manifest = root / 'version-manifest.json'; once(manifest, payload)
    state = dict(_manifest_state(config, manifest), source_actor_version=actor['actor_version'],
                 source_actor_receipt=actor, freshly_fitted_in_this_run=True,
                 imported_actual_round0_fit=False)
    once(complete, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True); parser.add_argument('--outer', type=int, required=True)
    args = parser.parse_args(); print(json.dumps(fit(args.base, args.outer), sort_keys=True))


if __name__ == '__main__':
    main()
