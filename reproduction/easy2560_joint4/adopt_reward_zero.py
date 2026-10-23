"""Reuse genuine official-actor round0 heads; never claim a new calibration."""
import json
import os
from pathlib import Path
import time
import uuid

from .control import own_base, publish, sha


def _scope(base, run):
    if run.get('dynrpo7b_selection') is True:
        from reproduction.dynrpo7b.infra_profile import native4, own_base as dynrpo_base
        if not native4(run):
            raise ValueError('DYNRPO7B_GROUNDING_REUSE_SCOPE_DIFFERS')
        return dynrpo_base(base)
    if not run.get('easy2560_selection') or run.get('question_probe') is not None:
        raise ValueError('EASY2560_GROUNDING_REUSE_SCOPE_DIFFERS')
    return own_base(base)


def link_verified(source, target, expected):
    source = Path(source); target = Path(target)
    if not source.is_file() or source.is_symlink() or sha(source) != expected:
        raise ValueError('EASY2560_ROUND0_SOURCE_SHA_DIFFERS: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    if sha(target) != expected: raise ValueError('EASY2560_ROUND0_TARGET_SHA_DIFFERS')


def adopt(base, run):
    base = _scope(base, run); old = Path(run['initial_reward_probe_source_base'])
    if old == base:
        raise ValueError('EASY2560_GROUNDING_REUSE_SCOPE_DIFFERS')
    previous = json.loads((old / 'logs/run-manifest.json').read_text())
    initial = json.loads((old / 'logs/formal-inline-initial-probe.json').read_text())
    archive = old / initial['archive']
    if archive.resolve().parent.parent != (old / 'artifacts').resolve():
        raise ValueError('EASY2560_INITIAL_PROBE_ARCHIVE_OUTSIDE_PREDECESSOR')
    if sha(archive) != initial['archive_sha256']:
        raise ValueError('EASY2560_INITIAL_PROBE_ARCHIVE_SHA_DIFFERS')
    source = json.loads(archive.read_text())
    if source['round'] != 0 or source['run_id'] != previous['run_id']:
        raise ValueError('EASY2560_ONLY_OFFICIAL_ACTOR_ROUND0_MAY_BE_REUSED')
    for key in ('initial_model', 'initial_model_sha256', 'model_revision', 'calibration_question_ids'):
        if previous[key] != run[key]: raise ValueError('EASY2560_ROUND0_SCIENTIFIC_INPUT_DIFFERS: ' + key)
    if source['encoder'] != run['initial_model'] or any(
            run['initial_model_sha256'].get(n) != h for n, h in source['encoder_sha256'].items()):
        raise ValueError('EASY2560_ROUND0_OFFICIAL_ENCODER_DIFFERS')
    if set(source['label_proof']['question_ids']) != set(run['calibration_question_ids']):
        raise ValueError('EASY2560_ROUND0_CALIBRATION_QUESTIONS_DIFFER')
    report = source['calibration_report']
    if report['configured_cv_folds'] != 2 or report['require_all_cv_folds'] is not True:
        raise ValueError('EASY2560_ROUND0_ORIGINAL_TWO_FOLD_RECIPE_REQUIRED')
    token = uuid.uuid4().hex
    directory = base / 'artifacts' / ('round_0-attempt-' + token)
    directory.mkdir(parents=True, exist_ok=False)
    origin_dir = archive.parent
    for name, hashes in (('heads', dict(source['head_sha256'], **{
            'calibration-report.json': source['calibration_report_sha256']})), ('dumps', source['dump_sha256'])):
        for filename, expected in hashes.items():
            if Path(filename).name != filename: raise ValueError('EASY2560_UNSAFE_ROUND0_MEMBER')
            link_verified(origin_dir / name / filename, directory / name / filename, expected)
    origin = dict(base=str(old), run_id=previous['run_id'], version=source['version'],
                  original_manifest=str(archive), original_manifest_sha256=sha(archive),
                  original_run_manifest_sha256=sha(old / 'logs/run-manifest.json'),
                  new_calibration_generated=False, inherited_actor_or_optimizer=False,
                  hardlinked_complete_original_calibration_artifacts=True)
    version = run['run_id'] + ':0:' + token
    live = dict(source, run_id=run['run_id'], version=version,
                encoder=run['initial_model'], commit=str(directory / 'heads/commit_probe.pkl'),
                boundary=str(directory / 'heads/boundary_probe.pkl'),
                published_unix=time.time(), reused_origin=origin, new_calibration_generated=False)
    publish(directory / 'version-manifest.json', live)
    publish(base / 'artifacts/live/manifest.json', live)
    receipt = dict(run_id=run['run_id'], version=version, source_commit=run['source_commit'],
                   reused_origin=origin, target_archive=str(directory / 'version-manifest.json'),
                   target_archive_sha256=sha(directory / 'version-manifest.json'),
                   new_calibration_generated=False, new_question_probe_loaded=False)
    publish(base / 'logs/initial-grounding-probe-reuse.json', receipt)
    return receipt


def verify_adopted(base, run, live, archive):
    """Independently reread the original fitted bytes and this run's rebinding."""
    base = _scope(base, run); archive = Path(archive)
    receipt = json.loads((base / 'logs/initial-grounding-probe-reuse.json').read_text())
    origin = live['reused_origin']; old = Path(run['initial_reward_probe_source_base'])
    if receipt['reused_origin'] != origin or origin['base'] != str(old):
        raise ValueError('EASY2560_REUSED_ROUND0_ORIGIN_DIFFERS')
    original = Path(origin['original_manifest'])
    if not original.resolve().is_relative_to(old.resolve()) or sha(original) != origin['original_manifest_sha256']:
        raise ValueError('EASY2560_REUSED_ROUND0_ORIGINAL_MANIFEST_CHANGED')
    previous = json.loads((old / 'logs/run-manifest.json').read_text())
    if sha(old / 'logs/run-manifest.json') != origin['original_run_manifest_sha256']:
        raise ValueError('EASY2560_REUSED_ROUND0_ORIGINAL_RUN_CHANGED')
    source = json.loads(original.read_text())
    if source['round'] != 0 or source['run_id'] != previous['run_id'] or source['version'] != origin['version']:
        raise ValueError('EASY2560_REUSED_ROUND0_ORIGINAL_IDENTITY_DIFFERS')
    for key in ('initial_model', 'initial_model_sha256', 'model_revision', 'calibration_question_ids'):
        if previous[key] != run[key]: raise ValueError('EASY2560_REUSED_ROUND0_SCIENTIFIC_INPUT_CHANGED')
    if (live['run_id'] != run['run_id'] or not live['version'].startswith(run['run_id'] + ':0:') or
            receipt['run_id'] != run['run_id'] or receipt['version'] != live['version'] or
            receipt['source_commit'] != run['source_commit'] or receipt['target_archive'] != str(archive) or
            receipt['target_archive_sha256'] != sha(archive) or receipt['new_calibration_generated'] is not False):
        raise ValueError('EASY2560_REUSED_ROUND0_TARGET_BINDING_DIFFERS')
    expected = dict(source, run_id=run['run_id'], version=live['version'],
                    encoder=run['initial_model'], commit=str(archive.parent / 'heads/commit_probe.pkl'),
                    boundary=str(archive.parent / 'heads/boundary_probe.pkl'),
                    published_unix=live['published_unix'], reused_origin=origin, new_calibration_generated=False)
    if live != expected or json.loads(archive.read_text()) != live:
        raise ValueError('EASY2560_REUSED_ROUND0_MANIFEST_CONTENT_DIFFERS')
    if origin['new_calibration_generated'] is not False or origin['inherited_actor_or_optimizer'] is not False:
        raise ValueError('EASY2560_REUSED_ROUND0_IS_NOT_WEIGHTS_ONLY_GROUNDING_REUSE')
    for directory, hashes in (('heads', dict(source['head_sha256'], **{
            'calibration-report.json': source['calibration_report_sha256']})), ('dumps', source['dump_sha256'])):
        for name, expected_sha in hashes.items():
            if (sha(original.parent / directory / name) != expected_sha or
                    sha(archive.parent / directory / name) != expected_sha):
                raise ValueError('EASY2560_REUSED_ROUND0_FITTED_BYTES_CHANGED')
    return receipt
