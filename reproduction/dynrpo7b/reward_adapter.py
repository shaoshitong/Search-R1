"""Reuse the official 7B round0 Soft Reward heads, separately from selection."""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .recipe import REWARD_SOURCE
from .deploy import sha

MANIFEST = (REWARD_SOURCE + '/artifacts/'
    'round_0-restored-cb419d0b950846208121053d3d382e62/version-manifest.json')
MANIFEST_SHA = '893fe8c21df2c30044d63a03884bef6ef98443f0010d38219e7dc96ae0b17286'
HEAD_SHA = dict(boundary_probe='a2c33da3d37f779a6792b36aea80b4e095c2e6cfaed2900a75042de2e0af7b75',
    commit_probe='b4e402c811e5d09a1df15716666528e008e9bc576ccb1a9fc0ccb4ef85a4a143')
ORIGINAL_DUMPS = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-20261021/'
    'artifacts/round_0-attempt-e0086aba4f4043e1acbf20dca55c634a/dumps')


def verified_dump_source(primary, expected, fallback=None):
    """Use the original static bytes only when the restored archive omits them."""
    primary = Path(primary)
    chosen = primary if primary.exists() else Path(fallback) if fallback is not None else primary
    if not chosen.is_file() or chosen.is_symlink() or sha(chosen) != expected:
        raise ValueError('Grounding calibration dump absent or SHA differs: ' + str(chosen))
    return chosen, dict(path=str(chosen), sha256=expected,
                        restored_archive_missing=chosen != primary)


def resolve_dump(archive, filename, expected):
    archive = Path(archive)
    primary = archive.parent / 'dumps' / filename
    fallback = ORIGINAL_DUMPS / filename if archive == Path(MANIFEST) else None
    return verified_dump_source(primary, expected, fallback)


def validate_source(run):
    """Small static manifest/head comparison; no model inference or refit."""
    path = Path(MANIFEST)
    if sha(path) != MANIFEST_SHA:
        raise ValueError('Pinned 7B round0 reward manifest SHA changed')
    document = json.loads(path.read_text())
    if document['round'] != 0 or document['encoder'] != run['initial_model']:
        raise ValueError('Reward round0 encoder differs from official initial 7B')
    if any(run['initial_model_sha256'].get(name) != value
           for name, value in document['encoder_sha256'].items()):
        raise ValueError('Reward encoder identity differs')
    labels = document['label_proof']
    protocol = labels['calibration_protocol']
    if (set(labels['question_ids']) != set(run['calibration_question_ids']) or
            labels['group_size'] != 5 or protocol['rollout_max_new'] != 500 or
            protocol['correction_feedback'] is not True):
        raise ValueError('Reward calibration questions/G5/correction protocol differs')
    for head, expected in HEAD_SHA.items():
        name = head + '.pkl'
        if document['head_sha256'][name] != expected or sha(path.parent / 'heads' / name) != expected:
            raise ValueError('Pinned reward head SHA changed: ' + name)
    return document


def adopt(base, run):
    validate_source(run)
    from reproduction.easy2560_joint4.adopt_reward_zero import adopt as reuse
    return reuse(base, run)


def verify_adopted(base, run, live, archive):
    from reproduction.easy2560_joint4.adopt_reward_zero import verify_adopted as verify
    return verify(base, run, live, archive)


def install_restored(base, run):
    """Relocate saved grounding bytes, preserving the saved version/encoder."""
    from .restore import source
    from .deploy import publish
    base = Path(base)
    _, receipt, _, _, _ = source(run['full_state_restore']['source_receipt'])
    original = Path(receipt['reward_probe_manifest']['path'])
    live = json.loads(original.read_text())
    if live['version'] != run['initial_probe_version']:
        raise ValueError('Saved grounding version differs from restored actor boundary')
    directory = base / 'artifacts' / ('round_%d-restored-%s' % (live['round'], uuid.uuid4().hex))
    directory.mkdir(parents=True)
    for category, files in (('heads', dict(live['head_sha256'], **{
            'calibration-report.json': live['calibration_report_sha256']})),
            ('dumps', live['dump_sha256'])):
        for filename, expected in files.items():
            src = original.parent / category / filename
            target = directory / category / filename
            target.parent.mkdir(exist_ok=True)
            if sha(src) != expected:
                raise ValueError('Saved grounding artifact SHA changed: ' + filename)
            try:
                os.link(src, target)
            except OSError:
                shutil.copy2(src, target)
    relocated = dict(live, commit=str(directory / 'heads/commit_probe.pkl'),
        boundary=str(directory / 'heads/boundary_probe.pkl'),
        restoration_origin=dict(source_archive=str(original), source_archive_sha256=sha(original),
            source_receipt=run['full_state_restore']['source_receipt'],
            new_calibration_generated=False), published_unix=time.time())
    publish(directory / 'version-manifest.json', relocated)
    publish(base / 'artifacts/live/manifest.json', relocated)
    return relocated
