"""CPU-only identity binding for this run's saved milestone and final QA."""
from pathlib import Path
import re

RUN_ID = '4a1cc2d500fe40979ef8ca4792959438'
SHARED_ROOT = Path('/mnt/shared/workspace/training_data_node_10/shaoshitong')
LOCAL_ROOT = Path('/mnt/data/shaoshitong')
NODES = ('h100-5', 'h100-8', 'h100-10', 'h100-11')


def retained_path(path):
    """Compare the same retained18 namespace without accessing another node's disk."""
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('QA paths cannot traverse parents')
    for root in (SHARED_ROOT, LOCAL_ROOT):
        if path.is_relative_to(root):
            return SHARED_ROOT / path.relative_to(root)
    raise ValueError('QA assets must stay on retained18 shared storage')


def binding(config):
    spec = config['qa_standalone']
    outer, count = spec['outer'], spec['expected_count']
    if (type(outer) is not int or outer not in (5, 50, 100, 150, 200)
            or type(count) is not int or count not in (512, 51713)
            or count == 51713 and outer != 200):
        raise ValueError('Only saved milestones or actor200 full51713 are authorized')
    if config['run_id'] != RUN_ID or config['node'] not in NODES:
        raise ValueError('QA run/node differs from the authorized H100 run')
    base = Path(spec['base'])
    if base.parent not in (SHARED_ROOT, LOCAL_ROOT) or config['base'] != str(base):
        raise ValueError('QA namespace is not the configured retained18 directory')
    if not re.fullmatch('search-g1-qa-i1i6-4a1cc2d5-outer%04d-[a-z0-9-]+' % outer, base.name):
        raise ValueError('QA directory does not identify its saved actor outer')
    unit = spec['unit']
    if not re.fullmatch('g1-qa-4a1cc2d5-o%d-[a-z0-9-]+[.]service' % outer, unit):
        raise ValueError('QA systemd unit does not identify its saved actor outer')
    if count == 51713 and ('-final-' not in base.name or '-final-' not in unit):
        raise ValueError('Final QA needs a distinct namespace and unit from dev512')
    if config['actor_version'] != RUN_ID + '-actor%04d' % outer:
        raise ValueError('QA actor version differs from its milestone')
    uuids = config['physical_gpu_uuids']
    if len(uuids) != 8 or len(set(uuids)) != 8:
        raise ValueError('QA requires eight distinct physical GPU bindings')
    key = 'validation_questions' if count == 512 else 'final_questions'
    retained_path(spec['model_dir'])
    retained_path(spec['source_cache'])
    retained_path(spec['source_evaluation_config'])
    return dict(outer=outer, expected_count=count, questions=config[key],
                questions_sha256=config[key + '_sha256'], unit=unit, base=str(base))


def verify_sources(config, read, sha):
    """Verify actual immutable files before allocating any model/GPU resources."""
    target = binding(config)
    spec = config['qa_standalone']
    source_path = Path(spec['source_evaluation_config'])
    if sha(source_path) != spec['source_evaluation_config_sha256']:
        raise ValueError('Original actor evaluation config changed')
    source = read(source_path)
    for key in ('run_id', 'actor_version', 'interaction_contract', 'validation_questions_sha256', 'final_questions_sha256'):
        if source[key] != config[key]:
            raise ValueError('Original actor evaluation identity changed: ' + key)
    if sha(target['questions']) != target['questions_sha256']:
        raise ValueError('Canonical QA questions changed')
    cache = Path(spec['source_cache'])
    cache_config = read(cache.parent / 'preparation/launch-config.json')
    if cache_config['run_id'] != config['run_id']:
        raise ValueError('Retriever cache belongs to another logical run')
    for key in ('retrieval_corpus', 'retrieval_index', 'retrieval_encoder'):
        if retained_path(cache_config[key]) != retained_path(config[key]):
            raise ValueError('Retriever assets differ from the original cache')
    # Keep the already accepted QA5 namespace loadable. New milestones/final
    # require the exact joint receipt, not merely an actor-version string.
    joint_path = spec.get('source_joint_receipt')
    if joint_path is None:
        if target['outer'] != 5 or target['expected_count'] != 512:
            raise ValueError('Later QA requires its actual complete joint receipt')
    else:
        joint_path = retained_path(joint_path)
        if sha(joint_path) != spec['source_joint_receipt_sha256']:
            raise ValueError('Saved joint receipt changed')
        joint = read(joint_path)
        expected_path = retained_path(source['base']) / 'outers' / ('outer-%06d' % target['outer']) / 'complete.json'
        if joint_path != expected_path or joint['run_id'] != RUN_ID or joint['outer'] != target['outer']:
            raise ValueError('Joint receipt does not identify the original saved milestone')
        actor, probe = joint['actor'], joint['probe']
        if (actor['actor_version'] != config['actor_version']
                or retained_path(actor['hf_model_dir']) != retained_path(spec['model_dir'])
                or actor['actual_adam_steps_per_rank'] != [20 * target['outer']] * 8
                or probe['actual_adam_steps_per_rank'] != [20 * target['outer']] * 8):
            raise ValueError('Saved model or actual actor/probe update boundary differs')
    return target
