"""Same-run frozen-A latest joint import; no submit, signal or live edit."""
import os
import uuid
from pathlib import Path

from reproduction.prefix20_online.launch import read, sha, once
from .ab_policy import FROZEN
from .b_progressive_resume import (
    PROTECTED, stopped, pending_pool, _import_complete, _validate_imported,
)

MODE = 'same_run_frozen_a_latest_joint_v1'


def enabled(config):
    return config.get('resume_joint', {}).get('recovery_mode') == MODE


def qualify(config, source, proof, joint):
    n = config['resume_joint']['completed_outer']
    if (not enabled(config) or config.get('selection_probe_experiment') != FROZEN
            or type(n) is not int or not 1 <= n <= 200
            or joint != read(Path(source)/'latest-joint.json')
            or joint['run_id'] != config['run_id'] or joint['outer'] != n
            or proof['status'] != 'AB_JOINT_FULL_SHA_CPU_PASS'
            or proof['run_id'] != config['run_id'] or proof['outer'] != n
            or proof['actual_cpu_deserialization'] is not True
            or joint['actor']['actual_adam_steps_per_rank'] != proof['actor_actual_adam_steps_per_rank']
            or joint['probe']['actual_adam_steps_per_rank'] != [0]*4
            or proof['probe_actual_adam_steps_per_rank'] != [0]*4):
        raise ValueError('Require THIS frozen-A actual latest full-state CPU qualification, with probe0')
    return n


def verify(config):
    from .i1_i6_joint_resume import source_manifest
    if (not enabled(config) or config.get('selection_probe_experiment') != FROZEN
            or config.get('b_progressive') or config['prefix_policy_tokens'] != 5):
        raise ValueError('Frozen-A recovery cannot import the progressive-B protocol')
    path = config['source_compatibility']; proof = read(path)
    if (sha(path) != config['source_compatibility_sha256'] or proof['mode'] != MODE
            or config['source_commit'] != proof['runtime_source_commit']
            or config['science_source_commit'] != proof['science_source_commit']):
        raise ValueError('Frozen-A source publication binding differs')
    actual = source_manifest(Path(config['base'])/'repo', config['source_commit'],
                             science_source_commit=config['science_source_commit'])
    if actual != proof['target_publication']:
        raise ValueError('Actual prepared source differs from the published target inventory')
    for name in PROTECTED:
        rel = 'reproduction/prefix_selection200/'+name
        if proof['parent_publication']['files_sha256'][rel] != actual['files_sha256'][rel]:
            raise ValueError('Unrequested actor/reward/grounding/probe math change: '+rel)
    return proof


def plan(base, config=None):
    base = Path(base); config = config or read(base/'preparation/launch-config.json')
    verify(config); spec = config['resume_joint']; source = Path(spec['source_base'])
    stopped(source)
    original = read(source/'preparation/launch-config.json')
    if (original['run_id'] != config['run_id']
            or original.get('selection_probe_experiment') != FROZEN
            or sha(source/'preparation/launch-config.json') != spec['source_config_sha256']):
        raise ValueError('Frozen-A source execution changed')
    from .i1_i6_joint_resume import SCIENCE_CONFIG
    if any(config.get(key) != original.get(key) for key in SCIENCE_CONFIG):
        raise ValueError('Frozen-A scientific configuration differs from its source')
    n = spec['completed_outer']; path = source/'outers'/('outer-%06d'%n)/'complete.json'
    joint = read(path); proof = read(spec['qualification_proof'])
    if (sha(path) != spec['source_joint_sha256']
            or sha(spec['qualification_proof']) != spec['qualification_proof_sha256']):
        raise ValueError('Actual latest joint or once-only CPU qualification changed')
    qualify(config, source, proof, joint)
    for row in proof['files']:
        st = Path(row['path']).stat()
        if (st.st_size, st.st_ino, st.st_mtime_ns) != (row['bytes'], row['inode'], row['mtime_ns']):
            raise ValueError('Already full-SHA qualified payload changed: '+row['path'])
    return dict(status='A_LATEST_FULL_JOINT_QUALIFIED_READY', source_base=str(source),
                completed_outer=n, next_outer=n+1, source_joint_sha256=sha(path), phase_reuse={},
                pending_pool=pending_pool(source, original, joint, n),
                actor_adam=joint['actor']['actual_adam_steps_per_rank'], probe_adam=[0]*4,
                payload_hash_repeated=False, recovery_mode=MODE)


def validate_imported(base, config):
    return _validate_imported(base, config, plan)


def import_complete(base):
    return _import_complete(base, plan, validate_imported, verify, arm='A')


def prepare_latest(source, base, staged_repo, commit):
    """Only after actual stopped writers and latest full-SHA/CPU qualification."""
    from .i1_i6_joint_resume import source_manifest
    from .contract import PREFIX_TOKENS
    if PREFIX_TOKENS != 5 or os.environ.get('SEARCH_G1_B_PREFIX_TOKENS', '5') != '5':
        raise ValueError('Frozen A must use a fresh prefix5 Python process')
    source, base, staged_repo = map(Path, (source, base, staged_repo))
    stopped(source); original = read(source/'preparation/launch-config.json')
    n = read(source/'latest-joint.json')['outer']
    if original.get('selection_probe_experiment') != FROZEN or type(n) is not int or not 1 <= n <= 200:
        raise ValueError('Only this frozen-A same-run actual latest boundary may be imported')
    proof = source/'preparation'/('ab-joint-%04d-independent-full-sha-cpu.json'%n)
    # Qualify before creating a new directory or importing any payload reference.
    spec = dict(adapter='i1_i6_joint_v1', recovery_mode=MODE, source_base=str(source),
                completed_outer=n, source_commit=original['source_commit'],
                source_execution_id=original['execution_id'],
                source_config_sha256=sha(source/'preparation/launch-config.json'),
                qualification_proof=str(proof), qualification_proof_sha256=sha(proof),
                source_joint_sha256=sha(source/'outers'/('outer-%06d'%n)/'complete.json'))
    qualify(dict(original, resume_joint=spec), source, read(proof), read(source/'latest-joint.json'))
    if base.exists():
        raise ValueError('Existing destination intent must be reconciled, never replayed')
    base.mkdir(); (base/'preparation').mkdir(); (base/'logs').mkdir(); (base/'repo').mkdir()
    files = []
    for path in sorted(staged_repo.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        target = base/'repo'/path.relative_to(staged_repo); target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        if sha(path) != sha(target):
            raise ValueError('Target source publication differs')
        files.append(dict(path=path.relative_to(staged_repo).as_posix(), sha256=sha(target)))
    once(base/'preparation/source-full-sha.json', dict(commit=commit, files=files))
    def rebase(value):
        if isinstance(value, dict): return {key: rebase(v) for key, v in value.items()}
        if isinstance(value, list): return [rebase(v) for v in value]
        if isinstance(value, str) and value.startswith(str(source)): return str(base)+value[len(str(source)):]
        return value
    c = rebase(original)
    science = read(source/'latest-joint.json')['actor']['binding']['source_commit']
    c.update(base=str(base), execution_id=uuid.uuid4().hex, source_commit=commit,
             science_source_commit=science, resume_joint=spec)
    for key in ('startup_sampling_resume', 'resume_import', 'resume_pending_probe', 'node_migration',
                'source_compatibility', 'source_compatibility_sha256'):
        c.pop(key, None)
    compat = dict(mode=MODE, runtime_source_commit=commit, science_source_commit=science,
                  parent_publication=source_manifest(source/'repo', original['source_commit']),
                  target_publication=source_manifest(base/'repo', commit, science_source_commit=science),
                  authorized_candidate_delta=None, authorized_infrastructure_delta=None)
    path = base/'preparation/frozen-a-latest-source-compatibility.json'; once(path, compat)
    c.update(source_compatibility=str(path), source_compatibility_sha256=sha(path))
    once(base/'preparation/launch-config.json', c)
    verify(c); result = import_complete(base)
    return dict(status='FROZEN_A_LATEST_JOINT_PREPARED', base=str(base), run_id=c['run_id'],
                commit=commit, config_sha256=sha(base/'preparation/launch-config.json'),
                imported_boundary=result['completed_outer'], next_outer=n+1,
                required_process_environment={'SEARCH_G1_B_PREFIX_TOKENS': '5'},
                no_new_actor_probe_or_rng_initialization=True)
