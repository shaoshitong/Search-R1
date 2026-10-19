"""Prepare private HPC3 paths using already qualified shared assets. Never run RL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

BASE = '/data/user/jxu957/experiments/dynrpo-reference-b-thread5-prepared-20261019'
PREVIOUS = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4')
SHARED_PYTHON = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10'
SHARED_LIB = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/lib'
SHARED_PACKAGES = '/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/env/lib/python3.10/site-packages'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(raw)
    return h.hexdigest()


def prepare(base):
    b = Path(base).resolve()
    assert str(b) == BASE
    assert not any((b / p).exists() for p in ('actor', 'probe', 'outers', 'latest-joint.json'))
    if (b / 'preparation/prepared-config.json').exists():
        c = json.loads((b / 'preparation/prepared-config.json').read_text())
        assert c['preparation_only'] and c['training_authorized'] is False
        return dict(status='EXISTING_PREPARATION_RECONCILED_NO_RESTART', base=str(b))
    previous = json.loads((PREVIOUS / 'preparation/launch-config.json').read_text())
    source = b / 'repo/reproduction/operations/reference_b_hpc3_prepared/reference-config.json'
    c = json.loads(source.read_text())
    asset_keys = ['model_dir', 'official_model_dir', 'actor_model_dir', 'encoder_model_dir',
                  'canonical_train', 'validation_questions', 'final_questions',
                  'source_question_pool', 'fixed_test_identity_path', 'fixed_test_features',
                  'grounding_calibration_questions', 'grounding_manifest',
                  'common_manifest_path', 'composite_manifest_path',
                  'probe_source_checkpoint', 'retrieval_index', 'retrieval_corpus', 'retrieval_encoder']
    assert c['model_files_sha256'] == previous['model_files_sha256']
    assert c['model_revision'] == previous['model_revision']
    assert c['common_manifest_sha256'] == previous['common_manifest_sha256']
    for key in asset_keys:
        assert Path(previous[key]).exists(), key
        c[key] = previous[key]
    c['probe_source_sha256'] = sha(c['probe_source_checkpoint'])
    st = Path(c['probe_source_checkpoint']).stat()
    c['probe_source_inode'], c['probe_source_bytes'] = st.st_ino, st.st_size
    for key in ('grounding_manifest_sha256', 'grounding_initial_manifest_sha256'):
        c[key] = sha(c['grounding_manifest'])
    for key in ('resume_joint', 'resume_cached_sampling', 'startup_sampling_resume',
                'runtime_compatibility', 'actor_checkpoint_binding_commit',
                'physical_gpu_uuids', 'auxiliary_gpu_uuids', 'expected_execution_node'):
        c.pop(key, None)
    publication = json.loads((b / 'preparation/source-publication.json').read_text())
    c.update(base=str(b), run_id=uuid.uuid4().hex, execution_id=None,
             source_commit=publication['commit'],
             observed_reference_runtime='bc8b16728b689ccbb97457e06f2ef0a65cb39fa7',
             reference_thread='01a0fcbc-0b46-7502-8b66-d8b21a0e032d',
             preparation_only=True, training_authorized=False,
             experiment_assignment_pending=False, initialization_assignment_pending=False,
             planned_initialization='official_actor_and_epoch5_ema_model_only',
             initial_student_outer=0, online_probe_adam_initial=0,
             old_actor_state_imported=False, old_online_probe_optimizer_imported=False,
             full_state_resume=False, checkpoint_imported=False,
             physical_gpu_uuids=[], auxiliary_gpu_uuids=[],
             physical_uuid_binding_pending=True, execution_backend='hpc3_prepared_only',
             environment_root=str(b / 'env'), pool_state_path=str(b / 'question-pool/state.json'),
             prefix_probe_receipt=str(b / 'probe/latest.json'),
             retrieval_data_cache=previous['retrieval_data_cache'],
             retriever_gpu_lib=str(b / 'retriever-gpu-lib'),
             retrieval_port=18860, grounding_probe_base_port=18870,
             search_url='http://127.0.0.1:18860/retrieve',
             gpu_uuid_lock_root=str(b / 'gpu-uuid-locks'),
             shared_existing_foreign_jobs_authorized=False,
             actor_version='official0-planned-not-loaded',
             b_progressive_schedule_note='Boundary50 schedules Prefix20; no CP50 import in this fresh planned run')
    for directory in ('env/bin', 'cache', 'tmp', 'logs', 'preparation'):
        (b / directory).mkdir(parents=True, exist_ok=True)
    for src, dst in [(PREVIOUS / 'env/overlay', b / 'env/overlay'),
                     (Path(previous['retriever_gpu_lib']), b / 'retriever-gpu-lib')]:
        assert src.is_dir() and not dst.exists()
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assets = {k: c[k] for k in asset_keys}
    small_sha = {}
    for key in ('canonical_train', 'validation_questions', 'final_questions',
                'source_question_pool', 'fixed_test_identity_path', 'grounding_calibration_questions',
                'grounding_manifest', 'common_manifest_path', 'composite_manifest_path'):
        small_sha[key] = sha(c[key])
    for key in ('canonical_train', 'validation_questions', 'final_questions',
                'source_question_pool', 'fixed_test_identity_path', 'grounding_calibration_questions'):
        expected_key = {'fixed_test_identity_path': 'fixed_test_identity_sha256',
                        'grounding_calibration_questions': 'grounding_calibration_sha256'}.get(key, key + '_sha256')
        assert small_sha[key] == c[expected_key], key
    c.update(asset_paths=assets, small_asset_sha256=small_sha)
    # Runtime packages are reused read-only; private overlay/code/caches are independent.
    wrapper = '''#!/usr/bin/env bash
set -euo pipefail
export SEARCH_G1_PHYSICAL_WORLD_SIZE=8
export SEARCH_G1_B_PREFIX_TOKENS=5
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="__BASE__/repo:__BASE__/env/overlay:__BASE__/retriever-gpu-lib:__PACKAGES__"
export LD_LIBRARY_PATH="__LIB__${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="__BASE__/cache/huggingface"
export TORCH_EXTENSIONS_DIR="__BASE__/cache/torch_extensions"
export TMPDIR="__BASE__/tmp"
unset PYTHONHOME
exec __PYTHON__ "$@"
'''.replace('__BASE__', str(b)).replace('__PACKAGES__', SHARED_PACKAGES).replace('__LIB__', SHARED_LIB).replace('__PYTHON__', SHARED_PYTHON)
    f = b / 'env/bin/python-cpu-check'
    f.write_text(wrapper)
    f.chmod(0o755)
    (b / 'preparation/prepared-config.json').write_text(json.dumps(c, sort_keys=True, indent=2) + '\n')
    result = dict(status='REFERENCE_B_HPC3_PREPARED_NOT_STARTED', base=str(b),
                  prepared_unix=time.time(), source_commit=publication['commit'],
                  reference_runtime=c['observed_reference_runtime'], world_size=8,
                  training_authorized=False, initialization=c['planned_initialization'],
                  actual_optimizer_created=False, checkpoint_imported=False,
                  slurm_submitted=False, physical_gpu_binding_pending=True,
                  config_sha256=sha(b / 'preparation/prepared-config.json'))
    (b / 'preparation/prepared-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base)))


if __name__ == '__main__':
    main()
