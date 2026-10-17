"""Prepare two reference-equivalent four-rank slots; never submit training.

Run on H100 using the already copied, SHA-verified reference assets. The user
will assign experiments later. Configurations are templates, not launch-config
files, and no optimizer, actor, probe or CUDA context is created here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from .h100_replica_hardware import LAYOUT, PREFIX, validate as validate_layout

REFERENCE_RUNTIME = '75d536575593891876df339467da73b3c4cbdc7c'
REFERENCE_REPAIR = 'd3a03054009fe3bf0068dfefabca941df95df101'
REFERENCE_THREAD = '01a1488e-807d-7ca3-b024-96b8fe97486c'
WARMUP_SHA = 'b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def prepare(assets, output, source_commit, infrastructure):
    assets, output = Path(assets), Path(output)
    complete = json.loads((assets / 'copy-complete.json').read_text())
    if complete['status'] != 'REFERENCE_ASSETS_REMOTE_COPY_SOURCE_TARGET_FULL_SHA_PASS':
        raise ValueError('Use actual reference source/target full-SHA copy completion')
    rows = {r['role']: r for r in complete['files']}
    source_row = rows['probe_source_checkpoint']
    if source_row['independent_target_sha256'] != WARMUP_SHA:
        if (source_row.get('original_full_checkpoint_sha256') != WARMUP_SHA
                or source_row.get('transport_model_only') is not True):
            raise ValueError('Use the reference epoch5 EMA, with explicit model-only transport provenance')
        # Do not transport raw/Adam/RNG that online initialization never imports.
        # Compare actual CPU tensor bytes with the reference extraction proof.
        import torch
        from .prefix5_model import warmup_model_weights
        saved = torch.load(source_row['target'], map_location='cpu', weights_only=False)
        model, buffers, _ = warmup_model_weights(saved, epoch=5, model_kind='ema',
            source_run='8af926a49630428097b94027864da68b')
        actual = {prefix + '.' + k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                  for prefix, values in [('model', model), ('buffers', buffers)] for k, v in values.items()}
        if actual != source_row['model_and_buffer_sha256']:
            raise ValueError('Transported EMA tensors differ from the reference checkpoint')
        del saved, model, buffers
    ref = json.loads(Path(rows['reference_launch_config']['target']).read_text())
    for role, key in [('canonical_train', 'canonical_train_sha256'),
                      ('validation_questions', 'validation_questions_sha256'),
                      ('final_questions', 'final_questions_sha256'),
                      ('common_manifest_path', 'common_manifest_sha256'),
                      ('composite_manifest_path', 'composite_manifest_sha256'),
                      ('source_question_pool', 'source_question_pool_sha256'),
                      ('fixed_test_identity_path', 'fixed_test_identity_sha256'),
                      ('grounding_calibration_questions', 'grounding_calibration_sha256')]:
        if rows[role]['independent_target_sha256'] != ref[key]:
            raise ValueError('Copied reference data identity differs: ' + role)
    copied = {k: r['target'] for k, r in rows.items()}
    original_grounding = json.loads(Path(copied['grounding_manifest']).read_text())
    # Only physical file paths change. Keep original head bytes, labels, layer
    # choices, encoder identities, versions and calibration protocol.
    grounding = copy.deepcopy(original_grounding)
    grounding.update(encoder=infrastructure['model_dir'],
        boundary=copied['grounding_boundary'], commit=copied['grounding_commit'])
    mapped_manifest = assets / 'grounding-manifest-h100-paths.json'
    if not mapped_manifest.exists():
        write(mapped_manifest, grounding)
    elif json.loads(mapped_manifest.read_text()) != grounding:
        raise ValueError('Existing mapped grounding manifest differs')
    binding_only = assets / 'warmup-feature-binding-only'
    binding_only.mkdir(exist_ok=True)
    binding = dict(role='warmup_checkpoint_original_feature_identity_only_not_a_tensor_cache',
        feature_manifest_sha256=ref['fixed_test_manifest_sha256'],
        actual_periodic_diagnostic='fresh_same_actor_fixed512_G5_generated_by_original_runtime')
    if not (binding_only / 'binding.json').exists():
        write(binding_only / 'binding.json', binding)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    gpus = {int(row.split(',')[0]): row.split(',')[1].strip() for row in gpu_rows}
    configs, comparison = {}, {}
    for slot, cards, port in [('A', [0, 1, 2, 3], 18410), ('B', [4, 5, 6, 7], 18510)]:
        c = copy.deepcopy(ref)
        for key in ('resume_joint', 'resume_import', 'resume_pending_probe', 'continue_reuse',
                    'source_compatibility', 'source_compatibility_sha256',
                    'candidate_schedule_authorization', 'candidate_schedule_authorization_sha256',
                    'candidate_schedule_delta', 'infrastructure_repair'):
            c.pop(key, None)
        run_id = uuid.uuid4().hex
        base = PREFIX + slot.lower() + '-20261018-' + run_id[:12]
        c.update(base=base, run_id=run_id, execution_id=uuid.uuid4().hex,
            source_commit=source_commit, actor_version=run_id + '-official0',
            pool_state_path=base + '/question-pool/state.json',
            prefix_probe_receipt=base + '/probe/latest.json',
            physical_gpu_indices=cards, auxiliary_gpu_indices=cards,
            physical_gpu_uuids=[gpus[i] for i in cards], auxiliary_gpu_uuids=[gpus[i] for i in cards],
            grounding_physical_gpu=cards[-1], grounding_refit_visible_device=cards[-1],
            grounding_physical_gpu_by_rank=cards,
            execution_hardware_layout=LAYOUT, retrieval_gpu_shards=4,
            gpu_uuid_lock_root='/mnt/data/shaoshitong/.search-g1-gpu-uuid-locks',
            shared_existing_foreign_jobs_authorized=True,
            environment_root=infrastructure['environment_root'],
            model_dir=infrastructure['model_dir'], official_model_dir=infrastructure['model_dir'],
            encoder_model_dir=infrastructure['model_dir'], actor_model_dir=infrastructure['model_dir'],
            retrieval_index=infrastructure['retrieval_index'], retrieval_corpus=infrastructure['retrieval_corpus'],
            retrieval_encoder=infrastructure['retrieval_encoder'], retriever_gpu_lib=infrastructure['retriever_gpu_lib'],
            retrieval_data_cache=base + '/retrieval-cache', retrieval_port=port,
            search_url='http://127.0.0.1:%d/retrieve' % port, grounding_probe_base_port=port + 10,
            fixed_test_features=str(binding_only),
            probe_source_checkpoint=copied['probe_source_checkpoint'],
            probe_source_sha256=source_row['independent_target_sha256'],
            probe_source_bytes=source_row['bytes'], original_warmup_checkpoint_sha256=WARMUP_SHA,
            probe_source_cpu_proof=copied['probe_source_cpu_proof'],
            canonical_train=copied['canonical_train'], validation_questions=copied['validation_questions'],
            final_questions=copied['final_questions'], source_question_pool=copied['source_question_pool'],
            fixed_test_identity_path=copied['fixed_test_identity_path'],
            common_manifest_path=copied['common_manifest_path'], composite_manifest_path=copied['composite_manifest_path'],
            grounding_calibration_questions=copied['grounding_calibration_questions'],
            grounding_manifest=str(mapped_manifest), grounding_manifest_sha256=sha(mapped_manifest),
            grounding_initial_manifest_sha256=sha(mapped_manifest),
            fresh_actor_optimizer_scheduler_rng_sampler=True, retained_warmup_model_only=True,
            full_state_resume=False, old_actor_state_imported=False,
            old_online_probe_optimizer_imported=False, science_source_commit=source_commit,
            resolve_idle_auxiliary_uuid_at_prepare=False,
            probe_source_inode=Path(copied['probe_source_checkpoint']).stat().st_ino,
            reference_thread=REFERENCE_THREAD,
            reference_runtime_commit=REFERENCE_RUNTIME, reference_infrastructure_repair=REFERENCE_REPAIR,
            experiment_assignment_pending=True, preparation_only=True)
        from .launch import validate_config
        validate_layout(c)
        validate_config(c, Path(base))
        # All changed values are explicit. The review compares scientific
        # settings and source functions, not an undocumented filtered config.
        delta = {k: dict(reference=ref.get(k), replica=c.get(k))
                 for k in sorted(set(ref) | set(c)) if ref.get(k) != c.get(k)}
        write(output / 'slots' / slot / 'config.template.json', c)
        write(output / 'slots' / slot / 'exact-config-delta.json', delta)
        configs[slot] = str(output / 'slots' / slot / 'config.template.json')
        comparison[slot] = dict(config_sha256=sha(Path(configs[slot])), changed_fields=sorted(delta),
            training_cards=cards, retrieval_cards=cards, experiment_assignment_pending=True)
    result = dict(status='TWO_FOUR_CARD_REFERENCE_TEMPLATES_PREPARED_NO_TRAINING_SUBMITTED',
        reference_thread=REFERENCE_THREAD, reference_runtime=REFERENCE_RUNTIME,
        reference_cache_repair=REFERENCE_REPAIR, source_commit=source_commit,
        source_config_sha256=rows['reference_launch_config']['independent_target_sha256'],
        original_warmup_sha256=WARMUP_SHA,
        transported_model_checkpoint_sha256=source_row['independent_target_sha256'],
        assets_copy_proof=str(assets / 'copy-complete.json'),
        slots=comparison, configs=configs, training_submitted=False,
        algorithm_audit_pending=True)
    write(output / 'prepared.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--infrastructure', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.assets, a.output, a.source_commit,
        json.loads(Path(a.infrastructure).read_text()))))


if __name__ == '__main__':
    main()
