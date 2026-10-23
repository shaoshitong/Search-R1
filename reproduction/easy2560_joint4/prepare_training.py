"""Bind the fresh or explicitly restored Easy run, without a selection head."""
import argparse
import json
import os
from pathlib import Path
import time

from .control import BASE, MODEL, PREDECESSOR, MICROS, own_base, publish, restore_binding
from search_g1.easy7b_infra import LEAN_PROFILES, FA3_PROFILES


def selection_settings(predecessor, physical, restoring=False, infra_profile='baseline'):
    from search_g1.easy7b_infra import profile_flags
    infra = profile_flags(infra_profile)
    if physical not in MICROS: raise ValueError('EASY2560_UNSUPPORTED_PHYSICAL_MICRO')
    old = Path(predecessor)
    previous = old / 'logs/run-manifest.json'
    pool = (json.loads(previous.read_text()).get('selection_eligible_pool_path')
            if previous.is_file() else None)
    settings = dict(easy2560_selection=True, data_selection_mode='mixed_first', mixed_first_selection=True,
                selection_candidate_questions=2560, selection_selected_questions=512,
                selection_fixed_candidate_budget=True,
                selection_eligible_pool_path=pool or str(old / 'assets/prefix20/eligible-question-pool.json'),
                actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4,
                all_correct_virtual_zero=False, startup_admission_mode='actual_first_five',
                easy2560_rolling_checkpoint=True, save_interval=1,
                checkpoint_milestone_steps=[5, 50, 100, 150, 200],
                personal_storage_budget_bytes=2 * 10**12)
    # 7B must fit its own official-actor round0 labels. No inherited 3B heads.
    if infra_profile != 'baseline':
        settings['infra_optimizations'] = infra
    if infra_profile in LEAN_PROFILES:
        settings.update(performance_profile='lean_v1', extra_policy_monitor_forwards=False,
                        infra_profile=infra_profile)
    return settings


def prepare(base, commit, actor_micro=8, restore_receipt=None, infra_profile='baseline',
            external_retriever_receipt=None):
    base = own_base(base)
    if os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') != '4' or not os.environ.get('SLURM_JOB_ID', '').isdecimal():
        raise ValueError('EASY2560_ACTUAL_NATIVE4_SLURM_REQUIRED')
    submitted = json.loads((base / 'preparation/training-submit-claim.json').read_text())
    if submitted.get('infra_profile', 'baseline') != infra_profile:
        raise ValueError('EASY2560_SUBMITTED_INFRA_PROFILE_DIFFERS')
    from .retrieval_proxy import external_binding
    external = external_binding(external_retriever_receipt)
    if submitted.get('external_retriever') != external:
        raise ValueError('EASY2560_SUBMITTED_EXTERNAL_RETRIEVER_DIFFERS')
    if submitted['source_commit'] != commit or submitted['actor_micro_batch_per_rank'] != actor_micro:
        raise ValueError('EASY2560_SUBMITTED_SOURCE_OR_MICRO_DIFFERS')
    restoration = restore_binding(restore_receipt)
    if any(submitted.get(key) != value for key, value in restoration.items()):
        raise ValueError('EASY2560_SUBMITTED_RESTORE_RECEIPT_DIFFERS')
    reuse = json.loads((base / 'preparation/environment-reuse.json').read_text())
    predecessor = reuse['predecessor']
    manifest = base / 'preparation/official-model-manifest.json'
    if not manifest.is_file():
        from .model7b import create
        create(Path(MODEL), manifest)
    settings = selection_settings(predecessor, actor_micro, restoring=bool(restore_receipt), infra_profile=infra_profile)
    if infra_profile in FA3_PROFILES:
        from .flash3_runtime import binding as flash3_binding
        settings['infra_flash_attention_3'] = flash3_binding(base)
    if external is not None:
        settings['external_retriever'] = external
    from reproduction.common8.formal_contract import SKIP_INITIAL_EVALUATION_POLICY
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE, validate
    settings['pretraining_baseline_policy'] = ('reuse_historical' if restore_receipt else
                                               SKIP_INITIAL_EVALUATION_POLICY)
    settings['infra_rollout'] = validate(dict(DEFAULT_ROLLOUT_PHASE), 4)
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    settings['retriever_version'] = sha_json(ASSETS)
    from reproduction.common8.prepare_run import prepare as prepare_actor
    run = prepare_actor(base, Path(MODEL), manifest, commit,
                        rollout_memory_utilization=.1,
                        local_retrieval_assets='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki',
                        restore_receipt=restoration['restore_receipt'],
                        execution_site='hpc3', actor_micro_batch_per_rank=actor_micro,
                        actor_logical_micro_batch_per_rank=4, easy2560_selection=True,
                        easy2560_settings=settings)
    from reproduction.common8.configuration import resolved
    _, config, _ = resolved(base)
    if config.data.selection_mode != 'mixed_first' or config.algorithm.all_correct_virtual_zero:
        raise ValueError('EASY2560_RESOLVED_SELECTION_OR_VIRTUAL0_DIFFERS')
    proof = dict(stage='easy2560_native4_configuration_prepared', run_id=run['run_id'],
                 source_commit=commit, slurm_job_id=os.environ['SLURM_JOB_ID'],
                 target_outers=200, training_world_size=4,
                 infra_profile=infra_profile,
                 external_retriever_receipt_sha256=external['receipt_sha256'] if external else None,
                 actor_micro_batch_per_rank=actor_micro, actor_logical_micro_batch_per_rank=4,
                 gradient_accumulation_steps=32 // actor_micro,
                 question_selection_probe_enabled=False, teacher_warmup_updates=0,
                 all_correct_virtual_zero=False, candidate_questions=2560, selected_questions=512,
                 restore_receipt=restoration['restore_receipt'],
                 restore_receipt_sha256=restoration['restore_receipt_sha256'],
                 restored_completed_outer=(run.get('full_state_restore') or {}).get('completed_outer'),
                 actual_cuda_restore_accepted=False,
                 rolling_full_checkpoint_every_outer=True,
                 rolling_retention='publish new complete boundary before retiring previous non-milestone',
                 actual_training_progress_pending=True, checked_unix=time.time())
    publish(base / 'logs/easy2560-configuration-prepared.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', default=BASE)
    p.add_argument('--commit', required=True)
    p.add_argument('--restore-receipt')
    p.add_argument('--external-retriever-receipt')
    from search_g1.easy7b_infra import PROFILES
    p.add_argument('--infra-profile', choices=PROFILES, default='baseline')
    p.add_argument('--actor-micro', type=int, choices=MICROS, default=8); a = p.parse_args()
    print(json.dumps(prepare(a.base, a.commit, a.actor_micro, a.restore_receipt, a.infra_profile,
                             a.external_retriever_receipt)), flush=True)


if __name__ == '__main__': main()
