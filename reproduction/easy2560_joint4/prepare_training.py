"""Bind a fresh actor, Easy candidate stream, and no question-selection head."""
import argparse
import json
import os
from pathlib import Path
import time

from .control import BASE, MODEL, PREDECESSOR, MICROS, own_base, publish


def selection_settings(predecessor, physical):
    if physical not in MICROS: raise ValueError('EASY2560_UNSUPPORTED_PHYSICAL_MICRO')
    return dict(easy2560_selection=True, data_selection_mode='mixed_first', mixed_first_selection=True,
                selection_candidate_questions=2560, selection_selected_questions=512,
                selection_fixed_candidate_budget=True,
                selection_eligible_pool_path=str(Path(predecessor) / 'assets/prefix20/eligible-question-pool.json'),
                actor_micro_batch_per_rank=physical, actor_logical_micro_batch_per_rank=4,
                all_correct_virtual_zero=False, startup_admission_mode='actual_first_five',
                initial_reward_probe_source_base=str(predecessor), personal_storage_budget_bytes=2 * 10**12)


def prepare(base, commit, actor_micro=32):
    base = own_base(base)
    if os.environ.get('SEARCH_G1_TRAINING_WORLD_SIZE') != '4' or not os.environ.get('SLURM_JOB_ID', '').isdecimal():
        raise ValueError('EASY2560_ACTUAL_NATIVE4_SLURM_REQUIRED')
    submitted = json.loads((base / 'preparation/training-submit-claim.json').read_text())
    if submitted['source_commit'] != commit or submitted['actor_micro_batch_per_rank'] != actor_micro:
        raise ValueError('EASY2560_SUBMITTED_SOURCE_OR_MICRO_DIFFERS')
    reuse = json.loads((base / 'preparation/environment-reuse.json').read_text())
    predecessor = reuse['predecessor']
    manifest = base / 'preparation/official-model-manifest.json'
    if not manifest.is_file():
        from reproduction.hpc3_qprobe.model_manifest import create
        create(Path(MODEL), manifest)
    from reproduction.common8.prepare_run import prepare as prepare_actor
    run = prepare_actor(base, Path(MODEL), manifest, commit,
                        rollout_memory_utilization=.1,
                        local_retrieval_assets='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/source/assets-root/assets/wiki',
                        execution_site='hpc3', actor_micro_batch_per_rank=actor_micro,
                        actor_logical_micro_batch_per_rank=4, easy2560_selection=True)
    run.update(selection_settings(predecessor, actor_micro))
    run.pop('question_probe', None)
    from reproduction.common8.formal_contract import SKIP_INITIAL_EVALUATION_POLICY
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE, validate
    run['pretraining_baseline_policy'] = SKIP_INITIAL_EVALUATION_POLICY
    run['infra_rollout'] = validate(dict(DEFAULT_ROLLOUT_PHASE), 4)
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    run['retriever_version'] = sha_json(ASSETS)
    target = base / 'logs/run-manifest.json'
    staging = target.with_name('run-manifest.easy2560.partial')
    with staging.open('x') as out:
        json.dump(run, out, indent=2); out.write('\n'); out.flush(); os.fsync(out.fileno())
    staging.replace(target)
    from reproduction.common8.configuration import resolved
    _, config, _ = resolved(base)
    if config.data.selection_mode != 'mixed_first' or config.algorithm.all_correct_virtual_zero:
        raise ValueError('EASY2560_RESOLVED_SELECTION_OR_VIRTUAL0_DIFFERS')
    proof = dict(stage='easy2560_native4_configuration_prepared', run_id=run['run_id'],
                 source_commit=commit, slurm_job_id=os.environ['SLURM_JOB_ID'],
                 target_outers=200, training_world_size=4,
                 actor_micro_batch_per_rank=actor_micro, actor_logical_micro_batch_per_rank=4,
                 gradient_accumulation_steps=32 // actor_micro,
                 question_selection_probe_enabled=False, teacher_warmup_updates=0,
                 all_correct_virtual_zero=False, candidate_questions=2560, selected_questions=512,
                 actual_training_progress_pending=True, checked_unix=time.time())
    publish(base / 'logs/easy2560-configuration-prepared.json', proof)
    return proof


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', default=BASE)
    p.add_argument('--commit', required=True)
    p.add_argument('--actor-micro', type=int, choices=MICROS, default=32); a = p.parse_args()
    print(json.dumps(prepare(a.base, a.commit, a.actor_micro)), flush=True)


if __name__ == '__main__': main()
