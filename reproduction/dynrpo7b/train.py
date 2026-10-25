"""Bind the independent DynRPO recipe and enter the mature native4 pipeline."""
import argparse
import json
import os
from pathlib import Path

from . import recipe
from .deploy import publish, sha
from .infra_profile import BASE, PROFILE, MODEL, WIKI, settings, own_base


def run_settings(base, actor_micro=8):
    from reproduction.common8.formal_contract import SKIP_INITIAL_EVALUATION_POLICY
    value = settings(actor_micro)
    value.update(question_probe=recipe.probe_settings(base),
        mixed_first_selection=False, all_correct_virtual_zero=False,
        objective_revision="cp50_mathfix_v1", all_correct_reward_one=True,
        selection_allow_all_wrong=False,
        selection_candidate_questions=recipe.CANDIDATES,
        selection_selected_questions=recipe.SELECTED,
        selection_fixed_candidate_budget=True,
        selection_prefix_tokens=80, selection_prefix_after_100=160,
        probe_update_every_outer=10, probe_update_epochs=3,
        probe_update_candidate_questions=recipe.UPDATE_CANDIDATES,
        probe_update_rollout_source='reuse_preupdate_actor_G5',
        startup_admission_mode='actual_first_five',
        pretraining_baseline_policy=SKIP_INITIAL_EVALUATION_POLICY,
        initial_reward_probe_source_base=recipe.REWARD_SOURCE,
        easy2560_rolling_checkpoint=True, save_interval=1,
        checkpoint_milestone_steps=list(recipe.MILESTONES))
    return value


def prepare_only(base, commit, actor_micro=8):
    """CPU prospectus only; never create a compute-node Ray root on login."""
    from reproduction.easy2560_joint4.model7b import expected
    base = own_base(base)
    value = dict(source_commit=commit, base=str(base), model=MODEL,
        official_model=expected(), settings=run_settings(base, actor_micro),
        allocation_preparation_pending=True, training_submitted=False,
        ray_created=False, gpu_inference_performed=False)
    publish(base / 'preparation/dynrpo7b-cpu-recipe.json', value)
    return value


def validate_retriever_scope(base, external, restore_receipt=None):
    """A resumed run can retain its own original2-GPU producer unchanged.

    external_binding already verifies the real producer ready/stage64 bytes.
    This reads only the small restore header; bind_run subsequently validates
    the full native4 restore receipt. Fresh runs still require a local service.
    """
    ready=Path(external['service_ready_path'])
    if ready.is_relative_to(Path(base)/'retrieval'):
        return external
    if restore_receipt:
        header=json.loads(Path(restore_receipt).read_text())
        if (header.get('kind')=='dynrpo7b_native4' and header.get('status') in (
                'DYNRPO7B_ROLLING_READY','DYNRPO7B_ACTOR_SAVED_CALLBACKS_PENDING')):
            origin=Path(header['source_base'])
            if origin.is_absolute() and ready.is_relative_to(origin/'retrieval'):
                return external
            if origin.is_absolute():
                # A second resume's direct parent may itself reuse the first
                # producer. Preserve that parent's exact binding, rather than
                # pretending the producer's READY moved into the new base.
                parent_path=origin/'logs/run-manifest.json'
                if parent_path.is_file():
                    expected=json.loads(parent_path.read_text()).get('external_retriever',{})
                    keys=('service_ready_path','service_job_id','service_ready_sha256','endpoint')
                    if all(expected.get(key)==external.get(key) and key in expected for key in keys):
                        return external
    raise ValueError('Use this run\'s independent retrieval service or its bound restore source producer')


def prepare(base, commit, actor_micro=8, restore_receipt=None,
            external_retriever_receipt=None):
    base = own_base(base)
    target = base / 'logs/run-manifest.json'
    if target.exists():
        run = json.loads(target.read_text())
        if not run.get('dynrpo7b_selection') or run['source_commit'] != commit:
            raise ValueError('Existing run manifest belongs to another recipe/source')
        return run
    if not os.environ.get('SLURM_JOB_ID', '').isdecimal():
        raise ValueError('Actual run preparation requires its real Slurm allocation')
    if not external_retriever_receipt:
        raise ValueError('An independent ready 2-GPU retrieval receipt is required')
    from reproduction.easy2560_joint4.model7b import expected
    manifest = base / 'preparation/official-model-manifest.json'
    if not manifest.exists():
        publish(manifest, expected())
    from reproduction.easy2560_joint4.retrieval_proxy import external_binding
    external = external_binding(external_retriever_receipt)
    validate_retriever_scope(base,external,restore_receipt)
    values = run_settings(base, actor_micro)
    values['external_retriever'] = external
    from reproduction.easy2560_joint4.flash3_runtime import binding
    values['infra_flash_attention_3'] = binding(base)
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    values['retriever_version'] = sha_json(ASSETS)
    from reproduction.common8.prepare_run import prepare as bind_run
    run = bind_run(base, Path(MODEL), manifest, commit,
        rollout_memory_utilization=.1, local_retrieval_assets=WIKI,
        execution_site='hpc3', actor_micro_batch_per_rank=actor_micro,
        actor_logical_micro_batch_per_rank=4, dynrpo7b_settings=values,
        restore_receipt=restore_receipt)
    # A CPU Hydra resolution catches inconsistent recipe overrides before model
    # initialization, while avoiding synthetic CUDA/gradient comparison jobs.
    from reproduction.common8.configuration import resolved
    _, config, overrides = resolved(base)
    publish(base / 'preparation/dynrpo7b-configured.json', dict(
        run_id=run['run_id'], source_commit=commit,
        infra_profile=PROFILE, training_gpus=4, retrieval_gpus=2,
        actor_physical_micro=actor_micro, actor_logical_micro=4,
        prefix_tokens=[80, 160], prefix_switch_after_outer=100,
        probe_every=10, probe_epochs=3, prepared=True,
        actual_training_progress_pending=True))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--actor-micro', type=int, default=8, choices=(4, 8, 16, 32))
    parser.add_argument('--restore-receipt')
    parser.add_argument('--external-retriever-receipt')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare_only(args.base, args.commit, args.actor_micro)), flush=True)
        return
    run = prepare(args.base, args.commit, args.actor_micro, args.restore_receipt,
                  args.external_retriever_receipt)
    from .infra_profile import configure_ray
    configure_ray(run)
    from .pipeline import launch
    launch(own_base(args.base), run)


if __name__ == '__main__':
    main()
