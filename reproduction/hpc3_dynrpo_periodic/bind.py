"""Bind the continuous twenty-epoch probe to an independent four-card run."""
import os
import shutil

from .common import BASE, MODEL, REUSED, TEACHER, own, read, sha, publish, verify_source
from .warmup import PREVIOUS


def recipe():
    from reproduction.common8.qprobe_qwen_bind import recipe as original
    from search_g1.qprobe_periodic_model import HEAD_KIND
    result = original(PREVIOUS/'token-features')
    result.update(head_kind=HEAD_KIND,head_epochs_per_outer=0,replay_window_outer=0,
        bootstrap_head_epochs=20,refit_every_outer=25,refit_epochs=20,refit_groups=5120,
        online_student_features=True,inherit_probe_optimizer=True,
        output_dir=str(BASE/'artifacts/question-probe'),teacher_warmup_cache=str(TEACHER))
    return result


def main():
    own(BASE)
    import torch
    from reproduction.common8.prepare_run import prepare
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    from search_g1.qprobe_periodic_model import PeriodicQwenProbe, restore_engine
    from search_g1.qprobe_periodic_features import QuestionMetadata
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
    from reproduction.a800.owned_processes import publish as replace_manifest
    commit = os.environ['SEARCH_G1_SOURCE_COMMIT']
    verify_source(commit)
    if torch.cuda.is_initialized():
        raise ValueError('Periodic binding must occur before CUDA admission')
    if (BASE/'logs/run-manifest.json').exists():
        raise ValueError('Existing run binding must be resumed, never recreated')
    ready = read(BASE/'warmup/ready.json')
    raw = BASE/'warmup/continued-head.pt'
    if ready['head_adam_steps'] != 400 or sha(raw) != ready['head_sha256']:
        raise ValueError('Continuous twenty-epoch initial probe is not accepted')
    metadata = QuestionMetadata(PREVIOUS/'token-features')
    if metadata.spec_sha != ready['token_feature_spec_sha256']:
        raise ValueError('Canonical question metadata belongs to another representation')
    source_manifest = PREVIOUS/'preparation/official-model-manifest.json'
    manifest = BASE/'preparation/official-model-manifest.json'
    if not manifest.exists():
        shutil.copyfile(source_manifest,manifest)
    if sha(manifest) != sha(source_manifest):
        raise ValueError('Official model manifest changed')
    run = prepare(BASE,MODEL,manifest,commit,rollout_memory_utilization=.1,
        local_retrieval_assets=str(REUSED/'source/assets-root/assets/wiki'),execution_site='hpc3')
    saved = torch.load(raw,map_location='cpu',weights_only=True)
    head = PeriodicQwenProbe(restore_engine(saved['head'],epochs=20),metadata.spec_sha,
        actor_snapshot_id=run['run_id']+':0',phases=[dict(outer=0,fit_epochs=20,
        head_adam_steps_before=0,head_adam_steps_after=400,
        inherited_first_ten_epochs=True,initial_feature_manifest_sha256=ready['token_feature_manifest_sha256'])])
    directory = BASE/'artifacts/question-probe/continuous-warmup-adoption'
    directory.mkdir(parents=True,exist_ok=False)
    path = directory/'initial.pt'
    receipt = head.save(path,actor_snapshot_id=run['run_id']+':0')
    loaded = PeriodicQwenProbe.load(path,expected_actor_snapshot_id=run['run_id']+':0',
        expected_checkpoint_sha=receipt['sha256'],expected_feature_spec_sha=metadata.spec_sha)
    if head.version != loaded.version or loaded._adam_steps() != 400:
        raise ValueError('Actual continuous initial probe save/load failed')
    proof = directory/'ready.json'
    publish(proof,dict(stage='CONTINUOUS_NATIVE_WARMUP20_ADOPTED',head_adam_steps=400,
        head_sha256=receipt['sha256'],source_warmup_sha256=ready['head_sha256'],
        optimizer_reset=False,student_parameters_copied=False,actual_saved_state_readback=True),once=True)
    settings = recipe()
    settings.update(initial_head_checkpoint=str(path),initial_head_checkpoint_sha256=receipt['sha256'],
        initial_head_ready=str(proof),initial_head_ready_sha256=sha(proof))
    proposed = dict(run,question_probe=settings,retriever_version=sha_json(ASSETS),
        mixed_first_selection=False,infra_rollout=dict(DEFAULT_ROLLOUT_PHASE),
        question_probe_refit_outers=list(range(0,200,25)),extra_checkpoint_steps=list(range(25,201,25)))
    from reproduction.common8.qprobe_configuration import arguments
    arguments(proposed,BASE)
    replace_manifest(BASE/'logs/run-manifest.json',proposed)
    from reproduction.common8.configuration import resolved
    resolved(BASE)
    publish(BASE/'preparation/initial0-adopted.json',dict(run_id=run['run_id'],
        head_adam_steps=400,student_outers=0,probe_updated_only_at=list(range(0,200,25)),
        dynamic_features_remote_ram_only=True,student_parameters_copied_on_refit=False),once=True)


if __name__=='__main__':main()
