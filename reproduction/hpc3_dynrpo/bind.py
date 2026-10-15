"""Adopt this run's own verified warmup using the unchanged native model APIs."""
import os
from .common import BASE,MODEL,REUSED,TEACHER,own,read,sha,publish,progress,teacher_labels


def main():
    own(BASE)
    import torch
    from reproduction.common8.prepare_run import prepare
    from reproduction.common8.qprobe_qwen_bind import recipe
    from reproduction.common8.stage_retrieval import ASSETS
    from search_g1.qprobe_features import sha_json
    from search_g1.qprobe_token_features import TokenFeatureCache
    from search_g1.qprobe_qwen_block_online import OnlineQwenBlockQProbe,QwenBlockOnlineConfig
    from search_g1.rollout_phase_contract import DEFAULT_ROLLOUT_PHASE
    from reproduction.a800.owned_processes import publish as runtime_publish
    if torch.cuda.is_initialized(): raise ValueError('Binding must be CPU-only before pipeline admission')
    commit=os.environ['SEARCH_G1_SOURCE_COMMIT']
    ready=read(BASE/'warmup/ready.json'); raw=BASE/'warmup/fresh-head.pt'
    if ready['head_adam_steps']!=200 or sha(raw)!=ready['head_sha256']:
        raise ValueError('Own warmup10 is not fully accepted')
    cache=TokenFeatureCache(BASE/'token-features',expected_spec_sha=ready['token_feature_spec_sha256'])
    if sha(cache.path/'manifest.json')!=ready['token_feature_manifest_sha256']:
        raise ValueError('Adopted bank differs from the own actual warmup')
    labels,provenance=teacher_labels();qids=[g['qid'] for g in labels]
    if provenance!=ready['label_producer_provenance']:
        raise ValueError('Own teacher provenance changed')
    if (BASE/'logs/run-manifest.json').exists():
        raise ValueError('Run already prepared; inspect existing binding rather than reinitialize')
    run=prepare(BASE,MODEL,BASE/'preparation/official-model-manifest.json',commit,
        rollout_memory_utilization=.1,local_retrieval_assets=str(REUSED/'source/assets-root/assets/wiki'),execution_site='hpc3')
    settings=recipe(cache.path)
    settings['teacher_warmup_cache']=str(TEACHER)
    settings['output_dir']=str(BASE/'artifacts/question-probe')
    directory=BASE/'artifacts/question-probe/native-qwen-warmup-adoption';directory.mkdir(parents=True)
    head=OnlineQwenBlockQProbe.adopt_warmup(raw,labels,qids,expected_checkpoint_sha=ready['head_sha256'],
        expected_feature_spec_sha=cache.spec_sha,provenance=dict(actor_snapshot_id=run['run_id']+':0',
        warmup_ready=str(BASE/'warmup/ready.json'),warmup_ready_sha256=sha(BASE/'warmup/ready.json'),
        warmup_source_commit=commit,teacher_identity_sha256=provenance['teacher_identity_sha256']),
        config=QwenBlockOnlineConfig.from_mapping(settings),feature_cache=cache,device='cpu')
    if head._adam_steps()!=200 or head.replay_labels or head.replay_features or head.protocol is not None:
        raise ValueError('Warmup Adam must be inherited while formal replay starts empty')
    checkpoint=directory/'initial.pt'; receipt=head.save(checkpoint,actor_snapshot_id=run['run_id']+':0')
    loaded=OnlineQwenBlockQProbe.load(checkpoint,expected_actor_snapshot_id=run['run_id']+':0',
        expected_feature_spec_sha=cache.spec_sha,expected_checkpoint_sha=receipt['sha256'],device='cpu')
    if loaded.version!=head.version or loaded._adam_steps()!=200 or not torch.equal(loaded.rng.get_state(),head.rng.get_state()):
        raise ValueError('Actual saved warmup state failed CPU readback')
    adopted=dict(stage='QWEN_NATIVE_FORMAL_INITIAL_HEAD_ADOPTED',run_id=run['run_id'],source_commit=commit,
        head_receipt=receipt,raw_warmup_checkpoint_sha256=ready['head_sha256'],feature_spec_sha=cache.spec_sha,
        actual_head_adam_updates=200,completed_outer=0,warmup_questions=5120,teacher_trajectories_reused=25600,
        trajectories_regenerated=False,no_head_refit=True,teacher_replay_excluded=True,native_block_proof=ready['native_block_proof'],
        initialization_provenance=head.initialization_provenance,actual_saved_head_cpu_readback=True,checked_unix=__import__('time').time())
    ready_path=directory/'ready.json'; publish(ready_path,adopted,once=True)
    settings.update(initial_head_checkpoint=str(checkpoint),initial_head_checkpoint_sha256=receipt['sha256'],
        initial_head_ready=str(ready_path),initial_head_ready_sha256=sha(ready_path))
    proposed=dict(run,question_probe=settings,retriever_version=sha_json(ASSETS),mixed_first_selection=False,
                  infra_rollout=dict(DEFAULT_ROLLOUT_PHASE))
    from reproduction.common8.qprobe_configuration import arguments
    arguments(proposed,BASE)
    runtime_publish(BASE/'logs/run-manifest.json',proposed)
    from reproduction.common8.configuration import resolved
    resolved(BASE)
    from search_g1.qprobe_runtime import QProbeRuntime
    runtime=QProbeRuntime(settings,BASE,feature_cache=cache,head=head)
    adoption=runtime.bootstrap_initial_head()
    publish(BASE/'preparation/initial0-adopted.json',dict(run_id=run['run_id'],adopted=True,proof=adoption,
            actor_outers=0,warmup_adam_steps=200,formal_replay_groups=0),once=True)
    progress('OWN_INITIAL0_ADOPTED_READY_FOR_FORMAL',run_id=run['run_id'],warmup_adam_steps=200,formal_outers=0)


if __name__=='__main__':
    main()
