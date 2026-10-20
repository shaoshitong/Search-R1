"""Typed CPU-only preparation; deliberately no GPU submission/training action."""
import argparse,copy,fcntl,json,platform,time,unittest
from pathlib import Path
from . import peer_append_train as a

TARGET=a.A/'direct-hidden-count30016-preparation-v1'

def configured(original,selection,publication,source_bytes):
    if (original['source_checkpoint']['sha256']!='02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
            or original['source_adam_steps']!=10464 or original['update_budget']!=256
            or original['source_world_size']!=4 or original['current_world_size']!=1
            or original['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise')):
        raise ValueError('Exact original sealed E1 and peer grouping protocol required')
    c=copy.deepcopy(original)
    c.update(source_commit=publication['commit'],code_hashes={n:r['sha256'] for n,r in publication['files'].items()},
        g10_complete=selection['new_bank'],plan=selection['append_plan'],original_plan=selection['source_plan'],
        source_cursor=0,source_adam_steps=10464,update_budget=938,fit_questions=30016,fit_real_trajectories=300160,
        recipe=dict(id='hidden-direct-count6-from-E1',trajectory_BCE_weight=1.,count_NLL_weight=1.,product_group_loss_weight=0.),
        source_world_size=4,current_world_size=1,hardware_topology_changed=True,canonical_source_rng_rank=0,
        original_E1_archive_world_size=4,legacy_optimizer_reset=False,
        new_head=dict(prefix_tokens=20,pooled_token_source='post_block27_policy_slots_only',
            attention='one learned scalar score per actual prefix token; masked softmax; zero pool for empty prefix',
            dimension=3584,embedding_dimension=128,projection='Linear+GELU',
            symmetric_statistics=['mean','population_variance','coordinatewise_max'],classifier=[384,128,6],
            q_direct='sum(softmax(count_logits)[1:5])',new_Adam_steps=0,learning_rate=5e-6,
            weight_decay=0.,EMA_decay=.99,initialization_seed=1729,initialization_preserves_legacy_RNG=True),
        frozen_plan_consumer_semantics='Use all938 frozen batches from E1 cursor0; historical256 fields describe reference append provenance, not completed updates in this new experiment',
        reference_append_source_selection=a.SELECTION,source_reference_control_config=selection['source_config'],
        equality_to_append_source_state_claim=False,feature_backend='huggingface_frozen_causal_prefill',
        prepare_only=True,actual_full_E1_restore_and_GPU_joint_next_update_still_required=True,
        evaluation_cumulative_cursors=[256,512,768,938])
    c['source_checkpoint']['bytes']=source_bytes
    c['domain_scope']['question_counts']={'nq':30016}
    return c

def prepare():
    selection=a.accepted_selection();old=a.read(a.checked(selection['source_config']));pub=a.read(TARGET/'source-publication.json')
    for n,v in pub['files'].items():a.checked(dict(path=str(Path(pub['code_root'])/n),**v))
    source=Path(old['source_checkpoint']['path'])
    if source.resolve()!=source or not source.is_file():raise ValueError('Existing exact E1 source must remain remote')
    c=configured(old,selection,pub,source.stat().st_size)
    groups,tail,plan,bank=a.accepted_inputs(c);a.validate_evaluations(c,groups)
    manifests=a.read(a.checked(bank['outputs']['features-layer26.json']))
    a.immutable(TARGET/'config.json',c)
    result=dict(status='HIDDEN_DIRECT_COUNT_PREPARED_ONLY_NO_GPU_TRAINING_SUBMITTED',config=a.proof(TARGET/'config.json'),
        source_checkpoint=c['source_checkpoint'],source_legacy_Adam=10464,new_head_Adam=0,
        source_cursor=0,target_cursor=938,target_legacy_Adam=11402,target_new_head_Adam=938,
        actual_questions=30016,actual_G10_trajectories=300160,feature_manifests=len(manifests),
        same_frozen_batch_order=selection['append_plan'],actual_existing_E1_CP_tensor_loaded=False,
        source_legacy_parameters_readout_Adam_not_reset=True,new_head_parameters=512518,
        genuine_hidden_direct_count_not_p_statistics=True,final_pool_read=False,time=time.time())
    a.immutable(TARGET/'preparation.json',result);return result

def qualify():
    started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_direct_hidden_count')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('Focused joint-head CPU qualification failed')
    done=dict(status='HIDDEN_DIRECT_COUNT_SYNTHETIC_CPU_QUALIFICATIONS_PASSED',tests=result.testsRun,
        seconds=time.monotonic()-started,actual_model_or_feature_assets_loaded=False,GPU_used=False,
        actual_E1_tensor_restore_or_GPU_bitexact_claim=False,time=time.time())
    a.immutable(TARGET/'cpu-qualification.json',done);return done

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('qualify','prepare'),required=True);args=p.parse_args()
    if platform.system()!='Linux':p.error('Actual preparation is remote CPU only')
    TARGET.mkdir(parents=True,exist_ok=True)
    with (TARGET/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(qualify() if args.action=='qualify' else prepare()),flush=True)
