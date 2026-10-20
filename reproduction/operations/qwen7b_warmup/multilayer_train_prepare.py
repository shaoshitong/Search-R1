"""Metadata-only real R2/Q1 configuration preparation for three fresh probes.

Reads SHA-bound JSON/code only. Official model tensor/shard content is verified
and loaded later on GPU nodes. Merged replay bulk-payload acceptance is bound,
not redundantly rescanned on login. No job submission occurs in this module.
"""
import argparse,json
from pathlib import Path
from . import probe_ablation_train as u
from .replay_layer_train import TARGET,build_plan,render_sbatch
from .probe_ablation_model import CAPTURE_PAIRS
from .replay_bank_prepare import q1_model_binding

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
R2=PARENT+'/replay-matched9088-bank-v2-20261021/prepared/bank-complete.json'
R2_SHA='c593363dfaca0edf3f142e1fa5f93bd7f461432ce7b0f1e186101039ac163dc4'
LAYER_SHA={'20':'378053edf25605803b6752f633c985bbb8e2bbdd08845da42767a9d1e61edf09',
    '23':'242c788f66a8c28012c58e42f95e7b76a09cba8abadc8ff538e1fb43cb245c65',
    '26':'d248a82c87a1a59297829f74d4cfdf977368f9f16fec129873b7e7ea6684aced'}


def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def immutable(p,value):
    if p.exists():
        if u.read(p)!=value:raise ValueError('Different immutable multilayer input: '+str(p))
    else:u.atomic(p,value)


def check_complete(value):
    if (value['status']!='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED' or
            value['questions']!=9088 or value['trajectories']!=45440 or
            value['roles']!=dict(fit=6528,diagnostic512=512,selection2048=2048) or
            value['full_replay_shared_file_SHA_verified'] is not True or
            value['source_runtime']!='huggingface_frozen_causal_prefill' or
            value['native_bitexact_equivalence_claim'] is not False or
            value['actor_updates']!=0 or value['new_responses_generated']!=0 or
            set(value['features_by_capture_layer'])!={'20','23','26'}):
        raise ValueError('Require actual completed four-shard matched9088 uniform HF replay banks')
    if any(value['features_by_capture_layer'][layer]['sha256']!=want for layer,want in LAYER_SHA.items()):
        raise ValueError('Known actual R2 feature-list SHA differs')


def make_config(reference_ref,reference,plan_ref,replay_ref,replay,model_dir,model_meta,model_files,index,layer,hashes,commit):
    if type(layer) is not int or layer not in CAPTURE_PAIRS:raise ValueError('Explicit official zero-based21/24/27 branch required')
    cap=str(CAPTURE_PAIRS[layer]);prefix='model.layers.%d.'%layer
    shards={name for key,name in index['weight_map'].items() if key.startswith(prefix)}
    if not shards:raise ValueError('Official block absent from model index')
    official={name:dict(path=str(Path(model_dir)/name),sha256=model_files[name]['sha256'],bytes=model_files[name]['bytes']) for name in sorted(shards)}
    return dict(block_layer=layer,epochs=4,update_budget=512,global_groups=51,micro_groups=2,save_every=16,
        lr=5e-6,ema_decay=.99,head_seed=1729,training_seed=1729,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.),
        source_matched_config=reference_ref,reference_plan=reference['plan'],plan=plan_ref,bank=reference['bank'],selection_bank=reference['selection_bank'],
        replay_bank_complete=replay_ref,replay_features=replay['features_by_capture_layer'][cap],model_dir=model_dir,
        model_config=model_meta['config.json'],model_inventory=model_meta['official-inventory.json'],model_index=model_meta['model.safetensors.index.json'],
        official_block_shards=official,code_hashes=hashes,source_commit=commit,feature_backend='huggingface_frozen_causal_prefill',
        fresh_official_block=True,E1_checkpoint_inherited=False,Adam_initial_step=0,EMA_initialization='copy_of_fresh_raw',
        fresh_common_flatten_head=True,native_bitexact_claim=False,untouched_final_read=False,screening_only=True)


def prepare(base):
    if str(base)!=TARGET or base.resolve()!=base:raise ValueError('Use isolated multilayer-four-epoch-v1 only')
    replay_ref=ref(R2)
    if replay_ref['sha256']!=R2_SHA or replay_ref['bytes']!=2756:raise ValueError('Fresh actual R2 complete SHA/size differs')
    replay=u.read(R2);check_complete(replay)
    for key in ('source_preparation','group_identities'):u.checked(replay[key])
    for item in replay['capture_completions']:u.checked(item)
    counts={}
    for layer,item in replay['features_by_capture_layer'].items():
        manifests=u.read(u.checked(item))
        if len(manifests)!=144:raise ValueError('Require all144 actual chunk manifests per capture layer')
        total=0
        for descriptor in manifests:
            own=u.read(u.checked(descriptor))
            if (own['status']!='LOSSLESS_HF_REPLAY_FEATURES_COMPLETE' or own['capture_layer']!=int(layer) or
                    tuple(own[k] for k in ('dimension','question_max_tokens','prefix_tokens'))!=(3584,273,20) or
                    own['source_runtime']!='huggingface_frozen_causal_prefill' or own['actor_optimizer_updates']!=0):
                raise ValueError('Actual replay chunk layer/geometry/backend differs')
            total+=own['count']
        if total!=45440:raise ValueError('Actual per-layer trajectory coverage differs')
        counts[layer]=dict(manifests=len(manifests),trajectories=total)
    reference_suite=Path(PARENT)/'algorithm-improvement-20261021/matched-loss128-v2'
    ready=u.read(reference_suite/'preparation.json');reference_ref=ready['variants']['event']['config'];reference=u.read(u.checked(reference_ref))
    for key in ('bank','selection_bank','plan'):u.checked(reference[key])
    reference_plan=u.read(reference['plan']['path']);plan=build_plan(reference_plan)
    identities=u.read(replay['group_identities']['path'])['groups'];by={g['qid']:g for g in identities}
    bank=u.read(reference['bank']['path']);selection=u.read(reference['selection_bank']['path'])['selection_validation'];diagnostic=bank['diagnostic']
    lookup={g['qid']:g for g in bank['train']};fit=[lookup[q] for q in reference_plan['cohort_qids']]
    if len(by)!=9088:raise ValueError('Actual replay identities not unique9088')
    for role,groups in [('fit',fit),('diagnostic512',diagnostic),('selection2048',selection)]:
        if len(groups)!={'fit':6528,'diagnostic512':512,'selection2048':2048}[role]:raise ValueError('Fixed role count differs')
        for g in groups:
            actual=by[g['qid']]
            if actual['role']!=role or actual['trajectory_ids']!=g['trajectory_ids'] or actual['question_hash']!=g['question_hash']:
                raise ValueError('Actual R2/source matched question or own-G5 identities differ')
    selection_proof,model_binding=q1_model_binding();model_dir=model_binding['official_view']
    model_meta={name:ref(Path(model_dir)/name) for name in ('config.json','official-inventory.json','model.safetensors.index.json')}
    index=u.read(model_meta['model.safetensors.index.json']['path'])
    pub=u.read(base/'source-publication.json');root=Path(pub['code_root'])
    if root.parent!=base or root.name!='code-'+pub['commit'][:12]:raise ValueError('Isolated immutable code root required')
    for name,item in pub['files'].items():u.checked(dict(path=str(root/name),**item))
    hashes={name:item['sha256'] for name,item in pub['files'].items()};(base/'inputs').mkdir(exist_ok=True)
    path=base/'inputs/four-epoch-plan.json';immutable(path,plan);plan_ref=ref(path);variants={}
    for layer in CAPTURE_PAIRS:
        name='layer%d'%layer;dest=base/name;dest.mkdir(exist_ok=True);(dest/'logs').mkdir(exist_ok=True)
        config=make_config(reference_ref,reference,plan_ref,replay_ref,replay,model_dir,model_meta,selection_proof['model_files'],index,layer,hashes,pub['commit'])
        immutable(dest/'config.json',config);script=render_sbatch(str(dest),str(root),layer);path=dest/'fit.sbatch'
        if path.exists():
            if path.read_text()!=script:raise ValueError('Immutable Slurm script differs')
        else:path.write_text(script)
        variants[name]=dict(base=str(dest),job_name='q7b-hflayer-'+str(layer),config=ref(dest/'config.json'),sbatch=ref(path))
    done=dict(status='THREE_FRESH_OFFICIAL_LAYER_FOUR_EPOCH_SUITE_PREPARED_ACTUAL_R2_METADATA_BOUND',commit=pub['commit'],code_root=str(root),
        replay_bank_complete=replay_ref,reference_config=reference_ref,source_plan=plan_ref,cpu_tests=ref(base/'actual-cpu-tests.log'),
        model_binding=model_binding,variants=variants,actual_layer_counts=counts,fit_questions=6528,diagnostic_questions=512,selection_questions=2048,
        epochs=4,updates=512,updates_per_epoch=128,global_groups=51,micro_groups=2,head_seed=1729,
        official_initial_block_by_branch=True,new_head_by_branch=True,Adam_initial_step=0,EMA_initial_equals_raw=True,
        checkpoint_tensors_loaded_on_login=False,model_tensors_loaded_on_login=False,bulk_replay_payload_SHA_repeated_on_login=False,
        one_ReplayFeatureReader_per_GPU_process_all_roles=True,native_bitexact_claim=False,no_dependency=True,no_automatic_submission=True,untouched_final_read=False)
    immutable(base/'preparation.json',done);return done


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.base)),flush=True)
