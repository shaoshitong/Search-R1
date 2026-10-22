"""Consolidate existing published metrics; no model, rollout or final-pool IO."""
from pathlib import Path
import hashlib
import json

D = Path(__file__).resolve().parent
source = D / 'native-P80-existing-evaluations-actual.json'
v = json.loads(source.read_bytes())
rows = []
candidates = []
for layer, experiment in v['experiments'].items():
    for epoch in range(4):
        item = experiment['evaluations'][str(epoch)]
        for role, states in item['value']['metrics'].items():
            for state, m in states.items():
                confusion = m['confusion']
                wrong, correct = m['all_wrong'], m['all_correct']
                classes = dict(
                    all_wrong=dict(questions=wrong['questions'], predicted_Mixed=wrong['predicted_Mixed'], predicted_nonMixed=wrong['questions']-wrong['predicted_Mixed']),
                    all_correct=dict(questions=correct['questions'], predicted_Mixed=correct['predicted_Mixed'], predicted_nonMixed=correct['questions']-correct['predicted_Mixed']),
                    Mixed=dict(questions=confusion['tp']+confusion['fn'], predicted_Mixed=confusion['tp'], predicted_nonMixed=confusion['fn']),
                )
                assert sum(x['questions'] for x in classes.values()) == 2048
                rows.append(dict(experiment='L%sP80' % layer, layer=int(layer), prefix=80, epoch=epoch, state=state, role=role,
                    BA=m['balanced_accuracy'], BCE=m['binary_Mixed_BCE'], Top512=m['selected512'], Mixed_recall=m['Mixed_recall'],
                    nonMixed_FPR=m['nonMixed_false_positive_rate'], confusion=confusion, classwise=classes,
                    evaluation_proof=item['proof'], prediction_path=str(Path(experiment['base'])/('epoch-%02d-%s-predictions.jsonl.gz' % (epoch, role)))))
    for cp in experiment['candidate_paths']:
        for state in ('raw', 'EMA'):
            candidate=dict(experiment='L%sP80' % layer, layer=int(layer), prefix=80, checkpoint=cp, state_key=state,
                actual_published_E3_SHA=experiment['latest.json']['value'] if cp['path'].endswith('03.pt') else None,
                not_frozen_for_final=True, prediction_score='direct_Mixed_logit', classification_threshold='logit>=0',
                native_final4096_v3_existing_geometry_gate_allows=(int(layer),80) in ((27,20),(21,20),(27,40),(27,80),(21,80),(15,20),(18,20),(24,20)))
            candidates.append(candidate)
# Actual source text is inspected and hashed; no model execution occurs here.
P = D.parents[2]/'operations/qwen7b_warmup'
names=['online_mixed_prefix_explore.py','online_mixed_depth_expansion.py','online_mixed_depth_resume.py','native_mixed_group_classifier.py','test_native_mixed_group_classifier.py','score_frozen_final4096.py']
proofs=[dict(path=str((P/name).relative_to(D.parents[3])), bytes=(P/name).stat().st_size, sha256=hashlib.sha256((P/name).read_bytes()).hexdigest())for name in names]
architecture=dict(official_revision='a09a35458c702b33eeacc393d103063234e8bc28', actor_model='Qwen2.5-7B-Instruct',
    official_model_dir='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/assets/qwen25-7b-instruct-a09a35458c70',
    layer_index_semantics='Exact runtime index: copied actor.layers[layer]; frozen actor.layers[:layer] supplies input, last included block index layer-1. Do not relabel zero-based runtime index as one-based layer ordinal.',
    online_input='Each of the real G5 trajectories retains its own causal tool-context prefill. Not the new packed one-question+Mprefix branch.',
    native_block_hidden_size=3584, QMAX=273, prefix=80, real_trajectory_rows_per_question=5,
    head=dict(projection_dimension=256, noncausal_transformer_layers=2, attention_heads=8, dropout=0, question_hidden='average shared question slots across five trajectories', policy_hidden='all five times80 slots retained; within-prefix position embeddings repeated across trajectories; no trajectory-index embedding', CLS_output_logits=1),
    loss='Direct binary true-Mixed BCE only, inverse class-frequency weights from each complete epoch plan. No trajectory BCE.',
    actual_epoch_counts=v['fixture_metadata']['plans'], learning_rate=5e-6, EMA_decay=.99, raw_updates_EMA=True,
    questions=8192, epochs=3, optimizer_updates=192, global_batch=128, GPUs=2, local_batch=64, micro_questions=8, accumulation=8,
    model_and_head_seed=9137, sampler_seed=20261022, source_seeds_currently_hardcoded=True,
    two_seed_reproduction='Not run. New independent code/output/locks must expose and bind both model/head seed and own-G5 plan seed; keep paired layer arms identical within a seed. Preserve the fixed train/dev G5 identities. Existing seed9137 and plans20261022 are one matched run, not two-seed evidence.')
permutation=dict(existing_native_test='test_native_mixed_group_classifier.py:test_group_permutation_and_shared_question_averaging',
    test_kind='Small CPU synthetic hidden-state fixture with permutation [4,0,3,1,2], tolerance rtol1e-5/atol2e-6; does not load the actual candidate checkpoint.',
    native_architecture='Permutation-invariant by shared question averaging, repeated within-prefix embeddings and noncausal group attention/CLS. Floating-point equality still needs tolerance.',
    actual_candidate_model_permutation_audit_found=False,
    related_sampler_audit='packed-variance-full169028/sampler-hardnegative-diagnostic/actual-epoch-*.json reverses trajectory order only to verify target variance; explicitly prediction_permutation_invariance_not_measured=true.',
    packed_architecture='The new question+Mprefix packed causal context with absolute positions is order-sensitive; random training order is augmentation, not a guarantee of invariant prediction.',
    new_inference_run=False)
out=dict(status='EXISTING_L15_L18_L21_NATIVE_P80_E0_TO_E3_CONSOLIDATED',source=dict(path=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
    rows=rows,candidates=candidates,architecture=architecture,trajectory_permutation_audit_search=permutation,source_proofs=proofs,
    fixed_train=v['inputs/fixed-train2048.json'],fixed_dev=v['inputs/validation-groups.json'],fixed_dev_G5_digest=v['fixture_metadata']['fixed_G5_sha256'],
    dev_is_already_viewed=True,final4096_read=False,new_inference=False,new_training=False, Slurm_mutated=False,
    scope='Same 8192 training questions, frozen ownG5 epoch plans and fixed train/dev groups. These are single-seed observed validation metrics; no generalization guarantee, no final candidate freeze and no causal comparison against full packed Z.')
(D/'native-P80-L15-L18-L21-consolidated.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Native Prefix80, existing matched layer runs','','All values are already published fixed-pool evaluations; no new model execution or final4096 read. Training plan each epoch contains3489Mixed and4703nonMixed questions, with inverse-frequency BCE weights. The fixed training probe has888Mixed/210all-correct/950all-wrong; fixed viewed validation has663Mixed/330all-correct/1055all-wrong.','','| Layer | State | Epoch | Train BA | Train BCE | Train Top512 Mixed | Viewed dev BA | Dev BCE | Dev Top512 Mixed |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for layer in (15,18,21):
 for state in ('raw','EMA'):
  for epoch in range(4):
   a=next(x for x in rows if x['layer']==layer and x['state']==state and x['epoch']==epoch and x['role']=='fixed_train2048')
   b=next(x for x in rows if x['layer']==layer and x['state']==state and x['epoch']==epoch and x['role']=='fixed_validation2048')
   lines.append(f"|{layer}|{state}|{epoch}|{a['BA']:.2%}|{a['BCE']:.6f}|{a['Top512']['Mixed']}|{b['BA']:.2%}|{b['BCE']:.6f}|{b['Top512']['Mixed']}|")
lines += ['', 'At E3, raw dev BA is59.80/59.12/59.94% for L15/L18/L21, with Top512 realMixed219/223/215. EMA dev BA is59.16/59.14/58.75%, with Top213/222/214. The highest BA and highest Top are different metrics; neither establishes generalization from this viewed pool. Complete E0…3 raw/EMA class confusion counts and candidate/prediction paths are in the JSON.','','All three use a frozen official actor and one trainable copied native block plus a dim256 two-layer eight-head noncausal group head, direct Mixed BCE, LR5e-6, EMA.99, 8192questions×3epochs,192updates,global128/two GPUs. Runtime layer numbers are exact zero-based actor.layers indices; each capture uses actor.layers[:layer]. These are five authentic causal trajectory prefills, not the newer repacked variable-M model.','','For a new independent seed, the current hardcoded model/head seed9137 and own-view-plan seed20261022 need an explicit immutable override and independent roots. Pair the same new model/plan seed across depth conditions; do not replace fixed dev G5. No second-seed job has been created by this report.','','An existing small CPU hidden-fixture test checks native G5 permutation invariance. No actual candidate checkpoint permutation audit was found. The full packed sampler audit checks target invariance only, explicitly not prediction invariance. The packed causal architecture itself is order-sensitive.','','Existing final4096 v3 loads generic native block/prefix geometry, but its predeclared allowed tuple gate currently lacks(15,80) and(18,80). If either is later frozen as a final candidate, a new immutable scorer publication must explicitly add that geometry before final access. This report does not alter that gate or unblind final data.']
(D/'native-P80-L15-L18-L21-consolidated.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(rows=len(rows),candidates=len(candidates),source_time=v['time'],output='native-P80-L15-L18-L21-consolidated.json')))
