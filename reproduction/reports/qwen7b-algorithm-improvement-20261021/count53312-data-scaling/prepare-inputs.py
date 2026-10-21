"""One finite metadata preparation; no torch, score, tensor copy or GPU job."""
import json,pathlib,hashlib,os,sys
D=pathlib.Path(__file__).resolve().parent

def read(p):return json.loads(pathlib.Path(p).read_bytes())
def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
def digest(v):return hashlib.sha256(raw(v)[:-1]).hexdigest()
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 assert proof(r['path'])=={k:r[k]for k in ('path','bytes','sha256')},'Actual source descriptor changed'
 return pathlib.Path(r['path'])
def save(p,v):
 data=raw(v)
 if p.exists():assert p.read_bytes()==data,'Own immutable metadata differs'
 else:p.write_bytes(data)
 return proof(p)

c=read(D/'SOURCE-INPUTS.json');p=read(D/'PROTOCOL.json');b=pathlib.Path(c['target']);b.mkdir(exist_ok=True);baseline=c['baseline'];t=c['threebank']
for key in ('family_inputs','baseline_config','baseline_publication','cache_complete','baseline_CP0','baseline_epoch0_evaluation','Count20_complete','cohort','fit_view_epochs','source_plan','frozen_NQ_halves'):checked(baseline[key])
f=read(baseline['family_inputs']['path']);bc=read(baseline['baseline_config']['path']);done=read(baseline['Count20_complete']['path'])
assert bc['head_LR']==1e-4 and bc['EMA_decay']==.99 and bc['grouping']==p['unchanged']['grouping'] and done['head_Adam']==18760 and not done['actor_updates'] and not done['legacy_updates']
for key in ('complete','future1666_append_plan','future53312_fit','original938_plan','internal_holdout_selection','internal_holdout_groups','identity_exclusion_acceptance','merged_features_layer26','merged_groups'):checked(t[key])
accepted=read(t['complete']['path']);old=read(baseline['source_plan']['path']);append=read(t['future1666_append_plan']['path']);cohort=read(baseline['cohort']['path']);fit=read(t['future53312_fit']['path'])['train']
assert accepted['fit_questions']==53312 and accepted['historical_prefix_preserved_exactly'] and accepted['fixed_evaluation_identity_overlap']==0
assert len(fit)==53312 and len(append['batches'])==1666 and append['batches'][:938]==old['batches'] and append['batch_epochs'][:938]==old['batch_epochs']
qs=[g['qid']for g in fit];assert qs[:30016]==cohort['fit_qids'] and qs==append['cohort_qids'] and len(set(qs))==53312 and len({g['question_hash']for g in fit})==53312
assert [q for batch in append['batches']for q in batch]==qs and all(len(batch)==32 for batch in append['batches'])
held=read(t['internal_holdout_groups']['path'])['internal_holdout'];hm={g['qid']:g for g in held};assert len(hm)==5120 and set(hm)==set(cohort['holdout_qids']) and not(set(qs)&set(hm))
assert bc['inputs']['internal_holdout_groups']==t['internal_holdout_groups'] and bc['inputs']['internal_holdout_selection']==t['internal_holdout_selection']
counts={'NQ':0,'HotpotQA':0};tids=[]
for g in fit:
 domain=json.loads(g['qid'])[0];counts['NQ'if domain=='nq'else'HotpotQA']+=1;assert domain in ('nq','hotpotqa') and len(g['trajectory_ids'])==10 and len(g['labels'])==10;tids+=g['trajectory_ids']
assert counts=={'NQ':39884,'HotpotQA':13428} and len(set(tids))==533120
assert not(set(tids)&{tid for g in held for tid in g['trajectory_ids']})
# The former append-only budget wording is not copied into this independent same-CP0 head experiment.
plan=dict(role='COUNT53312_SAME_CP0_20EPOCH_ORDER_AND_VIEWS',source_identity_order_plan=t['future1666_append_plan'],questions=53312,global_questions=32,epochs=20,updates_per_epoch=1666,update_budget=33320,matched_update18760=18760,batches=append['batches'],batch_epochs=append['batch_epochs'],cohort_qids=qs,grouping=bc['grouping'],views_per_question=4,view_weight=.25,effective_total_weight_per_question=1,original938_batches_sha256=digest(old['batches']),original938_plan=baseline['source_plan'],historical_prefix_intentionally_trained_each_epoch=True,initial_state=baseline['baseline_CP0'],initial_state_is_trained_Count20=False,selection_uses_labels=False,selection_uses_scores=False)
protocol=save(b/'protocol.json',p);planref=save(b/'scale-plan.json',plan)
newids=[tid for g in fit[30016:]for tid in g['trajectory_ids']]
newcache=save(b/'additional-cache-cohort.json',dict(role='ONLY_NEW23296_FIT_FROZEN_E1_NATIVE_CACHE_INPUT',qids=qs[30016:],source_groups=t['future53312_fit'],questions=23296,trajectories=232960,ordered_own_G10_IDs_sha256=digest(newids),source_layer26_features=t['merged_features_layer26'],no_new_rollout=True,frozen_E1_updates=0,cache_materialized=False))
sc=save(b/'scale-cohort.json',dict(fit_qids=qs,holdout_qids=cohort['holdout_qids']))
viewref=save(b/'fit-view-epochs.json',{q:append['batch_epochs'][i]for i,batch in enumerate(append['batches'])for q in batch})
packet=dict(role='FROZEN_E1_COUNT53312_DATA_SCALING_INPUTS',protocol=protocol,baseline_config=baseline['baseline_config'],baseline_publication=baseline['baseline_publication'],cache_complete=baseline['cache_complete'],baseline_cache_complete=baseline['cache_complete'],baseline_CP0=baseline['baseline_CP0'],baseline_CP0_four_hashes=baseline['baseline_CP0_four_hashes'],Count20_complete=baseline['Count20_complete'],baseline_epoch0_evaluation=baseline['baseline_epoch0_evaluation'],baseline_cohort=baseline['cohort'],baseline_orders=baseline['orders'],baseline_fit_view_epochs=baseline['fit_view_epochs'],baseline_source_plan=baseline['source_plan'],merged58432_complete=t['complete'],accepted_identity_plan=t['future1666_append_plan'],scale_plan=planref,cohort=sc,fit_view_epochs=viewref,fit_groups=t['future53312_fit'],merged_groups=t['merged_groups'],merged_features_layer26=t['merged_features_layer26'],held5120_identity=t['internal_holdout_selection'],held5120_groups=t['internal_holdout_groups'],internal_holdout_selection=t['internal_holdout_selection'],internal_holdout_groups=t['internal_holdout_groups'],identity_exclusion_acceptance=t['identity_exclusion_acceptance'],frozen_NQ_halves=baseline['frozen_NQ_halves'],small_checkpoint_verifier=baseline['small_checkpoint_verifier'],grouping=bc['grouping'],additional_cache_cohort=newcache,cache_layout=dict(old_questions=35136,old_trajectories=351360,new_questions=23296,new_trajectories=232960,total_questions=58432,total_trajectories=584320,old_hidden_bytes=50370969600,new_hidden_bytes=232960*20*3584*2,total_hidden_bytes=83768115200,batch320_hidden_bytes=45875200,backend='same frozen native E1 API BF16',old_physical_order='baseline fit30016 then held5120',new_physical_order='expanded fit suffix23296',entire_bank_to_GPU_forbidden=True),cache_extension_materialized=False,initial_CP0_restore_required_on_allocation=True,source_old_family_protocol_preserved=f['family_protocol'],original_warmup_unseen_claim=False,external_payload_opened=False,final_payload_opened=False,GPU_submitted=False)
ref=save(b/'INPUT-PACKET.json',packet);receipt=save(b/'READY.json',dict(status='COUNT53312_ACTUAL_SOURCE_METADATA_READY_NATIVE_CACHE_EXTENSION_REQUIRED',packet=ref,protocol=protocol,plan=planref,fit_questions=53312,fit_domains=counts,held_questions=5120,old938_prefix_equal=True,new_batches728=728,update_budget33320=33320,matched_update18760=18760,cache_extension_trajectories=232960,cache_extension_complete=False,model_tensor_or_CP0_payload_loaded=False,new_GPU_submitted=False,external_or_final_outcomes_opened=False))
print(json.dumps(dict(status='COUNT53312_ACTUAL_SOURCE_METADATA_READY_NATIVE_CACHE_EXTENSION_REQUIRED',packet=ref,READY=receipt,protocol=protocol,scale_plan=planref,new_cache_cohort=newcache,fit_domains=counts),sort_keys=True))
