"""Fixed512/2048 epoch and matched-update nodes for separate full169028 trial.

Never scores the retired internal Count5120 training pool or final4096.
Checkpoint precedes inference; RNG and both native buffer sets are restored.
"""
from pathlib import Path
from . import probe_ablation_train as u,evaluate_frozen_validation as ev
from . import full169028_protocol as protocol


def accepted_HF_validation(spec):
 """Use the already qualified HF replay of the new label-blind fixed G5.

 This intentionally does not call the old native-vLLM accepted_validation reader.
 """
 from .probe_g10_train import frozen_validation
 from .replay_feature_reader import ReplayFeatureReader
 complete=u.read(u.checked(spec['validation_complete']));audit=u.read(u.checked(spec['validation_reader_audit']))
 if (complete.get('status')!='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'or complete.get('questions')!=2048 or complete.get('trajectories')!=10240 or complete.get('source_runtime')!='huggingface_frozen_causal_prefill'or complete.get('capture_layers')!=[26]or complete.get('source_complete',{}).get('sha256')!='5ea5197a841d7bd0b226a82350a2436a8ae3d860143a2ea39fe59b7ecb6837d5'or complete.get('fixed_G5_sha256')!='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7'or complete.get('new_G5_selected')is not False or complete.get('labels_read')is not False or complete.get('final_pool_read')is not False or complete.get('actor_updates')!=0 or complete.get('probe_updates')!=0):raise ValueError('Actual qualified NEW fixedG5 same-HF validation required')
 if (audit.get('status')!='ALL_10240_NEW_FIXED_G5_HF_ROWS_RECONSTRUCTED_RETAINED_SHA_PASSED'or audit.get('complete')!=spec['validation_complete']or audit.get('features')!=complete['features']or audit.get('all_manifest_and_payload_SHA256')is not True or audit.get('all_reconstructed_row_SHA256')is not True):raise ValueError('Independent actual10240 HF retained-row acceptance required')
 for key in ('source_complete','source_selection','source_groups','source_fixed_G5','features'):u.checked(complete[key])
 groups=u.read(complete['source_groups']['path'])['validation'];plan=u.read(complete['source_fixed_G5']['path']);frozen=frozen_validation(groups,plan);manifests=u.read(complete['features']['path'])
 if manifests!=complete['feature_manifests_by_capture_layer']['26']:raise ValueError('Actual HF validation manifest list changed')
 return frozen,ReplayFeatureReader(manifests,26),dict(complete=spec['validation_complete'],reader_audit=spec['validation_reader_audit'],fixed_G5_sha256=complete['fixed_G5_sha256'],backend='huggingface_frozen_causal_prefill',native_bitexact_claim=False)


def evaluate(base,c,model,ema,opt,device,checkpoint):
 base=Path(base);cursor=checkpoint['cursor']
 protocol.validate_config(base,c);node=protocol.evaluation_node(cursor)
 if node is None:raise ValueError('Only declared full-data epoch or matched-update checkpoints')
 epoch=node['epoch'];folder=base/'evaluations'/node['folder'];folder.mkdir(exist_ok=True,parents=True);done=folder/'complete.json'
 if done.exists():
  prior=u.read(done)
  if prior['checkpoint']!=checkpoint or prior.get('comparison_node')!=node or prior['config_sha256']!=u.sha(base/'config.json'):raise ValueError('Existing immutable main evaluation differs')
  for name in ('raw','EMA'):
   for item in prior['metrics'][name].values():ev.checked(item['predictions'])
  return prior
 from .replay_feature_reader import ReplayFeatureReader
 spec=c['evaluation'];diagnostic=u.read(u.checked(spec['diagnostic_groups']))['diagnostic'];dfeatures=u.read(u.checked(spec['diagnostic_features']));validation,vr,vproof=accepted_HF_validation(spec)
 if len(diagnostic)!=512 or len(validation)!=2048:raise ValueError('Only real fixed512/2048 evaluation coverage')
 forbidden=u.read(u.checked(c['exclusions']));by={r['qid']:r['question_hash']for r in forbidden['groups']}
 if any(g['qid']not in by or(g.get('question_hash')and by[g['qid']]!=g['question_hash'])for g in diagnostic+validation):raise ValueError('Evaluation identities must be excluded from training')
 dr=ReplayFeatureReader(dfeatures,26);before=u.state_hashes(model,ema,opt,device);rng=u.rng_state(device);buffers=(u.buffers(model),u.buffers(ema));metrics={}
 try:
  for label,probe in [('raw',model),('EMA',ema)]:
   metrics[label]={}
   for name,reader,groups in [('diagnostic512',dr,diagnostic),('fixed_selection2048',vr,validation)]:
    destination=folder/(label+'-'+name+'.jsonl.gz');value=ev.evaluate(probe,reader,groups,device,destination,base/'evaluation-progress.json',label+'-'+name,micro_groups=2);metrics[label][name]=dict(value,predictions=ev.proof(destination))
 finally:
  u.restore_rng(rng,device);u.restore_buffers(model,buffers[0]);u.restore_buffers(ema,buffers[1]);dr.close();vr.close()
 if u.state_hashes(model,ema,opt,device)!=before:raise ValueError('Read-only evaluation changed native raw/EMA/Adam/buffers/RNG')
 result=dict(epoch=epoch,comparison_node=node,cursor=cursor,checkpoint=checkpoint,config_sha256=u.sha(base/'config.json'),metrics=metrics,training_components_unchanged=True,native_buffers_and_RNG_restored=True,Count5120_scored=False,final4096_scored=False,validation_proof=vproof);u.atomic(done,result);return result
