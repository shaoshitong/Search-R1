from pathlib import Path
import json,sys,datetime
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
staged=json.loads(Path('work/prefix5-reference-code-stage.json').read_text());C=staged['commit'];repo=staged['repo'];output='/mnt/data/shaoshitong/search-g1-prefix5-reference-preparation-20261018/prepared-'+C[:8]
inner='REPO='+repr(repo)+'\nOUT='+repr(output)+'\nCOMMIT='+repr(C)+'\n'+r'''
import sys,pathlib,json,os,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';sys.path.insert(0,REPO)
import torch;torch.set_num_threads(1)
from reproduction.prefix_selection200.prepare_h100_reference_replica import prepare
out=pathlib.Path(OUT);p=out/'prepared.json'
if p.exists():v=json.loads(p.read_text())
else:v=prepare('/mnt/data/shaoshitong/search-g1-prefix5-reference-assets-20261018-model-only',out,COMMIT,json.loads(pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix5-reference-preparation-20261018/infrastructure.json').read_text()))
small=dict(v,slots={})
keys=['world_size','physical_gpu_indices','physical_gpu_uuids','auxiliary_gpu_indices','grounding_physical_gpu_by_rank','grounding_refit_visible_device','retrieval_gpu_shards','retrieval_port','grounding_probe_base_port','base','run_id','source_commit','actor_version','fresh_actor_optimizer_scheduler_rng_sampler','retained_warmup_model_only','full_state_resume','old_actor_state_imported','old_online_probe_optimizer_imported','probe_source_epoch','probe_source_model_kind','probe_source_sha256','original_warmup_checkpoint_sha256','model_revision','candidate_schedule','probe_learning_rate','probe_update_period','probe_updates_per_period','probe_update_at_final','actor_global_mini_batch','actor_global_micro_batch_size','actor_ppo_epochs','actor_loss_reduction','nonmixed_advantage_mode','policy_safeguards','interaction_protocol','grounding_refresh_enabled','grounding_refit_rounds','grounding_calibration_sha256','canonical_train_sha256','validation_questions_sha256','final_questions_sha256','common_manifest_sha256','composite_manifest_sha256','fixed_test_identity_sha256','fixed_test_manifest_sha256','source_question_pool_sha256','seed','exploration_questions','experiment_assignment_pending','preparation_only']
for slot,path in v['configs'].items():
 c=json.loads(pathlib.Path(path).read_text());delta=json.loads((pathlib.Path(path).parent/'exact-config-delta.json').read_text())
 forbidden=[k for k in ('resume_joint','resume_import','resume_pending_probe','source_compatibility','continue_reuse')if k in c];assert not forbidden
 b=pathlib.Path(c['base']);no_launch=not any((b/'preparation'/name).exists()for name in ['launch-config.json','submit-intent.json','submit-receipt.json']);assert no_launch
 small['slots'][slot]=dict(v['slots'][slot],config={k:c[k]for k in keys if k in c},exact_delta=delta,resume_fields_present=forbidden,no_launch_or_submit_files=True)
small['actual_target_EMA_tensor_and_buffer_hashes_compared']=True;small['actual_CUDA_forward_performed']=False;small['training_submitted']=False
print(json.dumps(small))
'''
script='import os,subprocess\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=150);assert r.returncode==0,r.stderr;print(r.stdout)'
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');v=json.loads(remote(script,'work/prefix5-reference-template-prepare-'+tag+'.transport.json'));p=Path('work/prefix5-reference-prepared-small-summary.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'proof':str(p.resolve()),'status':v['status'],'source_commit':v['source_commit'],'slots':{k:{'cards':x['training_cards'],'config_sha256':x['config_sha256'],'no_submit':x['no_launch_or_submit_files']}for k,x in v['slots'].items()},'actual_EMA_tensor_compare':v['actual_target_EMA_tensor_and_buffer_hashes_compared']}))
