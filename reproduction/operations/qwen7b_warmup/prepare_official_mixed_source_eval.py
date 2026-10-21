"""Stdlib-only official fresh initialization + fixed HF evaluation contract.

Never opens a posttraining probe checkpoint, optimizer/tensor, evaluation result,
or large official weight payload. The eventual GPU worker must fresh-hash weights.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';REVISION='a09a35458c702b33eeacc393d103063234e8bc28';MODEL='Qwen/Qwen2.5-7B-Instruct'
MODEL_DIR=B/'assets/qwen25-7b-instruct-a09a35458c70';TARGET=A/'joint-mixed-direct-official-169028-v1';OLD=A/'joint-balanced-native-E4-124194-v1/config.json';INVENTORY=B/'replay-layer-qualification-20261021/official-snapshot-view/official-inventory.json';SELECTION=A/'hpc3-tail9344-independent-L26-prepared-v2/shard-00/inputs/selection.json'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def proof(p):
 p=Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(Path(p).read_bytes())
def checked(ref):
 actual=proof(ref['path']);assert all(actual[k]==ref[k]for k in('path','bytes','sha256')),'Small metadata SHA mismatch';return read(ref['path'])
def build():
 start=time.time();assert os.uname().sysname!='Darwin';allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:2]);inventory_ref=proof(INVENTORY);inventory=read(INVENTORY);assert inventory['sha']==REVISION and inventory.get('id',MODEL)==MODEL;official={x['rfilename']:x for x in inventory['siblings']};assert len(official)==len(inventory['siblings'])
 migration_ref=proof(MODEL_DIR/'complete.json');migration=read(migration_ref['path']);assert migration['revision']==REVISION and migration['model']==MODEL and migration['status']=='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED'and migration['fresh_all_files_SHA256']is True
 selection_ref=proof(SELECTION);selection=read(SELECTION);assert selection['model_revision']==REVISION;declared=selection['model_files'];assert declared['official-inventory.json']=={k:inventory_ref[k]for k in('bytes','sha256')}
 model_files={};checks={};small_refs={};weights=[]
 for name,item in sorted(migration['files'].items()):
  assert Path(name).name==name and name in official and type(item['bytes'])is int and len(item['sha256'])==64;p=MODEL_DIR/name;assert p.stat().st_size==item['bytes'];source=official[name];assert source['size']==item['bytes']
  if name.endswith('.safetensors'):
   assert source['lfs']['sha256']==item['sha256'];weights.append(name);checks[name]='inventory_LFS_SHA_and_size_match_accepted_migration;allocation_fresh_payload_SHA_required'
  else:
   assert item['bytes']<16*1024**2;raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['sha256'];blob=hashlib.sha1(('blob %d\0'%len(raw)).encode()+raw).hexdigest()
   if source.get('lfs'):assert source['lfs']['sha256']==item['sha256']
   else:assert source['blobId']==blob
   if 'git_blob_sha1'in item:assert item['git_blob_sha1']==blob
   checks[name]='fresh_metadata_SHA_and_official_LFS_or_Git_blob_match';small_refs[name]=dict(path=str(p),bytes=len(raw),sha256=item['sha256'])
  model_files[name]=dict(bytes=item['bytes'],sha256=item['sha256'])
 assert len(weights)==4
 for name,item in declared.items():
  if name!='official-inventory.json':assert model_files[name]==item,'Accepted feature selection/model asset differs'
 index=read(MODEL_DIR/'model.safetensors.index.json');assert set(index['weight_map'].values())==set(weights);config=read(MODEL_DIR/'config.json');assert(config['num_hidden_layers'],config['hidden_size'],config['num_attention_heads'],config['num_key_value_heads'],config['intermediate_size'])==(28,3584,28,4,18944)
 cfg_ref=proof(OLD);old=read(OLD);assert old['fit_questions']==124194 and old['model_dir']==str(MODEL_DIR);evaluation=old['evaluation'];exclusions_ref=old['exclusions'];exclusions=checked(exclusions_ref);assert exclusions['role']=='joint_training_fixed_evaluation_identity_exclusions'and exclusions['scope_counts']==dict(diagnostic=512,selection=2048,final=4096)and exclusions['final_labels_or_rollouts_read']is False and len(exclusions['groups'])==6656 and len({x['qid']for x in exclusions['groups']})==6656
 for group in exclusions['groups']:assert set(group)==set(('qid','question_hash'))
 validation=checked(evaluation['validation_complete']);assert validation['status']=='NEW_VALIDATION2048_FIXED_G5_QUALIFIED_HF_REPLAY_LAYER26_COMPLETE'and validation['questions']==2048 and validation['trajectories']==10240 and validation['fixed_G5_sha256']=='c55de2b4f66f45dcd94a8e246bfca97e53bdedc4ca85180c75e60c574dfc04b7';checked(evaluation['validation_reader_audit']);diag_list=checked(evaluation['diagnostic_features']);assert isinstance(diag_list,list)and diag_list
 diag_bank=checked(evaluation['diagnostic_HF_bank']);allowed_refs=checked(diag_bank['features_by_capture_layer']['26']);assert all(x in allowed_refs for x in diag_list)
 result=dict(status='OFFICIAL_FRESH_NATIVE_AND_DIRECT_MIXED_GROUP_SOURCE_EVAL_METADATA_BOUND',namespace=str(TARGET),model=MODEL,revision=REVISION,model_dir=str(MODEL_DIR),official_inventory=inventory_ref,inventory_schema=dict(kind='HuggingFace_model_revision_API_blobs_true',revision_key='sha',model_id_key='id',file_list_key='siblings',filename_key='rfilename',size_key='size',large_file_SHA256_key='lfs.sha256',small_file_Git_blob_key='blobId',top_level_keys=sorted(inventory),sibling_count=len(inventory['siblings'])),model_config=small_refs['config.json'],model_index=small_refs['model.safetensors.index.json'],model_files=model_files,metadata_checks=checks,accepted_asset_migration=migration_ref,feature_selection_model_files_crosscheck=selection_ref,official_inventory_is_explicit_external_ref=INVENTORY.parent!=MODEL_DIR,initialization=dict(native_block_source='fresh_official_Qwen2.5_7B_layer27',native_Adam_steps=0,group_head_Adam_steps=0,new_group_head=True,posttraining_probe_checkpoint_read=False,old_probe_Adam_or_RNG_inherited=False),evaluation=evaluation,evaluation_config_source=cfg_ref,exclusions=exclusions_ref,evaluation_questions=dict(diagnostic512=512,selection_fixed_G5=2048),evaluation_backend=dict(train_feature_capture_layer=26,prefix_tokens=20,question_max_tokens=273,selection='qualified_HF_batch32/16384_fixed_label_blind_G5',auxiliary_diagnostic='accepted_HF_scalar_R2_diagnostic512_only',native_bitexact_claim=False),all_official_weight_payloads_fresh_SHA_required_on_GPU_before_first_forward=True,official_weight_payloads_read_by_this_CPU_preparer=False,posttraining_probe_checkpoint_or_tensor_loaded=False,final4096_outcomes_read=False,actor_updates=0,GPU_submitted=False,epoch=time.time(),elapsed_seconds=time.time()-start)
 TARGET.mkdir(exist_ok=True);path=TARGET/'OFFICIAL-SOURCE-AND-EVAL-CONTRACT.json'
 if path.exists():
  old_result=read(path);assert old_result['official_inventory']==inventory_ref and old_result['evaluation_config_source']==cfg_ref and old_result['model_files']==model_files;return dict(contract=proof(path),value=old_result)
 raw=canonical(result)+bytes([10])
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(contract=proof(path),value=result)
if __name__=='__main__':print(json.dumps(build(),sort_keys=True))
