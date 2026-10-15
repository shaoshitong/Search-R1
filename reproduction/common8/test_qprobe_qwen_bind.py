"""Small metadata-only binding checks; no native model, weights or GPU."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from reproduction.common8 import qprobe_qwen_bind as binding
from reproduction.common8.qprobe_configuration import arguments
from search_g1.qprobe_qwen_block_online import QwenBlockOnlineConfig


class NativeBindingTests(unittest.TestCase):
    def test_qualified_bank_bind_uses_admitted_stat_seal_and_never_falls_back(self):
        from reproduction.a800_common import qprobe_qwen_formal_ops as ops
        with tempfile.TemporaryDirectory() as directory:
            fixture_root=Path(directory).resolve()
            base=fixture_root/'target';(base/'preparation').mkdir(parents=True)
            bank=fixture_root/'bank';cache_path=bank/'token-features';cache_path.mkdir(parents=True)
            manifest=dict(spec_sha=binding.TOKEN_SPEC_SHA,count=169028)
            (cache_path/'manifest.json').write_text(json.dumps(manifest))
            for name in ops.BANK_FILES[1:]:(cache_path/name).write_bytes(b'metadata fixture only')
            files=[]
            for name in ops.BANK_FILES:
                path=cache_path/name;st=path.stat()
                files.append(dict(path=str(path),device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns))
            run=dict(source_commit='a'*40,source_sha256=dict(producer='b'*64))
            intent=dict(base=str(base),source_commit=run['source_commit'],source_sha256=run['source_sha256'],bank_base=str(bank),qualified_bank_origin=dict(source_commit=ops.QUALIFIED_BANK_SOURCE))
            intent_path=base/'preparation/intent.json';intent_path.write_text(json.dumps(intent))
            admitted=dict(cache_dir=str(cache_path),files=files,source_commit=run['source_commit'],reused_qualified_bank=True,
                original_full_sha_seal_revalidated=True,bank_reencoded=False,packed_features_rehashed=False,
                actor_or_head_updates_inherited=False,formal_completed_outers=0,feature_spec_sha256=binding.TOKEN_SPEC_SHA,
                feature_manifest_sha256=binding.sha_file(cache_path/'manifest.json'),complete_sha256=ops.QUALIFIED_BANK_COMPLETE,
                qualified_bank_origin=dict(source_commit=ops.QUALIFIED_BANK_SOURCE,admitted_sha256=ops.QUALIFIED_BANK_ADMIT))
            cache=SimpleNamespace(path=cache_path,manifest=manifest,verify_all=Mock())
            def required(_base,_intent):
                self.assertEqual((_base,_intent),(base,intent));ops.check_bank_file_seal(admitted);return admitted
            with patch.object(ops,'validate_intent') as validate,patch.object(ops,'required_bank',side_effect=required):
                proof=binding._verify_token_bank(base,run,cache)
                self.assertTrue(proof['reused_qualified_bank']);self.assertFalse(proof['full_file_sha_verified_in_this_bind'])
                validate.assert_called_once_with(intent);cache.verify_all.assert_not_called()
                admitted['original_full_sha_seal_revalidated']=False
                with self.assertRaisesRegex(ValueError,'SEAL_NOT_ADMITTED'):binding._verify_token_bank(base,run,cache)
                admitted['original_full_sha_seal_revalidated']=True
                packed=cache_path/'packed_features.npy';st=packed.stat();os.utime(packed,ns=(st.st_atime_ns,st.st_mtime_ns+1))
                with self.assertRaises(AssertionError):binding._verify_token_bank(base,run,cache)
                cache.verify_all.assert_not_called()
            intent_path.write_text('{}')
            ordinary=binding._verify_token_bank(base,run,cache)
            self.assertFalse(ordinary['reused_qualified_bank']);cache.verify_all.assert_called_once_with()

    def metadata(self):
        sources = {'fixture-'+str(index):'7'*64 for index in range(489)}
        sources['search_g1/qprobe_qwen_block_model.py'] = binding.NATIVE_MODEL_SHA
        source_map = hashlib.sha256(binding._canonical(sources)).hexdigest()
        publication = dict(source_commit=binding.WARMUP_SOURCE,remote_commit=binding.WARMUP_SOURCE,
            source_map_sha256=source_map,runtime_file_count=490,remote_fresh_object_store_disk_git_hash_equal=True,
            source_sha256=sources)
        task = dict(source_commit=binding.WARMUP_SOURCE,teacher_identity_sha256=binding.TEACHER_SHA,
            epochs=10,batch_size=256,micro_batch_size=32,learning_rate=3e-4,auxiliary_weight=.3,
            target_statistic='sample_std',head_seed=1729,top_k=512,exploration_questions=0,block_index=35,
            architecture='native_transformers_Qwen2DecoderLayer',native_attention_implementation='sdpa',global_max_tokens=244)
        native = dict(transformers_version='4.46.3',attention_implementation='sdpa',
            implementations=dict(decoder_layer=dict(qualified_name=binding.NATIVE_CLASS)))
        config = dict(model_type='qwen2',hidden_size=2048,num_hidden_layers=36,num_attention_heads=16,
            num_key_value_heads=2,intermediate_size=11008,rms_norm_eps=1e-6,rope_theta=1e6,
            max_position_embeddings=32768,hidden_act='silu',attention_dropout=0.,use_sliding_window=False)
        original = dict(source_layer=35,tensor_count=12,tensor_index_sha256='d'*64,
            official_shards={'official.safetensors':dict(sha256='f'*64)},tensors={})
        copied = dict(source_layer=35,tensor_count=12,all_copied_parameters_trainable=True,tensors={})
        for name,shape in binding._block_shapes().items():
            original['tensors'][name] = dict(source_key='model.layers.35.'+name,shape=shape,source_dtype='bfloat16',
                source_sha256='c'*64,bytes=math.prod(shape)*2,shard='official.safetensors')
            copied['tensors'][name] = dict(original['tensors'][name],copied_fp32_sha256='e'*64,exact_cast_equal=True)
        checks = dict(passed=True,tests_run=2,skipped=0,failures=0,errors=0,cuda_initialized=False,log_sha256='b'*64)
        prepared = dict(stage='QWEN_LAST_BLOCK_WARMUP10_CPU_PREPARED',source_map_sha256=source_map,
            source_sha256=sources,task_config=task,native_implementation=native,official_model_config=config,
            official_model_config_sha256='a'*64,official_block_source=original,cpu_checks=checks,
            feature_cache_dir='/data/frozen-token-cache',token_feature_manifest_sha256='9'*64)
        ready = dict(stage='QWEN_LAST_BLOCK_FRESH_HEAD_WARMUP10_COMPLETE',source_commit=binding.WARMUP_SOURCE,
            source_map_sha256=source_map,questions=5120,teacher_trajectories_reused=25600,trajectories_generated=0,
            block_index=35,global_max_tokens=244,token_feature_spec_sha256=binding.TOKEN_SPEC_SHA,
            head_optimizer_updates=200,head_adam_steps=200,epoch_records=11,minibatch_records=200,
            old_head_inherited=False,encoder_reencoded=False,no_reencode=True,encoder_loaded_in_this_process=False,
            trajectories_regenerated=False,formal_training_started=False,task_config=task,
            replay_contract=dict(window_completed_formal_outers=8,teacher_warmup_in_replay=False,ages_included=list(range(8)),
                normalize_weights=False,weight_formula='2*2**(-age) if 0 <= age <= 7 else 0',reduction='mean(weight * per_group_loss)'),
            native_block_proof=dict(native_class=binding.NATIVE_CLASS,block_index=35,official_revision=binding.OFFICIAL_REVISION,
                source_tensors_verified=True,official_block_copy_verified=True,head_initialized=True,fresh_linear=True,
                fresh_adam=True,transformers_version='4.46.3'),native_implementation=native,
            official_model_config_sha256='a'*64,official_block_source=original,initial_block_proof=copied,
            initial_readout_proof=dict(fresh=True,seed=1729,weight_sha256='1'*64,bias_sha256='2'*64),cpu_checks=checks,
            head_sha256='3'*64,metrics_sha256='4'*64,prepared_sha256='5'*64,token_feature_manifest_sha256='9'*64,
            feature_cache_dir='/data/frozen-token-cache')
        run = dict(initial_model_sha256={'config.json':'a'*64,'model.safetensors.index.json':'d'*64,'official.safetensors':'f'*64})
        return ready,prepared,publication,run,source_map

    def test_exact_native_recipe_and_current_experiment_metadata_fail_closed(self):
        recipe = binding.recipe('/data/full-bank/token-features')
        self.assertEqual((recipe['seed'],recipe['head_seed']),(20261013,1729))
        self.assertEqual((recipe['head_hidden_size'],recipe['hidden_size'],recipe['feature_dim']),(2048,2048,2048))
        self.assertEqual((recipe['num_attention_heads'],recipe['num_key_value_heads'],recipe['intermediate_size']),(16,2,11008))
        self.assertEqual(recipe['head_device'],'cuda:0')
        self.assertTrue(recipe['teacher_replay_excluded'])
        self.assertEqual(recipe['replay_window_outer'],8)
        self.assertNotIn('replay_half_life_outer',recipe)
        self.assertEqual(QwenBlockOnlineConfig.from_mapping(recipe).seed,1729)
        recipe.update(initial_head_checkpoint='/data/new/initial.pt',initial_head_checkpoint_sha256='a'*64,
            initial_head_ready='/data/new/ready.json',initial_head_ready_sha256='b'*64)
        self.assertTrue(arguments(dict(question_probe=recipe),'/data/new'))
        ready,prepared,publication,run,source_map = self.metadata()
        with patch.object(binding,'WARMUP_SOURCE_MAP',source_map):
            self.assertIs(binding.validate_ready_metadata(ready,prepared,publication,run),ready)
            mutations = [
                ('source_commit','f3807ac6642761fcf06a7772255879b1cf291105'),('head_adam_steps',199),
                ('epoch_records',21),('old_head_inherited',True),('encoder_reencoded',True),
                ('trajectories_generated',25600),('formal_training_started',True)]
            for key,value in mutations:
                bad = copy.deepcopy(ready); bad[key] = value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    binding.validate_ready_metadata(bad,prepared,publication,run)
            for section,key,value in [
                ('task_config','epochs',20),('task_config','micro_batch_size',8),
                ('replay_contract','normalize_weights',True),('replay_contract','teacher_warmup_in_replay',True),
                ('native_block_proof','official_block_copy_verified',False),('initial_readout_proof','fresh',False)]:
                bad = copy.deepcopy(ready); bad[section][key] = value
                with self.subTest(section=section,key=key),self.assertRaises(ValueError):
                    binding.validate_ready_metadata(bad,prepared,publication,run)
            bad = copy.deepcopy(ready); bad['initial_block_proof']['tensors']['self_attn.k_proj.weight']['shape'] = [2048,2048]
            with self.assertRaisesRegex(ValueError,'PARAMETER_COPY'):
                binding.validate_ready_metadata(bad,prepared,publication,run)
        with self.assertRaisesRegex(ValueError,'THIS_NATIVE_WARMUP_READY_PATH'):
            binding._read_warmup(Path('/data/old-token-warmup/ready.json'),'a'*64,run)

    def test_all_actual_histories_and_exclusive_claim_without_model_allocation(self):
        parts = dict(binary_bce=.6,regression_mse_raw=.06,regression_mse_normalized=.2,regression_contribution=.06,binary_accuracy=.7)
        metrics = dict(fit_epochs=10,training_groups=5120,actual_head_adam_updates=200,expected_head_adam_updates=200,
            epoch_history=[dict(parts,epoch=epoch,actual_head_adam_updates=epoch*20,total_loss=.66) for epoch in range(11)],
            minibatch_history=[dict(parts,head_update_ordinal=index+1,head_adam_steps_before=index,
                head_adam_steps_after=index+1,epoch=index//20+1,minibatch_index=index%20,batch_size=256,
                pre_step_loss=.66) for index in range(200)])
        binding._histories(metrics)
        for field,value in [('head_adam_steps_after',199),('pre_step_loss',float('nan')),('binary_accuracy',1.1)]:
            bad = copy.deepcopy(metrics); bad['minibatch_history'][-1][field] = value
            with self.subTest(field=field),self.assertRaises(ValueError): binding._histories(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'claim.json'; binding._once(path,dict(attempt=1))
            with self.assertRaises(FileExistsError): binding._once(path,dict(attempt=2))
            self.assertEqual(json.loads(path.read_bytes()),dict(attempt=1))


if __name__ == '__main__': unittest.main()
