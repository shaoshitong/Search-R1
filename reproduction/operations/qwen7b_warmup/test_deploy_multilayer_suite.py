"""Small metadata/config/shell/closure tests; no live assets or GPU loads."""
import copy
import subprocess
import unittest
from .multilayer_train_prepare import TARGET,LAYER_SHA,check_complete,make_config
from .replay_layer_train import render_sbatch
from .deploy_multilayer_suite import self_test


def complete():
    return dict(status='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED',
        questions=9088,trajectories=45440,roles=dict(fit=6528,diagnostic512=512,selection2048=2048),
        full_replay_shared_file_SHA_verified=True,source_runtime='huggingface_frozen_causal_prefill',
        native_bitexact_equivalence_claim=False,actor_updates=0,new_responses_generated=0,
        features_by_capture_layer={layer:dict(path='/remote/features-'+layer,bytes=38306,sha256=sha) for layer,sha in LAYER_SHA.items()})


class MultilayerDeploymentTests(unittest.TestCase):
    def test_complete_strict_backend_count_and_layer_SHA_gate(self):
        source=complete();check_complete(source)
        for key,value in [('status','PARTIAL'),('questions',9087),('native_bitexact_equivalence_claim',True),
                          ('source_runtime','native_vllm'),('full_replay_shared_file_SHA_verified',False)]:
            bad=copy.deepcopy(source);bad[key]=value
            with self.assertRaises(ValueError):check_complete(bad)
        bad=copy.deepcopy(source);bad['features_by_capture_layer']['20']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'SHA'):check_complete(bad)

    def test_three_configs_same_plan_budget_head_seed_but_correct_layer_shards(self):
        reference=dict(plan=dict(path='/remote/reference-plan',sha256='1'*64),bank=dict(path='/remote/bank',sha256='2'*64),
            selection_bank=dict(path='/remote/selection',sha256='3'*64))
        metadata={name:dict(path='/remote/model/'+name,sha256='4'*64) for name in ('config.json','official-inventory.json','model.safetensors.index.json')}
        index=dict(weight_map={'model.layers.%d.self_attn.q_proj.weight'%layer:'shard-%d.safetensors'%layer for layer in (21,24,27)})
        files={'shard-%d.safetensors'%layer:dict(bytes=100,sha256=str(layer)*32) for layer in (21,24,27)}
        configs=[make_config(dict(path='/remote/reference',sha256='5'*64),reference,
            dict(path='/remote/plan4',sha256='6'*64),dict(path='/remote/complete',sha256='7'*64),complete(),
            '/remote/model',metadata,files,index,layer,{},'a'*40) for layer in (21,24,27)]
        for key in ('source_matched_config','reference_plan','plan','bank','selection_bank','epochs','update_budget','global_groups','micro_groups','head_seed','training_seed','loss_kwargs'):
            self.assertTrue(all(c[key]==configs[0][key] for c in configs))
        for layer,capture,c in zip((21,24,27),(20,23,26),configs):
            self.assertEqual(c['replay_features'],complete()['features_by_capture_layer'][str(capture)])
            self.assertEqual(set(c['official_block_shards']),{'shard-%d.safetensors'%layer})
            self.assertEqual(c['Adam_initial_step'],0);self.assertFalse(c['E1_checkpoint_inherited'])
            self.assertEqual((c['epochs'],c['update_budget'],c['global_groups'],c['micro_groups']),(4,512,51,2))
            self.assertNotIn('source_checkpoint',c)

    def test_normal_no_dependency_three_one_GPU_two_hour_scripts(self):
        for layer in (21,24,27):
            script=render_sbatch(TARGET+'/layer'+str(layer),TARGET+'/code-'+'a'*12,layer)
            subprocess.run(['bash','-n'],input=script,text=True,check=True)
            self.assertIn('#SBATCH --qos=normal\n',script);self.assertIn('#SBATCH --gres=gpu:1\n',script)
            self.assertIn('#SBATCH --time=02:00:00\n',script);self.assertNotIn('--dependency',script)

    def test_published_AST_closure_and_actual_shell_command_bounds(self):
        result=self_test();self.assertFalse(result['training_launched']);self.assertFalse(result['remote_actions_executed'])
        self.assertEqual(result['chunk_bytes'],4096);self.assertLess(max(result['command_bytes'].values()),16000)


if __name__=='__main__':unittest.main()
