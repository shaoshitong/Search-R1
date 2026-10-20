"""Narrow recipe/source-binding and safe-shell rendering checks only."""
import copy
import inspect
import subprocess
import unittest
from . import deploy_followup_loss_suite as module
from .deploy_followup_loss_suite import (ASSET_KEYS,CRITICAL_FILES,SOURCE_SHA,
    recipe_specs,job_name,render_followup_sbatch,make_variant_config)


class FollowupPreparationTest(unittest.TestCase):
    def test_recipes_preserve_original_assets_plan_budget_and_e1(self):
        hashes={name:'a'*64 for name in CRITICAL_FILES}
        assets={name:dict(path='/remote/'+name,sha256='b'*64) for name in ASSET_KEYS}
        assets['source_checkpoint']['sha256']=SOURCE_SHA
        for base in ('trajectory','joint','event'):
            group=dict(trajectory_weight=1.)
            if base=='joint':group['binary_mixed_weight']=1.
            if base=='event':group['three_event_weight']=1.
            reference=dict(model_dir='/remote/model',update_budget=128,global_groups=51,
                           micro_groups=2,save_every=16,code_hashes=hashes,
                           recipe=dict(id=base,loss_kwargs=group),**assets)
            source=copy.deepcopy(reference);specs=recipe_specs(base)
            self.assertEqual(len(specs),7);self.assertEqual(len(recipe_specs(base,True)),8)
            for name in recipe_specs(base,True):
                config=make_variant_config(reference,base,name,hashes,'c'*40,dict(path='/remote/config',sha256='d'*64))
                for key in ASSET_KEYS:self.assertEqual(config[key],reference[key])
                self.assertEqual(config['update_budget'],128)
                self.assertEqual(config['recipe']['loss_kwargs']['trajectory_weight'],1.)
                self.assertIn('NOT the completed128-step result',config['followup_start'])
                self.assertTrue(config['no_automatic_submission'])
            self.assertEqual(reference,source)
            drift=dict(hashes);drift[CRITICAL_FILES[0]]='e'*64
            with self.assertRaises(ValueError):make_variant_config(reference,base,'within1',drift,'c'*40,{})
        self.assertEqual(recipe_specs('event',True)['inter1-hard16']['inter_hard_negative_k'],16)
        self.assertEqual(recipe_specs('joint')['var5']['stat_variance_weight'],5.)

    def test_sbatch_environment_safe_independent_identity_and_no_dependencies(self):
        base='/data/user/jxu957/experiments/followup/event/within1'
        root='/data/user/jxu957/experiments/followup-repo-1234'
        script=render_followup_sbatch('event','within1',base,root)
        self.assertIn('#SBATCH --job-name='+job_name('event','within1'),script)
        self.assertIn('#SBATCH --gres=gpu:1\n',script)
        self.assertIn('#SBATCH --qos=normal\n',script)
        self.assertIn('--base '+base,script)
        self.assertNotIn('--dependency',script);self.assertNotIn('emergency_acd',script)
        self.assertNotIn('__CODE_ROOT__',script)
        subprocess.run(['bash','-n'],input=script,text=True,check=True)
        exports='\n'.join(line for line in script.splitlines() if line.startswith('export '))
        result=subprocess.run(['bash','-euc',exports+'\nprintf %s "$QWEN7B_CODE_ROOT"'],text=True,capture_output=True,check=True)
        self.assertEqual(result.stdout,root)
        for unsafe in ('/tmp/run','/data/user/jxu957/experiments/x;rm','/data/user/jxu957/experiments/../bad'):
            with self.assertRaises(ValueError):render_followup_sbatch('event','within1',unsafe,root)

    def test_rendered_control_payload_has_complete_function_scope(self):
        functions='\n'.join(inspect.getsource(f) for f in (
            module.render_matched_sbatch,module.recipe_specs,module.job_name,
            module.render_followup_sbatch,module.validate_reference_config,module.make_variant_config))
        prefix="commit='abc'\ngroup_base='event'\ninclude_hard_negative=False\nhashes={}\npacked=''\n"
        prefix+='SOURCE_SHA='+repr(SOURCE_SHA)+'\nASSET_KEYS='+repr(ASSET_KEYS)+'\nCRITICAL_FILES='+repr(CRITICAL_FILES)+'\n'
        shared=prefix+module.COMMON+'\n'+functions+'\n'
        for name,body in [('prepare',module.PREPARE),('status',module.STATUS)]:
            self.assertLess(len((shared+body).encode()),64000)
            compile(shared+body,'remote-'+name,'exec')
        # Definitions only, no prepare/status, SSH, file writes or scheduler calls.
        namespace={};exec(shared,namespace)
        rendered=namespace['render_followup_sbatch']('event','within1',
            '/data/user/jxu957/experiments/followup/event/within1',
            '/data/user/jxu957/experiments/followup-repo-abc')
        self.assertIn('#SBATCH --job-name=q7b-loss128f-event-within1',rendered)


if __name__=='__main__':unittest.main()
