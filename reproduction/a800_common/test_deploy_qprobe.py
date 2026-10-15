"""CPU-only checks for the prospective control boundary; no SSH or model load."""
import ast
import copy
import hashlib
import inspect
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from reproduction.a800_common import deploy_qprobe as control
from reproduction.a800_common import precompute_qprobe as features


def intent():
    identifier = '1' * 32
    return dict(role=control.ROLE, host='baai-a800', deployment_id=identifier,
        source_commit='2' * 40, source_sha256={'one.py':'3' * 64},
        baseline_intent_sha256='4' * 64, published_ref='refs/heads/codex/new-qprobe',
        base='/data/shaoshitong/Search-G1-a800-fresh200-' + identifier,
        source_base='/data/shaoshitong/Search-G1-qprobe-source-' + identifier,
        baseline_base=control.BASELINE, baseline_commit=control.BASELINE_COMMIT,
        initial_probes_source_base=control.INITIAL_PROBES, teacher_warmup_cache=control.TEACHER_CACHE,
        launch_services=False, target_updates=200,gpu_placement_policy=control.PLACEMENT_POLICY,gpu_placement_bound=False)


def feature_origin():
    old = dict(run_id=control.FEATURE_SOURCE_RUN,execution_id=control.FEATURE_SOURCE_EXECUTION,
        source_commit=control.FEATURE_SOURCE_COMMIT,profile='common8',execution_backend='ssh_local',
        training_world_size=4,completed_updates=0,initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',
        reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
        initial_model_sha256={str(i):'a'*64 for i in range(9)},input_sha256={'train.parquet':'b'*64},
        calibration_question_ids=['one'],source_sha256={name:'c'*64 for name in control.FEATURE_PRODUCERS})
    new = dict(copy.deepcopy(old),run_id='d'*32,execution_id='e'*32,source_commit='f'*40)
    ready = dict(run_id=old['run_id'],source_commit=old['source_commit'],
        cache_dir=control.FEATURE_SOURCE_BASE+'/question-features',pid=2861772,start_ticks=604134066,sid=2861771,
        stage='question_feature_cache_complete',count=169028,feature_dim=2048,completed_training_outers=0,
        target_training_outers=200,exit_code=0,actual_visible_cuda_count=1,physical_index=6,
        physical_gpu_uuid=control.FEATURE_UUID,actual_cuda_uuids=[control.FEATURE_UUID],
        manifest_sha256=control.FEATURE_MANIFEST_SHA,feature_file_sha256=control.FEATURE_FILE_SHA,spec_sha=control.FEATURE_SPEC_SHA)
    terminal = dict(run_id=old['run_id'],execution_id=old['execution_id'],source_commit=old['source_commit'],
        exit_code=0,status='FEATURE_CHILD_EXITED',child=dict(pid=2861772,start_ticks=604134066,sid=2861771))
    model_sha = hashlib.sha256(json.dumps(new['initial_model_sha256'],sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    manifest = dict(count=169028,feature_dim=2048,spec_sha=control.FEATURE_SPEC_SHA,
        files={'features.npy':dict(bytes=100,sha256=control.FEATURE_FILE_SHA),'index.jsonl':dict(bytes=20,sha256='1'*64)},
        spec=dict(train_parquet_sha256='b'*64,encoder_compute_dtype='bfloat16',feature_dtype='float16',
            encoder_files_sha256=model_sha,encoder_revision=new['model_revision'],feature_layer=27,max_prompt_tokens=4096))
    return old,new,ready,terminal,manifest


class ProspectiveControlTests(unittest.TestCase):
    def test_fresh_sigma3_intent_is_exact_and_excludes_restore_or_different_cache(self):
        selected=dict(intent(),fresh_question_probe_loss=dict(control.SIGMA3_LOSS))
        self.assertIs(control.validate_intent(selected),selected)
        namespace={};exec(control.REMOTE_COMMON,namespace)
        namespace['check'](selected)
        mutations=(lambda row:row.update(saved_checkpoint_restore=dict(source_receipt='/data/prior/receipt.json',source_receipt_sha256='a'*64)),
            lambda row:row['fresh_question_probe_loss'].update(std_loss_weight=0.1),
            lambda row:row.update(teacher_warmup_cache='/data/other/teacher'),
            lambda row:row.update(initial_probes_source_base='/data/shaoshitong/Search-G1-a800-fresh200-'+'a'*32))
        for index,mutate in enumerate(mutations):
            changed=copy.deepcopy(selected);mutate(changed)
            with self.subTest(index=index),self.assertRaises(AssertionError):control.validate_intent(changed)
            with self.subTest(remote=index),self.assertRaises(AssertionError):namespace['check'](changed)
        with self.assertRaisesRegex(AssertionError,'excludes saved-state'):
            control.stage(SimpleNamespace(fresh_sigma3=True,restore_receipt='/data/prior/receipt.json'))

    def test_fresh_sigma3_gate_rejects_old_state_and_requires_loss_specific_review(self):
        selected=dict(intent(),fresh_question_probe_loss=dict(control.SIGMA3_LOSS))
        run=dict(source_commit=selected['source_commit'],profile='common8',execution_backend='ssh_local',
            completed_updates=0,target_updates=200,training_world_size=4,
            initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',
            model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
            question_probe=dict(control.SIGMA3_LOSS,teacher_warmup_cache=control.TEACHER_CACHE))
        self.assertIs(control.validate_fresh_sigma3_run(selected,run),run)
        for name in ('actor_checkpoint','reference_checkpoint','resume_path','inherited_policy_run_id',
                     'inherited_optimizer_run_id','inherited_scheduler_run_id','full_state_restore'):
            with self.subTest(name=name),self.assertRaises(AssertionError):
                control.validate_fresh_sigma3_run(selected,dict(run,**{name:'prior'}))
        for name in ('initial_head_checkpoint','snapshot_path','restore_state'):
            changed=copy.deepcopy(run);changed['question_probe'][name]='prior'
            with self.subTest(name=name),self.assertRaises(AssertionError):control.validate_fresh_sigma3_run(selected,changed)
        review=dict(role='a800_qprobe_independent_code_review',approved=True,independent=True,
            reviewer='independent',implementation_actor='implementation',source_commit=selected['source_commit'],
            source_sha256=selected['source_sha256'],reviewed_unix=time.time()-1,
            checks={key:True for key in control.REVIEW_CHECKS+control.SIGMA3_REVIEW_CHECKS})
        self.assertIs(control.validate_review(review,selected),review)
        review['checks'].pop(control.SIGMA3_REVIEW_CHECKS[0])
        with self.assertRaises(AssertionError):control.validate_review(review,selected)

    def test_teacher_identity_excludes_head_loss_but_keeps_rollout_and_scorer_sha(self):
        from search_g1.qprobe_runtime import strict_scoring_sources
        from search_g1.qprobe_warmup import teacher_identity
        manifest=dict(model_revision='official',training_world_size=4,initial_model_sha256={'model':'a'*64},
            input_sha256={'train.parquet':'b'*64},budget={'common_manifest_sha256':'c'*64},retriever_version='d'*64,
            source_sha256=dict(strict_scoring_sources(),**{'search_g1/qprobe_model.py':'1'*64,
                'reproduction/common8/qprobe_configuration.py':'2'*64,'reproduction/common8/qprobe_bind.py':'3'*64,
                'search_g1/qprobe_warmup.py':'4'*64,'search_g1/qprobe_teacher_worker.py':'5'*64}))
        kwargs=dict(feature_spec_sha='f'*64,sampling_config={'n_agent':5},agent_config={'max_prompt_length':4096},
                    max_turns=4,seed=20261013,questions=[{'qid':'one','question_hash':'q'}])
        expected=teacher_identity(manifest,**kwargs)
        changed=copy.deepcopy(manifest);changed['question_probe']=dict(control.SIGMA3_LOSS)
        for name in ('search_g1/qprobe_model.py','reproduction/common8/qprobe_configuration.py','reproduction/common8/qprobe_bind.py'):
            changed['source_sha256'][name]='9'*64
        self.assertEqual(expected,teacher_identity(changed,**kwargs))
        changed['source_sha256']['search_g1/qprobe_teacher_worker.py']='9'*64
        self.assertNotEqual(expected,teacher_identity(changed,**kwargs))
        changed['source_sha256']['search_g1/signals.py']='9'*64
        with self.assertRaisesRegex(ValueError,'STRICT_SCORER_DIFFERS'):teacher_identity(changed,**kwargs)

    def test_prepared_micro4_derives_and_resolves_the_actual_four_rank_contract(self):
        from reproduction.a800_common import prepare_run as preparation
        from reproduction.common8.formal_contract import formal_mathematics,verify_formal_config
        from reproduction.common8.configuration import arguments
        function=ast.parse(inspect.getsource(preparation.prepare)).body[0]
        identity_node=next(node for node in function.body if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='identity' for target in node.targets))
        identity=eval(compile(ast.Expression(identity_node.value),'actual_prepare_identity','eval'),vars(preparation))
        derivation=next(node for node in ast.walk(function) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Name) and node.func.id=='formal_mathematics')
        self.assertLess(identity_node.lineno,derivation.lineno)
        self.assertEqual(ast.dump(derivation.args[0]),ast.dump(ast.Name(id='identity',ctx=ast.Load())))
        self.assertEqual(identity['actor_micro_batch_per_rank'],4)
        repo=Path(control.__file__).resolve().parents[2]
        budget=json.loads((repo/'reproduction/common8/budget.json').read_text());budget.update(formal_mathematics(identity))
        self.assertEqual((budget['per_rank_actor_micro_batch'],budget['ppo_micro_batch_size'],budget['per_rank_log_prob_micro_batch']),(4,16,8))
        self.assertEqual((128//budget['world_size'])//budget['per_rank_actor_micro_batch'],8)
        self.assertEqual(budget['objective_reduction'],'original four-row micro masked_mean')
        self.assertEqual((budget['optimizer_opportunities_per_outer'],budget['lr_warmup_steps']),(20,57))
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve();(base/'logs').mkdir()
            run=dict(identity,budget=budget,contract_kind='fresh200_warmup57',policy_safety_mode='inline_first_five',
                completed_updates=0,target_updates=200,acceptance_update=5,model_revision=budget['model_revision'],
                initial_model='/data/search-g1/assets/qwen2.5-3b-instruct',reference_model='/data/search-g1/assets/qwen2.5-3b-instruct',
                checkpoint_dir=str(base/'checkpoints/search-g1-common8'),vllm_gpu_memory_utilization=0.25,
                source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in
                    ('reproduction/common8/recipe.json','reproduction/common8/budget.json','verl/trainer/config/ppo_trainer.yaml')})
            (base/'logs/run-manifest.json').write_text(json.dumps(run))
            _,overrides=arguments(base);document=dict(do_search=True)
            for assignment in overrides:
                key,raw=assignment.split('=',1)
                try:value=json.loads(raw)
                except ValueError:
                    try:value=ast.literal_eval(raw)
                    except (ValueError,SyntaxError):value=raw
                parts=key.lstrip('+').split('.');parent=document
                for part in parts[:-1]:parent=parent.setdefault(part,{})
                parent[parts[-1]]=value
            defaults=(repo/'verl/trainer/config/ppo_trainer.yaml').read_text()
            document['actor_rollout_ref']['actor']['clip_ratio']=float(re.search(r'^\s+clip_ratio: ([0-9.]+)$',defaults,re.MULTILINE).group(1))
            verify_formal_config(document,run,base)
            self.assertEqual(document['actor_rollout_ref']['actor']['ppo_micro_batch_size'],16)
            self.assertEqual(document['actor_rollout_ref']['ref']['log_prob_micro_batch_size'],32)
            document['actor_rollout_ref']['actor']['ppo_micro_batch_size']=8
            with self.assertRaises(AssertionError):verify_formal_config(document,run,base)

    def test_actual_flexible_mapping_drives_cuda_and_receipt_without_claiming_stage_indices(self):
        for inventory_count,selected in ((8,[0,2,3,4]),(8,[1,3,4,7]),(10,[0,2,8,9])):
            physical=[[str(index),'GPU-'+str(index),'81920','4','0'] for index in range(inventory_count)]
            run=dict(profile='common8',execution_backend='ssh_local',slurm_job_id=None,execution_id='a'*32,
                training_world_size=4,gpu_placement_policy=control.PLACEMENT_POLICY,gpu_placement_bound=True,
                retriever_physical_indices=[5,6],probe_physical_index=6,
                training_physical_indices=selected,auxiliary_physical_indices=[5,6],physical_gpu_inventory=physical,
                assigned_gpu_uuids={str(index):physical[index][1] for index in selected+[5,6]},
                training_gpu_uuids=[physical[index][1] for index in selected],
                excluded_physical_indices=sorted(set(range(inventory_count))-set(selected+[5,6])),
                gpu_placement_binding_receipt='logs/local-gpu-placement-binding.json',
                gpu_placement_binding_sha256='a'*64,gpu_placement_bound_unix=time.time())
            placement=control.original_deploy.launch_placement(run)
            self.assertEqual(placement['cuda_visible_devices'],','.join(map(str,selected)))
            self.assertEqual(placement['training_physical_indices'],selected)
            self.assertEqual(placement['training_gpu_uuids'],run['training_gpu_uuids'])
            for patch in ({'gpu_placement_bound':False},{'training_physical_indices':[0,2,3,5]},
                          {'training_gpu_uuids':['GPU-0']*4},{'excluded_physical_indices':[4,7]}):
                with self.assertRaises(AssertionError):control.original_deploy.launch_placement(dict(run,**patch))

    def test_qprobe_binding_precedes_claim_and_parent_cuda_comes_from_binding(self):
        gate=ast.parse(control.REMOTE_LAUNCH_GATE)
        bind_line=next(node.lineno for node in ast.walk(gate) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='bind_idle_local_placement')
        claim_line=next(node.lineno for node in ast.walk(gate) if isinstance(node,ast.Constant) and node.value=='logs/qprobe-launch-claim.json')
        self.assertLess(bind_line,claim_line)
        for code in (control.original_deploy.REMOTE_EXECUTE,control.original_deploy.REMOTE_LAUNCH_PREPARED):
            ast.parse(code)
            self.assertNotIn("CUDA_VISIBLE_DEVICES='0,1,2,3'",code)
            self.assertIn("placement['cuda_visible_devices']",code)

    def test_embedded_remote_programs_compile(self):
        for name in ('REMOTE_STAGE','REMOTE_PREPARE','REMOTE_FEATURE_ADOPT','REMOTE_BIND','REMOTE_INSPECT','REMOTE_LAUNCH_GATE'):
            ast.parse(getattr(control, name), filename=name)

    def test_feature_adoption_rejects_changed_origin_pipeline_assets_or_failed_encoding(self):
        original = feature_origin()
        self.assertTrue(control.validate_feature_origin(*original))
        mutations = (
            lambda rows: rows[0].update(source_commit='0'*40),
            lambda rows: rows[1]['initial_model_sha256'].update({'0':'0'*64}),
            lambda rows: rows[1]['input_sha256'].update({'train.parquet':'0'*64}),
            lambda rows: rows[1]['source_sha256'].update({control.FEATURE_PRODUCERS[0]:'0'*64}),
            lambda rows: rows[2].update(start_ticks=604134067),
            lambda rows: rows[2].update(actual_cuda_uuids=['wrong-uuid']),
            lambda rows: rows[3].update(exit_code=1),
            lambda rows: rows[4].update(spec_sha='0'*64),
            lambda rows: rows[4]['spec'].update(encoder_files_sha256='0'*64),
            lambda rows: rows[4]['spec'].update(encoder_compute_dtype='float32'),
            lambda rows: rows[4]['files']['features.npy'].update(bytes=0),
            lambda rows: rows[4]['files']['index.jsonl'].update(bytes=-1),
            lambda rows: rows[4]['files']['features.npy'].update(sha256='0'*64),
        )
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed=copy.deepcopy(original);mutate(changed)
                with self.assertRaises(AssertionError):
                    control.validate_feature_origin(*changed)

    def test_copied_features_keep_real_origin_encoding_and_new_cpu_copy_separate(self):
        old,new,ready,terminal,manifest=feature_origin()
        self.assertIs(control.validate_prepared_features(ready,old),ready)
        copied=dict(ready,run_id=new['run_id'],execution_id=new['execution_id'],source_commit=new['source_commit'],
            cache_dir=intent()['base']+'/question-features',kind='copied_verified_existing_encoding',
            fresh_gpu_encoding=False,actual_visible_cuda_count=None,model_loaded=False,
            copy_process=dict(pid=100,start_ticks=200,sid=99,uid=0),
            origin=dict(base=control.FEATURE_SOURCE_BASE,run_id=old['run_id'],execution_id=old['execution_id'],
                source_commit=old['source_commit'],encoding_pid=2861772,encoding_start_ticks=604134066,
                encoding_exit_code=0,encoding_cuda_uuid=control.FEATURE_UUID,
                immutable_encoder_producer_sha256=new['source_sha256']))
        for key in ('pid','sid','start_ticks','physical_index','physical_gpu_uuid','actual_cuda_uuids'):
            copied.pop(key)
        self.assertIs(control.validate_prepared_features(copied,new),copied)
        mutations=(lambda row:row.update(fresh_gpu_encoding=True),lambda row:row.update(actual_visible_cuda_count=1),
            lambda row:row['origin'].update(source_commit='0'*40),lambda row:row['origin'].update(encoding_start_ticks=200),
            lambda row:row['copy_process'].update(pid=2861772,start_ticks=604134066,sid=2861771),
            lambda row:row['origin']['immutable_encoder_producer_sha256'].update({control.FEATURE_PRODUCERS[0]:'0'*64}))
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed=copy.deepcopy(copied);mutate(changed)
                with self.assertRaises(AssertionError):control.validate_prepared_features(changed,new)

    def test_independent_review_cannot_approve_other_source_or_partial_contract(self):
        original = intent()
        review = dict(role='a800_qprobe_independent_code_review', approved=True, independent=True,
            reviewer='review-agent', implementation_actor='implementation-agent',
            source_commit=original['source_commit'], source_sha256=original['source_sha256'],
            checks={key:True for key in control.REVIEW_CHECKS}, reviewed_unix=time.time()-1)
        self.assertIs(control.validate_review(review, original), review)
        for patch in ({'source_commit':'5'*40},{'reviewer':'implementation-agent'},
                      {'checks':dict(review['checks'], frozen_teacher_5120_x_5=False)},
                      {'reviewed_unix':float('nan')}):
            changed = dict(review, **patch)
            with self.assertRaises(AssertionError):
                control.validate_review(changed, original)

    def test_feature_plan_never_uses_training_gpu_or_foreign_assets(self):
        plan = control.feature_plan(intent())
        self.assertFalse(plan['gpu_job_started'])
        self.assertEqual(plan['required_environment']['CUDA_VISIBLE_DEVICES'], '6')
        self.assertIn('reproduction.a800_common.precompute_qprobe', plan['argv'])
        self.assertTrue(plan['progress_path'].startswith(intent()['base']+'/'))
        with self.assertRaises(AssertionError):
            control.feature_plan(intent(), 0)

    def test_shared_teacher_cache_is_immutable_intent_and_rejects_controller_paths(self):
        first = intent();second = intent()
        second['deployment_id'] = '6'*32
        second['base'] = '/data/shaoshitong/Search-G1-a800-fresh200-' + second['deployment_id']
        second['source_base'] = '/data/shaoshitong/Search-G1-qprobe-source-' + second['deployment_id']
        self.assertEqual(control.validate_intent(first)['teacher_warmup_cache'],
                         control.validate_intent(second)['teacher_warmup_cache'])
        for path in ('/Users/shaoshitong/cache','/data/cache/../other','relative-cache'):
            with self.assertRaises(AssertionError):
                control.validate_teacher_cache(path)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ):
            namespace = {};exec(control.REMOTE_COMMON,namespace)
            cache = Path(directory).resolve()/'shared'/'teacher-v2'
            selection = dict(teacher_warmup_cache=str(cache))
            namespace['teacher_storage'](selection,create_parent=True)
            cache.mkdir();marker = cache/'identity.json';marker.write_text('accepted-existing-cache')
            namespace['teacher_storage'](selection,create_parent=True)
            self.assertEqual(marker.read_text(),'accepted-existing-cache')
            alias = Path(directory)/'cache-alias';alias.symlink_to(cache,target_is_directory=True)
            with self.assertRaises(AssertionError):
                namespace['teacher_storage'](dict(teacher_warmup_cache=str(alias)))

    def test_feature_cuda_uuid_must_equal_admitted_physical_uuid(self):
        admitted = 'GPU-00000000-0000-0000-0000-000000000005'
        other = 'GPU-00000000-0000-0000-0000-000000000006'
        with mock.patch.dict(os.environ), mock.patch.object(features.sys,'modules',{}):
            reader = mock.Mock(return_value=[admitted])
            self.assertEqual(features.bind_cuda_uuid(admitted,reader),[admitted])
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],admitted)
            reader.assert_called_once_with(1)
            with self.assertRaisesRegex(AssertionError,'Actual CUDA UUID differs'):
                features.bind_cuda_uuid(admitted,lambda count:[other])

    def test_launch_pins_clean_source_and_rejects_cached_modified_deployer(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('config','user.name','test');git('config','user.email','test@example.invalid')
            original_repo = Path(control.__file__).resolve().parents[2]
            for relative in ('reproduction/a800_common/deploy.py','reproduction/a800_common/deploy_qprobe.py',
                             'verl/version/version','verl/single_controller/version/version'):
                target = repo/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((original_repo/relative).read_bytes())
            git('add','.');git('commit','-m','published fixture')
            commit = git('rev-parse','HEAD')
            selected = dict(source_commit=commit,source_sha256=control.published_source(repo,commit))
            with mock.patch.object(control.original_deploy,'__file__',str(repo/'reproduction/a800_common/deploy.py')):
                self.assertEqual(control.verified_launcher(repo,selected)[0],commit)
                with mock.patch.object(control.original_deploy,'REMOTE_LAUNCH_PREPARED',control.REMOTE_LAUNCH_PREPARED+'\n# dirty cached import'):
                    with self.assertRaises(AssertionError):
                        control.verified_launcher(repo,selected)
                with (repo/'reproduction/a800_common/deploy.py').open('a') as stream:
                    stream.write('\n# uncommitted imported deployer\n')
                with self.assertRaises(AssertionError):
                    control.verified_launcher(repo,selected)

    def test_timeout_consumes_claim_and_preserves_inspection_path(self):
        with tempfile.TemporaryDirectory() as directory:
            handle = Path(directory)/'handle.json'
            route = ['ssh','baai-a800','python3 -c pass']
            with mock.patch.object(control,'ssh_command',return_value=route), \
                 mock.patch.object(control.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',1)) as call:
                result = control.invoke(handle,intent(),'prepare','pass',b'{}',1)
                self.assertEqual(result['failure_type'],'TimeoutExpired')
                self.assertIsNone(result['exit_code'])
                self.assertTrue(handle.with_name('handle.prepare-once.json').is_file())
                self.assertTrue(call.call_args.args[0][-1].startswith('/data/search-g1/env/bin/python -c '))
                with self.assertRaises(FileExistsError):
                    control.invoke(handle,intent(),'prepare','pass',b'{}',1)
                self.assertEqual(call.call_count,1)

    def test_bundle_contains_exact_generic_ref_with_baseline_prerequisite(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)/'repo';repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.DEVNULL,text=True).strip()
            git('init');git('config','user.name','test');git('config','user.email','test@example.invalid')
            (repo/'one.py').write_text('original = 1\n');git('add','one.py');git('commit','-m','baseline')
            baseline = git('rev-parse','HEAD')
            (repo/'one.py').write_text('original = 2\n');git('add','one.py');git('commit','-m','prospective')
            commit = git('rev-parse','HEAD');ref = 'refs/heads/codex/new-qprobe';git('update-ref',ref,commit)
            bundle = Path(directory)/'source.bundle'
            with mock.patch.object(control,'BASELINE_COMMIT',baseline):
                control.create_bundle(repo,commit,ref,bundle)
            self.assertEqual(subprocess.check_output(['git','bundle','list-heads',str(bundle)],text=True).strip(),commit+' '+ref)
            verify = subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle)],capture_output=True,text=True,check=True)
            self.assertIn(baseline, verify.stdout)
            self.assertLess(bundle.stat().st_size,control.MAX_BUNDLE)


if __name__ == '__main__':
    unittest.main()
