"""CPU dispatch/artifact tests; no fixture proves real native policy execution."""
import json
import stat
import os
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reproduction.common8.run_pipeline import CALIBRATION_ROLES, Pipeline, adoption_inputs, execution_role, main
from reproduction.common8.verify_data import digest


class PipelineCalibrationTests(unittest.TestCase):
    def formal_pipeline(self,root):
        from reproduction.common8.test_configuration import ConfigurationTests
        base,run=ConfigurationTests().fixture(root)
        discovery=base.parent/'old immutable diagnostic';discovery.mkdir();discovery=discovery.resolve()
        descriptor=base/'preparation';descriptor.mkdir()
        for name in ('diagnostic.json','terminal.json'):
            (descriptor/name).write_text('{"scope":"CPU dispatch fixture, not actual admission"}')
        run.update(policy_calibration_discovery_base=str(discovery.resolve()),
            diagnostic_proof=str((descriptor/'diagnostic.json').resolve()),
            diagnostic_terminal_proof=str((descriptor/'terminal.json').resolve()),
            train_log=str(base/'logs/train.log'))
        pipeline=Pipeline.__new__(Pipeline);pipeline.base=base;pipeline.python='cpu-only-python'
        pipeline.run=run;pipeline.start=Mock(return_value=object());pipeline.wait=Mock();pipeline.stage=Mock()
        return pipeline

    def test_inline_formal200_fits_actual_initial_probes_without_old_diagnostic_admission(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline=self.formal_pipeline(root)
            sources=adoption_inputs(pipeline.run,pipeline.base)
            self.assertEqual(sources,{})
            pipeline.initial_probe_calibration()
            command=pipeline.start.call_args.args[1]
            self.assertIn('reproduction.common8.refit',command)
            self.assertIn('--step',command);self.assertIn('0',command)
            self.assertNotIn('--diagnostic-proof',command);self.assertNotIn('--terminal-proof',command)
            pipeline.start.reset_mock();stage=pipeline.train_and_verify()
            self.assertEqual(stage,'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP')
            self.assertEqual([c.args[0] for c in pipeline.start.call_args_list],['trainer','final-acceptance'])

    def test_inline_formal200_unknown_or_missing_mode_rejected_before_launch(self):
        for mode in (None,'legacy'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as root:
                pipeline=self.formal_pipeline(root)
                if mode is None:pipeline.run.pop('policy_safety_mode')
                else:pipeline.run['policy_safety_mode']=mode
                with self.assertRaises((AssertionError,KeyError)):
                    pipeline.initial_probe_calibration()
                pipeline.start.assert_not_called()

    def test_formal_markers_reject_child_role_before_owned_registration_or_native_admission(self):
        for role in CALIBRATION_ROLES:
            for marker in ('top','budget_only','top_warmup_only'):
                with self.subTest(role=role,marker=marker),tempfile.TemporaryDirectory() as root:
                    pipeline=self.formal_pipeline(root);run=pipeline.run;run['policy_calibration_role']=role
                    if marker=='budget_only':
                        run.pop('contract_kind');run.pop('policy_safety_mode');run['target_updates']=1
                    elif marker=='top_warmup_only':
                        run.clear();run.update(policy_calibration_role=role,lr_warmup_steps=57)
                    with patch('reproduction.common8.run_pipeline.OwnedProcesses') as owned:
                        with self.assertRaises((AssertionError,KeyError)):Pipeline(pipeline.base,run)
                    owned.assert_not_called()
                    with self.assertRaises((AssertionError,KeyError)):execution_role(run)

    def test_gpu_sample_file_and_parent_are_synced_and_failures_prevent_finished_stage(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as root:
                base=Path(root);(base/'logs').mkdir();pipeline=Pipeline.__new__(Pipeline)
                pipeline.base=base;pipeline.run=dict(run_id='CPU-sampler-fixture',source_sha256={'fixture.py':'a'*64})
                pipeline.physical_indices={0};pipeline.max_used={};pipeline.sample_error=None
                pipeline.job_id='123';pipeline.local=False
                pipeline.stop_sampling=threading.Event();pipeline.stage=Mock();seen=[];real=os.fsync
                def rows(*args):
                    pipeline.stop_sampling.set()
                    return [['0','constructed-uuid','1','81920','0']]
                def sync(fd):
                    directory=stat.S_ISDIR(os.fstat(fd).st_mode);seen.append('directory' if directory else 'file')
                    if directory and fail:raise OSError('constructed sampler parent durability failure')
                    return real(fd)
                with patch.dict(os.environ,SLURM_JOB_ID='123'),patch('reproduction.common8.run_pipeline.gpu_rows',side_effect=rows), \
                     patch('reproduction.a800.owned_processes.identity',return_value=dict(pid=123,start_ticks=456,sid=123,uid=0)), \
                     patch.object(os,'fsync',side_effect=sync):pipeline.sample()
                self.assertEqual(seen,['file','directory'])
                if fail:
                    self.assertIn('parent durability failure',pipeline.sample_error)
                    with self.assertRaisesRegex(RuntimeError,'finish durably'):pipeline.finish_sampling(None)
                    self.assertEqual(pipeline.stage.call_args.args[0],'FAILED')
                else:
                    self.assertIsNone(pipeline.sample_error);pipeline.finish_sampling(None);pipeline.stage.assert_not_called()

    def test_transient_query_fork_failure_records_gap_then_real_sample(self):
        import errno
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);(base/'logs').mkdir();pipeline=Pipeline.__new__(Pipeline)
            pipeline.base=base;pipeline.run=dict(run_id='CPU-query-retry',source_sha256={'fixture.py':'a'*64})
            pipeline.physical_indices={0};pipeline.max_used={};pipeline.sample_error=None
            pipeline.job_id='123';pipeline.local=False;pipeline.stop_sampling=threading.Event()
            calls=[]
            def rows(*args):
                calls.append(1)
                if len(calls)==1:raise BlockingIOError(errno.EAGAIN,'constructed temporary fork exhaustion')
                pipeline.stop_sampling.set()
                return [['0','constructed-uuid','2','81920','5']]
            with patch('reproduction.common8.run_pipeline.gpu_rows',side_effect=rows), \
                    patch('reproduction.a800.owned_processes.identity',return_value=dict(pid=123,start_ticks=456,sid=123,uid=0)), \
                    patch.object(pipeline.stop_sampling,'wait',return_value=False):pipeline.sample()
            self.assertIsNone(pipeline.sample_error);self.assertEqual(len(calls),2)
            errors=(base/'logs/gpu-sampling-errors.jsonl').read_text().splitlines()
            self.assertEqual(len(errors),1);self.assertTrue(json.loads(errors[0])['retrying'])
            samples=(base/'logs/gpu-samples-CPU-query-retry.jsonl').read_text().splitlines()
            self.assertEqual(len(samples),1);self.assertEqual(json.loads(samples[0])['gpus'][0]['used_mib'],2)

    def test_live_or_failed_sampler_cannot_acknowledge_finished(self):
        for alive,error in ((True,None),(False,'constructed failed sampler')):
            pipeline=Pipeline.__new__(Pipeline);pipeline.stop_sampling=threading.Event();pipeline.sample_error=error;pipeline.stage=Mock()
            sampler=Mock();sampler.is_alive.return_value=alive
            with self.assertRaisesRegex(RuntimeError,'finish durably'):pipeline.finish_sampling(sampler)
            sampler.join.assert_called_once_with(35);self.assertTrue(pipeline.stop_sampling.is_set())
            self.assertEqual(pipeline.stage.call_args.args[0],'FAILED')

    def pipeline(self, base, role=None, adoption_sources=True):
        pipeline=Pipeline.__new__(Pipeline)
        pipeline.base=Path(base);pipeline.python=str(Path(base)/'env/bin/python')
        pipeline.run=dict(run_id='child',train_log=str(Path(base)/'logs/train.log'),initial_model='/original')
        if role is not None: pipeline.run['policy_calibration_role']=role
        if role!=CALIBRATION_ROLES[0] and adoption_sources:
            names=['policy_calibration_discovery_base']
            if role is None:names.append('policy_calibration_confirmation_base')
            for name in names:
                path=Path(base)/name;path.mkdir()
                pipeline.run[name]=str(path.resolve())
        pipeline.start=Mock(return_value=object());pipeline.wait=Mock();pipeline.stage=Mock()
        return pipeline

    def test_production_dispatch_retains_full_verifier(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline=self.pipeline(root)
            pipeline.calibration_measurements=Mock(side_effect=AssertionError('not a child'))
            stage=pipeline.train_and_verify()
        self.assertEqual(stage,'FINISHED_FULL200_FINAL51713_AND_OWNED_CLEANUP')
        self.assertEqual([call.args[0] for call in pipeline.start.call_args_list],['trainer','final-acceptance'])
        self.assertIn('reproduction.common8.verify_full',pipeline.start.call_args_list[1].args[1])
        pipeline.calibration_measurements.assert_not_called()

    def test_children_require_segment_and_never_start_full_verifier(self):
        for role in CALIBRATION_ROLES:
            with self.subTest(role=role),tempfile.TemporaryDirectory() as root:
                pipeline=self.pipeline(root,role);pipeline.calibration_measurements=Mock()
                stage=pipeline.train_and_verify()
                self.assertEqual(stage,'FINISHED_POLICY_CALIBRATION_MEASUREMENTS_AND_OWNED_CLEANUP')
                self.assertEqual([call.args[0] for call in pipeline.start.call_args_list],['trainer'])
                pipeline.calibration_measurements.assert_called_once_with()
                self.assertEqual(pipeline.start.call_args.args[1],['bash','reproduction/common8/train.sh'])
                self.assertNotIn('301',pipeline.start.call_args.args[1])

    def test_missing_confirmation_adoption_constructor_probe_and_train_launch_nothing(self):
        role=CALIBRATION_ROLES[1]
        with tempfile.TemporaryDirectory() as root:
            with patch('reproduction.common8.run_pipeline.OwnedProcesses') as owned,self.assertRaisesRegex(
                    RuntimeError,'Authenticated initial-probe adoption'):
                Pipeline(Path(root),dict(policy_calibration_role=role))
            owned.assert_not_called()
            pipeline=self.pipeline(root,role,adoption_sources=False)
            for method in (pipeline.initial_probe_calibration,pipeline.train_and_verify):
                with self.assertRaisesRegex(RuntimeError,'Authenticated initial-probe adoption'):method()
            pipeline.start.assert_not_called()

    def test_missing_confirmation_adoption_direct_execute_cleans_only_owned_without_launch(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'logs').mkdir()
            pipeline=self.pipeline(root,CALIBRATION_ROLES[1],adoption_sources=False)
            pipeline.owned=SimpleNamespace(jobs={},stop_all=Mock())
            pipeline.stop_sampling=threading.Event();pipeline.sample_error=None;pipeline.sample=Mock()
            with patch('reproduction.common8.run_pipeline.gpu_rows',return_value=[]),patch(
                    'reproduction.common8.run_pipeline.threading.Thread') as sampler,self.assertRaisesRegex(
                    RuntimeError,'Authenticated initial-probe adoption'):
                pipeline.execute(18099)
            pipeline.owned.stop_all.assert_called_once_with()
            sampler.assert_not_called();pipeline.sample.assert_not_called();pipeline.start.assert_not_called()
            cleanup=json.loads((Path(root)/'logs/cleanup-status.json').read_text())
            self.assertTrue(cleanup['cleaned']);self.assertEqual(cleanup['run_id'],'child')
            self.assertFalse(any(call.args[0].startswith('FINISHED') for call in pipeline.stage.call_args_list))

    def test_missing_adoption_main_rejects_before_native_admission_or_pipeline(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'logs').mkdir()
            (Path(root)/'logs/run-manifest.json').write_text(json.dumps(dict(policy_calibration_role=CALIBRATION_ROLES[1])))
            with patch.object(sys,'argv',['run_pipeline','--base',root,'--export-port','18099']),patch(
                    'reproduction.common8.native_allocation.admit') as allocation,patch(
                    'reproduction.common8.run_pipeline.Pipeline') as pipeline,self.assertRaisesRegex(
                    RuntimeError,'Authenticated initial-probe adoption'):
                main()
            allocation.assert_not_called();pipeline.assert_not_called()

    def test_only_discovery_refits_production_and_confirmation_adopt_original_heads(self):
        for role in (None,)+CALIBRATION_ROLES:
            with self.subTest(role=role),tempfile.TemporaryDirectory() as root:
                pipeline=self.pipeline(root,role)
                pipeline.initial_probe_calibration()
                name,command,gpus=pipeline.start.call_args.args
                if role==CALIBRATION_ROLES[0]:
                    self.assertEqual(name,'initial-calibration');self.assertIn('reproduction.common8.refit',command)
                    self.assertEqual(gpus,[7]);self.assertIn('/original',command)
                else:
                    self.assertEqual(name,'initial-probe-adoption');self.assertEqual(tuple(gpus),tuple(range(8)))
                    self.assertEqual(command,[pipeline.python,'-m','reproduction.common8.adopt_initial_probes',
                        '--base',root,'--discovery-base',pipeline.run['policy_calibration_discovery_base']])
                    self.assertNotIn('reproduction.common8.refit',command)
                pipeline.wait.assert_called_once_with(name,pipeline.start.return_value)

    def test_adoption_sources_must_be_explicit_canonical_existing_and_distinct(self):
        for variant in ('missing','relative','nonexistent','noncanonical','same_target','same_children'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                pipeline=self.pipeline(root)
                run=pipeline.run;key='policy_calibration_discovery_base'
                if variant=='missing':run.pop(key)
                elif variant=='relative':run[key]='relative'
                elif variant=='nonexistent':run[key]=str(Path(root)/'missing')
                elif variant=='noncanonical':run[key]=run[key]+'/.'
                elif variant=='same_target':run[key]=str(Path(root).resolve())
                else:run['policy_calibration_confirmation_base']=run[key]
                with self.assertRaisesRegex(RuntimeError,'Authenticated initial-probe adoption'):
                    adoption_inputs(run,Path(root))

    def test_pipeline_calls_real_adoption_api_then_existing_version_reader(self):
        # Reuse the adoption fixture's explicit native/fitted/canonical-data
        # adapters. Real64 rank-proof aggregation, paired512 arithmetic,
        # plan/segment/NPZ readers, exclusive copying and version checks execute.
        from reproduction.common8.test_adopt_initial_probes import AdoptionTests,write
        from reproduction.a800.probe_admission import calibrated_version
        fixture=AdoptionTests();fixture.setUp()
        try:
            confirmation=fixture.base.parent/'confirmation';confirmation.mkdir()
            run=dict(fixture.run,policy_calibration_confirmation_base=str(confirmation))
            write(fixture.base/'logs/run-manifest.json',run)
            pipeline=Pipeline.__new__(Pipeline)
            pipeline.base=fixture.base;pipeline.run=run;pipeline.python=sys.executable
            pipeline.stage=Mock();pipeline.wait=Mock();receipts=[]
            def start(name,command,gpus):
                self.assertEqual(name,'initial-probe-adoption');self.assertEqual(tuple(gpus),tuple(range(8)))
                self.assertEqual(command,[sys.executable,'-m','reproduction.common8.adopt_initial_probes',
                    '--base',str(fixture.base),'--discovery-base',str(fixture.source)])
                receipts.append(fixture.adopt())
                return receipts[-1]
            pipeline.start=start
            pipeline.initial_probe_calibration()
            live=json.loads((fixture.base/'artifacts/live/manifest.json').read_text())
            self.assertEqual(calibrated_version(run,live),receipts[0]['version'])
            self.assertNotEqual(live['version'],fixture.live['version'])
            self.assertEqual(live['head_sha256'],fixture.live['head_sha256'])
            self.assertFalse((fixture.base/'logs/initial-probes-accepted.json').exists())
            self.assertIsNone(json.loads((fixture.base/'logs/run-manifest.json').read_text())['initial_probe_version'])
            self.assertFalse(receipts[0]['production_admission'])
            Path(live['commit']).write_bytes(b'CPU fixture changed actual copied head bytes')
            with self.assertRaises(AssertionError):calibrated_version(run,live)
        finally:
            fixture.doCleanups()

    def test_real_schema2_verifier_rejects_corrupted_receipt_through_actor_binder(self):
        # Deliberately incomplete negative fixture: no actual/native acceptance
        # is created. This exercises the real verifier and binder, without a
        # verifier stub, and preserves every original byte after rejection.
        import hashlib
        from reproduction.common8.policy_calibration_plan import _identity,canonical_bytes
        from reproduction.common8.test_policy_calibration_plan import CalibrationPlanTests
        from search_g1.policy_safety import bind_thresholds
        fixture=CalibrationPlanTests();fixture.setUp();run=dict(fixture.parent)
        with tempfile.TemporaryDirectory() as root:
            base=Path(root).resolve();(base/'logs').mkdir()
            receipt=base/'logs/negative-receipt.json'
            original=b'{"scope":"incomplete negative CPU receipt"}'
            receipt.write_bytes(original+b' ')
            artifact=dict(schema_version=2,accepted=True,role='healthy_pretraining_policy_calibration',
                run_id=run['run_id'],parent_production_run_id=run['run_id'],
                initial_model_sha256=run['initial_model_sha256'],source_sha256=run['source_sha256'],
                input_sha256=run['input_sha256'],production_identity=_identity(run),
                discovery_run_id='cpu-only-discovery',confirmation_run_id='cpu-only-confirmation',
                discovery_job_id='100',confirmation_job_id='101',
                receipts_sha256={'logs/negative-receipt.json':hashlib.sha256(original).hexdigest()})
            manifest=base/'logs/run-manifest.json';manifest.write_bytes(canonical_bytes(run))
            target=base/'logs/policy-safety-calibration.json';target.write_bytes(canonical_bytes(artifact))
            config=SimpleNamespace(calibration_manifest=str(target),dual_clip=3.,target_old_policy_kl=.01,
                                   kl_log_bound=30.,negative_tail_limit=None)
            before={path:path.read_bytes() for path in (receipt,manifest,target)}
            with patch.dict(os.environ,SEARCH_G1_BASE=str(base)),self.assertRaisesRegex(
                    RuntimeError,'Immutable receipt path/hash differs'):
                bind_thresholds(config)
            self.assertEqual(before,{path:path.read_bytes() for path in before})
            self.assertFalse((base/'logs/policy-safety-admission-claim.json').exists())
            self.assertFalse((base/'logs/training-launch-claim.json').exists())

    def test_unknown_role_and_failed_child_evidence_never_fall_back(self):
        for role in (None,False,'fixture','discovery',{}):
            with self.subTest(role=role),self.assertRaises(AssertionError):
                execution_role(dict(policy_calibration_role=role))
        self.assertEqual(execution_role({}),'production')
        with tempfile.TemporaryDirectory() as root:
            pipeline=self.pipeline(root,CALIBRATION_ROLES[0])
            pipeline.calibration_measurements=Mock(side_effect=FileNotFoundError('no segment'))
            with self.assertRaises(FileNotFoundError): pipeline.train_and_verify()
            self.assertEqual([call.args[0] for call in pipeline.start.call_args_list],['trainer'])

    def artifacts(self, base):
        logs=Path(base)/'logs';logs.mkdir()
        role=CALIBRATION_ROLES[0];sha='b'*64
        pipeline=self.pipeline(base,role)
        pipeline.run.update(profile='common8',source_commit='a'*40,source_sha256={'file.py':'c'*64},
            initial_model='/original',reference_model='/original',initial_model_sha256={'weights':'d'*64},
            input_sha256={'validation.parquet':'e'*64,'train.parquet':'6'*64},model_revision='revision',parent_production_run_id='parent')
        current=dict(pipeline.run,policy_calibration_plan_sha256=sha)
        (logs/'run-manifest.json').write_text(json.dumps(current))
        plan=dict(role=role,parent_production_run_id='parent',identity={'fixture':True},
                  resolved_config_sha256='f'*64,initial_reward_fingerprint={'fixture':True},slurm_job_id='123')
        allocation=dict(accepted=True,run_id='child',slurm_job_id='123',
                        actual_cuda_runtime_uuids=['GPU-'+str(rank) for rank in range(8)])
        (logs/'native-allocation-admission.json').write_text(json.dumps(allocation))
        states=[];state_sha={};worker_sha={}
        for rank in range(8):
            worker=logs/f'optimizer-rank-{rank}.jsonl';worker.write_text('fixture structural records\n')
            worker_sha[str(rank)]=digest(worker)
            state=dict(run_id='child',execution_role=role,parent_production_run_id='parent',plan_sha256=sha,
                slurm_job_id='123',rank=rank,initial_optimizer_writes=0,initial_adam_state_entries=0,
                final_optimizer_writes=20,pid=100+rank,start_ticks=200+rank,
                parameter_tensor_count=1,parameter_numel=1,adam_state_entries=1,
                adam_state_tensor_count=3,adam_state_numel=3,adam_steps=[20.],
                all_parameters_and_adam_finite=True,cuda_uuid='GPU-'+str(rank))
            path=logs/f'policy-calibration-optimizer-rank-{rank}.json';path.write_text(json.dumps(state))
            state_sha[str(rank)]=digest(path);states.append(state)
        parents=[dict(data_source='nq' if i<256 else 'hotpotqa',upstream_group_sha256=format(i,'064x'),
                      tensor_sha256={key:'7'*64 for key in ('input_ids','attention_mask','position_ids')})
                 for i in range(512)]
        repeated=[row['upstream_group_sha256'] for row in parents for _ in range(5)]
        inputs=dict(run_id='child',plan_sha256=sha,slurm_job_id='123',actual_parent_count=512,
                    actual_trajectory_count=2560,n_agent=5,train_sha256='6'*64,parents=parents,
                    repeated_group_sha256=repeated,production_admission=False)
        groups=dict(run_id='child',plan_sha256=sha,actual_trajectory_count=2560,
                    balanced_group_sha256=list(reversed(repeated)),upstream_group_multiset_unchanged=True,
                    production_admission=False)
        training_sha={}
        for name,suffix,row in (('parents','inputs',inputs),('groups','groups',groups)):
            training_path=logs/('policy-calibration-training-'+suffix+'.json')
            training_path.write_text(json.dumps(row));training_sha[name]=digest(training_path)
        births=[dict(pid=row['pid'],start_ticks=row['start_ticks'],sid=300,uid=os.getuid(),state='S') for row in states]
        evaluations={};evaluation_sha={}
        for name,step in (('before',0),('after',1)):
            row=dict(run_id='child',plan_sha256=sha,slurm_job_id='123',step=step,
                     question_count=512,source_counts=dict(nq=256,hotpotqa=256),
                     question_ids=['q'+str(i) for i in range(512)],
                     data_sources=['nq']*256+['hotpotqa']*256,production_admission=False)
            path=logs/('policy-calibration-evaluation-'+name+'.json');path.write_text(json.dumps(row))
            evaluations[name]=row;evaluation_sha[name]=digest(path)
        structural=dict(role='policy_safety_calibration_structural_verification',execution_role=role,
                        run_id='child',calibration_sha256=sha,outer_count=1,optimizer_writes_per_rank=20,
                        global_monitors=[{} for _ in range(40)],production_admission=False)
        proof=dict(role=role+'_evidence',origin='actual_native_policy_segment',run_id='child',
                   parent_production_run_id='parent',slurm_job_id='123',plan_sha256=sha,
                   identity=plan['identity'],resolved_config_sha256=plan['resolved_config_sha256'],
                   initial_reward_fingerprint=plan['initial_reward_fingerprint'],
                   production_admission=False,healthy_checkpoint_accepted=False,
                   completed_outer_updates=1,optimizer_writes_per_rank=20,ranks=list(range(8)),
                   structural=structural,worker_log_sha256=worker_sha,actual_optimizer_states=states,
                   actual_worker_births_at_segment_close=births,actual_training_inputs=inputs,
                   actual_training_groups=groups,training_input_proof_sha256=training_sha,
                   optimizer_state_proof_sha256=state_sha,evaluation_sha256=evaluation_sha,
                   evaluation_before=evaluations['before'],evaluation_after=evaluations['after'])
        path=logs/'policy-calibration-segment.json';path.write_text(json.dumps(proof))
        return pipeline,plan,sha,path,proof

    def test_child_completion_rechecks_frozen_file_hashes_and_actual_state_summaries(self):
        with tempfile.TemporaryDirectory() as root:
            pipeline,plan,sha,path,_=self.artifacts(root)
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch(
                    'reproduction.common8.policy_calibration_plan.load_plan',return_value=(plan,sha)),patch(
                    'reproduction.common8.calibration_runtime.verify_live_optimizer_workers',
                    side_effect=AssertionError('Trainer has exited; do not require live workers here')):
                self.assertEqual(pipeline.calibration_measurements(),path)
            self.assertEqual(pipeline.stage.call_args.args[0],'POLICY_CALIBRATION_MEASUREMENTS_COMPLETE')
            self.assertFalse(pipeline.stage.call_args.kwargs['production_admission'])

    def test_stale_run_hash_wrong_budget_or_health_claim_rejects_completion(self):
        for variant in ('run','worker_hash','state_step','writes','monitors','health','paired_ids'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                pipeline,plan,sha,path,proof=self.artifacts(root)
                logs=Path(root)/'logs'
                if variant=='run': proof['run_id']='another'
                elif variant=='worker_hash': (logs/'optimizer-rank-7.jsonl').write_text('changed')
                elif variant=='state_step':
                    state_path=logs/'policy-calibration-optimizer-rank-7.json'
                    state=json.loads(state_path.read_text());state['adam_steps']=[19.]
                    state_path.write_text(json.dumps(state))
                elif variant=='writes': proof['optimizer_writes_per_rank']=19
                elif variant=='monitors': proof['structural']['global_monitors'].pop()
                elif variant=='health': proof['healthy_checkpoint_accepted']=True
                else:
                    evaluation=proof['evaluation_after'];evaluation['question_ids'][0]='wrong'
                    eval_path=logs/'policy-calibration-evaluation-after.json';eval_path.write_text(json.dumps(evaluation))
                    proof['evaluation_sha256']['after']=digest(eval_path)
                path.write_text(json.dumps(proof))
                with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch(
                        'reproduction.common8.policy_calibration_plan.load_plan',return_value=(plan,sha)),self.assertRaises(AssertionError):
                    pipeline.calibration_measurements()

    def test_training_hash_source_group_budget_and_captured_birth_mismatch_reject_completion(self):
        variants=('input_hash','group_hash','training_run','train_source','parent_source','parent_count',
                  'trajectory_count','n_agent','tensor_hash','repeated_group','balanced_group','group_flag',
                  'birth_missing','birth_pid','birth_ticks','birth_sid','birth_uid','birth_zombie')
        for variant in variants:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as root:
                pipeline,plan,sha,path,proof=self.artifacts(root)
                logs=Path(root)/'logs';inputs=proof['actual_training_inputs'];groups=proof['actual_training_groups']
                if variant=='input_hash': (logs/'policy-calibration-training-inputs.json').write_text('{}')
                elif variant=='group_hash': (logs/'policy-calibration-training-groups.json').write_text('{}')
                elif variant=='training_run': inputs['run_id']='another'
                elif variant=='train_source': inputs['train_sha256']='8'*64
                elif variant=='parent_source': inputs['parents'][0]['data_source']='fixture'
                elif variant=='parent_count': inputs['actual_parent_count']=True
                elif variant=='trajectory_count': groups['actual_trajectory_count']=2559
                elif variant=='n_agent': inputs['n_agent']=4
                elif variant=='tensor_hash': inputs['parents'][0]['tensor_sha256']['input_ids']='not-a-sha'
                elif variant=='repeated_group': inputs['repeated_group_sha256'][0]='9'*64
                elif variant=='balanced_group': groups['balanced_group_sha256'][0]='9'*64
                elif variant=='group_flag': groups['upstream_group_multiset_unchanged']=False
                elif variant=='birth_missing': proof['actual_worker_births_at_segment_close'].pop()
                elif variant=='birth_pid': proof['actual_worker_births_at_segment_close'][7]['pid']=100
                elif variant=='birth_ticks': proof['actual_worker_births_at_segment_close'][7]['start_ticks']=999
                elif variant=='birth_sid': proof['actual_worker_births_at_segment_close'][7]['sid']=False
                elif variant=='birth_uid': proof['actual_worker_births_at_segment_close'][7]['uid']=os.getuid()+1
                else: proof['actual_worker_births_at_segment_close'][7]['state']='Z'
                if variant not in ('input_hash','group_hash'):
                    for name,suffix,row in (('parents','inputs',inputs),('groups','groups',groups)):
                        training_path=logs/('policy-calibration-training-'+suffix+'.json')
                        training_path.write_text(json.dumps(row));proof['training_input_proof_sha256'][name]=digest(training_path)
                path.write_text(json.dumps(proof))
                with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch(
                        'reproduction.common8.policy_calibration_plan.load_plan',return_value=(plan,sha)),self.assertRaises(AssertionError):
                    pipeline.calibration_measurements()

    def test_shell_child_freeze_or_production_admission_after_probe_before_claim(self):
        script=Path(__file__).with_name('train.sh')
        for role in ('production',)+CALIBRATION_ROLES:
            with self.subTest(role=role),tempfile.TemporaryDirectory() as root:
                base=Path(root);(base/'env/bin').mkdir(parents=True);(base/'repo').mkdir()
                fake=base/'env/bin/python'
                fake.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then\ncase "${3-}" in\n'
                    'policy_calibration_discovery_base) printf "%s\\n" "$G1_TEST_DISCOVERY_BASE" ;;\n'
                    'policy_calibration_confirmation_base) printf "%s\\n" "$G1_TEST_CONFIRMATION_BASE" ;;\n'
                    '*) printf "%s\\n" "$G1_TEST_ROLE" ;;\nesac\nexit 0\nfi\n'
                    'printf "%s\\n" "$*" >> "$G1_TEST_TRACE"\n')
                fake.chmod(0o700)
                trace=base/'trace';env=dict(os.environ,SEARCH_G1_BASE=root,SEARCH_G1_TRAIN_LOG=str(base/'train.log'),
                    G1_TEST_ROLE=role,G1_TEST_TRACE=str(trace),G1_TEST_DISCOVERY_BASE=str(base/'discovery'),
                    G1_TEST_CONFIRMATION_BASE=str(base/'confirmation'))
                completed=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
                self.assertEqual(completed.returncode,0,completed.stderr)
                commands=trace.read_text().splitlines()
                expected=['-m reproduction.common8.probe_admission --verify']
                if role in CALIBRATION_ROLES: expected.append('-m reproduction.common8.freeze_calibration_plan')
                else: expected.append('-m reproduction.common8.production_safety_admission --base '+root+
                    ' --discovery-base '+str(base/'discovery')+' --confirmation-base '+str(base/'confirmation'))
                expected+=['-m reproduction.common8.claim_training','-m reproduction.common8.configuration']
                self.assertEqual(commands,expected)


if __name__=='__main__': unittest.main()
