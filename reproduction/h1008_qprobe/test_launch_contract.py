import copy
import json
from pathlib import Path
import tempfile
import unittest
from reproduction.h1008_qprobe.adopt_reward_zero import validate_zero,adopt,verify_adopted,digest
from reproduction.h1008_qprobe.submit import validate_predecessor
from reproduction.h1008_qprobe.first_five_admission import selected


class LaunchContractTests(unittest.TestCase):
    def binding(self,root):
        source=Path(root)/'old';base=Path(root)/'new';model=Path(root)/'official';model.mkdir()
        for b in (source,base):(b/'logs').mkdir(parents=True);(b/'artifacts').mkdir()
        initial={'config.json':'a'*64,'model.safetensors':'b'*64}
        common=dict(initial_model=str(model),reference_model=str(model),model_revision='aa8e72537993ba99e69dfaafa59ed015b17504d1',
            initial_model_sha256=initial,input_sha256={k:'c'*64 for k in ('manifest.json','validation.parquet','calibration-300.parquet','audit.json')},completed_updates=0,execution_site='h1008')
        old=dict(common,run_id='old');new=dict(common,run_id='new')
        archive=dict(run_id='old',round=0,version='old:0:real',encoder=str(model),encoder_sha256=initial,
            head_sha256={},dump_sha256={},label_proof={'question_ids':['q']},calibration_report=dict(unique_question_count=300,question_grouped_cv=True,require_all_cv_folds=True))
        directory=source/'artifacts/round_0-real';(directory/'heads').mkdir(parents=True);(directory/'dumps').mkdir()
        for n in ('commit_probe.pkl','boundary_probe.pkl'):
            p=directory/'heads'/n;p.write_bytes(b'fixture fitted calibration bytes '+n.encode());archive['head_sha256'][n]=digest(p)
        p=directory/'dumps/labels.json';p.write_text('{}');archive['dump_sha256']['labels.json']=digest(p)
        p=directory/'heads/calibration-report.json';p.write_text(json.dumps(archive['calibration_report']));archive['calibration_report_sha256']=digest(p)
        path=directory/'version-manifest.json';path.write_text(json.dumps(archive))
        for b,v in [(base,new),(source,old)]:(b/'logs/run-manifest.json').write_text(json.dumps(v))
        return base,source,new,old,archive,path
    def test_reject_trained_live150_wrong_encoder_or_calibration_before_copy(self):
        with tempfile.TemporaryDirectory() as d:
            base,source,new,old,a,p=self.binding(d)
            for mutate in ('round','encoder','calibration'):
                v=copy.deepcopy(a);target=copy.deepcopy(new)
                if mutate=='round':v['round']=150
                if mutate=='encoder':v['encoder_sha256']['config.json']='f'*64
                if mutate=='calibration':target['input_sha256']['calibration-300.parquet']='f'*64
                with self.assertRaises(ValueError):validate_zero(target,old,v)
                self.assertFalse((base/'logs/initial-zero-adoption-intent.json').exists())
    def test_round0_actual_source_target_sha_copy_and_historical_verification(self):
        with tempfile.TemporaryDirectory() as d:
            base,source,new,old,a,p=self.binding(d)
            receipt=adopt(base,source,p,digest(p));live=json.loads((base/'artifacts/live/manifest.json').read_text())
            self.assertEqual(live['round'],0);self.assertEqual(live['run_id'],'new')
            self.assertFalse(live['adoption_origin']['actor_optimizer_scheduler_rng_inherited'])
            archive_path=Path(live['commit']).parent.parent/'version-manifest.json'
            verify_adopted(base,dict(new,completed_updates=200),live,archive_path)
            Path(live['commit']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'FULL_SHA'):verify_adopted(base,new,live,archive_path)
    def test_old_evaluation_must_finish_all_rows_and_actual_exit0_before_new_start(self):
        result={'qa_count':51713};cleanup={'cleaned':True};terminal=dict(job_id=3395,state='COMPLETED',exit_code='0:0')
        self.assertTrue(validate_predecessor(result,cleanup,terminal))
        for r,c,t in [({'qa_count':51712},cleanup,terminal),(result,{'cleaned':False},terminal),(result,cleanup,dict(terminal,state='RUNNING')),(result,cleanup,dict(terminal,exit_code='1:0'))]:
            with self.assertRaises(ValueError):validate_predecessor(r,c,t)
    def test_prospective_admission_calls_real_unstarted_import_without_smoke_claim(self):
        from unittest.mock import patch
        from reproduction.common8 import formal_inline_safety as safety
        from reproduction.h1008_qprobe.first_five_admission import prepare
        run=dict(execution_site='h1008',training_world_size=8,startup_admission_mode='actual_first_five',
            question_probe={'probe_architecture':'qwen2_decoder3_count6'},run_id='new',source_commit='code',
            source_sha256={},initial_model_sha256={},input_sha256={},budget={},initial_probe_version='new:0:fit',
            behavior_probability_policy='sampling_journal_no_extra_hf_forward')
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'logs').mkdir();(base/'preparation').mkdir()
            allocation=dict(accepted=True,run_id='new',slurm_job_id='7000',actual_cuda_runtime_uuids=['gpu'+str(i) for i in range(8)],
                actual_allocation_detail='JobId=7000 JobState=RUNNING SubmitTime=s StartTime=t UserId=root Command=launch WorkDir=repo NodeList=h100-8')
            for n,v in [('logs/native-allocation-admission.json',allocation),('preparation/formal-inline-method.json',{})]+[(n,{}) for n in ['logs/retrieval-admission.json','logs/retrieval-http-admission.json','logs/evaluator-release-admission.json','logs/probe-coexistence-admission.json']]:
                (base/n).write_text(json.dumps(v))
            with patch.dict('os.environ',{'SLURM_JOB_ID':'7000'}), \
                    patch.object(safety,'verify_plan') as verify, \
                    patch('reproduction.common8.freeze_calibration_plan._unstarted') as unstarted:
                artifact=prepare(base,run,{}, {},{'archive':'recorded-zero'},{})
            unstarted.assert_called_once_with(base,run);verify.assert_called_once()
            self.assertFalse(artifact['healthy_checkpoint_accepted'])
            self.assertFalse(artifact['probability']['standalone_synthetic_gpu_smoke_performed'])
            self.assertTrue(artifact['probability']['actual_model_initialization_required_before_first_write'])

    def test_reduced_startup_route_only_targets_new_h1008_sequence_experiment(self):
        run=dict(execution_site='h1008',training_world_size=8,startup_admission_mode='actual_first_five',question_probe={'probe_architecture':'qwen2_decoder3_count6'})
        self.assertTrue(selected(run))
        for key,value in [('execution_site','hpc3'),('training_world_size',4),('startup_admission_mode','old')]:
            self.assertFalse(selected(dict(run,**{key:value})))

if __name__=='__main__':unittest.main()
