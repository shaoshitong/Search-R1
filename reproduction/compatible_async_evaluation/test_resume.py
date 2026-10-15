"""Remote CPU fixtures for terminal takeover; no real job, CUDA, or model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reproduction.compatible_async_evaluation import resume
from reproduction.compatible_async_evaluation.contract import digest,protocol,PROTOCOL_PATH
from reproduction.compatible_async_evaluation.journal import make_binding,append_batch


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);(self.base/'journal').mkdir();(self.base/'logs').mkdir()
        self.ids=['duplicate'+str(i//2) for i in range(512)];self.sources=['nq']*256+['hotpotqa']*256
        self.jb=make_binding('a'*32,'b'*64,protocol()['datasets']['validation']['sha256'],self.ids,self.sources,5,'validation')
        self.binding=dict(journal_binding=self.jb,checkpoint='/remote/real-checkpoint',checkpoint_manifest_sha256='b'*64,generation_config_sha256='c'*64)
        self.rows=[dict(ordinal=i,question_id=self.ids[i],data_source=self.sources[i],em=0,f1=0.,actual_calls=0,valid_action=1,turns=1,answer_present=False,answer_sha256=None,response_token_count=1,model_token_count=1) for i in range(64)]
        self.workers=[dict(rank=i,world_size=8,cuda_uuid=str(i),hf_parameter_dtypes=['torch.float32'],vllm_parameter_dtypes=['torch.float32'],hf_parameter_numel=1,optimizer_constructed=False,optimizer_steps=0,async_phase_enabled=True,recipe_sha256=digest(PROTOCOL_PATH),checkpoint=self.binding['checkpoint'],checkpoint_manifest_sha256='b'*64,generation_config_sha256='c'*64,actual_vllm_version='0.6.3',actual_vllm_config_dtype='torch.float32',actual_engine_max_model_len=4596) for i in range(8)]
        (self.base/'journal/loaded-workers-123.json').write_text(json.dumps(self.workers))

    def prefix(self):
        append_batch(self.base/'journal',self.jb,self.ids,self.sources,self.rows)
        return resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')

    def test_partial_duplicate_occurrences_and_exact_file_hashes(self):
        rows,files,proof=self.prefix()
        self.assertEqual(len(rows),64);self.assertEqual(rows[0]['question_id'],rows[1]['question_id'])
        self.assertEqual([r['ordinal'] for r in rows],list(range(64)))
        self.assertEqual(set(files),{'binding.json','batch-00000000-00000064.json'})
        self.assertEqual(proof,digest(self.base/'journal/loaded-workers-123.json'))

    def test_missing_binding_never_initializes_or_mutates_prior(self):
        with self.assertRaises(ValueError):resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')
        self.assertFalse((self.base/'journal/binding.json').exists())

    def test_corrupt_or_incomplete_batch_rejected(self):
        self.prefix();p=self.base/'journal/batch-00000000-00000064.json';old=p.read_text()
        p.write_text(old[:50])
        with self.assertRaises((ValueError,json.JSONDecodeError)):resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')
        p.write_text(old);p.rename(self.base/'journal/batch-00000001-00000065.json')
        with self.assertRaises(ValueError):resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')

    def test_actual_worker_checkpoint_or_precision_mismatch_rejected(self):
        self.prefix();self.workers[0]['vllm_parameter_dtypes']=['torch.bfloat16']
        (self.base/'journal/loaded-workers-123.json').write_text(json.dumps(self.workers))
        with self.assertRaises(ValueError):resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')

    def test_complete_or_non_batch_prefix_is_not_resumable(self):
        append_batch(self.base/'journal',self.jb,self.ids,self.sources,self.rows[:3])
        with self.assertRaises(ValueError):resume.validated_prefix(self.base,self.binding,self.ids,self.sources,'123')

    def test_scientific_byte_change_rejected_control_change_allowed(self):
        old={'verl/scorer.py':'a','reproduction/common8/validation_identity.py':'b','reproduction/compatible_async_evaluation/prepare.py':'old'}
        old.update({'reproduction/compatible_async_evaluation/'+n:'same' for n in resume.SCIENCE_EVAL})
        new=dict(old);new['reproduction/compatible_async_evaluation/prepare.py']='new'
        resume.validate_science(old,new)
        for name in ('verl/scorer.py','reproduction/compatible_async_evaluation/async_controller.py'):
            bad=dict(new);bad[name]='different'
            with self.assertRaises(ValueError):resume.validate_science(old,bad)

    def test_missing_resume_receipt_blocks_manifest_launch(self):
        (self.base/'preparation').mkdir();resume.once(self.base/'preparation/resume-intent.json',{'run_id':'old'})
        with self.assertRaises(ValueError):resume.verify_resume_receipt(self.base,{'resume_intent_sha256':digest(self.base/'preparation/resume-intent.json')})

    def test_retained_terminal_proof_requires_explicit_sha_and_same_execution(self):
        row=dict(slurm_job_id='123',state='FAILED',exit_code='1:0',native_proof_sha256='a'*64,process_ledger_sha256='b'*64)
        manifest=dict(run_id='a'*32,execution_id='d'*32)
        path=self.base/'terminal.json';value=dict(role='compatible_evaluation_actual_terminal_release',prior_base=str(self.base),**manifest,observations=[row,row])
        path.write_text(json.dumps(value))
        self.assertEqual(resume.retained_terminal(path,digest(path),self.base,manifest),row)
        for sha,m in [(None,manifest),('0'*64,manifest),(digest(path),dict(manifest,execution_id='e'*32))]:
            with self.assertRaises(ValueError):resume.retained_terminal(path,sha,self.base,m)

    def test_new_namespace_copies_prefix_and_publishes_receipt_without_replay(self):
        from types import SimpleNamespace
        self.prefix()
        (self.base/'preparation').mkdir()
        source_map={'reproduction/compatible_async_evaluation/'+n:'same' for n in resume.SCIENCE_EVAL}
        self.binding.update(run_id='a'*32,checkpoint_step=5,split='validation',
            checkpoint_manifest='/remote/source-receipt.json',checkpoint_files_sha256={'model':'d'*64},
            official_model='/remote/official',resolved_config_sha256='e'*64,
            validation_sha256=self.jb['validation_sha256'],recipe_id=self.jb['recipe_id'],
            recipe_sha256=self.jb['recipe_sha256'],source_root='/remote/oldcontrol',
            source_sha256=source_map,control_source_commit='f'*40)
        old_binding=self.base/'preparation/binding.json';old_binding.write_text(json.dumps(self.binding))
        manifest=dict(base=str(self.base),journal=str(self.base/'journal'),binding=str(old_binding),
            run_id='a'*32,execution_id='b'*32,checkpoint_step=5,split='validation',
            recipe_sha256=self.jb['recipe_sha256'],source_base='/remote/data',config='/remote/config.json',
            physical_gpu_uuids=[str(i) for i in range(8)],gpu_lock_directory='/remote/locks',
            retrieval_url='http://127.0.0.1:18091/retrieve',retrieval_index_path='/remote/index',
            retrieval_corpus_path='/remote/corpus',retrieval_encoder='/remote/e5',retriever_gpu_lib='/remote/faiss')
        (self.base/'preparation/compatible-evaluation-manifest.json').write_text(json.dumps(manifest))
        publication=self.base/'publication.json';publication.write_text(json.dumps({'files':source_map}))
        target=self.base.parent/(self.base.name+'-resume')
        self.addCleanup(lambda:__import__('shutil').rmtree(target,ignore_errors=True))
        args=SimpleNamespace(prior_base=str(self.base),base=str(target),control_publication_proof=str(publication),
            control_commit='1'*40,sharing_bindings='/remote/sharing.json',execution_id='c'*32,
            retrieval_url=None,terminal_proof=None,terminal_proof_sha256=None)
        def fake_prepare(a,*,resume_plan):
            root=Path(a.base)
            for folder in ('preparation','journal'):(root/folder).mkdir(parents=True)
            resume.once(root/'preparation/resume-intent.json',resume_plan)
            resume.once(root/'preparation/binding.json',self.binding)
            resume.once(root/'preparation/compatible-evaluation-manifest.json',dict(manifest,
                base=str(root),journal=str(root/'journal'),binding=str(root/'preparation/binding.json'),
                execution_id=a.execution_id,resume_intent_sha256=digest(root/'preparation/resume-intent.json')))
            return dict(base=str(root),submitted=False,actual_gpu_execution=False)
        observation=dict(slurm_job_id='123',state='FAILED',exit_code='1:0',native_proof_sha256='2'*64,process_ledger_sha256='3'*64)
        frame={'data_source':SimpleNamespace(tolist=lambda:self.sources)}
        with patch.object(resume,'verify_source_tree'),patch.object(resume,'terminal_snapshot',return_value=observation) as terminal,patch.object(resume,'prepare',side_effect=fake_prepare),patch('reproduction.common8.validation_identity.expected',return_value=({'sha256':self.jb['validation_sha256']},frame,self.ids)):
            result=resume.resume(args)
            self.assertEqual(terminal.call_count,2)
            self.assertEqual(result['completed_occurrences'],64)
            self.assertFalse(result['actual_gpu_resume_performed'])
            self.assertFalse((target/'logs/process-ledger.json').exists())
            for name in ('binding.json','batch-00000000-00000064.json'):
                self.assertEqual(digest(self.base/'journal'/name),digest(target/'journal'/name))
            new_manifest=resume.read(target/'preparation/compatible-evaluation-manifest.json')
            self.assertTrue(resume.verify_resume_receipt(target,new_manifest)['complete'])
            with self.assertRaisesRegex(ValueError,'OCCUPIED_OUTCOME'):
                resume.resume(args)

    def test_same_execution_or_ray_prefix_is_rejected(self):
        for value in ('a'*32,'a'*12+'b'*20,'not-an-id'):
            with self.assertRaises(ValueError):resume.validate_new_execution('a'*32,value)
        with patch.object(Path,'exists',return_value=False):
            resume.validate_new_execution('a'*32,'b'*32)

    def test_pinned_terminal_survives_accounting_purge_but_not_queue_failure(self):
        import subprocess
        manifest=dict(run_id='a'*32,execution_id='d'*32,checkpoint_step=5,split='validation',recipe_sha256='c'*64)
        proof=dict(manifest,accepted=True,node='h100-8',slurm_job_id='123',self_cgroup='0::/slurm/uid_123/job_123/step_batch')
        native=self.base/'logs/compatible-native-allocation.json';native.write_text(json.dumps(proof))
        birth=dict(pid=543210,start_ticks=1,sid=543210,uid=123,state='R')
        ledger=self.base/'logs/process-ledger.json';ledger.write_text(json.dumps(dict(run_id=manifest['run_id'],jobs={'worker':dict(leader=birth,ledger={'543210':birth})})))
        retained=dict(slurm_job_id='123',state='CANCELLED',exit_code='0:9',native_proof_sha256=digest(native),process_ledger_sha256=digest(ledger),original_births_released=True,bound_native_cgroups_unpopulated=True,owned_gpu_contexts_released=True)
        def purged(command):
            if command[0]=='squeue':return ''
            if command[0]=='nvidia-smi':return '543210\n'
            raise subprocess.CalledProcessError(1,command)
        # The foreign current PID has a different birth, so do not claim it is ours.
        current=dict(birth,start_ticks=99,sid=99,uid=999,state='R')
        with patch.object(resume,'command',side_effect=purged),patch('reproduction.a800.owned_processes.same',return_value=False),patch('reproduction.a800.owned_processes.identity',return_value=current):
            observed=resume.terminal_snapshot(self.base,manifest,retained)
            self.assertEqual(observed['state'],'CANCELLED')
            with self.assertRaisesRegex(ValueError,'ACCOUNTING_REQUIRED'):resume.terminal_snapshot(self.base,manifest)
        with patch.object(resume,'command',side_effect=subprocess.CalledProcessError(1,['squeue'])):
            with self.assertRaises(subprocess.CalledProcessError):resume.terminal_snapshot(self.base,manifest,retained)

    def test_scontrol_actual_terminal_is_enough_without_sacct(self):
        manifest=dict(run_id='a'*32,execution_id='d'*32,checkpoint_step=5,split='validation',recipe_sha256='c'*64)
        proof=dict(manifest,accepted=True,node='h100-8',slurm_job_id='123',self_cgroup='0::/slurm/job_123/step_batch')
        (self.base/'logs/compatible-native-allocation.json').write_text(json.dumps(proof))
        birth=dict(pid=543210,start_ticks=1,sid=543210,uid=123,state='R')
        (self.base/'logs/process-ledger.json').write_text(json.dumps(dict(run_id=manifest['run_id'],jobs={'worker':dict(leader=birth,ledger={'543210':birth})})))
        def actual(command):
            if command[0]=='scontrol':return 'JobId=123 JobState=FAILED ExitCode=1:0'
            if command[0]=='sacct':self.fail('sacct is unnecessary with actual native terminal')
            return ''
        with patch.object(resume,'command',side_effect=actual),patch('reproduction.a800.owned_processes.same',return_value=False),patch('reproduction.a800.owned_processes.identity',return_value=dict(birth,sid=99,uid=999)):
            snapshot=resume.terminal_snapshot(self.base,manifest)
            self.assertEqual(snapshot['state'],'FAILED')
            self.assertEqual(snapshot['actual_scontrol_stdout'],'JobId=123 JobState=FAILED ExitCode=1:0')

    def test_running_or_missing_accounting_is_not_terminal(self):
        manifest=dict(run_id='a'*32,execution_id='d'*32,checkpoint_step=5,split='validation',recipe_sha256='c'*64)
        proof=dict(manifest,accepted=True,node='h100-8',slurm_job_id='123')
        (self.base/'logs/compatible-native-allocation.json').write_text(json.dumps(proof))
        with patch.object(resume,'command',return_value='123|RUNNING\n'):
            with self.assertRaisesRegex(ValueError,'STILL_QUEUED'):resume.terminal_snapshot(self.base,manifest)
        with patch.object(resume,'command',return_value=''):
            with self.assertRaisesRegex(ValueError,'ACCOUNTING_REQUIRED'):resume.terminal_snapshot(self.base,manifest)

if __name__=='__main__':unittest.main()
