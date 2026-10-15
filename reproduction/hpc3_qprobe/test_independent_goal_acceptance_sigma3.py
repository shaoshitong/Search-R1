"""Synthetic compatibility tests with real file hashing, never actual200 proof."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as audit
from reproduction.hpc3_qprobe import test_independent_goal_acceptance719225 as fixtures


def write(path,value,*,raw=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    data=value if raw else json.dumps(value).encode()
    path.write_bytes(data)
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def run_manifest():
    return dict(run_id=audit.RUN,source_commit=audit.PRODUCER,training_world_size=4,
        actor_micro_batch_per_rank=4,model_revision=audit.OFFICIAL_MODEL_REVISION,
        question_probe=dict(audit.LOSS,head_hidden_size=64,candidate_questions=5120,
            selected_questions=512,responses_per_question=5,exploration_questions=64,seed=20261013))


class EarlyIdentityTests(unittest.TestCase):
    def test_fresh_goal_requires_none_continuation_and_a_single_new_history(self):
        proof=dict(status=audit.GOAL_STATUS,run_id=audit.RUN,completed_outer=200,update201_performed=False,
            actual_attempts=201,bitwise_continuation_of_failed_execution=None,
            actual_training_parts=[dict(first=1,last=200,base=str(audit.BASE),source_commit=audit.PRODUCER)])
        audit.fresh_goal_gate(proof)
        for field,value in [('bitwise_continuation_of_failed_execution',False),('actual_attempts',199),
                            ('update201_performed',True),('producer_final_cpu_recovery',dict(commit='old'))]:
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'fresh complete sigma3'):
                audit.fresh_goal_gate(dict(proof,**{field:value}))
        bad=copy.deepcopy(proof);bad['actual_training_parts'][0]['first']=6
        with self.assertRaisesRegex(ValueError,'fresh complete sigma3'):audit.fresh_goal_gate(bad)

    def test_missing200_before_heavy_import_source_accounting_or_any_writes(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d).resolve();write(base/'logs/actual-updates.jsonl',b'{"step":1}\n',raw=True)
            before={str(p):p.read_bytes() for p in base.rglob('*') if p.is_file()}
            with patch.object(audit,'BASE',base),patch.object(audit,'OWNER',os.getuid()),\
                 patch.object(audit.shared,'OWNER',os.getuid()),patch.object(audit,'publication') as source,\
                 patch.object(audit,'producer_acceptance') as accounting,patch.object(audit,'load_fixed') as load,\
                 patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',**{k:'1' for k in audit.CAPS})),\
                 patch.object(sys,'dont_write_bytecode',True):
                imported=__import__
                def guard(name,*args,**kwargs):
                    if name.split('.')[0] in ('torch','numpy','transformers'):
                        self.fail('heavy dependency reached before missing200 refusal')
                    return imported(name,*args,**kwargs)
                with patch('builtins.__import__',side_effect=guard),self.assertRaisesRegex(ValueError,'fresh1..200'):
                    audit.verify('a'*40,followthrough_commit=audit.FOLLOWTHROUGH)
                source.assert_not_called();accounting.assert_not_called();load.assert_not_called()
            self.assertEqual(before,{str(p):p.read_bytes() for p in base.rglob('*') if p.is_file()})

    def test_complete_fresh_history_accepts_new_identity_and_rejects_old_restore_or_loss(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d).resolve();run=run_manifest()
            ledger=b''.join((json.dumps(dict(step=i,run_id=audit.RUN))+'\n').encode() for i in range(1,201))
            write(base/'logs/actual-updates.jsonl',ledger,raw=True)
            path=base/'logs/run-manifest.json';write(path,run)
            with patch.object(audit,'BASE',base),patch.object(audit.shared,'OWNER',os.getuid()):
                self.assertEqual(audit.early_history()[0],run)
                for change in ('run','restore','loss','hidden'):
                    bad=copy.deepcopy(run)
                    if change=='run':bad['run_id']='c3856315d3f345d7920ebf5725a083b4'
                    if change=='restore':bad['full_state_restore']=dict(completed_outer=5)
                    if change=='loss':bad['question_probe']['std_loss_weight']=.1
                    if change=='hidden':bad['question_probe']['head_hidden_size']=128
                    write(path,bad)
                    with self.subTest(change=change),self.assertRaisesRegex(ValueError,'producer/loss/capacity'):
                        audit.early_history()
                write(path,run)
                with (base/'logs/actual-updates.jsonl').open('ab') as f:
                    f.write((json.dumps(dict(step=201,run_id=audit.RUN))+'\n').encode())
                with self.assertRaisesRegex(ValueError,'fresh1..200'):audit.early_history()

    def test_native_source_pin_rejects_old_EA_and_changed_sigma3_science(self):
        run=dict(source_sha256={'search_g1/qprobe_model.py':'a'*64,'reproduction/common8/runtime.py':'b'*64})
        native=dict(audit_commit=audit.FOLLOWTHROUGH,audit_id='actual-audit')
        restored=dict(source_commit=audit.FOLLOWTHROUGH,run_id=audit.RUN,
            final_restore_audit=dict(audit_id='actual-audit'),source_sha256=dict(run['source_sha256']))
        source=dict(files=dict(run['source_sha256']))
        audit.native_source_gate(run,native,restored,source,audit.FOLLOWTHROUGH)
        with self.assertRaisesRegex(ValueError,'authenticated sigma3'):
            audit.native_source_gate(run,dict(native,audit_commit=audit.shared.EA),restored,source,audit.FOLLOWTHROUGH)
        source['files']['search_g1/qprobe_model.py']='c'*64
        with self.assertRaisesRegex(ValueError,'scientific source differs'):
            audit.native_source_gate(run,native,restored,source,audit.FOLLOWTHROUGH)

    def test_followthrough_state_alone_is_not_a_completed_CPU_return(self):
        follow=SimpleNamespace(BASE=audit.BASE,RUN=audit.RUN,JOB=audit.JOB,PRODUCER=audit.PRODUCER,
            DIRECTORY=audit.FOLLOWTHROUGH_DIRECTORY,inspect=lambda:dict(actual_live_birth=False,
                result=None,process=dict(actual_process=dict(pid=1,start_ticks=2,sid=1,uid=audit.OWNER)),
                state=dict(full_objective_accepted=True)))
        with self.assertRaisesRegex(ValueError,'state alone'):
            audit.followthrough_evidence(follow,{},audit.FOLLOWTHROUGH)


class FreshOriginalTests(unittest.TestCase):
    def test_new_raw_and_private_bytes_freshly_rehashed_and_old_job_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.object(audit.shared,'OWNER',os.getuid()),\
             patch.object(audit,'OWNER',os.getuid()),\
             patch.object(fixtures.audit,'PRODUCER',audit.PRODUCER),patch.object(fixtures.audit,'RUN',audit.RUN):
            root=Path(d).resolve()/'control';base,task,output=fixtures.AttemptTests().fixture(root)
            task['producer']['producer_job']=audit.JOB
            with patch.object(audit,'ROOT',root),patch.object(audit,'BASE',base):
                audit.fresh_attempt(task,output)
                for key in ('raw_file','private_trace_file'):
                    path=Path(output[key]['path']);original=path.read_bytes()
                    path.write_bytes(b'x'*len(original))
                    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'fresh original full SHA'):
                        audit.fresh_attempt(task,output)
                    path.write_bytes(original)
                task['producer']['producer_job']='719225'
                with self.assertRaisesRegex(ValueError,'source/job/base'):
                    audit.fresh_attempt(task,output)


class TeacherAndHeadTests(unittest.TestCase):
    def copy_fixture(self,root):
        base=root/'sigma3';origin=root/'original';target=base/'teacher-warmup-cache'
        names=['identity.json','complete.json']+['chunk-%02d.json'%i for i in range(10)]+['rollout-%02d.pt'%i for i in range(10)]
        copied=[]
        for name in names:
            data=json.dumps(dict(kind=name)).encode() if name.endswith('.json') else ('actual-'+name).encode()
            proof=write(origin/name,data,raw=True);write(target/name,data,raw=True)
            copied.append(dict(name=name,sha256=proof['sha256'],bytes=proof['bytes']))
        write(base/'preparation/teacher-copy-verification.json',dict(origin=str(origin),target=str(target),
            files=copied,questions=5120,trajectories=25600,receipt=json.loads((target/'complete.json').read_bytes())))
        return base,origin,target

    def test_exact_teacher_copy_bytes_and_independent_inode_are_required(self):
        with tempfile.TemporaryDirectory() as d,patch.object(audit.shared,'OWNER',os.getuid()):
            base,origin,target=self.copy_fixture(Path(d).resolve())
            run=dict(question_probe=dict(teacher_warmup_cache=str(target)))
            with patch.object(audit,'BASE',base),patch.object(audit,'TEACHER_ORIGIN',origin):
                self.assertEqual(audit.teacher_copy_evidence(run)['files'],22)
                path=target/'rollout-00.pt';original=path.read_bytes();path.write_bytes(b'x'*len(original))
                with self.assertRaisesRegex(ValueError,'fresh original full SHA'):audit.teacher_copy_evidence(run)
                path.unlink();os.link(origin/path.name,path)
                with self.assertRaisesRegex(ValueError,'independent regular copy'):audit.teacher_copy_evidence(run)

    def test_full_synthetic_teacher_keeps_historical_run_ids_and_rejects_changed_raw(self):
        with tempfile.TemporaryDirectory() as d,patch.object(fixtures.audit,'OWNER',os.getuid()),\
             patch.object(audit.shared,'OWNER',os.getuid()),patch.object(audit,'OWNER',os.getuid()):
            root=Path(d).resolve()/'control';repository=root/'source'/'future'
            old,cache,run,payload=fixtures.TeacherTests().fixture(root,repository)
            base=root.parent/'sigma3';(base/'data').mkdir(parents=True)
            shutil.copyfile(old/'data/train.parquet',base/'data/train.parquet')
            old_source=root/'source'/run['source_commit']/'search_g1/signals.py'
            new_source=root/'source'/audit.PRODUCER/'search_g1/signals.py';new_source.parent.mkdir(parents=True)
            shutil.copyfile(old_source,new_source);run['source_commit']=audit.PRODUCER;run['run_id']=audit.RUN
            with patch.object(audit,'ROOT',root),patch.object(audit,'BASE',base),patch.object(audit,'BASELINE_PAYLOAD_SHA',payload):
                proof=audit.teacher_evidence(repository,run)
                self.assertEqual(proof['questions'],5120)
                self.assertEqual(proof['trajectories'],25600)
                self.assertNotIn(audit.RUN,audit.TEACHER_RUNS)
                path=cache/'raw-07.pt';path.write_bytes(b'x'*path.stat().st_size)
                with self.assertRaisesRegex(ValueError,'fresh original full SHA'):audit.teacher_evidence(repository,run)

    def test_initial_head_exact_loss_capacity_and_bootstrap_Adam(self):
        import torch
        from dataclasses import asdict
        from search_g1.qprobe_model import QProbeConfig,CountMomentProbe
        with tempfile.TemporaryDirectory() as d,patch.object(audit.shared,'OWNER',os.getuid()):
            base=Path(d).resolve();run=run_manifest()
            run['question_probe']['bootstrap_max_groups_per_outer']=5120
            config=QProbeConfig(seed=20261013,bootstrap_max_groups_per_outer=5120,**audit.LOSS)
            path=base/'artifacts/question-probe/teacher-warmup-head/initial.pt';proof=write(path,b'bounded head fixture',raw=True)
            metrics=dict(head_optimizer_updates=40,sigma_ce=1.,std_mse=.2,count_ce=2.,replay_weighted_loss=1.4)
            write(base/'artifacts/question-probe/teacher-warmup-ready.json',dict(head_checkpoint=proof,
                fitted_questions=5120,reused_questions=5120,completed_training_outers=0,head_metrics=metrics))
            head=dict(config=asdict(config),feature_dim=2048,actor_snapshot_id=audit.RUN+':0',completed_outer=0,update_count=0,
                completed_attempts={},replay_labels=[dict(qid=str(i)) for i in range(5120)],
                replay_features=SimpleNamespace(shape=(5120,2048)),model=CountMomentProbe(2048).state_dict(),
                optimizer=dict(state={i:dict(step=torch.tensor(40.)) for i in range(6)}))
            with patch.object(audit,'BASE',base),patch.object(torch,'load',return_value=head):
                self.assertEqual(audit.initial_head_evidence(run)['head_parameters'],135622)
                head['optimizer']['state'][0]['step']=torch.tensor(39.)
                with self.assertRaisesRegex(ValueError,'actual bootstrap Adam'):audit.initial_head_evidence(run)
                head['config']['std_loss_weight']=.1
                with self.assertRaisesRegex(ValueError,'configuration differs'):audit.initial_head_evidence(run)


class ArchiveHeldTests(unittest.TestCase):
    """Actual small receipt bytes; synthetic kernel/publication boundaries only."""
    def test_all_five_archives_and_reconciled_submit_need_real_bound_wait0(self):
        with tempfile.TemporaryDirectory() as d,patch.object(audit,'OWNER',os.getuid()),\
             patch.object(audit.shared,'OWNER',os.getuid()):
            root=Path(d).resolve();publication=dict(commit=audit.FOLLOWTHROUGH,publication_sha256='a'*64)
            producer=dict(run_id=audit.RUN);controller=dict(pid=100,start_ticks=1000,sid=100,uid=os.getuid())
            pair=dict(sha256='b'*64)
            follow=SimpleNamespace(control=SimpleNamespace(pair=lambda *args:(None,pair),
                inspect_audit=lambda base:dict(status='ACTUAL_SLURM_RECONCILED',job='800001',audit_base=str(root/'native'))))
            with patch.object(audit,'ROOT',root),patch.object(audit.ctl,'ROOT',root),patch.object(audit.ctl,'live',return_value=False):
                folders=[]
                for index,step in enumerate((5,50,100,150,200,None)):
                    final=step is None;component='audit-controller' if final else 'archive-'+str(step)
                    folder=root/'final-control'/audit.RUN/component;folders.append(folder)
                    token='%032x'%(index+1);role='detached_cpu_final_audit_submission' if final else 'detached_cpu_pair_archive'
                    claim=dict(schema_version=1,role=role,token=token,producer=producer,source=publication,
                        controller=controller,created_unix=1,training_permitted=False,direct_gpu_device_access_permitted=False,
                        native_gpu_audit_submission_permitted=final,pair=pair)
                    if not final:claim['step']=step
                    claim['command']=audit.ctl.command(root/'source'/audit.FOLLOWTHROUGH,
                        'audit-worker' if final else 'archive-worker',folder/'submit-claim.json',token)
                    claim_file=write(folder/'submit-claim.json',claim)
                    runner=dict(pid=200+index,start_ticks=2000+index,sid=200+index,uid=os.getuid())
                    child=dict(pid=300+index,start_ticks=3000+index,sid=runner['sid'],uid=os.getuid())
                    binding=dict(token=token,claim_sha256=claim_file['sha256'])
                    write(folder/'runner-binding.json',dict(binding,actual_process=runner,controller=controller))
                    write(folder/'task-binding.json',dict(binding,actual_process=child,supervisor=runner))
                    task=dict(binding,status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if final else 'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
                        actual_process=child,training_performed=False,direct_gpu_device_access_performed=False,
                        actor_model_loaded=False,native_gpu_audit_submission_attempted=final,completed_unix=2)
                    if final:task['submission']=dict(status='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPLAY')
                    else:
                        path=root/'archives'/audit.RUN/('global_step_'+str(step))/'complete.json'
                        transport=write(path,dict(run_id=audit.RUN,step=step,world_size=4))
                        task['archive']=dict(path=str(path),sha256=transport['sha256'])
                    task_file=write(folder/'task-result.json',task)
                    result=dict(binding,actual_task_birth=child,actual_supervisor=runner,role=role,actual_task_exit_code=0,
                        status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' if final else 'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0',
                        task_result_sha256=task_file['sha256'],finished_unix=3)
                    write(folder/'worker-result.json',result)
                proof,_=audit.held_archive_evidence({},producer,follow,publication,controller)
                self.assertEqual(set(proof),{'5','50','100','150','200','native_submit'})
                self.assertEqual(proof['native_submit']['actual_native_submission']['job'],'800001')
                path=folders[2]/'worker-result.json';changed=json.loads(path.read_bytes());changed['actual_task_exit_code']=1;write(path,changed)
                with self.assertRaisesRegex(ValueError,'actual held original child0'):
                    audit.held_archive_evidence({},producer,follow,publication,controller)


if __name__=='__main__':unittest.main()
