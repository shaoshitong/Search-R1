"""CPU guard/readback mechanisms only. These tests never claim CUDA restoration."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from reproduction.hpc3_qprobe import final_restore_audit as audit
from search_g1.full_state_restore import equal_tree
from search_g1.qprobe_model import OnlineQProbe, QProbeConfig
from search_g1.qprobe_runtime import QProbeRuntime
from search_g1.training_state import atomic_save, driver_rng


def pair():
    receipt = dict(status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS', kind='question_probe_pair',
                   step=200, world_size=4, run_id='a'*32, actor_snapshot_id='a'*32+':200',
                   attempt_progress=dict(completed_outer=200,cumulative_writes=3600,attempt_index=202,aborted_attempts=2),
                   driver_state=dict(completed_outer=200,source_consumed=1024000),
                   checkpoint='/data/synthetic/actor/global_step_200',source_base='/data/synthetic',
                   source_run_manifest={'full_read_sha256':'b'*64},reward_probe_version='reward150',
                   reward_probe_round=150,question_probe_state_path='/data/synthetic/qprobe.pt',feature_spec_sha='f'*64)
    parent = dict(run_id='a'*32,training_world_size=4,actor_micro_batch_per_rank=4,
                  execution_site='hpc3',execution_backend='slurm',source_commit='b'*40,
                  source_sha256={'search_g1/qprobe_model.py':'c'*64},question_probe={})
    return receipt,parent


class FinalAuditGuardTests(unittest.TestCase):
    def test_final_gate_refuses_nonfinal_other_topologies_and_wrong_actor(self):
        receipt,parent=pair();audit.final_pair_gate(receipt,parent)
        for field,value in [('step',199),('step',201),('step',200.),('world_size',8),
                            ('actor_snapshot_id','a'*32+':199'),('kind','weights_only')]:
            wrong=copy.deepcopy(receipt);wrong[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                audit.final_pair_gate(wrong,parent)
        wrong=copy.deepcopy(parent);wrong['actor_micro_batch_per_rank']=2
        with self.assertRaises(ValueError):audit.final_pair_gate(receipt,wrong)

    def test_separate_binder_keeps_origin_id_and_distinct_audit_source(self):
        receipt,parent=pair()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source-receipt.json';path.write_text(json.dumps(receipt))
            new=audit.bind_audit(parent,receipt,path,Path(folder)/'new','d'*40,
                                 dict(parent['source_sha256'],**{'new_audit.py':'e'*64}),'f'*32)
            self.assertEqual(new['run_id'],parent['run_id'])
            self.assertEqual(new['source_commit'],'d'*40)
            self.assertEqual(new['full_state_restore']['completed_outer'],200)
            self.assertIs(new['final_restore_audit']['fit_permitted'],False)
            self.assertIsNone(new['question_probe']['initial_head_checkpoint'])
            self.assertIsNone(new['question_probe']['snapshot_path'])
            self.assertEqual(new['full_state_restore']['source_receipt_sha256'],audit.digest(path))
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                audit.bind_audit(parent,receipt,path,Path(folder)/'new','d'*40,
                                 {'search_g1/qprobe_model.py':'0'*64},'f'*32)

    def test_native_owner_entry_and_actual_resource_budget_fail_closed(self):
        base=Path('/data/synthetic-audit');source=Path('/data/synthetic-source')
        job=dict(JobId='12345',UserId='jxu957(14090)',Account='jxu957',QOS='formal-user',
                 JobState='RUNNING',Partition='emergency_acd',NumNodes='1',NodeList='ACD1-12',
                 NumCPUs='48',MinMemoryNode='512G',TresPerNode='gres:gpu:4',
                 Command=str(source/audit.ENTRY),WorkDir=str(base),JobName='qprobe-final200audit')
        audit.native_owner_gate(job,base,source,'12345',14090)
        for key,value in [('UserId','mou176(21930)'),('Partition','acd_u'),('NodeList','ACD1-67'),
                          ('TresPerNode','gres:gpu:8'),('NumCPUs','96'),('WorkDir','/other'),
                          ('Command',str(source/'train.sh')),('JobState','PENDING')]:
            wrong=dict(job);wrong[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit.native_owner_gate(wrong,base,source,'12345',14090)
        with self.assertRaises(ValueError):audit.native_owner_gate(job,base,source,'12345',21930)

    def test_fresh_full_hash_detects_same_size_same_mtime_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'weights';path.write_bytes(b'original')
            st=path.stat();row=dict(path=str(path),inode=st.st_ino,bytes=st.st_size,
                                   mtime_ns=st.st_mtime_ns,full_read_sha256=audit.digest(path))
            self.assertEqual(audit.verify_record_bytes([row])[str(path)],row['full_read_sha256'])
            path.write_bytes(b'changed!');os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'fresh full source SHA'):
                audit.verify_record_bytes([row])

    def test_full_records_include_external_boundary_and_reward_dependencies(self):
        def record(name):
            return dict(path=name,inode=1,bytes=1,mtime_ns=1,full_read_sha256='a'*64)
        value=dict(files=[record('actor')],boundary_worker_files=[record('rng')],
                   boundary_driver=record('driver'),source_run_manifest=record('manifest'),
                   reward_probe_manifest=record('reward_manifest'),reward_files=[record('reward_weights')],
                   historical_artifacts=dict(checkpoints=[record('cp5')],development={'before':record('dev0')},
                                             reward_archives=[record('archive')],
                                             training_parts=[dict(source_run_manifest=record('prefix_manifest'))]),
                   checkpoint_origin=dict(source_receipt=record('saved_source'),actual_workers=record('workers'),
                                          actual_driver=record('driver'),actual_question_probe=record('qprobe'),
                                          actual_allocation=record('allocation')))
        self.assertEqual({v['path'] for v in audit.full_records(value)},
                         {'actor','rng','driver','manifest','reward_manifest','reward_weights','cp5','dev0','archive',
                          'prefix_manifest','saved_source','workers','qprobe','allocation'})
        self.assertEqual(len(audit.full_records(value)),14)  # Identical driver record is read once.

    def test_full_records_reject_conflicting_duplicate_provenance(self):
        row=dict(path='/synthetic/path',inode=1,bytes=1,mtime_ns=1,full_read_sha256='a'*64)
        for key,value in [('inode',2),('bytes',2),('mtime_ns',2),('full_read_sha256','b'*64)]:
            changed=dict(row,**{key:value})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'conflicting identity/SHA'):
                audit.full_records(dict(files=[row],historical_artifacts=dict(training_parts=[dict(source_run_manifest=changed)])))

    def test_optimizer_and_scheduler_locks_preserve_exact_state_dict(self):
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=torch.optim.AdamW([parameter],lr=1e-6)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        worker=SimpleNamespace(actor_optimizer=optimizer,actor_lr_scheduler=scheduler,
                               actor=SimpleNamespace(update_policy=lambda:None))
        before=copy.deepcopy(scheduler.state_dict());audit.lock_writes(worker)
        equal_tree(scheduler.state_dict(),before)
        for callback in (optimizer.step,scheduler.step,worker.actor.update_policy,audit.forbid):
            with self.assertRaisesRegex(RuntimeError,'FORBIDS'):callback()
        equal_tree(scheduler.state_dict(),before)
        self.assertFalse(optimizer.state)

    def test_typed_tree_hash_preserves_types_and_object_array_contents(self):
        first=dict(tensor=torch.tensor(2,dtype=torch.int64),ids=np.array(['q1','q2'],dtype=object),
                   rng=(1,[torch.tensor([0,1],dtype=torch.uint8)]))
        self.assertEqual(audit.tree_sha(first),audit.tree_sha(copy.deepcopy(first)))
        second=copy.deepcopy(first);second['ids'][1]='q3'
        self.assertNotEqual(audit.tree_sha(first),audit.tree_sha(second))
        second=copy.deepcopy(first);second['tensor']=torch.tensor(2,dtype=torch.int32)
        self.assertNotEqual(audit.tree_sha(first),audit.tree_sha(second))

    def test_entry_has_no_fit_validation_generation_or_update_call(self):
        syntax=ast.parse(Path(audit.__file__).read_text())
        forbidden={'fit','_validate','update_actor','update_policy','generate_sequences',
                   'complete_outer','bootstrap','step','_fit','_save_checkpoint'}
        calls=[n.func.attr for n in ast.walk(syntax) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&forbidden)
        self.assertIn('init_workers',calls)
        self.assertIn('final_audit_readback',calls)
        # The training binder remains an independent strict <200 gate.
        from search_g1.qprobe_checkpoint import bind_restore
        import inspect
        self.assertIn("receipt['step']<200",inspect.getsource(bind_restore))

    def transport_fixture(self,folder):
        top=Path(folder).resolve();payload=b'synthetic immutable payload; not model weights'
        sha=hashlib.sha256(payload).hexdigest();records={};receipts={};roots={}
        for role in ('source','archive','target'):
            root=top/role;root.mkdir();roots[role]=root
            value=root/'payload';value.write_bytes(payload);st=value.stat()
            row=dict(path=str(value),inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,full_read_sha256=sha)
            receipt=dict(run_id='a'*32,step=200,world_size=4,source_base=str(root),files=[row],
                         boundary_worker_files=[],boundary_driver=row,source_run_manifest=row,
                         reward_probe_manifest=row,reward_files=[],historical_artifacts={})
            path=root/'receipt.json';path.write_text(json.dumps(receipt));records[role]=row;receipts[role]=path
        proof=dict(schema_version=1,status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',run_id='a'*32,
                   step=200,world_size=4,source_manifest_sha256=sha,
                   files=[dict(source_path=records['source']['path'],archive_path=records['archive']['path'],
                               target_path=records['target']['path'],bytes=len(payload),sha256=sha)])
        for role,path in receipts.items():proof[role+'_receipt']=dict(path=str(path),sha256=audit.digest(path))
        for role in ('archive','target'):
            proof[role+'_file_set']=[records[role]['path'],str(receipts[role])]
        path=top/'transport.json';path.write_text(json.dumps(proof))
        return path,receipts,proof,roots

    def test_transport_requires_three_real_full_hash_sets_and_target_receipt(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(audit,'OWNER',os.getuid()):
            proof,receipts,_,roots=self.transport_fixture(folder)
            value=audit.verify_transport(proof,receipts['target'])
            self.assertEqual(value['full_read_file_count'],1)
            with self.assertRaisesRegex(ValueError,'verified target'):
                audit.verify_transport(proof,receipts['source'])
            (roots['target']/'extra').write_bytes(b'unlisted')
            with self.assertRaisesRegex(ValueError,'missing/extra'):
                audit.verify_transport(proof,receipts['target'])
            (roots['target']/'extra').unlink()
            (roots['archive']/'payload').write_bytes(b'X'*(roots['archive']/'payload').stat().st_size)
            with self.assertRaisesRegex(ValueError,'fresh source/archive/target'):
                audit.verify_transport(proof,receipts['target'])

    def test_optional_milestone_verifier_keeps_native_default_strictly200(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(audit,'OWNER',os.getuid()):
            path,receipts,proof,_=self.transport_fixture(folder)
            proof['step']=5
            for role,value in receipts.items():
                receipt=json.loads(value.read_text());receipt['step']=5;value.write_text(json.dumps(receipt))
                proof[role+'_receipt']['sha256']=audit.digest(value)
            path.write_text(json.dumps(proof))
            with self.assertRaises(ValueError):audit.verify_transport(path,receipts['target'])
            self.assertEqual(audit.verify_transport(path,receipts['target'],expected_step=5)['step'],5)
            for step in (0,1,199,201,True):
                with self.assertRaises(ValueError):audit.verify_transport(path,receipts['target'],expected_step=step)

    def test_foreign_checkpoint_and_training_provenance_real_bytecopies_pass_transport(self):
        """Real text bytecopies exercise dependencies, never claim CUDA/model validity."""
        from reproduction.hpc3_qprobe.archive_completed_pair import file_records,copy_checked,destination,relocate
        with tempfile.TemporaryDirectory() as folder,patch.object(audit,'OWNER',os.getuid()):
            top=Path(folder).resolve();origin=top/'resumed';origin.mkdir()
            old=top/'original';checkpoint=old/'actor/global_step_5'
            def record(path):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(('explicit synthetic dependency '+str(path.relative_to(top))).encode())
                st=path.stat()
                return dict(path=str(path),inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,
                    full_read_sha256=audit.digest(path))
            files=[record(checkpoint/'model.safetensors'),record(checkpoint/'training_state/question-probe-state.pt'),
                   record(checkpoint/'training_state/head.pt')]
            source_manifest=record(origin/'logs/run-manifest.json')
            original_manifest=record(old/'logs/run-manifest.json')
            foreign={key:record(origin/'proofs'/(key+'.json')) for key in
                ('source_receipt','actual_workers','actual_driver','actual_question_probe','actual_allocation')}
            receipt=dict(run_id='a'*32,step=5,world_size=4,kind='question_probe_pair',
                status='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS',source_base=str(origin),
                checkpoint=str(checkpoint),question_probe_state_path=files[1]['path'],head_path=files[2]['path'],
                files=files,boundary_worker_files=[record(origin/'pair5/rng/rank-0.pt')],
                boundary_driver=record(origin/'pair5/driver.pt'),source_run_manifest=source_manifest,
                reward_probe_manifest=record(origin/'pair5/reward-manifest.json'),reward_files=[],
                checkpoint_origin=dict(foreign,actual_saved_predevelopment_boundary_restored=False,
                    synthetic_fixture_only=True,lost_post_evaluation_rng_reconstructed=False),
                historical_artifacts=dict(training_parts=[dict(base=str(old),first=1,last=5,
                    source_run_manifest=original_manifest)],checkpoints=[],development={},reward_archives=[]),
                actual_cuda_restore_accepted=False)
            source=origin/'receipt.json';source.write_text(json.dumps(receipt))
            # The real publisher traverses all nested records. This was wider
            # than the old hard-coded native audit consumer dependency list.
            original_records={row['path']:row for row in file_records(receipt)}
            self.assertEqual(set(original_records),{row['path'] for row in audit.full_records(receipt)})
            roots={role:top/role for role in ('archive','target')}
            for root in roots.values():root.mkdir()
            payload=[]
            for name,row in sorted(original_records.items()):
                copies={role:destination(name,origin,root,checkpoint) for role,root in roots.items()}
                for path in copies.values():copy_checked(name,path,row['full_read_sha256'],row['bytes'])
                payload.append(dict(source_path=name,archive_path=str(copies['archive']),
                    target_path=str(copies['target']),bytes=row['bytes'],sha256=row['full_read_sha256']))
            receipts={}
            for role,root in roots.items():
                path=root/'receipt.json';moved=relocate(receipt,root);path.write_text(json.dumps(moved))
                self.assertEqual(Path(moved['checkpoint']),root/'restored-checkpoint')
                self.assertEqual(Path(moved['files'][2]['path']).parent,root/'restored-checkpoint/training_state')
                receipts[role]=dict(path=str(path),sha256=audit.digest(path))
            proof=dict(schema_version=1,status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',step=5,
                world_size=4,run_id='a'*32,source_manifest_sha256=source_manifest['full_read_sha256'],
                source_receipt=dict(path=str(source),sha256=audit.digest(source)),
                archive_receipt=receipts['archive'],target_receipt=receipts['target'],files=payload)
            for role in roots:proof[role+'_file_set']=[row[role+'_path'] for row in payload]+[receipts[role]['path']]
            complete=top/'complete.json';complete.write_text(json.dumps(proof))
            passed=audit.verify_transport(complete,receipts['target']['path'],expected_step=5)
            self.assertEqual(passed['full_read_file_count'],len(original_records))
            # A declared omission must fail even when every remaining byte is correct.
            omitted=copy.deepcopy(proof)
            omitted['files']=[row for row in payload if row['source_path']!=foreign['actual_workers']['path']]
            complete.write_text(json.dumps(omitted))
            with self.assertRaisesRegex(ValueError,'payload file set differs'):
                audit.verify_transport(complete,receipts['target']['path'],expected_step=5)
            complete.write_text(json.dumps(proof))
            copied_original=next(row for row in payload if row['source_path']==original_manifest['path'])
            target=Path(copied_original['target_path']);target.write_bytes(b'X'*target.stat().st_size)
            with self.assertRaisesRegex(ValueError,'fresh source/archive/target'):
                audit.verify_transport(complete,receipts['target']['path'],expected_step=5)

    def test_reward_install_reads_relocated_target_files_without_source_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            top=Path(folder).resolve();target=top/'target';target.mkdir()
            output=top/'audit';output.mkdir();files=[]
            for key,name in [('commit','commit_probe.pkl'),('boundary','boundary_probe.pkl'),
                             ('reward_encoder/config.json','config.json')]:
                path=target/name;path.write_bytes(('synthetic-'+key).encode())
                files.append(dict(relative_path=key,path=str(path),full_read_sha256=audit.digest(path)))
            old=dict(run_id='a'*32,version='reward150',round=150,commit='/missing/original/commit_probe.pkl',
                     boundary='/missing/original/boundary_probe.pkl',encoder='/missing/original/encoder',
                     head_sha256={Path(row['path']).name:row['full_read_sha256'] for row in files[:2]},
                     encoder_sha256={'config.json':files[-1]['full_read_sha256']})
            manifest=target/'reward-manifest.json';manifest.write_text(json.dumps(old))
            original=manifest.read_bytes()
            receipt=dict(source_base=str(target),reward_probe_manifest={'path':str(manifest)},
                         reward_files=files,run_id='a'*32,reward_probe_version='reward150',reward_probe_round=150)
            live=audit.install_audit_reward_probe(output,receipt)
            self.assertEqual(live['commit'],str(target/'commit_probe.pkl'))
            self.assertEqual((Path(live['encoder'])/'config.json').resolve(),target/'config.json')
            self.assertEqual(manifest.read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'already exists'):
                audit.install_audit_reward_probe(output,receipt)

    def test_real_saved_pair_read_source_consumes_relocated_archive_and_target(self):
        # The fixture has actual small CPU Adam/head/RNG files and explicitly
        # synthetic HF bytes. It proves consumer path mechanics, never CUDA.
        from search_g1.test_qprobe_checkpoint import PairFixture
        from search_g1.qprobe_checkpoint import read_source
        from reproduction.hpc3_qprobe.archive_completed_pair import copy_checked,destination,relocate
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder,patch.object(audit,'OWNER',os.getuid()):
                top=Path(folder).resolve();fixture=PairFixture(top/'source',outer=200,size=4)
                published=fixture.publish();source=Path(published['path'])
                _,receipt,_,origin,_=read_source(source)
                roots={role:top/role for role in ('archive','target')}
                for root in roots.values():root.mkdir()
                unique={row['path']:row for row in audit.full_records(receipt)};files=[]
                for value,row in sorted(unique.items()):
                    copied={role:destination(value,origin,root) for role,root in roots.items()}
                    for path in copied.values():copy_checked(value,path,row['full_read_sha256'],row['bytes'])
                    files.append(dict(source_path=value,archive_path=str(copied['archive']),
                                      target_path=str(copied['target']),bytes=row['bytes'],sha256=row['full_read_sha256']))
                records={}
                for role,root in roots.items():
                    path=root/'receipt.json';path.write_text(json.dumps(relocate(receipt,root)))
                    consumed=read_source(path)
                    self.assertTrue(consumed[2].is_relative_to(root))
                    self.assertEqual(consumed[1]['head_version'],receipt['head_version'])
                    self.assertFalse(consumed[1]['actual_cuda_restore_accepted'])
                    records[role]=dict(path=str(path),sha256=audit.digest(path))
                proof=dict(schema_version=1,status='QPROBE_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
                    step=200,world_size=4,run_id=receipt['run_id'],source_manifest_sha256=receipt['source_run_manifest']['full_read_sha256'],
                    source_receipt=dict(path=str(source),sha256=audit.digest(source)),
                    archive_receipt=records['archive'],target_receipt=records['target'],files=files)
                for role in roots:
                    proof[role+'_file_set']=[row[role+'_path'] for row in files]+[records[role]['path']]
                path=top/'complete.json';path.write_text(json.dumps(proof))
                actual=audit.verify_transport(path,Path(records['target']['path']))
                self.assertEqual(actual['full_read_file_count'],len(unique))
        finally:torch.set_num_threads(threads)


class ActualCpuHeadReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def fixture(self,folder):
        config=dict(candidate_questions=5120,selected_questions=512,exploration_questions=64,
                    responses_per_question=5,seed=42,head_epochs_per_outer=1,head_batch_size=32)
        cache=SimpleNamespace(feature_dim=4,spec_sha='f'*64)
        head=OnlineQProbe(4,QProbeConfig.from_mapping(config),feature_spec_sha=cache.spec_sha)
        for outer in range(1,201):
            head.update_completed_outer(outer,[dict(qid='q'+str(outer),question_hash='e'*64,
                complete=True,selection_source='random_exploration',strict_em=[1,1,0,0,0])],
                torch.tensor([[1.,2.,3.,4.]]),policy_snapshot_id='synthetic-policy-'+str(outer),
                sampling_contract_sha='sampling',strict_scoring_contract_sha='strict',retriever_version='index',
                run_id='a'*32,attempt_id=str(outer))
        runtime=QProbeRuntime(config,folder,feature_cache=cache,head=head)
        runtime.completed_attempts=set(head.completed_attempts.values())
        data=dict(candidate_order=list(range(8)),candidate_cursor=4,epoch=2,consumed_source_rows=1024000,
                  pending_source=None,pending_offset=0)
        state=runtime.save_checkpoint(Path(folder)/'state',200,data,'reward150','a'*32+':200')
        progress=dict(completed_outer=200,cumulative_writes=3600,attempt_index=202,aborted_attempts=2)
        boundary=atomic_save(Path(folder)/'boundary.pt',dict(candidate_stream=data,rng=driver_rng(),attempt_progress=progress))
        loaded=QProbeRuntime(config,folder,feature_cache=cache)
        loaded.restore(state['path'],'a'*32+':200')
        receipt=dict(question_probe_state_path=state['path'],head_path=state['head']['path'],
                     boundary_driver=boundary,head_version=loaded.version,feature_spec_sha=cache.spec_sha,run_id='a'*32)
        trainer=SimpleNamespace(_restored_attempt_progress=copy.deepcopy(progress),global_steps=200)
        stream=SimpleNamespace(state=lambda:data)
        return loaded,receipt,trainer,stream,data

    def test_real_head_adam_rng_replay_selector_and_data_readbacks(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,receipt,trainer,stream,_=self.fixture(folder)
            first=audit.driver_readback(trainer,runtime,stream,receipt,'first')
            second=audit.driver_readback(trainer,runtime,stream,receipt,'second')
            self.assertEqual(first['head_sha256'],second['head_sha256'])
            self.assertEqual(first['head_update_count'],200)
            self.assertIsNone(first['head_scheduler'])
            self.assertTrue(first['actual_candidate_stream_readback_equal'])

    def test_actual_head_adam_rng_replay_selector_and_stream_mutation_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime,receipt,trainer,stream,data=self.fixture(folder)
            snapshot=receipt['question_probe_state_path']
            mutations=[lambda:runtime.head.rng.set_state(torch.Generator().manual_seed(9).get_state()),
                       lambda:runtime.selection_rng.random(),
                       lambda:runtime.head.replay_labels[0].update(qid='changed'),
                       lambda:runtime.head.optimizer.state[next(iter(runtime.head.optimizer.state))]['exp_avg'].add_(1),
                       lambda:data.update(candidate_cursor=5)]
            for mutate in mutations:
                runtime.restore(snapshot,'a'*32+':200');data['candidate_cursor']=4
                mutate()
                with self.assertRaises(AssertionError):
                    audit.driver_readback(trainer,runtime,stream,receipt,'first')


class WorkerBoundaryMechanismTests(unittest.TestCase):
    def fixture(self):
        receipt,_=pair();uuids=['GPU-'+str(uuid.UUID(int=i+1)) for i in range(4)]
        allocation=dict(slurm_job_id='12345',actual_cuda_runtime_uuids=uuids,accepted=True,
                        training_world_size=4,run_id=receipt['run_id'])
        first=[dict(rank=i,phase='first',slurm_job_id='12345',actual_cuda_uuid=uuids[i],
                    process_identity=dict(pid=i+10,start_ticks=i+20,uid=14090,sid=10),
                    counters=dict(completed_outer=200,actual_optimizer_writes=3600),
                    actual_adam_scheduler_readback_equal=True,actual_rng_readback_equal=True,
                    model_shard_sha256='model',optimizer_sha256='adam',scheduler_sha256='lr',rng_sha256='rng',checked_unix=1.)
               for i in range(4)]
        second=copy.deepcopy(first)
        for value in second:value.update(phase='second',checked_unix=2.)
        driver=dict(phase='first',completed_outer=200,head_update_count=200,head_sha256='head',checked_unix=1.)
        later=dict(driver,phase='second',checked_unix=2.)
        return first,second,allocation,receipt,driver,later

    def test_all_four_two_readbacks_required_and_no_write_or_uuid_drift(self):
        values=self.fixture();audit.verify_two_boundaries(*values)
        moving=copy.deepcopy(values);moving[1][0]['process_identity'].update(state='S',command='changed Ray title')
        audit.verify_two_boundaries(*moving)
        for change in ('rank','uuid','owner','adam','model','scheduler','write','head'):
            first,second,allocation,receipt,driver,later=copy.deepcopy(values)
            if change=='rank':second.pop()
            elif change=='uuid':second[0]['actual_cuda_uuid']='wrong'
            elif change=='owner':second[0]['process_identity']['uid']=21930
            elif change=='adam':second[0]['optimizer_sha256']='changed'
            elif change=='model':second[0]['model_shard_sha256']='changed'
            elif change=='scheduler':second[0]['scheduler_sha256']='changed'
            elif change=='write':second[0]['counters']['actual_optimizer_writes']+=1
            elif change=='head':later['head_sha256']='changed'
            with self.subTest(change=change),self.assertRaises((ValueError,AssertionError)):
                audit.verify_two_boundaries(first,second,allocation,receipt,driver,later)


if __name__=='__main__':unittest.main()
