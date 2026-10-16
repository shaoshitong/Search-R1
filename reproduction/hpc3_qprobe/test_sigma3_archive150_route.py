"""Explicit150 source separation, original5f100 and complete storage gates."""
from contextlib import ExitStack
import base64
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.hpc3_qprobe import independent_goal_acceptance_sigma3 as reader
from reproduction.hpc3_qprobe import independent_goal_control_sigma3 as holder
from reproduction.hpc3_qprobe import recovery_lineage_stage as lineage
from reproduction.hpc3_qprobe.test_independent_goal_acceptance_sigma3 import write
from reproduction.hpc3_qprobe.test_independent_goal_control_sigma3 import output_fixture

PINS=(reader.ARCHIVE5_RECOVERY_SOURCE,'f979d53cdb0d348e87c5877d84403252e85cd784',
      reader.ARCHIVE50_SECOND_RECOVERY_SOURCE,reader.ARCHIVE100_FOURTH_RECOVERY_SOURCE,
      reader.ARCHIVE150_FIRST_RECOVERY_SOURCE)


def actual_chain():
    fixture=json.loads(Path(__file__).with_name('archive150_first_failure_fixture.json').read_bytes())
    folder=str(reader.ROOT/'followthrough/721485-archive100-recovery-fourth-20261016')
    old=json.loads(base64.b64decode(fixture[folder]['claim.json']['raw_base64']))['recovery']
    latest=dict(old,source={'commit':PINS[-1]},prior_archive100_fourth_recovery=old,
        storage_protocol='actual-full-owner-partition-du-v1',inherited_archive100=dict(source=old['source'],
            controller=dict(pid=560250,start_ticks=81264116,sid=560250,uid=reader.OWNER),
            sha256='06e7a3a265ef3a6184ba6f71eabab69ab12b40a29701137dccfc1287395a6d39',
            transport_sha256='5f113a12cd245201e7a842d9d59035909e3ed6f92c7793b04c19088ee49b528e'))
    return old,latest


class Explicit150(unittest.TestCase):
    def test_distinct_route_scope_and_cli_preserve_both100_and150_pins(self):
        name,folder=reader.followthrough_route(PINS[-1],*PINS)
        self.assertEqual((name,folder.name),('recover_sigma3_archive150_first','721485-archive150-recovery-first-20261016'))
        self.assertNotEqual(holder.scope(*PINS),holder.scope(*PINS[:-1]))
        self.assertEqual(holder.recovery_args(*PINS)[-4:],['--archive100-recovery-commit',PINS[3],'--archive150-recovery-commit',PINS[4]])
        for i in range(len(PINS)):
            bad=list(PINS);bad[i]='a'*40
            with self.subTest(i=i),self.assertRaises(ValueError):reader.followthrough_route(PINS[-1],*bad)
        with self.assertRaises(ValueError):reader.followthrough_route(PINS[3],*PINS)

    def test_separate_original5f_consumption_and_inheritance_cannot_be_retagged(self):
        old,latest=actual_chain()
        with patch.object(lineage,'consume',return_value=old) as consume:
            self.assertEqual(reader.prior_archive150_evidence(latest,PINS[-1],PINS[3]),old)
            consume.assert_called_once_with(PINS[3],'recover_sigma3_archive100_fourth')
        for key in ('prior_archive100_fourth_recovery','prior_archive100_third_recovery','observer_recovery','inherited_archive100'):
            bad=copy.deepcopy(latest);bad[key]={}
            with self.subTest(key=key),patch.object(lineage,'consume',return_value=old),self.assertRaises(ValueError):
                reader.prior_archive150_evidence(bad,PINS[-1],PINS[3])
        bad=copy.deepcopy(latest);bad['inherited_archive100']['source']=latest['source']
        with patch.object(lineage,'consume',return_value=old),self.assertRaisesRegex(ValueError,'retagged'):
            reader.prior_archive150_evidence(bad,PINS[-1],PINS[3])

    def test_native_source_is_new150_follower_science_is_unchanged(self):
        run={'source_sha256':{'search_g1/science.py':'a'*64}}
        native={'audit_commit':PINS[-1],'audit_id':'native'}
        audit={'source_commit':PINS[-1],'run_id':reader.RUN,'final_restore_audit':{'audit_id':'native'},
               'source_sha256':run['source_sha256']}
        reader.native_source_gate(run,native,audit,{'files':run['source_sha256']},PINS[-1],*PINS)
        with self.assertRaisesRegex(ValueError,'native audit source'):
            reader.native_source_gate(run,dict(native,audit_commit=PINS[3]),audit,{'files':run['source_sha256']},PINS[-1],*PINS)

    def test_missing200_refuses_before_source_traversal_creation_or_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();base=root/'run';(base/'logs').mkdir(parents=True)
            (base/'logs/actual-updates.jsonl').write_text('{"step":154}\n');before=sorted(map(str,root.rglob('*')))
            with patch.object(holder,'DIRECTORY',root/'new'),patch.object(holder,'caps'),\
                 patch.object(reader,'BASE',base),patch.object(reader.shared,'OWNER',os.getuid()),\
                 patch.object(holder.ctl,'source') as source,patch.object(holder.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'fresh1..200'):
                    holder.start('a'*40,followthrough_commit=PINS[-1],recovery_commit=PINS[0],observer_recovery_commit=PINS[1],
                        archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3],archive150_recovery_commit=PINS[4])
                source.assert_not_called();spawn.assert_not_called()
            self.assertEqual(before,sorted(map(str,root.rglob('*'))))

    def output(self):
        commit='a'*40;_,admission,proof=output_fixture(commit);old,latest=actual_chain()
        for value in (admission,proof):
            value.update(recovery_commit=PINS[0],observer_recovery_commit=PINS[1],archive_recovery_commit=PINS[2],
                         archive100_recovery_commit=PINS[3],archive150_recovery_commit=PINS[4])
        admission.update(followthrough_source=latest['source'],raw_source={'commit':PINS[1]},archive150_recovery=latest,
            archive100_recovery=old,archive100_prior_recovery=old['prior_archive100_recovery'],
            archive100_second_prior_recovery=old['prior_archive100_second_recovery'],
            archive100_third_prior_recovery=old['prior_archive100_third_recovery'],archive50_recovery=old['prior_archive50_second_recovery'],
            archive50_prior_recovery=old['prior_archive50_second_recovery']['prior_archive50_recovery'],
            archive5_recovery=old['legacy_archive5_recovery'],observer_recovery=old['observer_recovery'])
        proof['followthrough_commit']=PINS[-1];follow=proof['actual_followthrough_full_CPU_consumer'];follow['source']=latest['source']
        for key in ('archive5_recovery','observer_recovery','archive50_recovery','archive50_prior_recovery','archive100_recovery',
                    'archive100_prior_recovery','archive100_second_prior_recovery','archive100_third_prior_recovery','archive150_recovery'):
            follow[key]=copy.deepcopy(admission[key])
        proof['independent_all_normal_return_raw']['source']=admission['raw_source']
        proof['actual_archive_and_submit_held_children']={str(step):{'owner_partition_storage':dict(
            protocol='actual-full-owner-partition-du-v1',complete_partition_coverage=True,
            exact_usage_or_hardquota_claimed=False,allocated_upper_bound_bytes=1000)} for step in (150,200)}
        return commit,admission,proof

    def test_final_output_needs_separate150100_and_both_complete_partitions(self):
        commit,admission,proof=self.output();holder.accept_output(proof,commit,PINS[-1],admission,*PINS)
        attacks=[('archive150_recovery_commit',PINS[3]),('actual_native_two_readbacks_reconsumed',False),('update201_performed',True)]
        for key,value in attacks:
            bad=copy.deepcopy(proof);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        for step in ('150','200'):
            bad=copy.deepcopy(proof);bad['actual_archive_and_submit_held_children'][step]['owner_partition_storage']={}
            with self.subTest(step=step),self.assertRaisesRegex(ValueError,'complete owner partition'):
                holder.accept_output(bad,commit,PINS[-1],admission,*PINS)
        bad=copy.deepcopy(proof);bad['actual_followthrough_full_CPU_consumer']['archive150_recovery']={}
        with self.assertRaises(ValueError):holder.accept_output(bad,commit,PINS[-1],admission,*PINS)


class ActualHeld150(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.stack=ExitStack();self.addCleanup(self.stack.close)
        for target,name,value in ((reader,'ROOT',self.root),(reader.ctl,'ROOT',self.root),(reader,'OWNER',os.getuid()),
                                  (reader.shared,'OWNER',os.getuid()),(reader.ctl,'live',lambda _:False)):
            self.stack.enter_context(patch.object(target,name,value))
        self.producer={'run_id':reader.RUN};self.pair={'sha256':'b'*64};self.folders={};self.inherited={};self.storage={};readbacks={};transports={}
        self.sources={5:{'commit':PINS[0]},50:{'commit':PINS[2]},100:{'commit':PINS[3]},150:{'commit':PINS[4]}}
        self.controllers={step:dict(pid=10*(i+1),start_ticks=100*(i+1),sid=10*(i+1),uid=os.getuid())
                          for i,step in enumerate((5,50,100,150))}
        for index,step in enumerate((5,50,100,150,200,None)):
            final=step is None;key=step if step in (5,50,100) else 150;source=self.sources[key];controller=self.controllers[key]
            folder=self.root/'final-control'/reader.RUN/('audit-controller' if final else 'archive-'+str(step));self.folders[step]=folder
            token='%032x'%(index+1);role='detached_cpu_final_audit_submission' if final else 'detached_cpu_pair_archive'
            claim=dict(schema_version=1,role=role,token=token,producer=self.producer,source=source,controller=controller,
                created_unix=4,training_permitted=False,direct_gpu_device_access_permitted=False,
                native_gpu_audit_submission_permitted=final,pair=self.pair)
            if not final:claim['step']=step
            if step==100:claim['storage_wait_protocol']='actual-full-du-repair-v1'
            if step in (150,200):claim['storage_wait_protocol']='actual-full-owner-partition-du-v1'
            claim['command']=reader.ctl.command(self.root/'source'/source['commit'],'audit-worker' if final else 'archive-worker',folder/'submit-claim.json',token)
            claim_file=write(folder/'submit-claim.json',claim);binding={'token':token,'claim_sha256':claim_file['sha256']}
            runner=dict(pid=200+index,start_ticks=2000+index,sid=200+index,uid=os.getuid())
            child=dict(pid=300+index,start_ticks=3000+index,sid=runner['sid'],uid=os.getuid())
            write(folder/'runner-binding.json',dict(binding,actual_process=runner,controller=controller))
            write(folder/'task-binding.json',dict(binding,actual_process=child,supervisor=runner))
            task=dict(binding,status='CPU_FINAL_SUBMIT_CONTROLLER_RETURNED' if final else 'CPU_PAIR_SOURCE_ARCHIVE_TARGET_FULL_SHA_PASS',
                actual_process=child,training_performed=False,direct_gpu_device_access_performed=False,actor_model_loaded=False,
                native_gpu_audit_submission_attempted=final,completed_unix=5)
            if final:task['submission']={'status':'ACTUAL_SLURM_SUBMITTED'}
            else:
                path=self.root/'archives'/reader.RUN/('global_step_'+str(step))/'complete.json'
                transport=dict(run_id=reader.RUN,step=step,world_size=4,files=[{'bytes':10},{'bytes':20}],
                    **{role+'_receipt':{'path':'/fixture/'+str(step)+'/'+role,'sha256':'a'*64} for role in ('source','archive','target')})
                proof=write(path,transport);task['archive']={'path':str(path),'sha256':proof['sha256']}
                if step in (5,50,100):
                    transports[str(step)]=proof['sha256'];readbacks[str(step)]=dict(path=str(path),sha256=proof['sha256'],
                        step=step,full_read_file_count=2,full_read_bytes=30,verified_by_uid=os.getuid(),checked_unix=2.5,
                        **{role+'_receipt':transport[role+'_receipt'] for role in ('source','archive','target')})
                else:
                    scan=write(folder/'scan-complete.json',dict(completed_unix=4.5))
                    self.storage[step]=dict(protocol='actual-full-owner-partition-du-v1',complete_partition_coverage=True,
                        exact_usage_or_hardquota_claimed=False,allocated_upper_bound_bytes=1000,
                        complete={k:scan[k] for k in ('path','sha256','bytes')})
                    write(path.parent/'claim.json',dict(source_receipt=transport['source_receipt'],run_id=reader.RUN,step=step,
                        world_size=4,uid=os.getuid(),pid=child['pid'],started_unix=4.6,
                        storage_semantics='complete_two_owner_allocated_upper_bound',allocated_upper_bound_bytes=1000,
                        measured_allocated_bytes=1000,required_copy_bytes=60+2**30))
            task_file=write(folder/'task-result.json',task)
            result_file=write(folder/'worker-result.json',dict(binding,actual_task_birth=child,actual_supervisor=runner,role=role,
                actual_task_exit_code=0,status='ACTUAL_CPU_FINAL_SUBMIT_CHILD_COMPLETED0' if final else 'ACTUAL_CPU_ARCHIVE_CHILD_COMPLETED0',
                task_result_sha256=task_file['sha256'],finished_unix=6))
            if step in (5,50,100):self.inherited[str(step)]=dict(path=claim_file['path'],sha256=claim_file['sha256'],source=source,controller=controller,
                worker_result_sha256=result_file['sha256'],task_result_sha256=task_file['sha256'],transport_sha256=proof['sha256'])
        self.follow_folder=self.root/'followthrough/new150'
        write(self.follow_folder/'inherited-archives-full-sha.json',dict(source_commit=PINS[-1],run_id=reader.RUN,
            transports=transports,readbacks=readbacks,checked_unix=3,training_actions=0,permission_changes=0))
        hundred={'inherited_archive50':self.inherited['50']};latest={'inherited_archive100':self.inherited['100']}
        self.follow=SimpleNamespace(DIRECTORY=self.follow_folder,consume=lambda _:latest,control=SimpleNamespace(
            pair=lambda *_:(None,self.pair),inspect_audit=lambda _:dict(status='ACTUAL_SLURM_SUBMITTED',job='800001',audit_base=str(self.root/'native'))))
        legacy=SimpleNamespace(service=lambda:SimpleNamespace(inherited=lambda:{'existing_archive_claims':{'5':self.inherited['5']}}))
        partition=SimpleNamespace(PROTOCOL='actual-full-owner-partition-du-v1',consume=lambda folder:copy.deepcopy(self.storage[int(folder.name.split('-')[-1])]))
        self.stack.enter_context(patch.object(reader,'load_fixed',side_effect=lambda commit,name,*_:(partition if name=='owner_partition_storage' else legacy,{})))
        self.stack.enter_context(patch.object(reader,'publication',side_effect=lambda commit:(None,next(s for s in self.sources.values() if s['commit']==commit))))
        self.stack.enter_context(patch.object(reader,'prior_archive150_evidence',return_value=hundred))
        self.stack.enter_context(patch.object(reader,'archive50_recovery_evidence'))

    def consume(self):return reader.held_archive_evidence({},self.producer,self.follow,self.sources[150],self.controllers[150],
        observer_recovery_commit=PINS[1],archive_recovery_commit=PINS[2],archive100_recovery_commit=PINS[3],archive150_recovery_commit=PINS[4])

    def test_all_five_actual_held0_original100_and_new150200_sources_accept(self):
        value,_=self.consume();self.assertEqual(set(value),{'5','50','100','150','200','native_submit'})
        self.assertEqual(value['150']['owner_partition_storage'],self.storage[150])

    def test_original100_retag_and_missing200_wait_or_partition_refuse(self):
        path=self.folders[100]/'submit-claim.json';raw=path.read_bytes();bad=json.loads(raw);bad['source']=self.sources[150];write(path,bad)
        with self.assertRaisesRegex(ValueError,'original archive claim changed'):self.consume()
        path.write_bytes(raw)
        path=self.folders[200]/'worker-result.json';raw=path.read_bytes();path.unlink()
        with self.assertRaises(FileNotFoundError):self.consume()
        path.write_bytes(raw);self.storage[200]['complete_partition_coverage']=False
        with self.assertRaisesRegex(ValueError,'admission/budget'):self.consume()

    def test_over_budget_or_missing_initial_readback_cannot_pass(self):
        path=self.root/'archives'/reader.RUN/'global_step_150/claim.json';raw=path.read_bytes()
        bad=json.loads(raw);bad['required_copy_bytes']=2*10**12;write(path,bad)
        with self.assertRaisesRegex(ValueError,'admission/budget'):self.consume()
        path.write_bytes(raw)
        path=self.follow_folder/'inherited-archives-full-sha.json';bad=json.loads(path.read_bytes());bad['readbacks'].pop('100');write(path,bad)
        with self.assertRaisesRegex(ValueError,'prefix readback missing'):self.consume()


if __name__=='__main__':unittest.main()
